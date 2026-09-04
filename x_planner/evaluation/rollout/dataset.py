"""Memory V4 dataset/runtime registrations and mixed-task config generation."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import zlib
from pathlib import Path
from typing import Any

import yaml

from x2robot_dataset_v2.processors.text.base import register_text_processor
from x2robot_dataset_v2.processors.text.multimodal_jsonl_qwen3_5_text_processor import (
    QWEN_DIALOGUES_KEY,
    MultimodalJsonlQwen3_5TextProcessor,
)
from x2robot_dataset_v2.processors.vision.multimodal_jsonl_vision_processor import (
    _resolve_multimodal_sample_idx,
)
from x2robot_dataset_v2.readers.multimodal_jsonl_reader import load_indexed_jsonl_item
from x2robot_dataset_v2.utils.multimodal_utils import process_dialogue

from x_planner.data.discovery.common.atomic import atomic_write
from x_planner.data.context import dataset as _dataset  # registers stable vision
from x_planner.data.context.common import file_sha256, write_json
from x_planner.data.context.dataset import allocate_mix_counts, exact_mix_budget
from .prompt import render_user
from .schema import SNAPSHOT_SCHEMA_VERSION, dumps_assistant


TASKS = ("continuous", "initial_plan", "terminal")


@register_text_processor("v10_memory_v4_qwen3_5")
class MemoryV4TextProcessor(MultimodalJsonlQwen3_5TextProcessor):
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        episode = data.get("_episode")
        if episode is None:
            return data
        frame_local_idx = data.get("_frame_local_idx", 0)
        sample_idx = _resolve_multimodal_sample_idx(episode, frame_local_idx)
        item = load_indexed_jsonl_item(episode.path, sample_idx)
        raw = item.get("v4_sample")
        if not isinstance(raw, dict):
            raise ValueError("Memory V4 row is missing v4_sample")
        sample = copy.deepcopy(raw)
        kept = data.get("_v10_kept_image_positions")
        if kept is not None:
            positions = tuple(int(value) for value in kept)
            if not positions or len(set(positions)) != len(positions):
                raise ValueError("Memory V4 retained image positions are invalid")
            images = list(sample["images"])
            if min(positions) < 0 or max(positions) >= len(images):
                raise IndexError("Memory V4 retained image position is out of range")
            sample["images"] = [images[position] for position in positions]
        task_type = str(sample["task_type"])
        instruction = str(sample["task_instruction"])
        dialogues = [
            {"role": "user", "text": render_user(sample)},
            {
                "role": "assistant",
                "text": dumps_assistant(
                    sample["target"],
                    str(sample["profile"]),
                    task_type,
                    instruction=instruction,
                    is_terminal_window=bool(sample.get("is_terminal_window", False)),
                ),
            },
        ]
        num_images = len(sample["images"])
        rng = data.get("_rng") or random.Random(
            zlib.crc32(f"{episode.path}|{sample_idx}".encode())
        )
        processed = process_dialogue(
            dialogues, seed=rng.getrandbits(64), num_images=num_images
        )
        data[QWEN_DIALOGUES_KEY] = json.dumps(processed, ensure_ascii=False)
        image_tokens = sum(
            (turn.get("text", "") or "").count("<image>") for turn in processed
        )
        if image_tokens != num_images:
            raise ValueError(
                f"Memory V4 image placeholder mismatch: {image_tokens} != {num_images}"
            )
        data["_expected_image_count"] = image_tokens
        return data


def prepare(
    snapshot: Path,
    work_dir: Path,
    model_path: Path,
    *,
    max_length: int,
    tasks: tuple[str, ...],
    allow_small: bool,
    max_budget: int | None = None,
    resize_mode: str = "B_auto_near_640",
) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    work_dir = work_dir.resolve()
    model_path = model_path.resolve()
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("not a Memory V4 snapshot")
    if manifest.get("complete") is not True or manifest.get("partial_inputs") is not False:
        raise ValueError("training requires a complete Memory V4 snapshot")
    unknown = sorted(set(tasks) - set(TASKS))
    if unknown or not tasks:
        raise ValueError(f"invalid V4 tasks: {tasks}; unknown={unknown}")
    if not (
        (model_path / "model.safetensors").is_file()
        or (model_path / "model.safetensors.index.json").is_file()
    ):
        raise FileNotFoundError(f"incomplete model directory: {model_path}")
    weights_all = {
        key: float(value) for key, value in manifest["train_mix"]["weights"].items()
    }
    weights = {task: weights_all[task] for task in tasks}
    normalizer = sum(weights.values())
    weights = {task: value / normalizer for task, value in weights.items()}
    counts = {
        task: int(manifest["datasets"][f"{task}_train"]["samples"])
        for task in tasks
    }
    budget = exact_mix_budget(counts, weights)
    if max_budget is not None:
        if max_budget <= 0:
            raise ValueError("max_budget must be positive")
        budget = min(budget, int(max_budget))
    allocated = allocate_mix_counts(budget, weights)
    if any(allocated[task] > counts[task] for task in tasks):
        raise ValueError("allocated V4 mix exceeds available data")
    if not allow_small and any(value < 100 for value in counts.values()):
        raise ValueError("formal V4 task requires at least 100 samples")
    work_dir.mkdir(parents=True, exist_ok=True)
    step_state = work_dir / "v10_step_state.json"
    quarantine = work_dir / "media_quarantine.json"
    write_json(str(step_state), {"global_step": 0, "memory_noise_probability": 0.0})
    if not quarantine.exists():
        write_json(
            str(quarantine),
            {"schema_version": "v10_media_quarantine_v1", "sample_ids": []},
        )
    sources = [
        {
            "name": task,
            "source_type": "multimodal",
            "paths": [{
                "path": str(snapshot / "datasets" / task / "train"),
                "episode_type": "x2_multimodal",
                "task_name": f"memory_v4_{task}",
            }],
        }
        for task in tasks
    ]
    resize = json.loads(
        (snapshot / "snapshot_metadata.json").read_text(encoding="utf-8")
    )["resize_policy"]
    if resize_mode not in {"A_current", "B_auto_near_640", "C_original_capped"}:
        raise ValueError(f"unknown resolution mode: {resize_mode}")
    vision_type = "v10_resilient_video_frame" if resize_mode == "A_current" else "v10_memory_video_frame"
    vision_params: dict[str, Any] = {
        "image_factor": int(resize["factor"]),
        "min_pixels": 1024,
        "max_pixels": int(resize["pixel_cap"]),
        "max_pixels_split_by_images": resize_mode == "A_current",
        "decoder_backend": "av",
        "quarantine_path": str(quarantine),
        "media_failure_report_path": str(work_dir / "media_failures_runtime.jsonl"),
        "drop_failed_views": True,
    }
    if resize_mode != "A_current":
        vision_params.update({
            "pixel_cap": int(resize["pixel_cap"]),
            "split_pixel_cap_across_images": resize_mode == "C_original_capped",
            "resize_policy_id": (
                str(resize["policy_id"])
                if resize_mode == "B_auto_near_640"
                else "original_resolution_total_pixel_cap_v1"
            ),
            "target_long_edge": (
                int(resize["target_long_edge"])
                if resize_mode == "B_auto_near_640"
                else 1_000_000_000
            ),
        })
    data_config = {
        "dataset": {
            "train_test_split": 1.0,
            "multimodal_chunk_size": 200,
            "bad_sample_tolerance": {
                "enabled": True,
                "report_path": str(work_dir / "bad_samples_runtime.jsonl"),
                "include_traceback": True,
                "max_traceback_chars": 8000,
            },
            "sampler": {
                "seed": 42,
                "type": "default",
                "batch_size": 1,
                "task_balance": {
                    "type": "static_count",
                    "params": {
                        "counts_per_task": {
                            f"memory_v4_{task}": allocated[task] for task in tasks
                        }
                    },
                    "unit": "frames",
                },
                "task_balance_report_path": str(work_dir / "task_balance_report.json"),
            },
            "pipeline": ["vision", "text", "metadata"],
            "cache": {"enabled": False, "dir": str(work_dir / "dataset_cache")},
            "processors": {
                "vision": {"type": vision_type, "params": vision_params},
                "text": {"type": "v10_memory_v4_qwen3_5", "params": {}},
                "epilogue": {
                    "type": "v10_multimodal_qwen3_5",
                    "params": {
                        "processor_path": str(model_path),
                        "max_seq_length": max_length,
                        "padding_side": "right",
                        "packing": False,
                    },
                },
            },
            "sources": sources,
        }
    }
    config_path = work_dir / "data.yml"
    with atomic_write(str(config_path)) as handle:
        yaml.safe_dump(data_config, handle, allow_unicode=True, sort_keys=False)
    summary = {
        "schema_version": "memory_v4_prepare_v1",
        "snapshot": str(snapshot),
        "manifest_digest": manifest["content_digest"],
        "manifest_file_sha256": file_sha256(snapshot / "manifest.json"),
        "data_config": str(config_path),
        "data_config_digest": file_sha256(config_path),
        "step_state": str(step_state),
        "tasks": list(tasks),
        "available_counts": counts,
        "normalized_weights": weights,
        "total_budget": budget,
        "expected_counts": allocated,
        "allow_small": bool(allow_small),
        "resize_mode": resize_mode,
        "batch_6_required": True,
        "assistant_outputs_l3": False,
    }
    write_json(str(work_dir / "prepare_summary.json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.environ.get("XPLANNER_MODEL_PATH", "/path/to/Qwen3.5-9B")),
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--allow-small", action="store_true")
    parser.add_argument("--max-budget", type=int)
    parser.add_argument(
        "--resize-mode",
        default="B_auto_near_640",
        choices=("A_current", "B_auto_near_640", "C_original_capped"),
    )
    args = parser.parse_args()
    tasks = tuple(value.strip() for value in args.tasks.split(",") if value.strip())
    result = prepare(
        args.snapshot,
        args.work_dir,
        args.model_path,
        max_length=args.max_length,
        tasks=tasks,
        allow_small=args.allow_small,
        max_budget=args.max_budget,
        resize_mode=args.resize_mode,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
