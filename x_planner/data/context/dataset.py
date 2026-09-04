"""Memory V3 dataset/runtime registrations and mixed-task config generation."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import zlib
from pathlib import Path
from typing import Any, Mapping

import yaml

from x_planner.data.video_frames import iter_multimodal_image_refs
from x2robot_dataset_v2.processors.text.base import register_text_processor
from x2robot_dataset_v2.processors.text.multimodal_jsonl_qwen3_5_text_processor import (
    QWEN_DIALOGUES_KEY,
    MultimodalJsonlQwen3_5TextProcessor,
)
from x2robot_dataset_v2.processors.vision.base import register_vision_processor
from x2robot_dataset_v2.processors.vision.multimodal_jsonl_vision_processor import (
    VIDEO_META_KEY,
    VIDEO_OBSERVATIONS_KEY,
    _resolve_multimodal_sample_idx,
)
from x2robot_dataset_v2.readers.multimodal_jsonl_reader import load_indexed_jsonl_item
from x2robot_dataset_v2.utils.multimodal_utils import process_dialogue

from ..pipeline import epilogue as _v10_epilogue  # noqa: F401
from ..pipeline.vision import (
    V10QuarantinedSampleError,
    V10ResilientVideoFrameVisionProcessor,
    _append_jsonl,
    decode_refs_by_complete_view,
    load_quarantined_sample_ids,
)
from ..discovery.common.atomic import atomic_write
from .common import file_sha256, load_config, write_json
from .prompt import render_user
from .schema import dumps_assistant


PACKAGE_ROOT = Path(__file__).resolve().parent
TASKS = ("continuous", "initial_plan", "terminal")


def resize_policy_identifier(resize: Mapping[str, Any]) -> str:
    value = str(resize.get("policy_id") or "").strip()
    if not value:
        raise ValueError("Snapshot resize policy is missing canonical policy_id")
    return value


def auto_near_640_dimensions(
    height: int,
    width: int,
    *,
    target_long_edge: int = 640,
    factor: int = 32,
    pixel_cap: int = 589824,
) -> tuple[int, int]:
    """Aspect-preserving, factor-aligned resize that never enlarges an axis."""
    if height < factor or width < factor:
        raise ValueError(
            f"cannot satisfy factor={factor} without upsampling {width}x{height}"
        )
    scale = min(1.0, target_long_edge / max(height, width))
    target_h = height * scale
    target_w = width * scale
    if target_h * target_w > pixel_cap:
        cap_scale = math.sqrt(pixel_cap / (target_h * target_w))
        target_h *= cap_scale
        target_w *= cap_scale
    resized_h = max(factor, math.floor(target_h / factor) * factor)
    resized_w = max(factor, math.floor(target_w / factor) * factor)
    if resized_h > height or resized_w > width:
        raise ValueError(
            f"resize would upsample {width}x{height} to {resized_w}x{resized_h}"
        )
    if resized_h * resized_w > pixel_cap:
        raise ValueError("factor-aligned resize exceeds pixel cap")
    return resized_h, resized_w


@register_vision_processor("v10_memory_video_frame")
class MemoryV3VisionProcessor(V10ResilientVideoFrameVisionProcessor):
    def __init__(
        self,
        *args: Any,
        target_long_edge: int = 640,
        pixel_cap: int | None = None,
        split_pixel_cap_across_images: bool = False,
        resize_policy_id: str = "auto_near_640_no_upscale_v1",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.target_long_edge = int(target_long_edge)
        self.pixel_cap = int(pixel_cap or self.max_pixels)
        self.split_pixel_cap_across_images = bool(split_pixel_cap_across_images)
        self.resize_policy_id = str(resize_policy_id)

    def _compute_resize_dims(
        self, orig_height: int, orig_width: int, cam_type: Any
    ) -> tuple[int, int]:
        return auto_near_640_dimensions(
            orig_height,
            orig_width,
            target_long_edge=self.target_long_edge,
            factor=self.image_factor,
            pixel_cap=min(self.pixel_cap, self.max_pixels),
        )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        episode = data.get("_episode")
        if episode is None:
            return data
        frame_local_idx = data.get("_frame_local_idx", 0)
        sample_idx = _resolve_multimodal_sample_idx(episode, frame_local_idx)
        data["frame_idx"] = sample_idx
        data["uid"] = episode.path
        item = load_indexed_jsonl_item(episode.path, sample_idx)
        raw_sample = item.get("v3_sample")
        sample_id = str(
            item.get("data_id")
            or (raw_sample.get("sample_key") if isinstance(raw_sample, dict) else "")
            or f"{episode.path}:{sample_idx}"
        )
        if sample_id in self._quarantined_ids():
            self._report_once(
                sample_id=sample_id,
                episode_path=str(episode.path),
                event="quarantined_sample",
            )
            raise V10QuarantinedSampleError(f"Memory V3 sample is quarantined: {sample_id}")
        refs = list(iter_multimodal_image_refs(item.get("image")))
        positions, images, failures = decode_refs_by_complete_view(
            refs, episode.path, self.load_image_refs
        )
        for view, error in failures.items():
            self._report_once(
                sample_id=sample_id,
                episode_path=str(episode.path),
                event="dropped_view",
                view=view,
                error=error,
            )
        if failures and not self.drop_failed_views:
            raise RuntimeError(f"Memory V3 view decode failed: {failures}")
        if not images:
            raise RuntimeError(f"Memory V3 sample has no decodable view: {sample_id}; {failures}")
        data["_v10_kept_image_positions"] = positions
        data["_v10_dropped_views"] = sorted(failures)
        is_train = data.get("_is_train", True)
        rng = data.get("_augmentation_rng") or data.get("_rng")
        self._aug_seed = rng.getrandbits(32) if rng is not None else None
        saved_max_pixels = self.max_pixels
        if self.split_pixel_cap_across_images:
            self.max_pixels = max(
                self.image_factor * self.image_factor,
                saved_max_pixels // len(images),
            )
        try:
            data["_v3_resize_records"] = [
                {
                    "original_width": image.width,
                    "original_height": image.height,
                    "resized_width": self._compute_resize_dims(
                        image.height, image.width, "multi_modal"
                    )[1],
                    "resized_height": self._compute_resize_dims(
                        image.height, image.width, "multi_modal"
                    )[0],
                    "resize_policy_id": self.resize_policy_id,
                }
                for image in images
            ]
            result = self.process_multimodal(
                images, episode_type=episode.episode_type, is_train=is_train
            )
        finally:
            self.max_pixels = saved_max_pixels
            self._aug_seed = None
        data.update(result)
        if result.get("orig_height", 0) > 0 and result.get("orig_width", 0) > 0:
            data["_grounding_resize_info"] = {
                "orig_height": result["orig_height"],
                "orig_width": result["orig_width"],
                "resized_height": result["resized_height"],
                "resized_width": result["resized_width"],
            }
        videos, video_metas = self._process_videos(episode, item)
        data[VIDEO_OBSERVATIONS_KEY] = videos
        data[VIDEO_META_KEY] = json.dumps(video_metas)
        return data


@register_text_processor("v10_memory_qwen3_5")
class MemoryV3TextProcessor(MultimodalJsonlQwen3_5TextProcessor):
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        episode = data.get("_episode")
        if episode is None:
            return data
        frame_local_idx = data.get("_frame_local_idx", 0)
        sample_idx = _resolve_multimodal_sample_idx(episode, frame_local_idx)
        item = load_indexed_jsonl_item(episode.path, sample_idx)
        raw = item.get("v3_sample")
        if not isinstance(raw, dict):
            raise ValueError("Memory V3 row is missing v3_sample")
        sample = dict(raw)
        kept = data.get("_v10_kept_image_positions")
        if kept is not None:
            positions = tuple(int(value) for value in kept)
            if not positions or len(set(positions)) != len(positions):
                raise ValueError("Memory V3 retained image positions are invalid")
            images = list(sample["images"])
            if min(positions) < 0 or max(positions) >= len(images):
                raise IndexError("Memory V3 retained image position is out of range")
            sample["images"] = [images[position] for position in positions]
        task_type = str(sample["task_type"])
        dialogues = [
            {"role": "user", "text": render_user(sample)},
            {
                "role": "assistant",
                "text": dumps_assistant(sample["target"], str(sample["profile"]), task_type),
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
            raise ValueError(f"Memory V3 image placeholder mismatch: {image_tokens} != {num_images}")
        data["_expected_image_count"] = image_tokens
        return data


def exact_mix_budget(counts: Mapping[str, int], weights: Mapping[str, float]) -> int:
    active = [(int(counts[name]), float(weights[name])) for name in weights if weights[name] > 0]
    if not active or any(count <= 0 for count, _ in active):
        raise ValueError(f"every weighted V3 task needs data: counts={dict(counts)} weights={dict(weights)}")
    return max(1, math.floor(min(count / weight for count, weight in active)))


def allocate_mix_counts(budget: int, weights: Mapping[str, float]) -> dict[str, int]:
    if budget <= 0:
        raise ValueError("mix budget must be positive")
    total = sum(float(value) for value in weights.values())
    if total <= 0:
        raise ValueError("mix weights must sum to a positive value")
    normalized = {key: float(value) / total for key, value in weights.items()}
    raw = {key: budget * value for key, value in normalized.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    remaining = budget - sum(result.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    return result


def prepare(
    snapshot: Path,
    work_dir: Path,
    model_path: Path,
    *,
    max_length: int,
    tasks: tuple[str, ...],
    allow_small: bool,
    allow_partial: bool = False,
    max_budget: int | None = None,
    resize_mode: str = "B_auto_near_640",
) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    work_dir = work_dir.resolve()
    model_path = model_path.resolve()
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "v10_memory_snapshot_v1":
        raise ValueError("not a Memory V3 snapshot")
    if not allow_small and not allow_partial and manifest.get("complete") is not True:
        raise ValueError("formal training requires a complete Memory V3 snapshot")
    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        raise ValueError(f"unknown V3 tasks: {unknown}")
    if not tasks:
        raise ValueError("at least one V3 task is required")
    has_weights = (
        (model_path / "model.safetensors").is_file()
        or (model_path / "model.safetensors.index.json").is_file()
    )
    if not has_weights:
        raise FileNotFoundError(f"incomplete model directory: {model_path}")
    weights_all = {key: float(value) for key, value in manifest["train_mix"]["weights"].items()}
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
        raise ValueError(f"allocated mix exceeds available data: {allocated} > {counts}")
    if not allow_small and any(value < 100 for value in counts.values()):
        raise ValueError(f"formal training requires >=100 samples per selected task: {counts}")
    work_dir.mkdir(parents=True, exist_ok=True)
    step_state = work_dir / "v10_step_state.json"
    quarantine = work_dir / "media_quarantine.json"
    write_json(str(step_state), {"global_step": 0, "memory_v3_noise_probability": 0.0})
    if not quarantine.exists():
        write_json(str(quarantine), {"schema_version": "v10_media_quarantine_v1", "sample_ids": []})
    sources = [
        {
            "name": task,
            "source_type": "multimodal",
            "paths": [{
                "path": str(snapshot / "datasets" / task / "train"),
                "episode_type": "x2_multimodal",
                "task_name": f"memory_v3_{task}",
            }],
        }
        for task in tasks
    ]
    resize = json.loads((snapshot / "snapshot_metadata.json").read_text(encoding="utf-8"))["resize_policy"]
    if resize_mode not in {"A_current", "B_auto_near_640", "C_original_capped"}:
        raise ValueError(f"unknown resolution smoke mode: {resize_mode}")
    vision_type = (
        "v10_resilient_video_frame"
        if resize_mode == "A_current"
        else "v10_memory_video_frame"
    )
    vision_params = {
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
                resize_policy_identifier(resize)
                if resize_mode == "B_auto_near_640"
                else "original_resolution_total_pixel_cap_v1"
            ),
            "target_long_edge": (
                int(resize["target_long_edge"])
                if resize_mode == "B_auto_near_640" else 1_000_000_000
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
                            f"memory_v3_{task}": allocated[task] for task in tasks
                        },
                    },
                    "unit": "frames",
                },
                "task_balance_report_path": str(work_dir / "task_balance_report.json"),
            },
            "pipeline": ["vision", "text", "metadata"],
            "cache": {"enabled": False, "dir": str(work_dir / "dataset_cache")},
            "processors": {
                "vision": {
                    "type": vision_type,
                    "params": vision_params,
                },
                "text": {"type": "v10_memory_qwen3_5", "params": {}},
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
        "schema_version": "memory_v3_prepare_v1",
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
        "allow_partial": bool(allow_partial),
        "allow_small": bool(allow_small),
        "resize_mode": resize_mode,
        "batch_6_required_for_resolution_acceptance": True,
    }
    write_json(str(work_dir / "prepare_summary.json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path", type=Path,
        default=Path(os.environ.get("XPLANNER_MODEL_PATH", "/path/to/Qwen3.5-9B")),
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--allow-small", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--max-budget", type=int)
    parser.add_argument(
        "--resize-mode", default="B_auto_near_640",
        choices=("A_current", "B_auto_near_640", "C_original_capped"),
    )
    args = parser.parse_args()
    tasks = tuple(value.strip() for value in args.tasks.split(",") if value.strip())
    result = prepare(
        args.snapshot, args.work_dir, args.model_path,
        max_length=args.max_length, tasks=tasks, allow_small=args.allow_small,
        allow_partial=args.allow_partial,
        max_budget=args.max_budget, resize_mode=args.resize_mode,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
