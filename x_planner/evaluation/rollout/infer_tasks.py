"""Deterministic local generation smoke for the three Memory V4 tasks."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import zlib
from pathlib import Path
from typing import Any

from x2robot_dataset_v2.readers.multimodal_jsonl_reader import (
    get_indexed_jsonl_sample_count,
    load_indexed_jsonl_item,
)
from x2robot_dataset_v2.utils.multimodal_utils import process_dialogue

from x_planner.data.pipeline.runtime import apply_b30z_compatibility
from x_planner.data.pipeline.validation_runtime import render_generation_prompt
from x_planner.data.context.dataset import MemoryV3VisionProcessor
from .prompt import render_user
from .schema import dumps_assistant, loads_assistant


DEFAULT_ROWS = {
    "continuous": 3_362_672,
    "initial_plan": 4_233,
    "terminal": 21_979,
}


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _flatten_processed_images(value: dict[str, Any]) -> list[Any]:
    observations = value.get("image_observations")
    if not isinstance(observations, list) or not observations:
        raise RuntimeError("vision processor returned no image observations")
    images = [camera[-1] for camera in observations if camera]
    if len(images) != len(observations):
        raise RuntimeError("vision processor returned an empty camera observation")
    return images


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    return stripped


def _has_complete_root_json(text: str) -> bool:
    """Return true only when the generated text is exactly one complete JSON object."""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    try:
        value, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and not stripped[end:].strip()


class Generator:
    def __init__(self, checkpoint: Path, *, device: str) -> None:
        import torch
        from transformers import AutoProcessor
        from x_planner.trainer.builders import load_model
        from x2robot_dataset_v2.processors.epilogue.qwen3_5_epilogue import (
            build_qwen_messages,
        )

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this generation smoke")
        apply_b30z_compatibility()
        self.torch = torch
        self.checkpoint = str(checkpoint.resolve())
        self.processor = AutoProcessor.from_pretrained(str(checkpoint))
        self.model = load_model(str(checkpoint), torch.bfloat16, "sdpa")
        self.model.to(device)
        self.model.eval()
        self.model.config.use_cache = True
        if hasattr(self.model.config, "text_config"):
            self.model.config.text_config.use_cache = True
        self.vision = MemoryV3VisionProcessor(
            image_factor=32,
            min_pixels=1024,
            max_pixels=589824,
            pixel_cap=589824,
            target_long_edge=640,
            resize_policy_id="auto_near_640_no_upscale_v1",
            decoder_backend="av",
        )
        self.build_qwen_messages = build_qwen_messages
        self.device = device

    def generate(
        self,
        row: dict[str, Any],
        dataset_root: Path,
        *,
        max_new_tokens: int,
        prompt_override: str | None = None,
        image_mode: str = "normal",
        stop_after_root_json: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        sample = row["v4_sample"]
        prompt = prompt_override if prompt_override is not None else render_user(sample)
        refs = list(row["image"])
        raw_images = self.vision.load_image_refs(
            refs, str(dataset_root / "data.jsonl")
        )
        if image_mode == "shuffled":
            if len(raw_images) > 1:
                raw_images = raw_images[1:] + raw_images[:1]
        elif image_mode == "blank":
            from PIL import Image

            raw_images = [Image.new("RGB", image.size, "black") for image in raw_images]
        elif image_mode != "normal":
            raise ValueError(f"unknown image_mode: {image_mode!r}")
        original_sizes = [[image.width, image.height] for image in raw_images]
        processed = self.vision.process_multimodal(
            raw_images, episode_type="x2_multimodal", is_train=False
        )
        images = _flatten_processed_images(processed)
        resized_sizes = [[image.width, image.height] for image in images]
        dialogues = process_dialogue(
            [{"role": "user", "text": prompt}],
            seed=random.Random(
                zlib.crc32(str(sample["sample_key"]).encode())
            ).getrandbits(64),
            num_images=len(images),
        )
        messages, used = self.build_qwen_messages(dialogues, images)
        if used != len(images):
            raise RuntimeError(f"prompt consumed {used}/{len(images)} images")
        rendered = render_generation_prompt(self.processor, messages)
        inputs = self.processor(
            text=[rendered],
            images=images,
            padding=False,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        input_tokens = int(inputs["input_ids"].shape[1])
        image_grid = inputs.get("image_grid_thw")
        image_grid_thw = image_grid.detach().cpu().tolist() if image_grid is not None else []
        self.torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        generation_kwargs: dict[str, Any] = {}
        if stop_after_root_json:
            from transformers import StoppingCriteria, StoppingCriteriaList

            tokenizer = self.processor.tokenizer
            prefix_length = input_tokens

            class RootJsonStoppingCriteria(StoppingCriteria):
                def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                    if input_ids.shape[0] != 1:
                        raise RuntimeError("root JSON stopping requires batch size one")
                    suffix = tokenizer.decode(
                        input_ids[0, prefix_length:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    return _has_complete_root_json(suffix)

            generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [RootJsonStoppingCriteria()]
            )
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                **generation_kwargs,
            )
        elapsed = time.monotonic() - started
        output_ids = generated[0, input_tokens:]
        text = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        metadata = {
            "generation_seconds": elapsed,
            "input_tokens": input_tokens,
            "output_tokens": int(output_ids.numel()),
            "image_count": len(images),
            "image_grid_thw": image_grid_thw,
            "original_sizes_wh": original_sizes,
            "resized_sizes_wh": resized_sizes,
            "peak_allocated_mib": self.torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": self.torch.cuda.max_memory_reserved() / 1024**2,
            "image_mode": image_mode,
            "root_json_stop_enabled": bool(stop_after_root_json),
            "complete_root_json_at_decode_end": _has_complete_root_json(text),
        }
        return text, metadata


def _load_row(snapshot: Path, task: str, index: int) -> tuple[Path, dict[str, Any]]:
    root = snapshot / "datasets" / task / "validation"
    count = get_indexed_jsonl_sample_count(str(root))
    if not 0 <= index < count:
        raise IndexError(f"{task} row {index} is outside [0, {count})")
    row = load_indexed_jsonl_item(str(root), index)
    sample = row.get("v4_sample")
    if not isinstance(sample, dict):
        raise ValueError(f"{task} row {index} is missing v4_sample")
    terminal = bool(sample.get("is_terminal_window"))
    if task == "continuous" and terminal:
        raise ValueError("continuous smoke row must be nonterminal")
    if task == "terminal" and not terminal:
        raise ValueError("terminal smoke row must be terminal")
    if task == "initial_plan" and sample.get("task_type") != "initial_plan":
        raise ValueError("initial_plan smoke row has the wrong task_type")
    return root, row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--continuous-row", type=int, default=DEFAULT_ROWS["continuous"])
    parser.add_argument("--initial-plan-row", type=int, default=DEFAULT_ROWS["initial_plan"])
    parser.add_argument("--terminal-row", type=int, default=DEFAULT_ROWS["terminal"])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    snapshot = args.snapshot.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    rows = {
        "continuous": args.continuous_row,
        "initial_plan": args.initial_plan_row,
        "terminal": args.terminal_row,
    }
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    generator = Generator(checkpoint, device=args.device)
    records: list[dict[str, Any]] = []
    for task in ("continuous", "initial_plan", "terminal"):
        dataset_root, row = _load_row(snapshot, task, rows[task])
        sample = row["v4_sample"]
        raw_prediction, generation = generator.generate(
            row, dataset_root, max_new_tokens=args.max_new_tokens
        )
        prediction_text = _extract_json(raw_prediction)
        schema_valid = False
        parsed_prediction = None
        schema_error = None
        try:
            parsed_prediction = loads_assistant(
                prediction_text,
                str(sample["profile"]),
                str(sample["task_type"]),
                instruction=str(sample["task_instruction"]),
                is_terminal_window=bool(sample.get("is_terminal_window", False)),
            )
            schema_valid = True
        except Exception as exc:  # Preserve the exact model failure for inspection.
            schema_error = f"{type(exc).__name__}: {exc}"
        gt_text = dumps_assistant(
            sample["target"],
            str(sample["profile"]),
            str(sample["task_type"]),
            instruction=str(sample["task_instruction"]),
            is_terminal_window=bool(sample.get("is_terminal_window", False)),
        )
        record = {
            "task": task,
            "validation_row": rows[task],
            "sample_key": sample["sample_key"],
            "source_id": sample["source_id"],
            "profile": sample["profile"],
            "task_instruction": sample["task_instruction"],
            "is_terminal_window": bool(sample.get("is_terminal_window", False)),
            "prompt": render_user(sample),
            "prediction_raw": raw_prediction,
            "prediction": parsed_prediction,
            "prediction_schema_valid": schema_valid,
            "prediction_schema_error": schema_error,
            "gt_text": gt_text,
            "gt": sample["target"],
            "exact_match": parsed_prediction == sample["target"],
            **generation,
        }
        _write_json(output_dir / f"{task}.json", record)
        records.append(record)
        print(
            json.dumps(
                {
                    "task": task,
                    "sample_key": sample["sample_key"],
                    "schema_valid": schema_valid,
                    "exact_match": record["exact_match"],
                    **generation,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    summary = {
        "schema_version": "memory_v4_three_task_inference_smoke_v1",
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(checkpoint),
        "snapshot": str(snapshot),
        "device": args.device,
        "rows": rows,
        "sample_count": len(records),
        "schema_valid_count": sum(row["prediction_schema_valid"] for row in records),
        "exact_match_count": sum(row["exact_match"] for row in records),
        "results": [
            {
                key: row[key]
                for key in (
                    "task",
                    "sample_key",
                    "source_id",
                    "profile",
                    "task_instruction",
                    "prediction_schema_valid",
                    "prediction_schema_error",
                    "exact_match",
                    "input_tokens",
                    "output_tokens",
                    "image_count",
                    "generation_seconds",
                    "peak_allocated_mib",
                    "peak_reserved_mib",
                )
            }
            for row in records
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
