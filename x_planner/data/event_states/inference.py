"""Real-video V5 generation smoke using the exact training prompt contract."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .prompt import render_user
from .schema import validate_sample, validate_target


SCHEMA_VERSION = "v10_action_segment_v5_inference_smoke_v1"
TARGETS: tuple[tuple[str, str, str, str], ...] = (
    ("initial_plan", "robodojo", "initial_plan", "no_memory"),
    ("ongoing_no_memory", "robodojo", "ongoing", "no_memory"),
    ("ongoing_with_memory", "robodojo", "ongoing", "with_memory"),
    ("terminal", "robodojo", "terminal", "no_memory"),
    ("takeover", "takeover_q", "takeover", "no_memory"),
)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def extract_json(text: str) -> str:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        if len(lines) >= 3:
            value = "\n".join(lines[1:-1]).strip()
    return value


def parse_prediction(text: str, sample: Mapping[str, Any]) -> tuple[Any, str | None]:
    try:
        value = json.loads(extract_json(text))
        validated = validate_target(
            value,
            str(sample["category"]),
            sample["output_spec"],
        )
        return validated, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def flatten_processed_images(value: Mapping[str, Any]) -> list[Any]:
    observations = value.get("image_observations")
    if not isinstance(observations, list) or not observations:
        raise RuntimeError("vision processor returned no image observations")
    images = [camera[-1] for camera in observations if camera]
    if len(images) != len(observations):
        raise RuntimeError("vision processor returned an empty camera observation")
    return images


def initial_plan_structural_stop_boundary(
    text: str,
    *,
    maximum_actions: int,
    maximum_segments_per_action: int,
) -> dict[str, Any] | None:
    """Find a balanced JSON boundary before an over-limit initial plan item."""
    if maximum_actions <= 0 or maximum_segments_per_action <= 0:
        raise ValueError("initial plan structural limits must be positive")
    start = text.find("{")
    if start < 0:
        return None
    stack: list[str] = []
    in_string = False
    escaped = False
    completed_actions = 0
    completed_segments = 0
    last_action_boundary: int | None = None
    last_segment_boundary: int | None = None
    pairs = {"}": "{", "]": "["}
    action_array_stack = ["{", "["]
    segment_array_stack = ["{", "[", "{", "{", "["]
    for offset in range(start, len(text)):
        character = text[offset]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            continue
        if character in "[{":
            if character == "{" and stack == action_array_stack:
                if completed_actions >= maximum_actions and last_action_boundary is not None:
                    return {
                        "reason": "maximum_actions",
                        "cut_character_index": last_action_boundary,
                        "completed_actions": completed_actions,
                        "completed_segments_in_current_action": 0,
                    }
            elif character == "{" and stack == segment_array_stack:
                if completed_segments >= maximum_segments_per_action and last_segment_boundary is not None:
                    return {
                        "reason": "maximum_segments_per_action",
                        "cut_character_index": last_segment_boundary,
                        "completed_actions": completed_actions,
                        "completed_segments_in_current_action": completed_segments,
                    }
            stack.append(character)
            continue
        if character not in "]}":
            continue
        if not stack or stack[-1] != pairs[character]:
            return None
        stack.pop()
        if character != "}":
            continue
        if stack == segment_array_stack:
            completed_segments += 1
            last_segment_boundary = offset + 1
        elif stack == action_array_stack:
            completed_actions += 1
            completed_segments = 0
            last_action_boundary = offset + 1
            last_segment_boundary = None
    return None


class _InitialPlanStructuralStoppingCriteria:
    def __init__(
        self,
        tokenizer: Any,
        *,
        input_tokens: int,
        maximum_actions: int,
        maximum_segments_per_action: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.input_tokens = input_tokens
        self.maximum_actions = maximum_actions
        self.maximum_segments_per_action = maximum_segments_per_action
        self.report: dict[str, Any] | None = None

    def __call__(self, input_ids: Any, _scores: Any, **_kwargs: Any) -> Any:
        if int(input_ids.shape[0]) != 1:
            raise ValueError("Initial Plan structural stopping requires batch size 1")
        output_ids = input_ids[0, self.input_tokens:]
        text = self.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self.report = initial_plan_structural_stop_boundary(
            text,
            maximum_actions=self.maximum_actions,
            maximum_segments_per_action=self.maximum_segments_per_action,
        )
        return input_ids.new_full((1,), self.report is not None).bool()


def _indexed_leaf_candidates(
    snapshot: Path,
) -> list[tuple[tuple[str, str, str], Path]] | None:
    """Resolve composed leaves from one root manifest instead of PFS fan-out."""

    root_manifest_path = snapshot / "manifest.json"
    try:
        root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        root_manifest.get("schema_version") != "v10_action_segment_v5_snapshot_v3"
        or root_manifest.get("complete") is not True
        or not isinstance(root_manifest.get("leaves"), list)
    ):
        return None
    candidates: list[tuple[tuple[str, str, str], Path]] = []
    for item in root_manifest["leaves"]:
        if not isinstance(item, Mapping):
            raise ValueError("V5 root manifest contains a non-object leaf")
        relative = Path(str(item.get("relative_path") or ""))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) < 2
            or relative.parts[0] != "data"
        ):
            raise ValueError(f"unsafe V5 root-manifest leaf path: {relative}")
        candidates.append((
            (
                str(item.get("source") or ""),
                str(item.get("category") or ""),
                str(item.get("memory_variant") or ""),
            ),
            snapshot / relative,
        ))
    candidates.sort(key=lambda value: value[1].as_posix())
    return candidates


def _legacy_leaf_candidates(
    snapshot: Path,
) -> list[tuple[tuple[str, str, str], Path]]:
    candidates: list[tuple[tuple[str, str, str], Path]] = []
    for manifest_path in sorted((snapshot / "data").rglob("manifest.json")):
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("schema_version") != "v10_action_segment_v5_leaf_v2":
            continue
        candidates.append((
            (
                str(value.get("source") or ""),
                str(value.get("category") or ""),
                str(value.get("memory_variant") or ""),
            ),
            manifest_path.parent,
        ))
    return candidates


def select_rows(snapshot: Path) -> dict[str, tuple[Path, dict[str, Any]] | None]:
    from x2robot_dataset_v2.readers.multimodal_jsonl_reader import (
        load_indexed_jsonl_item,
    )

    candidates = _indexed_leaf_candidates(snapshot)
    if candidates is None:
        candidates = _legacy_leaf_candidates(snapshot)
    wanted = {(source, category, memory) for _, source, category, memory in TARGETS}
    leaves: dict[tuple[str, str, str], Path] = {}
    for key, leaf in candidates:
        if key in wanted and key not in leaves:
            manifest_path = leaf / "manifest.json"
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            if value.get("schema_version") != "v10_action_segment_v5_leaf_v2":
                raise ValueError(f"invalid indexed V5 leaf manifest: {manifest_path}")
            observed = (
                str(value.get("source") or ""),
                str(value.get("category") or ""),
                str(value.get("memory_variant") or ""),
            )
            if observed != key:
                raise ValueError(
                    f"V5 root/leaf routing mismatch: {leaf}: {key} != {observed}"
                )
            leaves[key] = leaf
    selected: dict[str, tuple[Path, dict[str, Any]] | None] = {}
    for name, source, category, memory in TARGETS:
        leaf = leaves.get((source, category, memory))
        if leaf is None:
            selected[name] = None
            continue
        row = load_indexed_jsonl_item(str(leaf), 0)
        sample = row.get("v5_sample")
        if not isinstance(sample, dict):
            raise ValueError(f"selected {name} row lacks v5_sample")
        validate_sample(sample)
        selected[name] = (leaf, row)
    return selected


class Generator:
    def __init__(
        self,
        checkpoint: Path,
        *,
        processor_path: Path | None = None,
        device: str,
        initial_plan_structure_limits: tuple[int, int] | None = None,
    ) -> None:
        import torch
        from transformers import AutoProcessor
        from x_planner.trainer.builders import load_model
        from ..pipeline.runtime import apply_b30z_compatibility
        from ..context.dataset import MemoryV3VisionProcessor
        from x2robot_dataset_v2.processors.epilogue.qwen3_5_epilogue import (
            build_qwen_messages,
        )

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for real V5 generation")
        apply_b30z_compatibility()
        self.torch = torch
        processor_path = checkpoint if processor_path is None else processor_path
        self.processor = AutoProcessor.from_pretrained(str(processor_path))
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
        if initial_plan_structure_limits is not None:
            maximum_actions, maximum_segments = initial_plan_structure_limits
            if maximum_actions <= 0 or maximum_segments <= 0:
                raise ValueError("initial plan structural limits must be positive")
        self.initial_plan_structure_limits = initial_plan_structure_limits

    def generate(
        self,
        row: Mapping[str, Any],
        leaf_root: Path,
        *,
        max_new_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        from ..pipeline.validation_runtime import render_generation_prompt
        from x2robot_dataset_v2.utils.multimodal_utils import process_dialogue

        sample = validate_sample(row["v5_sample"])
        canonical_prompt = render_user(sample)
        prompt = row.get("_rendered_prompt", canonical_prompt)
        if not isinstance(prompt, str) or not prompt.startswith(canonical_prompt):
            raise ValueError("inference prompt override must extend the canonical prompt")
        references = list(row.get("image") or sample["images"])
        raw_images = self.vision.load_image_refs(
            references,
            str(leaf_root / "data.jsonl"),
        )
        original_sizes = [[image.width, image.height] for image in raw_images]
        processed = self.vision.process_multimodal(
            raw_images,
            episode_type="x2_multimodal",
            is_train=False,
        )
        images = flatten_processed_images(processed)
        resized_sizes = [[image.width, image.height] for image in images]
        dialogues = process_dialogue(
            [{"role": "user", "text": prompt}],
            seed=random.Random(
                zlib.crc32(str(sample["sample_id"]).encode())
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
        if self.device.startswith("cuda"):
            self.torch.cuda.reset_peak_memory_stats()
        structural_stopper: _InitialPlanStructuralStoppingCriteria | None = None
        stopping_criteria = None
        if (
            sample["category"] == "initial_plan"
            and self.initial_plan_structure_limits is not None
        ):
            from transformers import StoppingCriteriaList

            maximum_actions, maximum_segments = self.initial_plan_structure_limits
            structural_stopper = _InitialPlanStructuralStoppingCriteria(
                self.processor.tokenizer,
                input_tokens=input_tokens,
                maximum_actions=maximum_actions,
                maximum_segments_per_action=maximum_segments,
            )
            stopping_criteria = StoppingCriteriaList([structural_stopper])
        started = time.monotonic()
        generation_options: dict[str, Any] = {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
        }
        if stopping_criteria is not None:
            generation_options["stopping_criteria"] = stopping_criteria
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                **generation_options,
            )
        elapsed = time.monotonic() - started
        output_ids = generated[0, input_tokens:]
        decoded_text = self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        structural_stop = None
        if structural_stopper is not None and structural_stopper.report is not None:
            structural_stop = dict(structural_stopper.report)
            structural_stop["maximum_actions"] = structural_stopper.maximum_actions
            structural_stop["maximum_segments_per_action"] = (
                structural_stopper.maximum_segments_per_action
            )
            structural_stop["decoded_characters_before_cut"] = len(decoded_text)
            decoded_text = decoded_text[:int(structural_stop["cut_character_index"])]
            structural_stop["decoded_characters_after_cut"] = len(decoded_text)
        text = decoded_text.strip()
        if not text:
            raise RuntimeError("model generated an empty V5 response")
        grid = inputs.get("image_grid_thw")
        return text, {
            "input_tokens": input_tokens,
            "output_tokens": int(output_ids.numel()),
            "image_count": len(images),
            "image_grid_thw": grid.detach().cpu().tolist() if grid is not None else [],
            "original_sizes_wh": original_sizes,
            "resized_sizes_wh": resized_sizes,
            "generation_seconds": elapsed,
            "initial_plan_structural_stop": structural_stop,
            "peak_allocated_mib": (
                self.torch.cuda.max_memory_allocated() / 1024**2
                if self.device.startswith("cuda") else None
            ),
            "peak_reserved_mib": (
                self.torch.cuda.max_memory_reserved() / 1024**2
                if self.device.startswith("cuda") else None
            ),
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--processor",
        type=Path,
        help="Processor/tokenizer asset directory; defaults to the checkpoint.",
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args(argv)
    checkpoint = args.checkpoint.resolve(strict=True)
    processor_path = (
        args.processor.resolve(strict=True)
        if args.processor is not None
        else checkpoint
    )
    snapshot = args.snapshot.resolve(strict=True)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    selected = select_rows(snapshot)
    available = [name for name, value in selected.items() if value is not None]
    if not available:
        raise ValueError("V5 snapshot exposes none of the inference smoke categories")
    generator = Generator(
        checkpoint,
        processor_path=processor_path,
        device=args.device,
    )
    records: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    for name, _source, _category, _memory in TARGETS:
        selected_row = selected[name]
        if selected_row is None:
            unavailable.append({
                "target": name,
                "reason": "snapshot_has_no_matching_leaf",
            })
            continue
        leaf, row = selected_row
        sample = validate_sample(row["v5_sample"])
        prediction_raw, generation = generator.generate(
            row,
            leaf,
            max_new_tokens=args.max_new_tokens,
        )
        prediction, schema_error = parse_prediction(prediction_raw, sample)
        record = {
            "target": name,
            "leaf": str(leaf),
            "sample_id": sample["sample_id"],
            "source": sample["source"],
            "category": sample["category"],
            "memory_variant": sample["memory_variant"],
            "task_instruction": sample["task_instruction"],
            "prompt": render_user(sample),
            "prediction_raw": prediction_raw,
            "prediction": prediction,
            "prediction_schema_valid": schema_error is None,
            "prediction_schema_error": schema_error,
            "ground_truth": sample["target"],
            "exact_match": prediction == sample["target"],
            **generation,
        }
        write_json(output / f"{name}.json", record)
        records.append(record)
        print(json.dumps({
            "target": name,
            "sample_id": sample["sample_id"],
            "prediction_schema_valid": schema_error is None,
            **generation,
        }, ensure_ascii=False, sort_keys=True), flush=True)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "passed": len(records) == len(available) and all(
            bool(record["prediction_raw"].strip()) for record in records
        ),
        "acceptance": "real_decode_prompt_load_generate_nonempty",
        "checkpoint": str(checkpoint),
        "checkpoint_mode": (
            "v10_trained_checkpoint"
            if (checkpoint / "v10_checkpoint_meta.json").is_file()
            else "weights_only"
        ),
        "processor": str(processor_path),
        "snapshot": str(snapshot),
        "device": args.device,
        "requested_targets": [value[0] for value in TARGETS],
        "generated_targets": [record["target"] for record in records],
        "unavailable_targets": unavailable,
        "generated_count": len(records),
        "schema_valid_count": sum(
            bool(record["prediction_schema_valid"]) for record in records
        ),
        "exact_match_count": sum(bool(record["exact_match"]) for record in records),
        "results": [{
            key: record[key]
            for key in (
                "target",
                "sample_id",
                "source",
                "category",
                "memory_variant",
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
        } for record in records],
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Generator",
    "TARGETS",
    "extract_json",
    "initial_plan_structural_stop_boundary",
    "parse_prediction",
    "select_rows",
]
