"""Run Initial Plan once, then autoregressive Short-Memory V4 inference."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from x2robot_dataset_v2.readers.multimodal_jsonl_reader import (
    load_indexed_jsonl_item,
)

from .infer_three_tasks_v4 import Generator, _extract_json
from .prompt_v4 import render_user
from .schema_v4 import dumps_assistant, loads_assistant


UNIT_FIELD = {"subtask": "subtask", "action": "action", "segment": "l0"}


def loads_inference_continuous(
    text: str, profile: str, instruction: str
) -> tuple[dict[str, Any], bool]:
    """Validate a continuous prediction without consulting its GT terminal label."""
    valid: list[tuple[dict[str, Any], bool]] = []
    errors: list[str] = []
    for predicted_terminal in (False, True):
        try:
            prediction = loads_assistant(
                text,
                profile,
                "continuous",
                instruction=instruction,
                is_terminal_window=predicted_terminal,
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"terminal={predicted_terminal}: {exc}")
        else:
            valid.append((prediction, predicted_terminal))
    if len(valid) != 1:
        raise ValueError(
            "continuous prediction must match exactly one terminal state; "
            + "; ".join(errors)
        )
    return valid[0]


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def predicted_short_memory(
    prediction: Mapping[str, Any], unit_type: str
) -> list[dict[str, Any]]:
    """Return exactly Prediction 1 at the active rollout scale."""
    try:
        field = UNIT_FIELD[unit_type]
    except KeyError as exc:
        raise ValueError(f"unknown rollout unit_type: {unit_type!r}") from exc
    predictions = prediction.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("rollout prediction has no Prediction 1")
    value = predictions[0].get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"Prediction 1 is missing {field}")
    caption = value.get("caption")
    progress = value.get("progress_percent")
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("Prediction 1 caption is invalid")
    if isinstance(progress, bool) or not isinstance(progress, int) or not 0 <= progress <= 100:
        raise ValueError("Prediction 1 progress_percent is invalid")
    return [{"caption": caption, "progress_percent": progress}]


def _find_episode_rows(list_path: Path, global_episode_key: str) -> list[int]:
    rows: list[int] = []
    with list_path.open(encoding="utf-8") as handle:
        for line in handle:
            if global_episode_key not in line:
                continue
            value = json.loads(line)
            if value.get("global_episode_key") == global_episode_key:
                rows.append(int(value["row_index"]))
    if not rows:
        raise RuntimeError(f"no continuous rows found for {global_episode_key}")
    if len(rows) != len(set(rows)):
        raise RuntimeError("continuous list contains duplicate row indices")
    return rows


def _load_row(root: Path, row_index: int) -> dict[str, Any]:
    row = load_indexed_jsonl_item(str(root), row_index)
    if not isinstance(row.get("v4_sample"), dict):
        raise ValueError(f"row {row_index} is missing v4_sample")
    return row


def _generation_record(
    *,
    task: str,
    row_index: int,
    sample: Mapping[str, Any],
    prompt: str,
    raw_prediction: str,
    prediction: Mapping[str, Any],
    metadata: Mapping[str, Any],
    input_short_memory: list[dict[str, Any]],
    input_long_memory: list[str],
) -> dict[str, Any]:
    gt = sample["target"]
    return {
        "task": task,
        "validation_row": row_index,
        "sample_key": sample["sample_key"],
        "anchor_frame": int(sample["anchor_frame"]),
        "is_terminal_window": bool(sample.get("is_terminal_window", False)),
        "task_instruction": sample["task_instruction"],
        "source_id": sample["source_id"],
        "profile": sample["profile"],
        "unit_type": sample["unit_type"],
        "input_short_memory": input_short_memory,
        "input_long_memory": input_long_memory,
        "prompt": prompt,
        "prediction_raw": raw_prediction,
        "prediction": dict(prediction),
        "gt_text": dumps_assistant(
            gt,
            str(sample["profile"]),
            str(sample["task_type"]),
            instruction=str(sample["task_instruction"]),
            is_terminal_window=bool(sample.get("is_terminal_window", False)),
        ),
        "gt": gt,
        "exact_match": prediction == gt,
        "images": copy.deepcopy(sample["images"]),
        **dict(metadata),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--initial-plan-row", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    snapshot = args.snapshot.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    initial_root = snapshot / "datasets" / "initial_plan" / "validation"
    continuous_root = snapshot / "datasets" / "continuous" / "validation"
    initial_row = _load_row(initial_root, args.initial_plan_row)
    initial_sample = initial_row["v4_sample"]
    global_key = str(initial_sample["global_episode_key"])
    continuous_indices = _find_episode_rows(
        snapshot / "lists" / "continuous_val.list", global_key
    )
    indexed_rows = [
        (index, _load_row(continuous_root, index)) for index in continuous_indices
    ]
    indexed_rows.sort(key=lambda value: int(value[1]["v4_sample"]["anchor_frame"]))
    continuous_indices = [value[0] for value in indexed_rows]
    continuous_rows = [value[1] for value in indexed_rows]
    anchors = [int(row["v4_sample"]["anchor_frame"]) for row in continuous_rows]
    if any(b - a != 20 for a, b in zip(anchors, anchors[1:])):
        raise RuntimeError(f"non-20-frame rollout anchors: {anchors}")
    if any(row["v4_sample"]["global_episode_key"] != global_key for row in continuous_rows):
        raise RuntimeError("cross-Episode row contamination")
    if any(row["v4_sample"]["profile"] != initial_sample["profile"] for row in continuous_rows):
        raise RuntimeError("Initial Plan and continuous profile mismatch")

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    generator = Generator(checkpoint, device=args.device)

    initial_raw, initial_metadata = generator.generate(
        initial_row, initial_root, max_new_tokens=args.max_new_tokens
    )
    initial_text = _extract_json(initial_raw)
    initial_prediction = loads_assistant(
        initial_text,
        str(initial_sample["profile"]),
        "initial_plan",
        instruction=str(initial_sample["task_instruction"]),
        is_terminal_window=False,
    )
    initial_record = _generation_record(
        task="initial_plan",
        row_index=args.initial_plan_row,
        sample=initial_sample,
        prompt=render_user(initial_sample),
        raw_prediction=initial_raw,
        prediction=initial_prediction,
        metadata=initial_metadata,
        input_short_memory=[],
        input_long_memory=[],
    )
    _write_json(output_dir / "initial_plan.json", initial_record)
    print(json.dumps({
        "task": "initial_plan",
        "row": args.initial_plan_row,
        "schema_valid": True,
        "generation_seconds": initial_metadata["generation_seconds"],
    }, sort_keys=True), flush=True)

    short_memory: list[dict[str, Any]] = []
    # This selected canonical Episode contains one L2 Task unit, hence no prior
    # completed same-level unit exists to archive into Long Memory.
    long_memory: list[str] = []
    rollout: list[dict[str, Any]] = []
    for row_index, row in zip(continuous_indices, continuous_rows):
        sample = copy.deepcopy(row["v4_sample"])
        sample["short_memory"] = copy.deepcopy(short_memory)
        sample["long_memory"] = copy.deepcopy(long_memory)
        active_row = copy.deepcopy(row)
        active_row["v4_sample"] = sample
        prompt = render_user(sample)
        raw_prediction, metadata = generator.generate(
            active_row, continuous_root, max_new_tokens=args.max_new_tokens
        )
        prediction_text = _extract_json(raw_prediction)
        prediction, predicted_terminal = loads_inference_continuous(
            prediction_text,
            str(sample["profile"]),
            str(sample["task_instruction"]),
        )
        record = _generation_record(
            task="continuous",
            row_index=row_index,
            sample=sample,
            prompt=prompt,
            raw_prediction=raw_prediction,
            prediction=prediction,
            metadata=metadata,
            input_short_memory=copy.deepcopy(short_memory),
            input_long_memory=copy.deepcopy(long_memory),
        )
        short_memory = predicted_short_memory(prediction, str(sample["unit_type"]))
        record["predicted_terminal"] = predicted_terminal
        record["output_short_memory_for_next_anchor"] = copy.deepcopy(short_memory)
        rollout.append(record)
        _write_json(output_dir / f"anchor_{record['anchor_frame']:06d}.json", record)
        print(json.dumps({
            "task": "continuous",
            "row": row_index,
            "anchor": record["anchor_frame"],
            "input_short_memory": record["input_short_memory"],
            "output_short_memory": short_memory,
            "schema_valid": True,
            "predicted_terminal": predicted_terminal,
            "generation_seconds": metadata["generation_seconds"],
        }, ensure_ascii=False, sort_keys=True), flush=True)

    summary = {
        "schema_version": "memory_v4_full_episode_rollout_v1",
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(checkpoint),
        "snapshot": str(snapshot),
        "global_episode_key": global_key,
        "task_instruction": initial_sample["task_instruction"],
        "profile": initial_sample["profile"],
        "source_id": initial_sample["source_id"],
        "initial_plan_row": args.initial_plan_row,
        "continuous_rows": continuous_indices,
        "anchors": anchors,
        "initial_plan_schema_valid": True,
        "continuous_schema_valid_count": len(rollout),
        "continuous_count": len(rollout),
        "exact_match_count": sum(record["exact_match"] for record in rollout),
        "gt_terminal_count": sum(record["is_terminal_window"] for record in rollout),
        "predicted_terminal_count": sum(record["predicted_terminal"] for record in rollout),
        "terminal_true_positive_count": sum(
            record["is_terminal_window"] and record["predicted_terminal"]
            for record in rollout
        ),
        "terminal_false_negative_count": sum(
            record["is_terminal_window"] and not record["predicted_terminal"]
            for record in rollout
        ),
        "terminal_false_positive_count": sum(
            not record["is_terminal_window"] and record["predicted_terminal"]
            for record in rollout
        ),
        "short_memory_mode": "previous_model_prediction_1_at_active_unit_scale",
        "gt_short_memory_used_as_input": False,
        "initial_plan_fed_to_continuous_prompt": False,
        "long_memory_mode": "empty_for_single_L2_episode",
        "records": [
            {
                key: record[key]
                for key in (
                    "validation_row",
                    "anchor_frame",
                    "sample_key",
                    "is_terminal_window",
                    "predicted_terminal",
                    "input_short_memory",
                    "output_short_memory_for_next_anchor",
                    "exact_match",
                    "input_tokens",
                    "output_tokens",
                    "image_count",
                    "generation_seconds",
                    "peak_allocated_mib",
                )
            }
            for record in rollout
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
