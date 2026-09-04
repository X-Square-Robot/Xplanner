"""Run leak-free Episode rollouts for Memory V4 audit modes."""

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

from x_planner.data.pipeline.constants import LEVEL_TO_FIELD, PROFILE_FIELDS, UNIT_LEVEL
from x_planner.data.pipeline.memory import MemoryBank, MemoryCodec, UnitObservation
from x_planner.data.context.prompt import _image_blocks, _source_rate_line
from x_planner.data.context.schema import TERMINAL_CAPTION
from .infer_tasks import Generator, _extract_json
from .prompt import _instruction_block, _tagged_block, render_user
from .rollout_episode import loads_inference_continuous
from .schema import dumps_assistant, loads_assistant


MODES = ("strict_causal", "legacy_diagnostic", "oracle_memory", "strict_causal_plan_probe")
STRICT_PROFILE = "full"
STRICT_FIELD = "subtask"
INITIAL_PLAN_SCHEMA = (
    '{"initial_plan":[{"index":1,"subtask":{"level":"L2","caption":"...",'
    '"actions":[{"index":1,"action":{"level":"L1","caption":"...",'
    '"segments":[{"index":1,"l0":{"level":"L0","source":"segment",'
    '"caption":"..."}}]}}]}}]}'
)
CONTINUOUS_SCHEMA = (
    '{"task_progress_percent":0,"predictions":['
    '{"index":1,"subtask":{"level":"L2","caption":"...","progress_percent":0},'
    '"action":{"level":"L1","caption":"...","progress_percent":0},'
    '"l0":{"level":"L0","source":"segment","caption":"...","progress_percent":0}},'
    '{"index":2,"subtask":{"level":"L2","caption":"...","progress_percent":0},'
    '"action":{"level":"L1","caption":"...","progress_percent":0},'
    '"l0":{"level":"L0","source":"segment","caption":"...","progress_percent":0}}]}'
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _strict_initial_prompt(sample: Mapping[str, Any]) -> str:
    views = ", ".join(dict.fromkeys(str(image["view"]) for image in sample["images"]))
    blocks = [
        "Build the complete ordered plan for the given Task (L3) from the earliest "
        "synchronized robot-camera observations.",
        "Hierarchy: Task (L3) > Subtask (L2) > Action (L1) > Segment (L0).",
        _instruction_block(sample),
        f"Available views: {views}.",
        "Return one English JSON object with only initial_plan. Include every Subtask "
        "(L2) in order, nest its Actions (L1), and nest each Action's Segments (L0). "
        "Do not output progress or a completion sentinel.",
        f"Exact JSON shape (repeat array entries as needed): {INITIAL_PLAN_SCHEMA}",
    ]
    rate = _source_rate_line(sample)
    if rate:
        blocks.append(rate)
    blocks.extend(_image_blocks(list(sample["images"])))
    return "\n\n".join(blocks)


def _plan_memory(plan: Mapping[str, Any] | None, long_memory: list[str]) -> dict[str, Any]:
    captions: list[str] = []
    if isinstance(plan, Mapping):
        for entry in plan.get("initial_plan", []):
            if not isinstance(entry, Mapping):
                continue
            value = entry.get(STRICT_FIELD)
            if isinstance(value, Mapping) and isinstance(value.get("caption"), str):
                captions.append(str(value["caption"]))
    codec = MemoryCodec()
    remaining = [
        caption
        for caption in captions
        if not any(codec.same(caption, completed) for completed in long_memory)
    ]
    return {"completed_l2": list(long_memory), "remaining_l2": remaining}


def _strict_continuous_prompt(
    sample: Mapping[str, Any],
    *,
    long_memory: list[str],
    short_memory: list[dict[str, Any]],
    plan: Mapping[str, Any] | None,
) -> str:
    views = ", ".join(dict.fromkeys(str(image["view"]) for image in sample["images"]))
    blocks = [
        "Track the given Task (L3) from synchronized robot-camera observations and "
        "the causal history below.",
        "Hierarchy: Task (L3) > Subtask (L2) > Action (L1) > Segment (L0).",
        _instruction_block(sample),
        f"Available views: {views}.",
        "Return one English JSON object with only task_progress_percent and predictions. "
        "Output exactly two indexed predictions. For each prediction output Subtask "
        "(L2), Action (L1), and Segment (L0): Prediction 1 is current and Prediction 2 "
        "is next. If no next unit exists, use 'the task is complete' only as the "
        "no-next-unit marker; it is not a command to stop the video.",
        f"Exact JSON shape (replace values, keep field names): {CONTINUOUS_SCHEMA}",
        _tagged_block(
            "long_memory",
            "[none]" if not long_memory else "\n".join(
                f"{index}. {value}" for index, value in enumerate(long_memory, 1)
            ),
        ),
        _tagged_block(
            "short_memory",
            "[none]" if not short_memory else json.dumps(
                short_memory, ensure_ascii=False, separators=(",", ":")
            ),
        ),
    ]
    if plan is not None:
        blocks.append(_tagged_block(
            "plan_memory",
            json.dumps(_plan_memory(plan, long_memory), ensure_ascii=False, separators=(",", ":")),
        ))
    rate = _source_rate_line(sample)
    if rate:
        blocks.append(rate)
    blocks.extend(_image_blocks(list(sample["images"])))
    prompt = "\n\n".join(blocks)
    if "Profile:" in prompt or "prediction unit" in prompt.lower():
        raise AssertionError("strict prompt contains hidden output-scale metadata")
    return prompt


def _load_row(snapshot: Path, task: str, index: int) -> tuple[Path, dict[str, Any]]:
    root = snapshot / "datasets" / task / "validation"
    row = load_indexed_jsonl_item(str(root), index)
    if not isinstance(row.get("v4_sample"), dict):
        raise ValueError(f"{task} row {index} has no v4_sample")
    return root, row


def _parse_initial(text: str, *, sample: Mapping[str, Any], strict: bool) -> dict[str, Any]:
    return loads_assistant(
        text,
        STRICT_PROFILE if strict else str(sample["profile"]),
        "initial_plan",
        instruction=str(sample["task_instruction"]),
        is_terminal_window=False,
    )


def _parse_continuous(
    text: str, *, sample: Mapping[str, Any], strict: bool
) -> tuple[dict[str, Any], bool]:
    return loads_inference_continuous(
        text,
        STRICT_PROFILE if strict else str(sample["profile"]),
        str(sample["task_instruction"]),
    )


def _prediction_state(
    prediction: Mapping[str, Any], *, field: str
) -> tuple[list[dict[str, Any]], UnitObservation]:
    predictions = prediction.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 2:
        raise ValueError("prediction must contain two entries")
    first = predictions[0].get(field)
    second = predictions[1].get(field)
    if not isinstance(first, Mapping) or not isinstance(second, Mapping):
        raise ValueError(f"prediction is missing {field}")
    caption = str(first["caption"])
    progress = int(first["progress_percent"])
    next_caption = str(second["caption"])
    if next_caption == TERMINAL_CAPTION:
        next_caption = None
    return (
        [{"caption": caption, "progress_percent": progress}],
        UnitObservation(caption, progress, next_caption),
    )


def _invalid_output_memory_update(bank: MemoryBank) -> dict[str, Any]:
    """Describe a rejected step without mutating the last valid causal state."""
    return {
        "committed": None,
        "transitioned": False,
        "reason": "invalid_output_hold_last_valid",
        "long_memory": list(bank.long_memory),
    }


def _flatten_gt_metrics(
    prediction: Mapping[str, Any] | None,
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    gt = sample["target"]
    result: dict[str, Any] = {
        "task_progress_absolute_error": None,
        "p1_caption_similarities": {},
        "p1_progress_absolute_errors": {},
    }
    if not isinstance(prediction, Mapping):
        return result
    predicted_progress = prediction.get("task_progress_percent")
    if isinstance(predicted_progress, int) and not isinstance(predicted_progress, bool):
        result["task_progress_absolute_error"] = abs(
            predicted_progress - int(gt["task_progress_percent"])
        )
    codec = MemoryCodec()
    predicted_units = prediction.get("predictions")
    if not isinstance(predicted_units, list) or not predicted_units:
        return result
    for field in PROFILE_FIELDS[str(sample["profile"])]:
        predicted = predicted_units[0].get(field)
        expected = gt["predictions"][0][field]
        if not isinstance(predicted, Mapping):
            continue
        result["p1_caption_similarities"][field] = codec.similarity(
            str(predicted.get("caption", "")), str(expected["caption"])
        )
        progress = predicted.get("progress_percent")
        if isinstance(progress, int) and not isinstance(progress, bool):
            result["p1_progress_absolute_errors"][field] = abs(
                progress - int(expected["progress_percent"])
            )
    return result


def _run_episode(
    generator: Generator,
    *,
    snapshot: Path,
    episode: Mapping[str, Any],
    mode: str,
    output: Path,
    initial_max_new_tokens: int,
    continuous_max_new_tokens: int,
    image_mode: str,
    stop_after_root_json: bool,
) -> dict[str, Any]:
    strict = mode in {"strict_causal", "strict_causal_plan_probe"}
    plan_probe = mode == "strict_causal_plan_probe"
    oracle = mode == "oracle_memory"
    initial_root, initial_row = _load_row(
        snapshot, "initial_plan", int(episode["initial_plan_row"])
    )
    initial_sample = initial_row["v4_sample"]
    initial_prompt = (
        _strict_initial_prompt(initial_sample) if strict else render_user(initial_sample)
    )
    initial_raw, initial_meta = generator.generate(
        initial_row,
        initial_root,
        max_new_tokens=initial_max_new_tokens,
        prompt_override=initial_prompt,
        image_mode=image_mode,
        stop_after_root_json=stop_after_root_json,
    )
    initial_prediction: dict[str, Any] | None = None
    initial_error: str | None = None
    try:
        initial_prediction = _parse_initial(
            _extract_json(initial_raw), sample=initial_sample, strict=strict
        )
    except Exception as exc:
        initial_error = f"{type(exc).__name__}: {exc}"
    initial_record = {
        "task": "initial_plan",
        "mode": mode,
        "prompt": initial_prompt,
        "prediction_raw": initial_raw,
        "prediction": initial_prediction,
        "prediction_schema_valid": initial_prediction is not None,
        "prediction_schema_error": initial_error,
        "gt": initial_sample["target"],
        "images": copy.deepcopy(initial_sample["images"]),
        "input_contract": "L3+earliest_visual_only",
        **initial_meta,
    }
    _atomic_json(output / "initial_plan.json", initial_record)

    bank = MemoryBank()
    vocabulary: list[str] = []
    if plan_probe and initial_prediction is not None:
        vocabulary = _plan_memory(initial_prediction, [])["remaining_l2"]
    bank.reset(str(episode["global_episode_key"]), canonical_captions=vocabulary)
    short_memory: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    continuous_root = snapshot / "datasets" / "continuous" / "validation"
    previous_anchor: int | None = None
    for row_index in episode["continuous_rows"]:
        _, row = _load_row(snapshot, "continuous", int(row_index))
        sample = copy.deepcopy(row["v4_sample"])
        anchor = int(sample["anchor_frame"])
        forced = bool(sample.get("forced_terminal_anchor", False))
        if previous_anchor is not None and anchor - previous_anchor != 20 and not forced:
            raise ValueError(f"non-20-frame causal sequence: {previous_anchor}->{anchor}")
        previous_anchor = anchor
        if oracle:
            input_long = copy.deepcopy(sample.get("long_memory") or [])
            input_short = copy.deepcopy(sample.get("short_memory") or [])
        else:
            input_long = list(bank.long_memory)
            input_short = copy.deepcopy(short_memory)
        sample["long_memory"] = copy.deepcopy(input_long)
        sample["short_memory"] = copy.deepcopy(input_short)
        active_row = copy.deepcopy(row)
        active_row["v4_sample"] = sample
        if strict:
            prompt = _strict_continuous_prompt(
                sample,
                long_memory=input_long,
                short_memory=input_short,
                plan=initial_prediction if plan_probe else None,
            )
        else:
            prompt = render_user(sample)
        raw, generation = generator.generate(
            active_row,
            continuous_root,
            max_new_tokens=continuous_max_new_tokens,
            prompt_override=prompt,
            image_mode=image_mode,
            stop_after_root_json=stop_after_root_json,
        )
        prediction: dict[str, Any] | None = None
        predicted_no_next = False
        error: str | None = None
        memory_update: dict[str, Any] | None = None
        try:
            prediction, predicted_no_next = _parse_continuous(
                _extract_json(raw), sample=sample, strict=strict
            )
            if not oracle:
                field = STRICT_FIELD if strict else LEVEL_TO_FIELD[UNIT_LEVEL[str(sample["profile"])]]
                short_memory, observation = _prediction_state(prediction, field=field)
                update = bank.step(str(episode["global_episode_key"]), observation)
                memory_update = {
                    "committed": update.committed,
                    "transitioned": update.transitioned,
                    "reason": update.reason,
                    "long_memory": list(update.long_memory),
                }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if not oracle:
                # A malformed generation is not a trustworthy new observation.  Keep
                # the last valid causal state instead of erasing it and amplifying one
                # schema failure into every later window.  The raw generation remains
                # invalid and is never truncated or counted as a valid prediction.
                memory_update = _invalid_output_memory_update(bank)
        gt_text = dumps_assistant(
            sample["target"],
            str(sample["profile"]),
            "continuous",
            instruction=str(sample["task_instruction"]),
            is_terminal_window=bool(sample.get("is_terminal_window", False)),
        )
        record = {
            "task": "continuous",
            "mode": mode,
            "validation_row": int(row_index),
            "sample_key": sample["sample_key"],
            "anchor_frame": anchor,
            "source_id": sample["source_id"],
            "training_profile": sample["profile"],
            "training_unit_type": sample["unit_type"],
            "prompt": prompt,
            "input_long_memory": input_long,
            "input_short_memory": input_short,
            "input_plan_memory": (
                _plan_memory(initial_prediction, input_long) if plan_probe else None
            ),
            "prediction_raw": raw,
            "prediction": prediction,
            "prediction_schema_valid": prediction is not None,
            "prediction_schema_error": error,
            "predicted_no_next_same_scale_unit": predicted_no_next,
            "gt_last_same_scale_unit_window": bool(sample.get("is_terminal_window", False)),
            "physical_task_completion_label": "unknown",
            "memory_update": memory_update,
            "output_short_memory_for_next_anchor": copy.deepcopy(short_memory),
            "gt_text": gt_text,
            "gt": sample["target"],
            "images": copy.deepcopy(sample["images"]),
            "input_contract": (
                "L3+visual+model_history" if strict else
                "L3+visual+gt_memory_oracle" if oracle else
                "L3+visual+model_history+training_profile_scale"
            ),
            **_flatten_gt_metrics(prediction, sample),
            **generation,
        }
        records.append(record)
        _atomic_json(output / f"anchor_{anchor:06d}.json", record)

    schema_valid = sum(record["prediction_schema_valid"] for record in records)
    task_errors = [
        record["task_progress_absolute_error"]
        for record in records
        if record["task_progress_absolute_error"] is not None
    ]
    caption_scores = [
        score
        for record in records
        for score in record["p1_caption_similarities"].values()
    ]
    progress_errors = [
        error
        for record in records
        for error in record["p1_progress_absolute_errors"].values()
    ]
    summary = {
        "schema_version": "memory_v4_causal_episode_eval_v1",
        "mode": mode,
        "checkpoint": generator.checkpoint,
        "snapshot": str(snapshot),
        "episode_id": episode["episode_id"],
        "global_episode_key": episode["global_episode_key"],
        "task_instruction": episode["task_instruction"],
        "source_id": episode["source_id"],
        "training_profile": episode["profile"],
        "image_mode": image_mode,
        "initial_plan_schema_valid": initial_prediction is not None,
        "continuous_count": len(records),
        "continuous_schema_valid_count": schema_valid,
        "continuous_schema_valid_rate": schema_valid / len(records),
        "task_progress_mae": sum(task_errors) / len(task_errors) if task_errors else None,
        "p1_caption_similarity_mean": (
            sum(caption_scores) / len(caption_scores) if caption_scores else None
        ),
        "p1_progress_mae": (
            sum(progress_errors) / len(progress_errors) if progress_errors else None
        ),
        "predicted_no_next_count": sum(
            record["predicted_no_next_same_scale_unit"] for record in records
        ),
        "gt_last_unit_window_count": sum(
            record["gt_last_same_scale_unit_window"] for record in records
        ),
        "physical_completion_metrics_reported": False,
        "physical_completion_reason": "No reliable post-completion label in the source Snapshot.",
        "gt_memory_used_as_input": oracle,
        "plan_memory_used_as_input": plan_probe,
        "hidden_training_profile_or_unit_type_in_strict_prompt": False if strict else None,
        "records": [{
            "validation_row": record["validation_row"],
            "anchor_frame": record["anchor_frame"],
            "prediction_schema_valid": record["prediction_schema_valid"],
            "prediction_schema_error": record["prediction_schema_error"],
            "input_long_memory": record["input_long_memory"],
            "input_short_memory": record["input_short_memory"],
            "output_short_memory_for_next_anchor": record["output_short_memory_for_next_anchor"],
            "memory_update": record["memory_update"],
            "predicted_no_next_same_scale_unit": record["predicted_no_next_same_scale_unit"],
            "gt_last_same_scale_unit_window": record["gt_last_same_scale_unit_window"],
            "input_tokens": record["input_tokens"],
            "output_tokens": record["output_tokens"],
            "image_count": record["image_count"],
            "generation_seconds": record["generation_seconds"],
            "peak_allocated_mib": record["peak_allocated_mib"],
        } for record in records],
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--set", choices=("review", "probe"), default="review")
    parser.add_argument("--episode-ids", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-mode", choices=("normal", "shuffled", "blank"), default="normal")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Legacy override: use one generation budget for both tasks.",
    )
    parser.add_argument("--initial-max-new-tokens", type=int, default=2048)
    parser.add_argument("--continuous-max-new-tokens", type=int, default=512)
    parser.add_argument("--stop-after-root-json", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    episodes = manifest[f"{args.set}_episodes"]
    requested = {value for value in args.episode_ids.split(",") if value}
    if requested:
        episodes = [episode for episode in episodes if episode["episode_id"] in requested]
        missing = requested - {episode["episode_id"] for episode in episodes}
        if missing:
            raise ValueError(f"unknown episode IDs: {sorted(missing)}")
    if not episodes:
        raise ValueError("no Episodes selected")
    if args.max_new_tokens is not None:
        args.initial_max_new_tokens = args.max_new_tokens
        args.continuous_max_new_tokens = args.max_new_tokens
    if args.initial_max_new_tokens <= 0 or args.continuous_max_new_tokens <= 0:
        raise ValueError("generation token budgets must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = Generator(args.checkpoint.resolve(), device=args.device)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summaries: list[dict[str, Any]] = []
    for episode in episodes:
        episode_output = output_dir / str(episode["episode_id"])
        if episode_output.exists():
            raise FileExistsError(episode_output)
        episode_output.mkdir(parents=True)
        summary = _run_episode(
            generator,
            snapshot=args.snapshot.resolve(),
            episode=episode,
            mode=args.mode,
            output=episode_output,
            initial_max_new_tokens=args.initial_max_new_tokens,
            continuous_max_new_tokens=args.continuous_max_new_tokens,
            image_mode=args.image_mode,
            stop_after_root_json=args.stop_after_root_json,
        )
        summaries.append(summary)
        print(json.dumps({
            "episode_id": summary["episode_id"],
            "mode": args.mode,
            "continuous_count": summary["continuous_count"],
            "schema_valid_rate": summary["continuous_schema_valid_rate"],
            "task_progress_mae": summary["task_progress_mae"],
        }, sort_keys=True), flush=True)
    aggregate = {
        "schema_version": "memory_v4_causal_eval_run_v1",
        "started_at": started,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(args.checkpoint.resolve()),
        "snapshot": str(args.snapshot.resolve()),
        "manifest": str(args.manifest.resolve()),
        "mode": args.mode,
        "set": args.set,
        "image_mode": args.image_mode,
        "initial_max_new_tokens": args.initial_max_new_tokens,
        "continuous_max_new_tokens": args.continuous_max_new_tokens,
        "stop_after_root_json": args.stop_after_root_json,
        "episode_count": len(summaries),
        "continuous_count": sum(item["continuous_count"] for item in summaries),
        "continuous_schema_valid_count": sum(
            item["continuous_schema_valid_count"] for item in summaries
        ),
        "summaries": summaries,
    }
    _atomic_json(output_dir / "run_summary.json", aggregate)


if __name__ == "__main__":
    main()
