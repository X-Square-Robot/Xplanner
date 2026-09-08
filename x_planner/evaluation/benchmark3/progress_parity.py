#!/usr/bin/env python3
"""Run a raw, no-video Benchmark3 V5.3 Action progress parity audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import copy
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import traceback
from typing import Any

from x_planner.data.event_states.inference import Generator
from x_planner.evaluation.whole_episode.rollout import (
    build_rollout_slots,
    jsonl_rows,
    load_episode_spec,
    run_rollout,
    select_rollout_slots,
    sha256_file,
)


RUN_SCHEMA_VERSION = "v5_3_benchmark3_action_progress_parity_v1"
EXPECTED_CHECKPOINT_STEP = "checkpoint-80500"
EXPECTED_JOB = "job-oc4o4xflrov1"
EXPECTED_MODEL_SHA256 = (
    "db0445c87c7bcbab918e38ae2cbaaa40a9446e1501427e79f93653a83a9ab1d7"
)
CONTEXTS = (
    {
        "name": "no_memory",
        "variant": "no_memory_no_initial",
        "context_source": "predicted",
    },
    {
        "name": "oracle_memory",
        "variant": "with_memory_no_initial",
        "context_source": "oracle",
    },
    {
        "name": "oracle_memory_oracle_plan",
        "variant": "with_memory_with_initial",
        "context_source": "oracle",
    },
    {
        "name": "rolling_memory",
        "variant": "with_memory_no_initial",
        "context_source": "predicted",
    },
    {
        "name": "rolling_memory_model_plan",
        "variant": "with_memory_with_initial",
        "context_source": "predicted",
    },
)
SCHEDULES = ("action_midpoints", "dense_stride10_action_boundaries")
PAIRS = (
    ("oracle_memory_vs_no_memory", "no_memory", "oracle_memory"),
    (
        "oracle_plan_increment",
        "oracle_memory",
        "oracle_memory_oracle_plan",
    ),
    ("rolling_memory_vs_no_memory", "no_memory", "rolling_memory"),
    (
        "model_plan_increment",
        "rolling_memory",
        "rolling_memory_model_plan",
    ),
)
FORBIDDEN_PROMPT_TEXT = (
    "Temporal contract for this anchor:",
    "Correction request:",
    "Demo plan constraint:",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def action_midpoint_frames(spec: Mapping[str, Any]) -> list[int]:
    total_frames = int(spec["total_frames"])
    frames = {
        (int(item["start_frame"]) + int(item["end_frame"]) - 1) // 2
        for item in spec["actions"]
    }
    return sorted(frame for frame in frames if 0 <= frame < total_frames - 1)


def dense_stride_action_boundary_frames(
    spec: Mapping[str, Any],
    *,
    stride: int,
) -> list[int]:
    if stride <= 0:
        raise ValueError("dense stride must be positive")
    total_frames = int(spec["total_frames"])
    frames = set(range(0, total_frames - 1, stride))
    for item in spec["actions"]:
        start = int(item["start_frame"])
        end = int(item["end_frame"])
        frames.update((start, (start + end - 1) // 2, end - 1))
    return sorted(frame for frame in frames if 0 <= frame < total_frames - 1)


def schedule_frames(
    spec: Mapping[str, Any],
    *,
    dense_stride: int,
) -> dict[str, list[int]]:
    return {
        "action_midpoints": action_midpoint_frames(spec),
        f"dense_stride{dense_stride}_action_boundaries": (
            dense_stride_action_boundary_frames(spec, stride=dense_stride)
        ),
    }


def _context(name: str) -> Mapping[str, str]:
    matches = [value for value in CONTEXTS if value["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"unknown context: {name}")
    return matches[0]


def _assert_raw_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    context: Mapping[str, str],
) -> None:
    violations: list[str] = []
    for row in rows:
        slot = str(row.get("slot_id"))
        if row.get("context_source") != context["context_source"]:
            violations.append(f"{slot}:context_source")
        prompt = str(row.get("prompt") or "")
        if any(text in prompt for text in FORBIDDEN_PROMPT_TEXT):
            violations.append(f"{slot}:assisted_prompt")
        if row.get("inference_prompt_suffix") is not None:
            violations.append(f"{slot}:initial_suffix")
        if row.get("selected_generation_prompt_suffix") is not None:
            violations.append(f"{slot}:retry_suffix")
        if row.get("prediction_repair") is not None:
            violations.append(f"{slot}:repair")
        if row.get("prediction_normalization") is not None:
            violations.append(f"{slot}:normalization")
        if row.get("prediction_fallback") is not None:
            violations.append(f"{slot}:fallback")
        if row.get("status") == "generated":
            attempts = row.get("prediction_attempts")
            if not isinstance(attempts, list) or len(attempts) != 1:
                violations.append(f"{slot}:attempt_count")
    if violations:
        raise RuntimeError(f"raw contract violations: {violations[:20]}")


def _run_context(
    *,
    spec: Mapping[str, Any],
    spec_path: Path,
    episode_name: str,
    schedule: str,
    anchors: Sequence[int],
    context: Mapping[str, str],
    generator: Any,
    output_root: Path,
    initial_max_new_tokens: int,
    execution_max_new_tokens: int,
) -> dict[str, Any]:
    run_root = output_root / schedule / str(context["name"])
    run_root.mkdir(parents=True, exist_ok=True)
    all_slots = build_rollout_slots(
        spec,
        anchor_frames=anchors,
        minimum_units_per_profile=1,
        dense_gap_policy="schema_proxy",
    )
    slots = select_rollout_slots(
        all_slots,
        profiles=("action_only",),
        context_variants=(str(context["variant"]),),
    )
    expected_slots = 1 + len(anchors) + 1
    if len(slots) != expected_slots:
        raise ValueError(
            f"{episode_name}/{schedule}/{context['name']} "
            f"has {len(slots)} slots, expected {expected_slots}"
        )
    predictions = run_root / "predictions.jsonl"
    rows = run_rollout(
        slots=slots,
        generator=generator,
        source_root=spec_path.parent,
        output_path=predictions,
        initial_max_new_tokens=initial_max_new_tokens,
        execution_max_new_tokens=execution_max_new_tokens,
        resume=True,
        initial_prompt_suffix=None,
        compact_initial_plan=False,
        enforce_temporal_decision_contract=False,
        execution_schema_retries=0,
        allow_execution_hold_fallback=False,
        allow_initial_plan_json_repair=False,
        allow_ongoing_early_end_normalization=False,
        allow_terminal_progress_normalization=False,
        memory_update_source="raw",
        protocol="blinded",
        context_source=str(context["context_source"]),
    )
    _assert_raw_contract(rows, context=context)
    metadata = {
        "schema_version": RUN_SCHEMA_VERSION,
        "episode_name": episode_name,
        "schedule": schedule,
        "context": context["name"],
        "context_variant": context["variant"],
        "context_source": context["context_source"],
        "anchor_count_excluding_end": len(anchors),
        "slot_count": len(rows),
        "generated": sum(row.get("status") == "generated" for row in rows),
        "raw_schema_valid": sum(
            bool(row.get("raw_prediction_schema_valid"))
            for row in rows
            if row.get("category") != "initial_plan"
        ),
        "predictions": str(predictions),
        "video_rendered": False,
    }
    write_json(run_root / "metadata.json", metadata)
    return metadata


def _episode_run(
    *,
    entry: Mapping[str, Any],
    generator: Any,
    output_root: Path,
    dense_stride: int,
    initial_max_new_tokens: int,
    execution_max_new_tokens: int,
) -> dict[str, Any]:
    episode_root = output_root / "episodes" / str(entry["name"])
    result_path = episode_root / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    spec_path = Path(str(entry["spec"])).resolve(strict=True)
    spec = load_episode_spec(
        spec_path,
        verify_sources=True,
        minimum_intervals_per_unit=1,
    )
    if spec["profiles"] != ["action_only"]:
        raise ValueError(f"{entry['name']} is not action_only")
    schedules = schedule_frames(spec, dense_stride=dense_stride)
    if tuple(schedules) != (
        "action_midpoints",
        f"dense_stride{dense_stride}_action_boundaries",
    ):
        raise RuntimeError("schedule naming mismatch")
    runs: list[dict[str, Any]] = []
    for schedule, anchors in schedules.items():
        for context in CONTEXTS:
            print(json.dumps({
                "event": "context_start",
                "episode": entry["name"],
                "schedule": schedule,
                "context": context["name"],
                "anchors_excluding_end": len(anchors),
            }, sort_keys=True), flush=True)
            runs.append(_run_context(
                spec=spec,
                spec_path=spec_path,
                episode_name=str(entry["name"]),
                schedule=schedule,
                anchors=anchors,
                context=context,
                generator=generator,
                output_root=episode_root,
                initial_max_new_tokens=initial_max_new_tokens,
                execution_max_new_tokens=execution_max_new_tokens,
            ))
    result = {
        "schema_version": RUN_SCHEMA_VERSION,
        "episode_name": entry["name"],
        "benchmark3_uid": spec["benchmark3_uid"],
        "spec": str(spec_path),
        "spec_sha256": spec["spec_sha256"],
        "total_frames": spec["total_frames"],
        "fps": spec["fps"],
        "action_count": len(spec["actions"]),
        "schedules": {
            name: {
                "anchors_excluding_end": len(frames),
                "first_anchor": frames[0] if frames else None,
                "last_anchor": frames[-1] if frames else None,
            }
            for name, frames in schedules.items()
        },
        "runs": runs,
        "video_rendered": False,
        "complete": True,
    }
    write_json(result_path, result)
    return result


def _worker_main(
    worker_index: int,
    device: str,
    entries: Sequence[Mapping[str, Any]],
    checkpoint: str,
    output_root: str,
    dense_stride: int,
    initial_max_new_tokens: int,
    execution_max_new_tokens: int,
) -> None:
    root = Path(output_root)
    worker_root = root / "workers" / f"worker_{worker_index:02d}"
    worker_root.mkdir(parents=True, exist_ok=True)
    try:
        generator = Generator(
            Path(checkpoint),
            processor_path=Path(checkpoint),
            device=device,
        )
        results: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for entry in entries:
            try:
                results.append(_episode_run(
                    entry=entry,
                    generator=generator,
                    output_root=root,
                    dense_stride=dense_stride,
                    initial_max_new_tokens=initial_max_new_tokens,
                    execution_max_new_tokens=execution_max_new_tokens,
                ))
            except Exception as exc:
                failure = {
                    "episode_name": entry["name"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                write_json(
                    root / "episodes" / str(entry["name"]) / "failure.json",
                    failure,
                )
        write_json(worker_root / "result.json", {
            "worker_index": worker_index,
            "device": device,
            "episodes": results,
            "failures": failures,
        })
        if failures:
            raise RuntimeError(f"worker {worker_index} had {len(failures)} failures")
    except BaseException as exc:
        write_json(worker_root / "failure.json", {
            "worker_index": worker_index,
            "device": device,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        raise


def _entry_weight(entry: Mapping[str, Any], *, dense_stride: int) -> int:
    spec = json.loads(Path(str(entry["spec"])).read_text(encoding="utf-8"))
    schedules = schedule_frames(spec, dense_stride=dense_stride)
    return sum((len(anchors) + 2) * len(CONTEXTS) for anchors in schedules.values())


def assign_entries(
    entries: Sequence[Mapping[str, Any]],
    worker_count: int,
    *,
    dense_stride: int,
) -> tuple[list[list[Mapping[str, Any]]], list[int]]:
    assignments: list[list[Mapping[str, Any]]] = [
        [] for _ in range(worker_count)
    ]
    loads = [0 for _ in range(worker_count)]
    weighted = [
        (_entry_weight(entry, dense_stride=dense_stride), entry)
        for entry in entries
    ]
    for weight, entry in sorted(
        weighted,
        key=lambda item: (item[0], str(item[1]["name"])),
        reverse=True,
    ):
        worker = min(range(worker_count), key=lambda index: (loads[index], index))
        assignments[worker].append(entry)
        loads[worker] += weight
    return assignments, loads


def raw_progress_score(row: Mapping[str, Any]) -> dict[str, Any]:
    valid = bool(
        row.get("status") == "generated"
        and row.get("raw_prediction_schema_valid")
    )
    prediction = row.get("raw_prediction")
    if not isinstance(prediction, Mapping):
        prediction = {}
    target = row.get("ground_truth")
    if not isinstance(target, Mapping):
        raise TypeError("execution row has no ground_truth")
    try:
        predicted_task = int(prediction["task_progress_percent"]) if valid else None
    except (KeyError, TypeError, ValueError):
        predicted_task = None
    target_task = int(target["task_progress_percent"])
    action_errors: list[int] = []
    action_predictions: dict[str, int | None] = {}
    action_targets: dict[str, int] = {}
    output_spec = row.get("output_spec")
    if not isinstance(output_spec, Mapping):
        output_spec = {}
    for prediction_index, name in ((0, "action_current"), (1, "action_next")):
        units = output_spec.get(f"prediction{prediction_index + 1}_units") or ()
        if "action" not in units:
            continue
        target_progress = int(
            target["predictions"][prediction_index]["action"]["progress_percent"]
        )
        try:
            predicted_progress = (
                int(
                    prediction["predictions"][prediction_index]["action"][
                        "progress_percent"
                    ]
                )
                if valid
                else None
            )
        except (KeyError, IndexError, TypeError, ValueError):
            predicted_progress = None
        absolute = (
            abs(predicted_progress - target_progress)
            if predicted_progress is not None
            else 100
        )
        action_errors.append(absolute)
        action_predictions[name] = predicted_progress
        action_targets[name] = target_progress
    return {
        "valid": valid,
        "task_error": (
            abs(predicted_task - target_task)
            if predicted_task is not None
            else 100
        ),
        "task_prediction": predicted_task,
        "task_target": target_task,
        "action_errors": action_errors,
        "action_current_error": (
            action_errors[0] if "action_current" in action_predictions else None
        ),
        "action_next_error": (
            action_errors[-1] if "action_next" in action_predictions else None
        ),
        "action_predictions": action_predictions,
        "action_targets": action_targets,
    }


def _mean(values: Sequence[float | int]) -> float | None:
    return (
        sum(float(value) for value in values) / len(values)
        if values
        else None
    )


def _metric_summary(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    field: str,
) -> dict[str, Any]:
    values: list[float] = []
    by_episode: dict[str, list[float]] = defaultdict(list)
    for row, score in zip(rows, scores):
        if field == "action_all":
            observed = [float(value) for value in score["action_errors"]]
        else:
            raw = score.get(field)
            observed = [] if raw is None else [float(raw)]
        values.extend(observed)
        by_episode[str(row["episode_name"])].extend(observed)
    episode_means = {
        episode: _mean(group)
        for episode, group in sorted(by_episode.items())
        if group
    }
    return {
        "value_count": len(values),
        "episode_count": len(episode_means),
        "micro_mae": _mean(values),
        "macro_episode_mae": _mean([
            value for value in episode_means.values() if value is not None
        ]),
        "per_episode_mae": episode_means,
        "missing_or_invalid_policy": 100,
    }


def _stability(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = (
        defaultdict(list)
    )
    for row, score in zip(rows, scores):
        grouped[str(row["episode_name"])].append((row, score))
    task_pairs = 0
    task_regressions = 0
    task_residual_steps: list[float] = []
    action_pairs = 0
    action_regressions = 0
    action_residual_steps: list[float] = []
    for group in grouped.values():
        group.sort(key=lambda item: int(item[0]["anchor_frame"]))
        for (left_row, left), (right_row, right) in zip(group, group[1:]):
            left_task = left["task_prediction"]
            right_task = right["task_prediction"]
            if left_task is not None and right_task is not None:
                task_pairs += 1
                task_regressions += int(right_task < left_task)
                left_residual = left_task - left["task_target"]
                right_residual = right_task - right["task_target"]
                task_residual_steps.append(abs(right_residual - left_residual))
            try:
                left_caption = left_row["ground_truth"]["predictions"][0]["action"][
                    "caption"
                ]
                right_caption = right_row["ground_truth"]["predictions"][0]["action"][
                    "caption"
                ]
            except (KeyError, IndexError, TypeError):
                continue
            left_action = left["action_predictions"].get("action_current")
            right_action = right["action_predictions"].get("action_current")
            if (
                left_caption == right_caption
                and left_action is not None
                and right_action is not None
            ):
                action_pairs += 1
                action_regressions += int(right_action < left_action)
                left_residual = (
                    left_action - left["action_targets"]["action_current"]
                )
                right_residual = (
                    right_action - right["action_targets"]["action_current"]
                )
                action_residual_steps.append(abs(right_residual - left_residual))
    return {
        "task_adjacent_valid_pairs": task_pairs,
        "task_regression_count": task_regressions,
        "task_regression_rate": (
            task_regressions / task_pairs if task_pairs else None
        ),
        "task_residual_step_abs_mean": _mean(task_residual_steps),
        "action_same_label_adjacent_valid_pairs": action_pairs,
        "action_same_label_regression_count": action_regressions,
        "action_same_label_regression_rate": (
            action_regressions / action_pairs if action_pairs else None
        ),
        "action_residual_step_abs_mean": _mean(action_residual_steps),
    }


def _memory_copy(
    rows: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    task_total = 0
    task_copy = 0
    action_total = 0
    action_copy = 0
    for row, score in zip(rows, scores):
        memory = row.get("memory_input")
        short = memory.get("short_memory") if isinstance(memory, Mapping) else None
        if not isinstance(short, Mapping) or not score["valid"]:
            continue
        task_prediction = score["task_prediction"]
        if isinstance(short.get("task_progress_percent"), int):
            task_total += 1
            task_copy += int(
                task_prediction == int(short["task_progress_percent"])
            )
        try:
            memory_action = int(
                short["prediction1"]["action"]["progress_percent"]
            )
        except (KeyError, TypeError, ValueError):
            continue
        action_prediction = score["action_predictions"].get("action_current")
        if action_prediction is not None:
            action_total += 1
            action_copy += int(action_prediction == memory_action)
    return {
        "task_comparable": task_total,
        "task_exact_copy": task_copy,
        "task_exact_copy_rate": task_copy / task_total if task_total else None,
        "action_comparable": action_total,
        "action_exact_copy": action_copy,
        "action_exact_copy_rate": (
            action_copy / action_total if action_total else None
        ),
    }


def aggregate_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ongoing = [row for row in rows if row.get("category") == "ongoing"]
    terminal = [row for row in rows if row.get("category") == "end"]
    scores = [raw_progress_score(row) for row in ongoing]
    terminal_scores = [raw_progress_score(row) for row in terminal]
    task_counter = Counter(
        score["task_prediction"]
        for score in scores
        if score["task_prediction"] is not None
    )
    action_counter = Counter(
        score["action_predictions"].get("action_current")
        for score in scores
        if score["action_predictions"].get("action_current") is not None
    )
    return {
        "ongoing_rows": len(ongoing),
        "episodes": len({str(row["episode_name"]) for row in ongoing}),
        "raw_schema_valid": sum(score["valid"] for score in scores),
        "raw_schema_valid_rate": (
            sum(score["valid"] for score in scores) / len(scores)
            if scores
            else None
        ),
        "ongoing": {
            "task": _metric_summary(ongoing, scores, "task_error"),
            "action_current": _metric_summary(
                ongoing, scores, "action_current_error"
            ),
            "action_next": _metric_summary(
                ongoing, scores, "action_next_error"
            ),
            "action_all_requested": _metric_summary(
                ongoing, scores, "action_all"
            ),
        },
        "terminal": {
            "rows": len(terminal),
            "task_mae": _mean([
                score["task_error"] for score in terminal_scores
            ]),
            "action_current_mae": _mean([
                score["action_current_error"]
                for score in terminal_scores
                if score["action_current_error"] is not None
            ]),
            "raw_schema_valid_rate": (
                sum(score["valid"] for score in terminal_scores)
                / len(terminal_scores)
                if terminal_scores
                else None
            ),
        },
        "stability": _stability(ongoing, scores),
        "memory_copy": _memory_copy(ongoing, scores),
        "prediction_histograms": {
            "task_progress_percent": {
                str(key): value for key, value in sorted(task_counter.items())
            },
            "action_current_progress_percent": {
                str(key): value for key, value in sorted(action_counter.items())
            },
        },
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    observed = [float(value) for value in values]
    if not observed:
        return {
            "count": 0,
            "mean": None,
            "ci95": [None, None],
            "iterations": iterations,
            "seed": seed,
        }
    generator = random.Random(seed)
    bootstrapped = sorted(
        sum(generator.choice(observed) for _ in observed) / len(observed)
        for _ in range(iterations)
    )
    lower = bootstrapped[int(0.025 * (iterations - 1))]
    upper = bootstrapped[int(0.975 * (iterations - 1))]
    return {
        "count": len(observed),
        "mean": _mean(observed),
        "ci95": [lower, upper],
        "iterations": iterations,
        "seed": seed,
    }


def paired_deltas(
    rows: Sequence[Mapping[str, Any]],
    *,
    schedule: str,
) -> dict[str, Any]:
    selected = [
        row for row in rows
        if row.get("schedule") == schedule
        and row.get("category") == "ongoing"
    ]
    indexed = {
        (
            str(row["episode_name"]),
            int(row["anchor_frame"]),
            str(row["audit_context"]),
        ): row
        for row in selected
    }
    anchors = sorted({(key[0], key[1]) for key in indexed})
    result: dict[str, Any] = {}
    for pair_name, left_name, right_name in PAIRS:
        task_by_episode: dict[str, list[float]] = defaultdict(list)
        action_by_episode: dict[str, list[float]] = defaultdict(list)
        anchor_count = 0
        for episode, anchor in anchors:
            left = indexed.get((episode, anchor, left_name))
            right = indexed.get((episode, anchor, right_name))
            if left is None or right is None:
                continue
            left_score = raw_progress_score(left)
            right_score = raw_progress_score(right)
            task_by_episode[episode].append(
                right_score["task_error"] - left_score["task_error"]
            )
            left_action = left_score["action_current_error"]
            right_action = right_score["action_current_error"]
            if left_action is not None and right_action is not None:
                action_by_episode[episode].append(right_action - left_action)
            anchor_count += 1
        task_episode_deltas = [
            float(_mean(values))
            for values in task_by_episode.values()
            if values
        ]
        action_episode_deltas = [
            float(_mean(values))
            for values in action_by_episode.values()
            if values
        ]
        result[pair_name] = {
            "left": left_name,
            "right": right_name,
            "direction": "negative_delta_is_better",
            "paired_anchors": anchor_count,
            "task_mae_delta_macro_episode": bootstrap_mean_ci(
                task_episode_deltas
            ),
            "action_current_mae_delta_macro_episode": bootstrap_mean_ci(
                action_episode_deltas
            ),
        }
    return result


def load_all_rows(
    entries: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    plans: list[dict[str, Any]] = []
    for entry in entries:
        result_path = output_root / "episodes" / str(entry["name"]) / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for run in result["runs"]:
            values = jsonl_rows(Path(str(run["predictions"])))
            for row in values:
                annotated = copy.deepcopy(row)
                annotated["episode_name"] = entry["name"]
                annotated["benchmark3_uid"] = result["benchmark3_uid"]
                annotated["schedule"] = run["schedule"]
                annotated["audit_context"] = run["context"]
                if annotated.get("category") == "initial_plan":
                    plans.append(annotated)
                else:
                    rows.append(annotated)
    return rows, plans


def prompt_integrity(
    rows: Sequence[Mapping[str, Any]],
    plans: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for row in (*rows, *plans):
        prompt = str(row.get("prompt") or "")
        reasons: list[str] = []
        if any(text in prompt for text in FORBIDDEN_PROMPT_TEXT):
            reasons.append("forbidden_assist_text")
        if row.get("selected_generation_prompt_suffix") is not None:
            reasons.append("retry")
        if row.get("prediction_repair") is not None:
            reasons.append("repair")
        if row.get("prediction_normalization") is not None:
            reasons.append("normalization")
        if row.get("prediction_fallback") is not None:
            reasons.append("fallback")
        attempts = row.get("prediction_attempts")
        if row.get("status") == "generated" and (
            not isinstance(attempts, list) or len(attempts) != 1
        ):
            reasons.append("generation_attempt_count")
        if reasons:
            violations.append({
                "episode_name": row.get("episode_name"),
                "schedule": row.get("schedule"),
                "context": row.get("audit_context"),
                "slot_id": row.get("slot_id"),
                "reasons": reasons,
            })
    return {
        "passed": not violations,
        "policy": (
            "canonical raw prompt; greedy; no suffix, retry, repair, "
            "fallback, or normalization"
        ),
        "checked_rows": len(rows) + len(plans),
        "violations": violations,
    }


def initial_plan_summary(plans: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [
        row for row in plans
        if row.get("status") == "generated"
        and row.get("raw_prediction_schema_valid")
    ]
    action_counts: list[int] = []
    for row in valid:
        prediction = row.get("raw_prediction")
        plan = prediction.get("initial_plan") if isinstance(prediction, Mapping) else None
        if isinstance(plan, list):
            action_counts.append(len(plan))
    return {
        "rows": len(plans),
        "raw_schema_valid": len(valid),
        "raw_schema_valid_rate": len(valid) / len(plans) if plans else None,
        "predicted_action_count_mean": _mean(action_counts),
        "predicted_action_count_histogram": {
            str(key): value for key, value in sorted(Counter(action_counts).items())
        },
    }


def aggregate(
    rows: Sequence[Mapping[str, Any]],
    plans: Sequence[Mapping[str, Any]],
    *,
    dense_stride: int,
) -> dict[str, Any]:
    schedules = (
        "action_midpoints",
        f"dense_stride{dense_stride}_action_boundaries",
    )
    groups: dict[str, Any] = {}
    for schedule in schedules:
        groups[schedule] = {}
        for context in CONTEXTS:
            selected = [
                row for row in rows
                if row.get("schedule") == schedule
                and row.get("audit_context") == context["name"]
            ]
            groups[schedule][str(context["name"])] = aggregate_group(selected)
    integrity = prompt_integrity(rows, plans)
    if not integrity["passed"]:
        raise RuntimeError("prompt integrity audit failed")
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "primary_metric": (
            "ongoing raw task/action progress MAE; invalid or missing prediction=100"
        ),
        "row_count": len(rows),
        "initial_plan": initial_plan_summary(plans),
        "prompt_integrity": integrity,
        "groups": groups,
        "paired_deltas": {
            schedule: paired_deltas(rows, schedule=schedule)
            for schedule in schedules
        },
        "bootstrap": {
            "unit": "episode",
            "iterations": 10_000,
            "seed": 42,
        },
        "video_rendered": False,
    }


def markdown_report(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# Benchmark3 Action Progress Parity Audit",
        "",
        "Raw greedy inference only. Missing or schema-invalid progress is scored as 100 absolute error.",
        "",
        "| Schedule | Context | Task micro MAE | Task macro MAE | Action-current micro MAE | Action-current macro MAE | Task regression rate | Action same-label regression rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for schedule, contexts in metrics["groups"].items():
        for context, result in contexts.items():
            ongoing = result["ongoing"]
            stability = result["stability"]
            values = (
                schedule,
                context,
                ongoing["task"]["micro_mae"],
                ongoing["task"]["macro_episode_mae"],
                ongoing["action_current"]["micro_mae"],
                ongoing["action_current"]["macro_episode_mae"],
                stability["task_regression_rate"],
                stability["action_same_label_regression_rate"],
            )
            formatted = [
                values[0],
                values[1],
                *[
                    "NA" if value is None else f"{float(value):.4f}"
                    for value in values[2:]
                ],
            ]
            lines.append("| " + " | ".join(formatted) + " |")
    lines.extend([
        "",
        "## Paired context deltas",
        "",
        "Negative MAE delta means the right-hand context is better.",
        "",
    ])
    for schedule, pairs in metrics["paired_deltas"].items():
        lines.append(f"### {schedule}")
        lines.append("")
        for name, value in pairs.items():
            task = value["task_mae_delta_macro_episode"]
            action = value["action_current_mae_delta_macro_episode"]
            lines.append(
                f"- {name}: task {task['mean']} CI95={task['ci95']}; "
                f"action-current {action['mean']} CI95={action['ci95']}."
            )
        lines.append("")
    lines.extend([
        "## Integrity",
        "",
        f"- Prompt/raw-contract passed: {metrics['prompt_integrity']['passed']}",
        f"- Initial-plan raw schema-valid rate: {metrics['initial_plan']['raw_schema_valid_rate']}",
        "- Videos rendered: false",
        "",
    ])
    return "\n".join(lines)


def checkpoint_record(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    manifest_path = checkpoint / "pin_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = str(manifest.get("source_checkpoint") or "")
    model_hashes = manifest.get("model_sha256")
    if EXPECTED_JOB not in source or not source.endswith(EXPECTED_CHECKPOINT_STEP):
        raise ValueError(f"checkpoint pin source mismatch: {source}")
    if model_hashes != {"model.safetensors": EXPECTED_MODEL_SHA256}:
        raise ValueError(f"checkpoint model SHA mismatch: {model_hashes}")
    files = {
        str(item["name"]): item
        for item in manifest.get("files") or []
        if isinstance(item, Mapping)
    }
    model_path = checkpoint / "model.safetensors"
    if model_path.stat().st_size != int(files["model.safetensors"]["size"]):
        raise ValueError("checkpoint model size differs from pin manifest")
    trainer = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if int(trainer["global_step"]) != 80500:
        raise ValueError("trainer_state is not step 80500")
    return {
        "checkpoint_pin": str(checkpoint),
        "pin_manifest": str(manifest_path),
        "pin_manifest_sha256": sha256_file(manifest_path),
        "source_checkpoint": source,
        "global_step": 80500,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "model_size": model_path.stat().st_size,
        "verification": "pin manifest SHA identity plus model size and trainer step",
    }


def hardware_record() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip().splitlines(),
        "stderr": completed.stderr.strip(),
    }


def code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        root / "data/event_states/materialize_episode.py",
        root / "data/event_states/inference.py",
        root / "data/event_states/memory.py",
        root / "data/event_states/prompt.py",
        root / "data/event_states/schema.py",
        root / "evaluation/whole_episode/rollout.py",
    )
    return {str(path): sha256_file(path) for path in paths}


def load_batch(
    path: Path,
    *,
    dense_stride: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "v5_3_progress_parity_batch_v1":
        raise ValueError("batch schema mismatch")
    entries = value.get("episodes")
    if not isinstance(entries, list) or len(entries) != 20:
        raise ValueError("batch must contain exactly 20 episodes")
    names = [str(entry.get("name")) for entry in entries]
    if len(set(names)) != 20:
        raise ValueError("batch episode names are not unique")
    uids: list[str] = []
    preflight: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("selection") != "benchmark3_holdout":
            raise ValueError("batch contains a non-Benchmark3 episode")
        spec_path = Path(str(entry["spec"]))
        spec = load_episode_spec(
            spec_path,
            verify_sources=True,
            minimum_intervals_per_unit=1,
        )
        if spec["profiles"] != ["action_only"] or len(spec["actions"]) < 4:
            raise ValueError(f"invalid action-only spec: {spec_path}")
        uid = str(spec["benchmark3_uid"])
        uids.append(uid)
        schedules = schedule_frames(spec, dense_stride=dense_stride)
        preflight.append({
            "name": entry["name"],
            "uid": uid,
            "spec": str(spec_path.resolve()),
            "spec_sha256": spec["spec_sha256"],
            "total_frames": spec["total_frames"],
            "fps": spec["fps"],
            "action_count": len(spec["actions"]),
            "schedule_anchor_counts_excluding_end": {
                name: len(frames) for name, frames in schedules.items()
            },
            "estimated_model_calls": sum(
                (len(frames) + 2) * len(CONTEXTS)
                for frames in schedules.values()
            ),
        })
    if len(set(uids)) != 20:
        raise ValueError("batch Benchmark3 UIDs are not unique")
    return value, {
        "batch_spec": str(path),
        "batch_spec_sha256": sha256_file(path),
        "episode_count": 20,
        "pure_benchmark3": True,
        "unique_uids": True,
        "episodes": preflight,
        "estimated_model_calls": sum(
            item["estimated_model_calls"] for item in preflight
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-spec", type=Path, required=True)
    parser.add_argument("--checkpoint-pin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:2,cuda:3,cuda:4,cuda:6")
    parser.add_argument("--dense-stride", type=int, default=10)
    parser.add_argument("--initial-max-new-tokens", type=int, default=4096)
    parser.add_argument("--execution-max-new-tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.dense_stride <= 0:
        raise ValueError("dense stride must be positive")
    devices = tuple(
        value.strip() for value in args.devices.split(",") if value.strip()
    )
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("devices must be a non-empty unique list")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_record(args.checkpoint_pin)
    batch, data_preflight = load_batch(
        args.batch_spec,
        dense_stride=args.dense_stride,
    )
    entries = batch["episodes"]
    assignments, loads = assign_entries(
        entries,
        len(devices),
        dense_stride=args.dense_stride,
    )
    experiment = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "running",
        "started_at": utc_now(),
        "command": [sys.executable, *sys.argv],
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "devices": list(devices),
        "generation": {
            "do_sample": False,
            "prompt_protocol": "blinded_canonical_raw",
            "initial_max_new_tokens": args.initial_max_new_tokens,
            "execution_max_new_tokens": args.execution_max_new_tokens,
            "retries": 0,
            "repairs": False,
            "normalizations": False,
            "fallbacks": False,
        },
        "matrix": {
            "schedules": [
                "action_midpoints",
                f"dense_stride{args.dense_stride}_action_boundaries",
            ],
            "contexts": list(CONTEXTS),
            "profile": "action_only",
        },
        "checkpoint": checkpoint,
        "data": data_preflight,
        "code_sha256": code_hashes(),
        "hardware_at_start": hardware_record(),
        "worker_estimated_loads": loads,
        "worker_assignments": [
            [str(entry["name"]) for entry in group] for group in assignments
        ],
        "video_rendered": False,
    }
    write_json(output / "experiment_record.json", experiment)
    write_json(output / "run_manifest.json", {
        **experiment,
        "run_fingerprint": stable_json_sha256({
            "checkpoint": checkpoint,
            "batch_spec_sha256": data_preflight["batch_spec_sha256"],
            "matrix": experiment["matrix"],
            "generation": experiment["generation"],
            "code_sha256": experiment["code_sha256"],
        }),
    })
    context = multiprocessing.get_context("spawn")
    processes: list[multiprocessing.Process] = []
    for worker_index, (device, assigned) in enumerate(
        zip(devices, assignments)
    ):
        process = context.Process(
            target=_worker_main,
            args=(
                worker_index,
                device,
                assigned,
                str(args.checkpoint_pin.resolve()),
                str(output),
                args.dense_stride,
                args.initial_max_new_tokens,
                args.execution_max_new_tokens,
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failed = [
        {
            "worker": index,
            "pid": process.pid,
            "exitcode": process.exitcode,
        }
        for index, process in enumerate(processes)
        if process.exitcode != 0
    ]
    if failed:
        experiment.update({
            "status": "failed",
            "finished_at": utc_now(),
            "worker_failures": failed,
        })
        write_json(output / "experiment_record.json", experiment)
        raise RuntimeError(f"worker processes failed: {failed}")
    rows, plans = load_all_rows(entries, output)
    metrics = aggregate(
        rows,
        plans,
        dense_stride=args.dense_stride,
    )
    write_json(output / "metrics.json", metrics)
    (output / "report.md").write_text(
        markdown_report(metrics),
        encoding="utf-8",
    )
    experiment.update({
        "status": "complete",
        "finished_at": utc_now(),
        "metrics": str(output / "metrics.json"),
        "report": str(output / "report.md"),
        "execution_rows": len(rows),
        "initial_plan_rows": len(plans),
        "hardware_at_end": hardware_record(),
    })
    write_json(output / "experiment_record.json", experiment)
    print(json.dumps({
        "status": "complete",
        "output": str(output),
        "metrics": str(output / "metrics.json"),
        "report": str(output / "report.md"),
        "execution_rows": len(rows),
        "video_rendered": False,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
