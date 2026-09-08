#!/usr/bin/env python3
"""Pure metrics and audit helpers for controlled V5.3 whole-episode inference."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import copy
import math
from statistics import median
from typing import Any

from .rollout import stable_json_sha256, token_f1


AUDIT_SCHEMA_VERSION = "v5_3_context_end_audit_v1"
CONTEXT_VARIANTS = (
    "no_memory_no_initial",
    "with_memory_no_initial",
    "with_memory_with_initial",
)


def label_boundary_midpoint_frames(spec: Mapping[str, Any]) -> list[int]:
    """Return deterministic start/mid/end-1 anchors from all labelled intervals."""

    total_frames = int(spec["total_frames"])
    if total_frames < 2:
        raise ValueError("an audited episode requires at least two frames")
    frames = {0}
    for group in ("actions", "segments"):
        values = spec.get(group)
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{group} must be an array")
        for offset, interval in enumerate(values):
            if not isinstance(interval, Mapping):
                raise TypeError(f"{group}[{offset}] must be an object")
            start = int(interval["start_frame"])
            end = int(interval["end_frame"])
            if not 0 <= start < end:
                raise ValueError(f"{group}[{offset}] has invalid half-open interval")
            for frame in (start, (start + end - 1) // 2, end - 1):
                if 0 <= frame < total_frames - 1:
                    frames.add(frame)
    return sorted(frames)


def dense_stride_frames(
    spec: Mapping[str, Any],
    *,
    stride: int = 10,
) -> list[int]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    total_frames = int(spec["total_frames"])
    return sorted(
        set(label_boundary_midpoint_frames(spec))
        | set(range(0, total_frames - 1, stride))
    )


def duration_quantile_episode_names(
    entries: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Select five holdout episodes at nearest duration quartile positions."""

    holdout = sorted(
        (
            (int(entry["total_frames"]), str(entry["name"]))
            for entry in entries
            if entry.get("selection") == "benchmark3_holdout"
        ),
        key=lambda value: (value[0], value[1]),
    )
    if len(holdout) < 5:
        raise ValueError("at least five holdout episodes are required")
    last = len(holdout) - 1
    positions = [((last * numerator) + 2) // 4 for numerator in range(5)]
    return [holdout[position][1] for position in positions]


def _caption(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    return str(value.get("caption") or "")


def monotonic_caption_alignment(
    predicted: Sequence[str],
    expected: Sequence[str],
) -> dict[str, Any]:
    """Maximum-similarity order-preserving one-to-one alignment."""

    left = [str(value) for value in predicted]
    right = [str(value) for value in expected]
    rows = len(left) + 1
    columns = len(right) + 1
    scores = [[0.0 for _ in range(columns)] for _ in range(rows)]
    choices = [["" for _ in range(columns)] for _ in range(rows)]
    for i in range(1, rows):
        choices[i][0] = "skip_prediction"
    for j in range(1, columns):
        choices[0][j] = "skip_target"
    for i in range(1, rows):
        for j in range(1, columns):
            options = [
                (scores[i - 1][j], 1, "skip_prediction"),
                (scores[i][j - 1], 0, "skip_target"),
            ]
            similarity = token_f1(left[i - 1], right[j - 1])
            if similarity > 0:
                options.append((scores[i - 1][j - 1] + similarity, 2, "match"))
            best_score, _priority, choice = max(options, key=lambda value: (value[0], value[1]))
            scores[i][j] = best_score
            choices[i][j] = choice
    matches: list[dict[str, Any]] = []
    i, j = len(left), len(right)
    while i or j:
        choice = choices[i][j]
        if choice == "match":
            similarity = token_f1(left[i - 1], right[j - 1])
            matches.append({
                "prediction_index": i - 1,
                "target_index": j - 1,
                "similarity": similarity,
            })
            i -= 1
            j -= 1
        elif choice == "skip_prediction":
            i -= 1
        else:
            j -= 1
    matches.reverse()
    accepted = [item for item in matches if item["similarity"] >= 0.5]
    denominator = max(len(left), len(right), 1)
    return {
        "prediction_count": len(left),
        "target_count": len(right),
        "similarity_sum": scores[-1][-1],
        "normalized_score": scores[-1][-1] / denominator,
        "target_coverage_at_0_5": len(accepted) / max(len(right), 1),
        "extra_prediction_rate_at_0_5": max(len(left) - len(accepted), 0)
        / max(len(left), 1),
        "matches": matches,
    }


def _duplicate_rate(values: Sequence[str]) -> float:
    normalized = [" ".join(str(value).casefold().split()) for value in values]
    return (len(normalized) - len(set(normalized))) / max(len(normalized), 1)


def initial_plan_metrics(
    prediction: Mapping[str, Any] | None,
    target: Mapping[str, Any],
    *,
    schema_valid: bool,
) -> dict[str, Any]:
    predicted_plan = (
        prediction.get("initial_plan") if isinstance(prediction, Mapping) else None
    )
    target_plan = target.get("initial_plan")
    predicted_items = (
        list(predicted_plan)
        if isinstance(predicted_plan, Sequence)
        and not isinstance(predicted_plan, (str, bytes))
        else []
    )
    target_items = (
        list(target_plan)
        if isinstance(target_plan, Sequence)
        and not isinstance(target_plan, (str, bytes))
        else []
    )
    predicted_actions = [
        _caption(item.get("action")) if isinstance(item, Mapping) else ""
        for item in predicted_items
    ]
    target_actions = [
        _caption(item.get("action")) if isinstance(item, Mapping) else ""
        for item in target_items
    ]
    action_alignment = monotonic_caption_alignment(predicted_actions, target_actions)
    segment_scores: list[float] = []
    segment_coverages: list[float] = []
    predicted_segment_count = 0
    target_segment_count = 0
    predicted_segment_captions: list[str] = []
    for match in action_alignment["matches"]:
        predicted_action = predicted_items[int(match["prediction_index"])].get("action")
        target_action = target_items[int(match["target_index"])].get("action")
        predicted_segments = (
            predicted_action.get("segments")
            if isinstance(predicted_action, Mapping)
            else []
        )
        target_segments = (
            target_action.get("segments") if isinstance(target_action, Mapping) else []
        )
        predicted_segments = (
            list(predicted_segments)
            if isinstance(predicted_segments, Sequence)
            and not isinstance(predicted_segments, (str, bytes))
            else []
        )
        target_segments = (
            list(target_segments)
            if isinstance(target_segments, Sequence)
            and not isinstance(target_segments, (str, bytes))
            else []
        )
        predicted_captions = [
            _caption(item.get("segment")) if isinstance(item, Mapping) else ""
            for item in predicted_segments
        ]
        target_captions = [
            _caption(item.get("segment")) if isinstance(item, Mapping) else ""
            for item in target_segments
        ]
        predicted_segment_count += len(predicted_captions)
        target_segment_count += len(target_captions)
        predicted_segment_captions.extend(predicted_captions)
        alignment = monotonic_caption_alignment(predicted_captions, target_captions)
        segment_scores.append(float(alignment["normalized_score"]))
        segment_coverages.append(float(alignment["target_coverage_at_0_5"]))
    return {
        "schema_valid": bool(schema_valid),
        "scored_as_failure": not schema_valid,
        "action_count": len(predicted_items),
        "target_action_count": len(target_items),
        "action_duplicate_rate": _duplicate_rate(predicted_actions),
        "action_alignment": action_alignment,
        "segment_count_in_aligned_actions": predicted_segment_count,
        "target_segment_count_in_aligned_actions": target_segment_count,
        "segment_duplicate_rate": _duplicate_rate(predicted_segment_captions),
        "segment_alignment_mean": (
            sum(segment_scores) / len(segment_scores) if segment_scores else 0.0
        ),
        "segment_target_coverage_at_0_5": (
            sum(segment_coverages) / len(segment_coverages)
            if segment_coverages
            else 0.0
        ),
        "strict_action_alignment_score": (
            float(action_alignment["normalized_score"]) if schema_valid else 0.0
        ),
    }


def _prediction_view(
    row: Mapping[str, Any],
    view: str,
) -> tuple[Mapping[str, Any] | None, bool]:
    if view == "raw":
        value = row.get("raw_prediction")
        valid = bool(row.get("raw_prediction_schema_valid"))
    elif view == "effective":
        value = row.get("prediction")
        valid = bool(row.get("prediction_schema_valid"))
    else:
        raise ValueError("view must be raw or effective")
    return (value if isinstance(value, Mapping) else None), valid


def _unit_score(
    prediction: Mapping[str, Any] | None,
    target: Mapping[str, Any],
    *,
    prediction_index: int,
    unit: str,
    valid: bool,
) -> tuple[float, int, int | None]:
    try:
        predicted_unit = prediction["predictions"][prediction_index][unit]
        target_unit = target["predictions"][prediction_index][unit]
        predicted_caption = str(predicted_unit["caption"])
        target_caption = str(target_unit["caption"])
        predicted_progress = int(predicted_unit["progress_percent"])
        target_progress = int(target_unit["progress_percent"])
    except (KeyError, IndexError, TypeError, ValueError):
        return 0.0, 100, None
    if not valid:
        return 0.0, 100, None
    signed = predicted_progress - target_progress
    return token_f1(predicted_caption, target_caption), abs(signed), signed


def execution_row_score(row: Mapping[str, Any], *, view: str) -> dict[str, Any]:
    prediction, valid = _prediction_view(row, view)
    target = row.get("ground_truth")
    if not isinstance(target, Mapping):
        raise TypeError("execution row lacks ground_truth")
    output_spec = row.get("output_spec")
    if not isinstance(output_spec, Mapping):
        output_spec = {"prediction1_units": [], "prediction2_units": []}
    try:
        predicted_task = int(prediction["task_progress_percent"]) if valid else None
    except (KeyError, TypeError, ValueError):
        predicted_task = None
    target_task = int(target["task_progress_percent"])
    task_signed = predicted_task - target_task if predicted_task is not None else None
    ground_truth_available = bool(row.get("scoring_available", True))
    unit_scores: dict[str, Any] = {}
    captions: list[float] = []
    if ground_truth_available:
        for prediction_index in range(2):
            for unit in output_spec.get(f"prediction{prediction_index + 1}_units", []):
                caption, absolute, signed = _unit_score(
                    prediction,
                    target,
                    prediction_index=prediction_index,
                    unit=str(unit),
                    valid=valid,
                )
                key = f"prediction{prediction_index + 1}_{unit}"
                unit_scores[key] = {
                    "caption_token_f1": caption,
                    "progress_abs_error": absolute,
                    "progress_signed_error": signed,
                }
                captions.append(caption)
    decision = prediction.get("execution_decision") if prediction is not None else None
    return {
        "schema_valid": valid,
        "task_progress_abs_error": abs(task_signed) if task_signed is not None else 100,
        "task_progress_signed_error": task_signed,
        "caption_scoring_available": ground_truth_available,
        "caption_token_f1": (
            sum(captions) / len(captions)
            if captions else 0.0 if ground_truth_available else None
        ),
        "decision": decision,
        "decision_correct": bool(
            valid and decision == target.get("execution_decision")
        ),
        "unit_scores": unit_scores,
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[position]


def _error_summary(
    absolute: Sequence[float],
    signed: Sequence[float],
) -> dict[str, Any]:
    return {
        "count": len(absolute),
        "mae": sum(absolute) / len(absolute) if absolute else None,
        "median_abs_error": median(absolute) if absolute else None,
        "p90_abs_error": _percentile(absolute, 0.9),
        "signed_bias": sum(signed) / len(signed) if signed else None,
        "error_gt_10_rate": (
            sum(value > 10 for value in absolute) / len(absolute)
            if absolute else None
        ),
        "error_gt_20_rate": (
            sum(value > 20 for value in absolute) / len(absolute)
            if absolute else None
        ),
    }


def aggregate_execution_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    view: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for context in CONTEXT_VARIANTS:
        selected = [
            row for row in rows
            if row.get("category") in {"ongoing", "end"}
            and row.get("context_variant") == context
        ]
        scores = [execution_row_score(row, view=view) for row in selected]
        progress: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: {"absolute": [], "signed": []}
        )
        captions: list[float] = []
        for score in scores:
            progress["task"]["absolute"].append(score["task_progress_abs_error"])
            if score["task_progress_signed_error"] is not None:
                progress["task"]["signed"].append(score["task_progress_signed_error"])
            if score["caption_token_f1"] is not None:
                captions.append(score["caption_token_f1"])
            for key, unit in score["unit_scores"].items():
                progress[key]["absolute"].append(unit["progress_abs_error"])
                if unit["progress_signed_error"] is not None:
                    progress[key]["signed"].append(unit["progress_signed_error"])
        task_values_by_episode: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for row in selected:
            prediction, valid = _prediction_view(row, view)
            if (
                valid
                and prediction is not None
                and isinstance(prediction.get("task_progress_percent"), int)
            ):
                task_values_by_episode[str(row.get("episode_name") or "episode")].append((
                    int(row["anchor_frame"]),
                    int(prediction["task_progress_percent"]),
                ))
        regressions: list[dict[str, Any]] = []
        for episode, values in sorted(task_values_by_episode.items()):
            values.sort()
            regressions.extend([
                {
                    "episode_name": episode,
                    "previous_anchor": left[0],
                    "anchor": right[0],
                    "previous_progress": left[1],
                    "progress": right[1],
                }
                for left, right in zip(values, values[1:])
                if right[1] < left[1]
            ])
        result[context] = {
            "rows": len(selected),
            "generated": sum(row.get("status") == "generated" for row in selected),
            "schema_valid": sum(score["schema_valid"] for score in scores),
            "schema_valid_rate": (
                sum(score["schema_valid"] for score in scores) / len(scores)
                if scores else None
            ),
            "caption_token_f1_mean": (
                sum(captions) / len(captions) if captions else None
            ),
            "decision_accuracy": (
                sum(score["decision_correct"] for score in scores) / len(scores)
                if scores else None
            ),
            "progress": {
                key: _error_summary(value["absolute"], value["signed"])
                for key, value in sorted(progress.items())
            },
            "task_progress_regression_count": len(regressions),
            "task_progress_regressions": regressions,
        }
    return result


def paired_context_deltas(
    rows: Sequence[Mapping[str, Any]],
    *,
    view: str,
) -> dict[str, Any]:
    indexed = {
        (
            str(row.get("episode_name") or "episode"),
            str(row.get("category")),
            int(row.get("anchor_frame", -1)),
            str(row.get("context_variant")),
        ): row
        for row in rows
        if row.get("category") in {"ongoing", "end"}
    }
    pairs = (
        ("memory_without_initial", "no_memory_no_initial", "with_memory_no_initial"),
        ("initial_plan_increment", "with_memory_no_initial", "with_memory_with_initial"),
    )
    result: dict[str, Any] = {}
    anchor_keys = sorted({(key[0], key[1], key[2]) for key in indexed})
    for name, left_context, right_context in pairs:
        caption_deltas: list[float] = []
        task_error_deltas: list[float] = []
        for episode, category, anchor in anchor_keys:
            left = indexed.get((episode, category, anchor, left_context))
            right = indexed.get((episode, category, anchor, right_context))
            if left is None or right is None:
                continue
            left_score = execution_row_score(left, view=view)
            right_score = execution_row_score(right, view=view)
            if (
                right_score["caption_token_f1"] is not None
                and left_score["caption_token_f1"] is not None
            ):
                caption_deltas.append(
                    right_score["caption_token_f1"] - left_score["caption_token_f1"]
                )
            task_error_deltas.append(
                right_score["task_progress_abs_error"]
                - left_score["task_progress_abs_error"]
            )
        result[name] = {
            "left": left_context,
            "right": right_context,
            "paired_anchors": len(task_error_deltas),
            "paired_caption_anchors": len(caption_deltas),
            "caption_token_f1_delta_mean": (
                sum(caption_deltas) / len(caption_deltas)
                if caption_deltas else None
            ),
            "task_progress_mae_delta_mean": (
                sum(task_error_deltas) / len(task_error_deltas)
                if task_error_deltas else None
            ),
            "direction": {
                "caption_token_f1": "positive_is_better",
                "task_progress_mae": "negative_is_better",
            },
        }
    return result


def _confusion(records: Sequence[tuple[bool, bool]]) -> dict[str, Any]:
    tp = sum(target and predicted for target, predicted in records)
    fn = sum(target and not predicted for target, predicted in records)
    fp = sum(not target and predicted for target, predicted in records)
    tn = sum(not target and not predicted for target, predicted in records)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    balanced = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": balanced,
    }


def end_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    view: str,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("category") not in {"ongoing", "end"}:
            continue
        grouped[(
            str(row.get("episode_name") or "episode"),
            str(row.get("context_variant")),
        )].append(row)
    by_context: dict[str, Any] = {}
    for context in CONTEXT_VARIANTS:
        decision_only: list[tuple[bool, bool]] = []
        strict: list[tuple[bool, bool]] = []
        first_end_offsets: list[int] = []
        first_raw_end_offsets: list[int] = []
        missing_first_end = 0
        missing_raw_first_end = 0
        terminal_outcome_correct: list[bool] = []
        terminal_progress_errors: list[int] = []
        terminal_caption_scores: list[float] = []
        terminal_exact_matches: list[bool] = []
        for (episode, row_context), episode_rows in grouped.items():
            if row_context != context:
                continue
            ordered = sorted(episode_rows, key=lambda row: int(row["anchor_frame"]))
            valid_end_frames: list[int] = []
            raw_end_frames: list[int] = []
            target_end_frame = max(
                int(row["anchor_frame"])
                for row in ordered
                if row.get("category") == "end"
            )
            for row in ordered:
                prediction, valid = _prediction_view(row, view)
                target_end = row.get("category") == "end"
                predicted_end = bool(
                    prediction is not None
                    and prediction.get("execution_decision") == "End"
                )
                decision_only.append((target_end, predicted_end))
                strict.append((
                    target_end,
                    predicted_end if valid else not target_end,
                ))
                if valid and predicted_end:
                    valid_end_frames.append(int(row["anchor_frame"]))
                if predicted_end:
                    raw_end_frames.append(int(row["anchor_frame"]))
                if target_end:
                    terminal_score = execution_row_score(row, view=view)
                    terminal_caption_scores.append(
                        float(terminal_score["caption_token_f1"])
                    )
                    terminal_exact_matches.append(bool(
                        valid and prediction == row.get("ground_truth")
                    ))
                    outcome = (
                        prediction.get("decision_detail")
                        if prediction is not None else None
                    )
                    terminal_outcome_correct.append(bool(
                        valid
                        and isinstance(outcome, Mapping)
                        and outcome.get("outcome") == "completed"
                    ))
                    try:
                        progress = int(prediction["task_progress_percent"]) if valid else 0
                    except (KeyError, TypeError, ValueError):
                        progress = 0
                    terminal_progress_errors.append(abs(progress - 100))
            if valid_end_frames:
                first_end_offsets.append(min(valid_end_frames) - target_end_frame)
            else:
                missing_first_end += 1
            if raw_end_frames:
                first_raw_end_offsets.append(min(raw_end_frames) - target_end_frame)
            else:
                missing_raw_first_end += 1
        by_context[context] = {
            "decision_only_confusion": _confusion(decision_only),
            "strict_confusion_invalid_is_wrong": _confusion(strict),
            "first_valid_end_lead_lag_frames": first_end_offsets,
            "first_valid_end_lead_lag_mean": (
                sum(first_end_offsets) / len(first_end_offsets)
                if first_end_offsets else None
            ),
            "episodes_without_valid_end": missing_first_end,
            "first_raw_end_lead_lag_frames": first_raw_end_offsets,
            "first_raw_end_lead_lag_mean": (
                sum(first_raw_end_offsets) / len(first_raw_end_offsets)
                if first_raw_end_offsets else None
            ),
            "episodes_without_raw_end": missing_raw_first_end,
            "terminal_outcome_completed_accuracy": (
                sum(terminal_outcome_correct) / len(terminal_outcome_correct)
                if terminal_outcome_correct else None
            ),
            "terminal_progress_mae": (
                sum(terminal_progress_errors) / len(terminal_progress_errors)
                if terminal_progress_errors else None
            ),
            "terminal_caption_token_f1_mean": (
                sum(terminal_caption_scores) / len(terminal_caption_scores)
                if terminal_caption_scores else None
            ),
            "terminal_exact_match_rate": (
                sum(terminal_exact_matches) / len(terminal_exact_matches)
                if terminal_exact_matches else None
            ),
        }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "view": view,
        "contexts": by_context,
    }


def assist_event_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        if row.get("prediction_repair"):
            counter["initial_plan_json_repair"] += 1
        normalization = row.get("prediction_normalization")
        if isinstance(normalization, Mapping):
            counter[f"normalization:{normalization.get('policy', 'unknown')}"] += 1
        retries = max(0, len(row.get("prediction_attempts") or []) - 1)
        counter["schema_retry"] += retries
        if row.get("prediction_fallback"):
            counter["hold_last_valid_fallback"] += 1
        prompt = str(row.get("prompt") or "")
        if "Temporal contract for this anchor:" in prompt:
            counter["temporal_prompt_suffix"] += 1
        update = row.get("memory_update")
        if isinstance(update, Mapping):
            counter[f"memory:{update.get('reason', 'unknown')}"] += 1
    return dict(sorted(counter.items()))


def raw_effective_record(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_prediction")
    effective = row.get("prediction")
    return {
        "episode_name": row.get("episode_name"),
        "selection": row.get("selection"),
        "anchor_mode": row.get("anchor_mode"),
        "protocol": row.get("protocol"),
        "profile": row.get("profile"),
        "context_variant": row.get("context_variant"),
        "category": row.get("category"),
        "anchor_frame": row.get("anchor_frame"),
        "status": row.get("status"),
        "raw_schema_valid": row.get("raw_prediction_schema_valid"),
        "effective_schema_valid": row.get("prediction_schema_valid"),
        "raw_sha256": stable_json_sha256(raw),
        "effective_sha256": stable_json_sha256(effective),
        "changed": raw != effective,
        "raw_decision": raw.get("execution_decision") if isinstance(raw, Mapping) else None,
        "effective_decision": (
            effective.get("execution_decision")
            if isinstance(effective, Mapping) else None
        ),
        "repair": copy.deepcopy(row.get("prediction_repair")),
        "normalization": copy.deepcopy(row.get("prediction_normalization")),
        "retry_count": max(0, len(row.get("prediction_attempts") or []) - 1),
        "fallback": copy.deepcopy(row.get("prediction_fallback")),
    }


def memory_trace_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "episode_name": row.get("episode_name"),
        "selection": row.get("selection"),
        "anchor_mode": row.get("anchor_mode"),
        "protocol": row.get("protocol"),
        "profile": row.get("profile"),
        "context_variant": row.get("context_variant"),
        "category": row.get("category"),
        "anchor_frame": row.get("anchor_frame"),
        "memory_input": copy.deepcopy(row.get("memory_input")),
        "memory_update": copy.deepcopy(row.get("memory_update")),
        "raw_schema_valid": row.get("raw_prediction_schema_valid"),
        "effective_schema_valid": row.get("prediction_schema_valid"),
    }


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "CONTEXT_VARIANTS",
    "aggregate_execution_metrics",
    "assist_event_counts",
    "dense_stride_frames",
    "duration_quantile_episode_names",
    "end_audit",
    "execution_row_score",
    "initial_plan_metrics",
    "label_boundary_midpoint_frames",
    "memory_trace_record",
    "monotonic_caption_alignment",
    "paired_context_deltas",
    "raw_effective_record",
]
