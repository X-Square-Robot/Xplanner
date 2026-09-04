"""Aggregate strict, diagnostic, and ablation outputs from evaluate."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _raw_class(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty"
    decoder = json.JSONDecoder()
    try:
        _, end = decoder.raw_decode(stripped)
    except Exception:
        return "no_json_prefix"
    suffix = stripped[end:].strip()
    if not suffix:
        return "clean_json"
    if len(suffix) <= 8:
        return "tiny_trailing_suffix"
    if "trial_id" in suffix:
        return "trailing_trial_id"
    if suffix.startswith("{") or suffix.startswith("["):
        return "repeated_json"
    return "other_trailing_text"


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _aggregate(name: str, root: Path) -> dict[str, Any]:
    episode_dirs = (
        [root]
        if (root / "summary.json").is_file()
        else sorted(path.parent for path in root.glob("*/summary.json"))
    )
    if not episode_dirs:
        raise ValueError(f"{name}: no completed Episode summaries under {root}")
    records: list[dict[str, Any]] = []
    episode_records: list[list[dict[str, Any]]] = []
    initial_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for episode_dir in episode_dirs:
        summaries.append(_read(episode_dir / "summary.json"))
        initial_records.append(_read(episode_dir / "initial_plan.json"))
        current = [_read(path) for path in sorted(episode_dir.glob("anchor_*.json"))]
        episode_records.append(current)
        records.extend(current)
    valid = [record for record in records if record["prediction_schema_valid"]]
    invalid = [record for record in records if not record["prediction_schema_valid"]]
    raw_classes = Counter(_raw_class(str(record.get("prediction_raw") or "")) for record in records)
    gt_positive = [record for record in records if record["gt_last_same_scale_unit_window"]]
    gt_negative = [record for record in records if not record["gt_last_same_scale_unit_window"]]
    tp = sum(record["predicted_no_next_same_scale_unit"] for record in gt_positive)
    fp = sum(record["predicted_no_next_same_scale_unit"] for record in gt_negative)
    regressions = 0
    for current_records in episode_records:
        progress_sequence = [
            (int(record["anchor_frame"]), int(record["prediction"]["task_progress_percent"]))
            for record in current_records
            if record["prediction_schema_valid"]
            and isinstance(record.get("prediction", {}).get("task_progress_percent"), int)
        ]
        regressions += sum(
            current[1] < previous[1]
            for previous, current in zip(progress_sequence, progress_sequence[1:])
        )
    held = sum(
        (record.get("memory_update") or {}).get("reason")
        == "invalid_output_hold_last_valid"
        for record in invalid
    )
    memory_updates = [record.get("memory_update") or {} for record in records]
    memory_reasons = Counter(str(update.get("reason") or "missing") for update in memory_updates)
    memory_transition_count = sum(bool(update.get("transitioned")) for update in memory_updates)
    memory_commit_count = sum(update.get("committed") is not None for update in memory_updates)
    continuous_cap = None
    run_summary = root / "run_summary.json"
    if run_summary.is_file():
        continuous_cap = _read(run_summary).get("continuous_max_new_tokens")
    cap_hits = (
        sum(int(record["output_tokens"]) >= int(continuous_cap) for record in records)
        if continuous_cap else None
    )
    return {
        "name": name,
        "root": str(root.resolve()),
        "mode": summaries[0]["mode"],
        "checkpoint": summaries[0]["checkpoint"],
        "image_mode": summaries[0]["image_mode"],
        "episode_count": len(episode_dirs),
        "initial_plan_count": len(initial_records),
        "initial_plan_schema_valid_count": sum(
            record["prediction_schema_valid"] for record in initial_records
        ),
        "continuous_count": len(records),
        "continuous_schema_valid_count": len(valid),
        "continuous_schema_valid_rate": len(valid) / len(records),
        "first_json_recoverable_count": sum(
            count for kind, count in raw_classes.items() if kind not in {"empty", "no_json_prefix"}
        ),
        "raw_output_classes": dict(sorted(raw_classes.items())),
        "continuous_token_cap": continuous_cap,
        "continuous_token_cap_hit_count": cap_hits,
        "input_tokens_mean": _safe_mean([float(record["input_tokens"]) for record in records]),
        "output_tokens_mean": _safe_mean([float(record["output_tokens"]) for record in records]),
        "generation_seconds_mean": _safe_mean(
            [float(record["generation_seconds"]) for record in records]
        ),
        "peak_allocated_mib_max": max(float(record["peak_allocated_mib"]) for record in records),
        "task_progress_mae_on_schema_valid": _safe_mean([
            float(record["task_progress_absolute_error"])
            for record in valid
            if record["task_progress_absolute_error"] is not None
        ]),
        "task_progress_regression_count_between_valid_steps": regressions,
        "invalid_steps_holding_last_valid_memory": held,
        "memory": {
            "transition_count": memory_transition_count,
            "commit_count": memory_commit_count,
            "update_reasons": dict(sorted(memory_reasons.items())),
        },
        "no_next_same_scale_unit": {
            "true_positive": tp,
            "false_positive": fp,
            "ground_truth_positive": len(gt_positive),
            "ground_truth_negative": len(gt_negative),
            "recall": tp / len(gt_positive) if gt_positive else None,
            "premature_rate_over_nonlast_windows": fp / len(gt_negative) if gt_negative else None,
            "physical_completion_metric": None,
            "physical_completion_reason": (
                "V4 terminal labels mark the final same-scale unit window, not observed "
                "physical completion; source videos generally lack a post-completion label."
            ),
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Memory V4 causal evaluation metrics",
        "",
        "Strict schema validity never accepts a recoverable JSON prefix with trailing text. "
        "The recoverable-prefix count is diagnostic only.",
        "",
        "| run | mode | image | Episodes | Initial valid | Continuous valid | progress MAE | no-next recall | premature |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        end = run["no_next_same_scale_unit"]
        initial = f'{run["initial_plan_schema_valid_count"]}/{run["initial_plan_count"]}'
        continuous = f'{run["continuous_schema_valid_count"]}/{run["continuous_count"]} ({run["continuous_schema_valid_rate"]:.1%})'
        mae = run["task_progress_mae_on_schema_valid"]
        recall = end["recall"]
        premature = end["premature_rate_over_nonlast_windows"]
        lines.append(
            f'| {run["name"]} | {run["mode"]} | {run["image_mode"]} | '
            f'{run["episode_count"]} | {initial} | {continuous} | '
            f'{mae if mae is not None else "n/a"} | '
            f'{recall if recall is not None else "n/a"} | '
            f'{premature if premature is not None else "n/a"} |'
        )
    lines.extend([
        "",
        "`no-next` evaluates the existing final-same-scale-unit sentinel only. It must not "
        "be reported as physical task-completion recall or used to stop a video.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="NAME=RUN_DIRECTORY")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for value in args.run:
        if "=" not in value:
            raise ValueError(f"invalid --run {value!r}; expected NAME=PATH")
        name, path = value.split("=", 1)
        runs.append(_aggregate(name, Path(path)))
    report = {"schema_version": "memory_v4_causal_metrics_v1", "runs": runs}
    _atomic(args.output_json.resolve(), report)
    markdown = _markdown(report)
    output_markdown = args.output_markdown.resolve()
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_markdown.with_suffix(output_markdown.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(markdown)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_markdown)
    print(json.dumps({"runs": len(runs), "output": str(args.output_json.resolve())}))


if __name__ == "__main__":
    main()
