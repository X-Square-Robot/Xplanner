"""Build an exact prompt/prediction/GT gallery from completed causal rollouts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _stage_records(episode: Path) -> list[tuple[str, dict[str, Any]]]:
    initial = _read(episode / "initial_plan.json")
    anchors = [_read(path) for path in sorted(episode.glob("anchor_*.json"))]
    if len(anchors) < 3:
        raise ValueError(f"{episode}: need at least three continuous anchors")
    terminal = next(
        (record for record in reversed(anchors) if record["gt_last_same_scale_unit_window"]),
        anchors[-1],
    )
    middle_candidates = [record for record in anchors[1:-1] if record is not terminal]
    middle = middle_candidates[len(middle_candidates) // 2] if middle_candidates else anchors[1]
    return [("initial", initial), ("start", anchors[0]), ("process", middle), ("terminal", terminal)]


def _compact(record: dict[str, Any], *, stage: str, source: str) -> dict[str, Any]:
    return {
        "audit_contract": (
            "L3+earliest_visual_only" if stage == "initial"
            else "L3+visual+previous_valid_model_outputs"
        ),
        "source_id": source,
        "stage": stage,
        "anchor_frame": record.get("anchor_frame"),
        "prompt": record["prompt"],
        "input_long_memory": record.get("input_long_memory"),
        "input_short_memory": record.get("input_short_memory"),
        "images": record.get("images"),
        "prediction_raw": record.get("prediction_raw"),
        "prediction": record.get("prediction"),
        "prediction_schema_valid": record.get("prediction_schema_valid"),
        "prediction_schema_error": record.get("prediction_schema_error"),
        "predicted_no_next_same_scale_unit": record.get(
            "predicted_no_next_same_scale_unit"
        ),
        "gt_last_same_scale_unit_window": record.get("gt_last_same_scale_unit_window"),
        "physical_task_completion_label": (
            "unknown" if stage != "initial" else None
        ),
        "gt": record.get("gt"),
    }


def _markdown(item: dict[str, Any]) -> str:
    title = f"{item['source_id']} / {item['stage']}"
    lines = [
        f"# {title}",
        "",
        f"- Audit contract: `{item['audit_contract']}`",
        f"- Anchor: `{item['anchor_frame']}`",
        f"- Schema valid: `{item['prediction_schema_valid']}`",
        f"- Predicted no-next-same-scale: `{item['predicted_no_next_same_scale_unit']}`",
        f"- GT final-same-scale-window: `{item['gt_last_same_scale_unit_window']}`",
        f"- Physical task completion: `{item['physical_task_completion_label']}`",
        "",
        "## Exact user prompt",
        "",
        "```text",
        item["prompt"],
        "```",
        "",
        "## Causal memory input",
        "",
        "```json",
        _json_block({
            "long_memory": item["input_long_memory"],
            "short_memory": item["input_short_memory"],
        }),
        "```",
        "",
        "## Model raw output",
        "",
        "```json",
        str(item["prediction_raw"]),
        "```",
        "",
        "## Parsed model output",
        "",
        "```json",
        _json_block(item["prediction"]),
        "```",
        "",
        "## Ground truth (review only; never model input)",
        "",
        "```json",
        _json_block(item["gt"]),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--episode",
        action="append",
        required=True,
        help="SOURCE=EPISODE_ID; repeat once per source",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    selections = []
    for value in args.episode:
        source, episode_id = value.split("=", 1)
        episode = args.rollout_root.resolve() / episode_id
        for stage, record in _stage_records(episode):
            item = _compact(record, stage=stage, source=source)
            stage_root = output / source
            _atomic(stage_root / f"{stage}.json", _json_block(item) + "\n")
            _atomic(stage_root / f"{stage}.md", _markdown(item))
            selections.append({
                "source_id": source,
                "episode_id": episode_id,
                "stage": stage,
                "anchor_frame": item["anchor_frame"],
                "prediction_schema_valid": item["prediction_schema_valid"],
            })
    _atomic(output / "selection_manifest.json", _json_block({
        "schema_version": "memory_v4_causal_review_gallery_v1",
        "rollout_root": str(args.rollout_root.resolve()),
        "selection_count": len(selections),
        "selections": selections,
    }) + "\n")
    _atomic(output / "README.md", "\n".join([
        "# Memory V4 strict-causal example gallery",
        "",
        "Every example uses only L3, current/earlier visual observations, and previous valid model outputs. Ground truth is included only for offline review.",
        "",
        "`terminal` means the dataset's final same-scale-unit window. It is not a verified physical-completion observation and must not stop video inference.",
        "",
        f"Selections: {len(selections)} (3 sources × 4 stages).",
        "",
    ]))
    print(json.dumps({"output": str(output), "selections": len(selections)}))


if __name__ == "__main__":
    main()
