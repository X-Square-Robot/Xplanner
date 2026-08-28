"""Render a full-video causal V4 audit with model memory, prediction, and GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .build_full_episode_rollout_video_v4 import (
    CYAN,
    GREEN,
    ORANGE,
    RED,
    VIEW_BOXES,
    _continuous_lines,
    _decode_video,
    _encode_video,
    _memory_lines,
    _plan_lines,
    _read_json,
    _render,
    _view_paths,
    _write_json,
)


def _failure_lines(record: dict[str, Any], *, limit: int = 420) -> list[str]:
    raw = " ".join(str(record.get("prediction_raw") or "").split())
    if len(raw) > limit:
        raw = raw[:limit] + " … [full raw output in JSON sidecar]"
    return [
        "SCHEMA INVALID — state was not updated",
        str(record.get("prediction_schema_error") or "unknown schema error"),
        f"Raw: {raw or '[empty]'}",
    ]


def _model_plan_lines(record: dict[str, Any]) -> list[str]:
    prediction = record.get("prediction")
    return _plan_lines(prediction) if isinstance(prediction, dict) else _failure_lines(record)


def _model_continuous_lines(record: dict[str, Any]) -> list[str]:
    prediction = record.get("prediction")
    return (
        _continuous_lines(prediction)
        if isinstance(prediction, dict)
        else _failure_lines(record)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rollout = args.rollout_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    summary = _read_json(rollout / "summary.json")
    initial = _read_json(rollout / "initial_plan.json")
    anchors = sorted(
        (_read_json(path) for path in rollout.glob("anchor_*.json")),
        key=lambda record: int(record["anchor_frame"]),
    )
    if not anchors:
        raise ValueError("rollout has no continuous anchors")
    paths = _view_paths(initial)
    decoded = {}
    source_fps = {}
    for view, path in paths.items():
        decoded[view], source_fps[view] = _decode_video(path)
    frame_count = min(len(decoded[view]) for view in VIEW_BOXES)
    if frame_count <= int(anchors[-1]["anchor_frame"]):
        raise RuntimeError("source video ends before final anchor")
    frame_zero = {view: decoded[view][0] for view in VIEW_BOXES}
    intro = _render(
        frame_zero,
        instruction=summary["task_instruction"],
        status="STRICT CAUSAL / INITIAL PLAN / physical completion unknown",
        blocks=[
            ("INITIAL PLAN — MODEL", _model_plan_lines(initial), GREEN),
            ("INITIAL PLAN — GT (review only; never model input)", _plan_lines(initial["gt"]), ORANGE),
            ("CAUSAL INPUT STATE", ["Long Memory: [none]", "Short Memory: [none]"], CYAN),
        ],
    )
    intro.save(output / "poster_initial_plan.png", optimize=True)
    rendered_frames = []
    posters = []
    active = 0
    for frame_index in range(frame_count):
        while active + 1 < len(anchors) and int(anchors[active + 1]["anchor_frame"]) <= frame_index:
            active += 1
        record = anchors[active]
        model_end = "yes" if record["predicted_no_next_same_scale_unit"] else "no"
        gt_end = "yes" if record["gt_last_same_scale_unit_window"] else "no"
        blocks = [
            ("INITIAL PLAN — MODEL (not fed to continuous)", _model_plan_lines(initial), GREEN),
            ("STREAMING CAUSAL MEMORY INPUT", _memory_lines(record), CYAN),
            (
                f"CONTINUOUS — MODEL (no-next-same-scale={model_end})",
                _model_continuous_lines(record),
                GREEN if record["prediction_schema_valid"] else RED,
            ),
            (
                f"CONTINUOUS — GT review only (final-unit-window={gt_end})",
                _continuous_lines(record["gt"]),
                ORANGE,
            ),
        ]
        frame_views = {view: decoded[view][frame_index] for view in VIEW_BOXES}
        rendered = _render(
            frame_views,
            instruction=summary["task_instruction"],
            status=(
                f"frame={frame_index}/{frame_count - 1} anchor={record['anchor_frame']} "
                "| sentinel never stops video | physical completion unknown"
            ),
            blocks=blocks,
        )
        rendered_frames.append(rendered)
        if frame_index == int(record["anchor_frame"]):
            name = f"poster_anchor_{frame_index:06d}.png"
            rendered.save(output / name, optimize=True)
            posters.append(name)
    video = output / "full_episode_causal_audit_4k.mp4"
    _encode_video(video, intro, rendered_frames)
    manifest = {
        "schema_version": "memory_v4_causal_audit_video_v1",
        "rollout_dir": str(rollout),
        "checkpoint": summary["checkpoint"],
        "snapshot": summary["snapshot"],
        "task_instruction": summary["task_instruction"],
        "mode": summary["mode"],
        "input_contract": "L3+visual+previous_valid_model_outputs",
        "gt_memory_used_as_input": summary["gt_memory_used_as_input"],
        "initial_plan_fed_to_continuous": False,
        "sentinel_stops_video": False,
        "physical_completion_label": "unknown",
        "source_videos": paths,
        "source_fps": source_fps,
        "source_frame_count": frame_count,
        "anchors": [int(record["anchor_frame"]) for record in anchors],
        "video": str(video),
        "posters": ["poster_initial_plan.png", *posters],
    }
    _write_json(output / "video_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
