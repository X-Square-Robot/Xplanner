"""Render a full V4 Episode with only model predictions and model-causal memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .build_full_episode_rollout_video_v4 import (
    BACKGROUND,
    CYAN,
    GREEN,
    HEIGHT,
    ORANGE,
    PANEL,
    RED,
    VIEW_BOXES,
    WHITE,
    WIDTH,
    Block,
    _continuous_lines,
    _decode_video,
    _encode_video,
    _font,
    _memory_lines,
    _panel_height,
    _plan_lines,
    _read_json,
    _render,
    _view_paths,
    _wrap,
    _write_json,
)


PREFERRED_BODY_SIZE = 34
MINIMUM_BODY_SIZE = 24
INTRO_PANEL_X = 65
INTRO_PANEL_Y = 135
INTRO_PANEL_WIDTH = 3710
INTRO_PANEL_BOTTOM = 2120


def _failure_lines(record: dict[str, Any], *, limit: int = 420) -> list[str]:
    raw = " ".join(str(record.get("prediction_raw") or "").split())
    if len(raw) > limit:
        raw = raw[:limit] + " … [full model output in JSON sidecar]"
    return [
        "MODEL OUTPUT IS SCHEMA INVALID — causal state was held",
        str(record.get("prediction_schema_error") or "unknown schema error"),
        f"Raw model output: {raw or '[empty]'}",
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


def _memory_output_lines(record: dict[str, Any]) -> list[str]:
    update = dict(record.get("memory_update") or {})
    transitioned = "yes" if update.get("transitioned") else "no"
    lines = [
        f"Transition detected: {transitioned}; reason: {update.get('reason') or 'unknown'}"
    ]
    committed = update.get("committed")
    if committed:
        lines.append(f"Committed to Long Memory: {committed}")
    long_memory = list(update.get("long_memory") or ())
    lines.append(
        "Long Memory after update: [none]"
        if not long_memory
        else "Long Memory after update:"
    )
    lines.extend(f"  {value}" for value in long_memory)
    short_memory = list(record.get("output_short_memory_for_next_anchor") or ())
    if not short_memory:
        lines.append("Short Memory for next anchor: [none]")
    else:
        lines.append("Short Memory for next anchor:")
        lines.extend(
            f"  [{value['progress_percent']}%] {value['caption']}"
            for value in short_memory
        )
    return lines


def _intro_blocks(initial: dict[str, Any]) -> list[Block]:
    return [
        ("INITIAL PLAN — MODEL PREDICTION", _model_plan_lines(initial), GREEN),
        (
            "INITIAL CAUSAL MEMORY",
            ["Long Memory: [none]", "Short Memory: [none]"],
            CYAN,
        ),
    ]


def _render_initial_fullscreen(
    *, instruction: str, status: str, blocks: list[Block]
) -> tuple[Image.Image, int]:
    """Render all Initial Plan text on one 4K frame without silent clipping."""
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, WIDTH, 105), fill="#030b18")
    draw.text(
        (40, 20),
        f'Task instruction (L3): "{instruction}"',
        font=_font(34, bold=True),
        fill=WHITE,
    )
    status_font = _font(27, bold=True)
    status_width = draw.textbbox((0, 0), status, font=status_font)[2]
    draw.text((WIDTH - status_width - 45, 28), status, font=status_font, fill=CYAN)
    draw.rounded_rectangle(
        (25, 115, WIDTH - 25, HEIGHT - 25),
        radius=20,
        fill=PANEL,
        outline="#334155",
        width=3,
    )

    body_size = PREFERRED_BODY_SIZE
    available_height = INTRO_PANEL_BOTTOM - INTRO_PANEL_Y
    while (
        body_size > MINIMUM_BODY_SIZE
        and _panel_height(draw, blocks, body_size, INTRO_PANEL_WIDTH)
        > available_height
    ):
        body_size -= 1
    if _panel_height(draw, blocks, body_size, INTRO_PANEL_WIDTH) > available_height:
        raise RuntimeError(
            "semantic Initial Plan text does not fit the full-screen 4K review panel"
        )

    body = _font(body_size)
    title_font = _font(body_size + 4, bold=True)
    y = INTRO_PANEL_Y
    for title, lines, color in blocks:
        draw.text((INTRO_PANEL_X, y), title, font=title_font, fill=color)
        y += body_size + 12
        for text in lines:
            for line in _wrap(draw, text, body, INTRO_PANEL_WIDTH):
                draw.text((INTRO_PANEL_X, y), line, font=body, fill=WHITE)
                y += body_size + 5
        y += 14
    return canvas, body_size


def _stream_blocks(record: dict[str, Any]) -> list[Block]:
    predicted_end = "yes" if record.get("predicted_no_next_same_scale_unit") else "no"
    return [
        ("MODEL-CAUSAL MEMORY INPUT", _memory_lines(record), CYAN),
        (
            f"MODEL PREDICTION (no-next-same-scale={predicted_end})",
            _model_continuous_lines(record),
            GREEN if record.get("prediction_schema_valid") else RED,
        ),
        ("MEMORY UPDATE FROM MODEL OUTPUT", _memory_output_lines(record), ORANGE),
    ]


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

    intro, intro_body_size = _render_initial_fullscreen(
        instruction=summary["task_instruction"],
        status="MODEL ONLY / INITIAL PLAN / STRICT CAUSAL",
        blocks=_intro_blocks(initial),
    )
    intro.save(output / "poster_initial_plan.png", optimize=True)

    rendered_frames = []
    posters = []
    active = 0
    for frame_index in range(frame_count):
        while active + 1 < len(anchors) and int(anchors[active + 1]["anchor_frame"]) <= frame_index:
            active += 1
        record = anchors[active]
        frame_views = {view: decoded[view][frame_index] for view in VIEW_BOXES}
        rendered = _render(
            frame_views,
            instruction=summary["task_instruction"],
            status=(
                f"MODEL ONLY / frame={frame_index}/{frame_count - 1} "
                f"anchor={record['anchor_frame']} / strict causal"
            ),
            blocks=_stream_blocks(record),
            preferred_body_size=PREFERRED_BODY_SIZE,
            minimum_body_size=MINIMUM_BODY_SIZE,
        )
        rendered_frames.append(rendered)
        if frame_index == int(record["anchor_frame"]):
            name = f"poster_anchor_{frame_index:06d}.png"
            rendered.save(output / name, optimize=True)
            posters.append(name)

    video = output / "full_episode_model_predictions_only_4k.mp4"
    _encode_video(video, intro, rendered_frames)
    manifest = {
        "schema_version": "memory_v4_prediction_only_episode_video_v1",
        "display_contract": "model_predictions_and_model_causal_memory_only",
        "ground_truth_displayed": False,
        "rollout_dir": str(rollout),
        "checkpoint": summary["checkpoint"],
        "snapshot": summary["snapshot"],
        "task_instruction": summary["task_instruction"],
        "mode": summary["mode"],
        "input_contract": "L3+visual+previous_valid_model_outputs",
        "gt_memory_used_as_input": summary["gt_memory_used_as_input"],
        "initial_plan_fed_to_continuous": False,
        "initial_plan_schema_valid": bool(initial.get("prediction_schema_valid")),
        "sentinel_stops_video": False,
        "physical_completion_label": "unknown",
        "source_videos": paths,
        "source_fps": source_fps,
        "source_frame_count": frame_count,
        "anchors": [int(record["anchor_frame"]) for record in anchors],
        "memory_transition_count": sum(
            bool(record.get("memory_update", {}).get("transitioned"))
            for record in anchors
        ),
        "memory_commit_count": sum(
            record.get("memory_update", {}).get("committed") is not None
            for record in anchors
        ),
        "preferred_body_font_px": PREFERRED_BODY_SIZE,
        "minimum_body_font_px": MINIMUM_BODY_SIZE,
        "initial_plan_body_font_px": intro_body_size,
        "video": str(video),
        "posters": ["poster_initial_plan.png", *posters],
    }
    _write_json(output / "video_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
