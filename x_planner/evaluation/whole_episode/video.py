"""Render full-length V5.3 rollouts as Wall-Planner console videos."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from x_planner.data.event_states.materialize_episode import PROFILES
from x_planner.data.event_states.inference import write_json
from x_planner.data.event_states.schema import V5ValidationError, validate_target


WIDTH = 1920
HEIGHT = 1080
LAYOUT_VERSION = "planner_console_v8"
CURRENT_UNIT_PROGRESS_POLICY = "contiguous_current_caption_span_v1"
EARLY_END_SCHEMA_ERROR = "ongoing cannot use decision End"

BACKGROUND_TOP = (8, 12, 24)
BACKGROUND_BOTTOM = (10, 16, 31)
PANEL = (15, 23, 40)
PANEL_ALT = (12, 19, 34)
PANEL_SOFT = (18, 28, 48)
BORDER = (39, 51, 75)
BORDER_SOFT = (31, 42, 63)
WHITE = (238, 243, 250)
TEXT = (210, 220, 234)
MUTED = (130, 145, 169)
DIM = (86, 101, 128)
CYAN = (101, 231, 218)
BLUE = (108, 166, 255)
VIOLET = (180, 148, 255)
GREEN = (91, 224, 174)
ORANGE = (245, 186, 75)
RED = (255, 132, 151)

PROFILE_LABELS = {
    "action_only": "ACTION ONLY",
    "segment_only": "SEGMENT ONLY",
    "action_segment_joint": "ACTION + SEGMENT",
}
PROFILE_UNITS = {
    "action_only": ("action",),
    "segment_only": ("segment",),
    "action_segment_joint": ("action", "segment"),
}
UNIT_ACCENTS = {
    "ACTION": BLUE,
    "SEGMENT": VIOLET,
}
VARIANT_LABELS = {
    "no_memory_no_initial": "NO MEMORY",
    "with_memory_no_initial": "WITH MEMORY",
    "with_memory_with_initial": "MEMORY + INITIAL PLAN",
}
FOCUS_VARIANTS = ("no_memory_no_initial", "with_memory_no_initial")
SUPPORTED_FOCUS_VARIANTS = (*FOCUS_VARIANTS, "with_memory_with_initial")
INITIAL_PLAN_PAGE_SIZE = 6


@dataclass(frozen=True)
class TextLayout:
    font_size: int
    lines: tuple[str, ...]
    line_height: int
    truncated: bool


@lru_cache(maxsize=128)
def _font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    candidates = (
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        (
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
        ),
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


@lru_cache(maxsize=1)
def _background() -> Any:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_TOP)
    draw = ImageDraw.Draw(image)
    for y in range(HEIGHT):
        alpha = y / max(HEIGHT - 1, 1)
        color = tuple(
            round(start * (1.0 - alpha) + end * alpha)
            for start, end in zip(BACKGROUND_TOP, BACKGROUND_BOTTOM)
        )
        draw.line((0, y, WIDTH, y), fill=color)
    return image


def _fit(image: Any, width: int, height: int):
    from PIL import Image

    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), (4, 7, 13))
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def _text_width(draw: Any, text: str, font: Any) -> int:
    box = draw.textbbox((0, 0), text or " ", font=font)
    return int(box[2] - box[0])


def _split_token(draw: Any, token: str, font: Any, width: int) -> list[str]:
    if _text_width(draw, token, font) <= width:
        return [token]
    chunks: list[str] = []
    current = ""
    for character in token:
        candidate = current + character
        if current and _text_width(draw, candidate, font) > width:
            chunks.append(current)
            current = character
        else:
            current = candidate
    if current or not chunks:
        chunks.append(current)
    return chunks


def _wrap_text(draw: Any, text: str, font: Any, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            for chunk in _split_token(draw, word, font, width):
                candidate = chunk if not current else f"{current} {chunk}"
                if current and _text_width(draw, candidate, font) > width:
                    lines.append(current)
                    current = chunk
                else:
                    current = candidate
        lines.append(current)
    return lines


def _ellipsize(draw: Any, text: str, font: Any, width: int) -> str:
    suffix = "..."
    value = text
    while value and _text_width(draw, value + suffix, font) > width:
        value = value[:-1]
    return value.rstrip() + suffix


def _layout_text(
    draw: Any,
    text: str,
    *,
    width: int,
    height: int,
    max_size: int,
    min_size: int,
    max_lines: int | None = None,
    bold: bool = False,
) -> TextLayout:
    if width <= 0 or height <= 0:
        return TextLayout(min_size, (), min_size + 4, bool(text))
    chosen: tuple[int, list[str], int, int] | None = None
    for size in range(max_size, min_size - 1, -1):
        font = _font(size, bold=bold)
        line_height = size + max(4, size // 4)
        capacity = max(1, height // line_height)
        if max_lines is not None:
            capacity = min(capacity, max_lines)
        lines = _wrap_text(draw, text, font, width)
        chosen = (size, lines, line_height, capacity)
        if len(lines) <= capacity:
            return TextLayout(size, tuple(lines), line_height, False)
    assert chosen is not None
    size, lines, line_height, capacity = chosen
    visible = lines[:capacity]
    if visible:
        visible[-1] = _ellipsize(draw, visible[-1], _font(size, bold=bold), width)
    return TextLayout(size, tuple(visible), line_height, len(lines) > capacity)


def _draw_text_box(
    draw: Any,
    text: str,
    box: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int] = TEXT,
    max_size: int = 20,
    min_size: int = 14,
    max_lines: int | None = None,
    bold: bool = False,
) -> TextLayout:
    left, top, right, bottom = box
    layout = _layout_text(
        draw,
        text,
        width=right - left,
        height=bottom - top,
        max_size=max_size,
        min_size=min_size,
        max_lines=max_lines,
        bold=bold,
    )
    font = _font(layout.font_size, bold=bold)
    y = top
    for line in layout.lines:
        draw.text((left, y), line, font=font, fill=color)
        y += layout.line_height
    return layout


def _panel(
    draw: Any,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int] = PANEL,
    outline: tuple[int, int, int] = BORDER,
    radius: int = 16,
    width: int = 1,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _badge(
    draw: Any,
    x: int,
    y: int,
    text: str,
    *,
    color: tuple[int, int, int],
    fill: tuple[int, int, int] = PANEL_SOFT,
    size: int = 13,
) -> int:
    font = _font(size, bold=True)
    width = _text_width(draw, text, font) + 24
    draw.rounded_rectangle(
        (x, y, x + width, y + size + 15),
        radius=(size + 15) // 2,
        fill=fill,
        outline=color,
        width=1,
    )
    draw.ellipse((x + 9, y + 10, x + 15, y + 16), fill=color)
    draw.text((x + 20, y + 6), text, font=font, fill=color)
    return width


def _section_title(draw: Any, x: int, y: int, title: str, subtitle: str = "") -> None:
    draw.text((x, y), title, font=_font(14, bold=True), fill=MUTED)
    if subtitle:
        draw.text((x, y + 22), subtitle, font=_font(12), fill=DIM)


def _progress_bar(
    draw: Any,
    box: tuple[int, int, int, int],
    value: int | float,
    *,
    accent: tuple[int, int, int] = VIOLET,
) -> None:
    left, top, right, bottom = box
    clipped = max(0.0, min(100.0, float(value)))
    draw.rounded_rectangle(box, radius=(bottom - top) // 2, fill=(31, 41, 59))
    fill_right = left + round((right - left) * clipped / 100.0)
    if fill_right > left:
        draw.rounded_rectangle(
            (left, top, fill_right, bottom),
            radius=(bottom - top) // 2,
            fill=accent,
        )


def _view_rate(stream: Any) -> Fraction:
    if stream.average_rate is not None:
        return Fraction(stream.average_rate)
    return Fraction(20, 1)


def _active_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    profile: str,
    frame_index: int,
) -> dict[str, Mapping[str, Any]]:
    active: dict[str, Mapping[str, Any]] = {}
    candidates = sorted(
        (row for row in rows if row["profile"] == profile),
        key=lambda row: (
            int(row["anchor_frame"]),
            0 if row["category"] == "initial_plan" else 1,
            str(row["slot_id"]),
        ),
    )
    for row in candidates:
        if int(row["anchor_frame"]) > frame_index:
            continue
        variant = str(row["context_variant"])
        previous = active.get(variant)
        if previous is None or int(row["anchor_frame"]) >= int(previous["anchor_frame"]):
            active[variant] = row
    return active


def _initial_plan_state(
    rows: Sequence[Mapping[str, Any]], profile: str
) -> tuple[str, tuple[int, int, int]]:
    candidates = [
        row
        for row in rows
        if row["profile"] == profile and row["category"] == "initial_plan"
    ]
    if not candidates:
        return "INITIAL PLAN · NOT REQUESTED", DIM
    row = candidates[0]
    if row.get("status") != "generated":
        return "INITIAL PLAN · SKIPPED", ORANGE
    if not row.get("prediction_schema_valid"):
        return "INITIAL PLAN · INVALID JSON", RED
    prediction = row.get("prediction")
    if not isinstance(prediction, Mapping):
        return "INITIAL PLAN · UNAVAILABLE", RED
    count = len(prediction.get("initial_plan", []))
    return f"INITIAL PLAN · {count} STEPS", GREEN


def _initial_plan_row(
    rows: Sequence[Mapping[str, Any]], profile: str
) -> Mapping[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if row["profile"] == profile and row["category"] == "initial_plan"
        ),
        None,
    )


def _initial_plan_items(row: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if row is None or not row.get("prediction_schema_valid"):
        return []
    prediction = row.get("prediction")
    if not isinstance(prediction, Mapping):
        return []
    raw = prediction.get("initial_plan")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        return []
    return [item for item in raw if isinstance(item, Mapping)]


def _task_instruction(rows: Sequence[Mapping[str, Any]]) -> str:
    for row in rows:
        value = str(row.get("task_instruction") or "").strip()
        if value:
            return value
    return "Whole-episode rollout"


def _row_state(row: Mapping[str, Any] | None) -> tuple[str, tuple[int, int, int]]:
    if row is None:
        return "WAITING", DIM
    if row.get("status") != "generated":
        return "SKIPPED", ORANGE
    if _is_early_end(row):
        return "EARLY END", ORANGE
    if not row.get("prediction_schema_valid"):
        return "INVALID JSON", RED
    if not isinstance(row.get("prediction"), Mapping):
        return "MISSING OUTPUT", RED
    decision = str(row["prediction"].get("execution_decision") or "READY").upper()
    return decision, GREEN if decision == "END" else CYAN


def _is_end(row: Mapping[str, Any] | None) -> bool:
    if row is None or not row.get("prediction_schema_valid"):
        return False
    prediction = row.get("prediction")
    return (
        isinstance(prediction, Mapping)
        and str(prediction.get("execution_decision") or "").lower() == "end"
    )


def _early_end_prediction(
    row: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if (
        row is None
        or row.get("status") != "generated"
        or row.get("category") != "ongoing"
        or row.get("prediction_schema_valid")
        or EARLY_END_SCHEMA_ERROR
        not in str(row.get("prediction_schema_error") or "")
    ):
        return None
    raw = row.get("prediction_raw")
    output_spec = row.get("output_spec")
    if not isinstance(raw, str) or not isinstance(output_spec, Mapping):
        return None
    try:
        parsed = json.loads(raw)
        return validate_target(parsed, category="end", output_spec=output_spec)
    except (json.JSONDecodeError, V5ValidationError):
        return None


def _is_early_end(row: Mapping[str, Any] | None) -> bool:
    return _early_end_prediction(row) is not None


def _display_prediction(row: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if row is None:
        return None
    if row.get("prediction_schema_valid") and isinstance(row.get("prediction"), Mapping):
        return row["prediction"]
    return _early_end_prediction(row)


def _end_outcome(row: Mapping[str, Any]) -> str:
    prediction = row.get("prediction")
    if not isinstance(prediction, Mapping):
        return "COMPLETED"
    detail = prediction.get("decision_detail")
    if not isinstance(detail, Mapping):
        return "COMPLETED"
    return str(detail.get("outcome") or "completed").replace("_", " ").upper()


def _task_progress(row: Mapping[str, Any] | None) -> int | None:
    prediction = _display_prediction(row)
    if prediction is None:
        return None
    raw = prediction.get("task_progress_percent")
    return int(raw) if isinstance(raw, (int, float)) else None


def _memory_input(row: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if row is None:
        return {}
    value = row.get("memory_input")
    return value if isinstance(value, Mapping) else {}


def _visualization_memory(row: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if row is not None:
        value = row.get("visualization_memory_snapshot")
        if isinstance(value, Mapping):
            return value
    return _memory_input(row)


def _long_memory(row: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    value = _visualization_memory(row).get("long_memory")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _memory_entry_text(entry: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for unit, label in (("action", "A"), ("segment", "S")):
        value = entry.get(unit)
        if value:
            parts.append(f"{label}: {value}")
    return "  /  ".join(parts) or "Empty memory entry"


def _short_memory_text(row: Mapping[str, Any] | None) -> str:
    short = _visualization_memory(row).get("short_memory")
    if _is_end(row):
        prediction = _display_prediction(row)
        predictions = prediction.get("predictions") if prediction is not None else None
        if (
            isinstance(predictions, Sequence)
            and not isinstance(predictions, (str, bytes))
            and predictions
            and isinstance(predictions[0], Mapping)
        ):
            short = {
                "task_progress_percent": prediction.get("task_progress_percent"),
                "prediction1": predictions[0],
            }
    if not isinstance(short, Mapping):
        return "Memory is empty at the first causal anchor."
    prediction = short.get("prediction1")
    if not isinstance(prediction, Mapping):
        return "Short memory is present without prediction content."
    parts: list[str] = []
    for unit, label in (("action", "ACTION"), ("segment", "SEGMENT")):
        value = prediction.get(unit)
        if isinstance(value, Mapping) and value.get("caption"):
            parts.append(
                f"{label} [{value.get('progress_percent', 'n/a')}%]  {value['caption']}"
            )
    progress = short.get("task_progress_percent")
    prefix = f"TASK {progress}%\n" if isinstance(progress, (int, float)) else ""
    return prefix + "\n".join(parts or ["Short memory has no available unit."])


def _prediction_items(row: Mapping[str, Any] | None) -> Sequence[Any]:
    prediction = _display_prediction(row)
    if prediction is None:
        return ()
    values = prediction.get("predictions")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return ()
    return values


def _prediction_texts(
    row: Mapping[str, Any] | None,
    *,
    profile: str,
    prediction_index: int,
) -> list[tuple[str, str, int | None]]:
    values = _prediction_items(row)
    if prediction_index >= len(values) or not isinstance(values[prediction_index], Mapping):
        return []
    item = values[prediction_index]
    expected_units: set[str] | None = None
    if row is not None and isinstance(row.get("output_spec"), Mapping):
        raw_expected = row["output_spec"].get(
            f"prediction{prediction_index + 1}_units"
        )
        if isinstance(raw_expected, Sequence) and not isinstance(
            raw_expected, (str, bytes)
        ):
            expected_units = {str(unit) for unit in raw_expected}
    result: list[tuple[str, str, int | None]] = []
    for unit in PROFILE_UNITS[profile]:
        if expected_units is not None and unit not in expected_units:
            continue
        value = item.get(unit)
        if not isinstance(value, Mapping) or not value.get("available", True):
            result.append((unit.upper(), "Unavailable", None))
            continue
        caption = str(value.get("caption") or "Unavailable")
        raw_progress = value.get("progress_percent")
        progress = int(raw_progress) if isinstance(raw_progress, (int, float)) else None
        result.append((unit.upper(), caption, progress))
    return result


def _current_unit_progress_timeline(
    active_timeline: Sequence[Mapping[str, Mapping[str, Any]]],
    *,
    profile: str,
    variant: str,
) -> list[dict[str, dict[str, int | None]]]:
    """Interpolate display progress within each contiguous current-caption span.

    Model-reported progress is retained separately for display. This function
    changes no prediction records and uses only the already-rendered timeline.
    """
    result: list[dict[str, dict[str, int | None]]] = [
        {} for _ in active_timeline
    ]
    for unit in PROFILE_UNITS[profile]:
        label = unit.upper()
        keys: list[str | None] = []
        model_progress: list[int | None] = []
        for active in active_timeline:
            row = active.get(variant)
            values = _prediction_texts(
                row,
                profile=profile,
                prediction_index=0,
            )
            selected = next((value for value in values if value[0] == label), None)
            if selected is None or selected[1] == "Unavailable":
                keys.append(None)
                model_progress.append(None)
                continue
            keys.append(" ".join(selected[1].lower().split()))
            model_progress.append(selected[2])
        start = 0
        while start < len(keys):
            key = keys[start]
            if key is None:
                start += 1
                continue
            end = start
            while end + 1 < len(keys) and keys[end + 1] == key:
                end += 1
            width = end - start
            for frame_index in range(start, end + 1):
                span_progress = (
                    100
                    if width == 0
                    else round(100 * (frame_index - start) / width)
                )
                result[frame_index][label] = {
                    "span_progress": span_progress,
                    "model_progress": model_progress[frame_index],
                }
            start = end + 1
    return result


def _error_text(row: Mapping[str, Any] | None) -> str | None:
    if row is None:
        return "Waiting for the first causal anchor."
    if row.get("status") != "generated":
        return f"Dependency unavailable: {row.get('skip_reason', 'unknown reason')}"
    if _is_early_end(row):
        return None
    if not row.get("prediction_schema_valid"):
        error = str(row.get("prediction_schema_error") or "Schema validation failed")
        raw = str(row.get("prediction_raw") or "").replace("\n", " ")
        excerpt = raw[:220] + ("..." if len(raw) > 220 else "")
        return f"{error}\nRAW  {excerpt}"
    if not isinstance(row.get("prediction"), Mapping):
        return "Parsed prediction is missing."
    return None


def _technical_line(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return "anchor --  ·  input --  ·  output --  ·  latency --"
    anchor = row.get("anchor_frame", "--")
    input_tokens = row.get("input_tokens")
    output_tokens = row.get("output_tokens")
    latency = row.get("generation_seconds")
    latency_text = f"{float(latency):.2f}s" if isinstance(latency, (int, float)) else "--"
    return (
        f"anchor {anchor}  ·  input {input_tokens if input_tokens is not None else '--'}  ·  "
        f"output {output_tokens if output_tokens is not None else '--'}  ·  latency {latency_text}"
    )


def _draw_topbar(
    canvas: Any,
    *,
    task: str,
    frame_index: int,
    total_frames: int,
    fps: float,
    mode: str,
    state_override: str | None = None,
) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    _panel(draw, (24, 18, WIDTH - 24, 84), fill=(14, 22, 39))
    draw.rounded_rectangle((42, 31, 84, 73), radius=11, fill=(29, 48, 71), outline=CYAN)
    draw.text((55, 38), "W", font=_font(21, bold=True), fill=CYAN)
    draw.text((100, 31), "Wall-Planner", font=_font(20, bold=True), fill=WHITE)
    draw.text((100, 57), "EMBODIED PLANNING CONSOLE", font=_font(10, bold=True), fill=DIM)
    draw.text((540, 29), "ACTIVE TASK", font=_font(10, bold=True), fill=DIM)
    _draw_text_box(
        draw,
        task,
        (540, 46, 1110, 74),
        color=TEXT,
        max_size=17,
        min_size=13,
        max_lines=1,
        bold=True,
    )
    draw.text((1160, 29), mode, font=_font(11, bold=True), fill=MUTED)
    seconds = frame_index / max(fps, 1e-6)
    draw.text(
        (1160, 50),
        f"FRAME {frame_index + 1:04d}/{total_frames:04d}  ·  {seconds:05.2f}s",
        font=_font(14, bold=True),
        fill=TEXT,
    )
    state = state_override or ("END" if frame_index == total_frames - 1 else "RUNNING")
    _badge(draw, 1680, 36, state, color=GREEN, fill=(18, 48, 46), size=12)


def _draw_observations(
    canvas: Any,
    *,
    images: Sequence[Any],
    views: Sequence[str],
    box: tuple[int, int, int, int],
    frame_index: int,
    total_frames: int,
    compact: bool = False,
) -> None:
    from PIL import ImageDraw

    draw = ImageDraw.Draw(canvas)
    left, top, right, bottom = box
    _panel(draw, box)
    _section_title(draw, left + 18, top + 15, "VISUAL OBSERVATIONS", "SYNCHRONIZED MULTI-VIEW INPUT")
    _badge(draw, right - 94, top + 14, "LIVE", color=GREEN, fill=(16, 43, 42), size=10)
    camera_top = top + 57
    timeline_height = 42 if not compact else 0
    camera_bottom = bottom - 17 - timeline_height
    gap = 10
    inner_width = right - left - 36
    cell_width = (inner_width - gap * (len(images) - 1)) // len(images)
    for index, (view, image) in enumerate(zip(views, images)):
        cell_left = left + 18 + index * (cell_width + gap)
        cell_box = (cell_left, camera_top, cell_left + cell_width, camera_bottom)
        draw.rounded_rectangle(cell_box, radius=11, fill=(4, 7, 13), outline=BORDER_SOFT)
        fitted = _fit(image, cell_width - 2, camera_bottom - camera_top - 2)
        canvas.paste(fitted, (cell_left + 1, camera_top + 1))
        draw.rounded_rectangle(
            (cell_left + 10, camera_top + 10, cell_left + 154, camera_top + 37),
            radius=7,
            fill=(4, 7, 13),
        )
        draw.text(
            (cell_left + 19, camera_top + 15),
            view.replace("_", " ").upper(),
            font=_font(11, bold=True),
            fill=WHITE,
        )
    if not compact:
        timeline_top = bottom - 35
        draw.text((left + 18, timeline_top - 2), "EPISODE", font=_font(10, bold=True), fill=DIM)
        progress_left = left + 91
        progress_right = right - 95
        value = 100.0 * frame_index / max(total_frames - 1, 1)
        _progress_bar(draw, (progress_left, timeline_top + 2, progress_right, timeline_top + 10), value)
        draw.text(
            (right - 77, timeline_top - 3),
            f"{round(value):3d}%",
            font=_font(11, bold=True),
            fill=MUTED,
        )


def _draw_branch_header(
    draw: Any,
    box: tuple[int, int, int, int],
    *,
    row: Mapping[str, Any] | None,
    variant: str,
    accent: tuple[int, int, int],
) -> None:
    left, top, right, bottom = box
    draw.text(
        (left, top),
        _display_context_label(row, variant),
        font=_font(16, bold=True),
        fill=accent,
    )
    state, state_color = _row_state(row)
    _badge(draw, left, top + 28, state, color=state_color, fill=PANEL_ALT, size=10)
    progress = _task_progress(row)
    progress_text = "--" if progress is None else str(progress)
    draw.text((left + 150, top + 32), "MODEL TASK", font=_font(10, bold=True), fill=DIM)
    draw.text((left + 240, top + 27), f"{progress_text}%", font=_font(18, bold=True), fill=WHITE)
    _progress_bar(draw, (left + 300, top + 36, right, top + 44), progress or 0, accent=accent)
    draw.text((left, bottom - 16), _technical_line(row), font=_font(10), fill=DIM)


def _draw_unit_progress(
    draw: Any,
    box: tuple[int, int, int, int],
    *,
    unit: str,
    progress: int | None,
    model_progress: int | None = None,
) -> None:
    left, top, right, _bottom = box
    accent = UNIT_ACCENTS.get(unit, CYAN)
    unit_text = f"{unit} SPAN" if model_progress is not None else unit
    unit_font = _font(10, bold=True)
    draw.text((left, top), unit_text, font=unit_font, fill=accent)
    value_text = "--%" if progress is None else f"{progress}%"
    if model_progress is not None:
        value_text += f" · MODEL {model_progress}%"
    value_font = _font(9 if model_progress is not None else 10, bold=True)
    value_width = _text_width(draw, value_text, value_font)
    value_left = right - value_width
    bar_left = left + _text_width(draw, unit_text, unit_font) + 12
    bar_right = max(bar_left + 1, value_left - 10)
    _progress_bar(
        draw,
        (bar_left, top + 5, bar_right, top + 11),
        progress if progress is not None else 0,
        accent=accent,
    )
    draw.text((value_left, top), value_text, font=value_font, fill=TEXT)


def _draw_prediction_card(
    draw: Any,
    box: tuple[int, int, int, int],
    *,
    row: Mapping[str, Any] | None,
    profile: str,
    prediction_index: int,
    title: str,
    accent: tuple[int, int, int],
    max_size: int,
    min_size: int,
    compact: bool = False,
    unit_progress_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    left, top, right, bottom = box
    _panel(draw, box, fill=PANEL_ALT, outline=accent, radius=13)
    draw.text((left + 15, top + 12), title, font=_font(12, bold=True), fill=accent)
    error = _error_text(row)
    if error is not None:
        if prediction_index > 0:
            error = (
                "Next subtask unavailable because this anchor did not produce "
                "a schema-valid prediction."
            )
        _draw_text_box(
            draw,
            error,
            (left + 15, top + 42, right - 15, bottom - 13),
            color=RED if row is not None else MUTED,
            max_size=max_size,
            min_size=min_size,
            max_lines=5 if not compact else 3,
        )
        return
    values = _prediction_texts(row, profile=profile, prediction_index=prediction_index)
    if not values:
        text = (
            "No next subtask is present in this anchor's output contract."
            if prediction_index > 0
            else "Prediction unit unavailable."
        )
        _draw_text_box(
            draw,
            text,
            (left + 15, top + 42, right - 15, bottom - 13),
            color=ORANGE if prediction_index > 0 and _is_early_end(row) else MUTED,
            max_size=max_size,
            min_size=min_size,
        )
        return
    content_top = top + 39
    content_height = bottom - content_top - 10
    section_height = max(1, content_height // len(values))
    for index, (unit, caption, progress) in enumerate(values):
        section_top = content_top + index * section_height
        if index:
            draw.line((left + 15, section_top, right - 15, section_top), fill=BORDER_SOFT)
        override = (
            unit_progress_overrides.get(unit)
            if unit_progress_overrides is not None
            else None
        )
        display_progress = (
            int(override["span_progress"])
            if isinstance(override, Mapping)
            and isinstance(override.get("span_progress"), (int, float))
            else progress
        )
        model_progress = (
            int(override["model_progress"])
            if isinstance(override, Mapping)
            and isinstance(override.get("model_progress"), (int, float))
            else None
        )
        _draw_unit_progress(
            draw,
            (left + 15, section_top + 6, right - 15, section_top + 20),
            unit=unit,
            progress=display_progress,
            model_progress=model_progress,
        )
        _draw_text_box(
            draw,
            caption,
            (left + 15, section_top + 25, right - 15, section_top + section_height - 5),
            color=TEXT,
            max_size=max_size,
            min_size=min_size,
            max_lines=2 if compact else None,
        )


def _draw_end_card(
    draw: Any,
    box: tuple[int, int, int, int],
    *,
    row: Mapping[str, Any],
    profile: str,
    max_size: int,
    min_size: int,
) -> None:
    left, top, right, bottom = box
    _panel(draw, box, fill=(11, 29, 35), outline=GREEN, radius=14, width=2)
    draw.ellipse(
        (left + 18, top + 18, left + 66, top + 66),
        fill=(20, 60, 52),
        outline=GREEN,
        width=2,
    )
    draw.text((left + 31, top + 25), "✓", font=_font(22, bold=True), fill=GREEN)
    draw.text(
        (left + 82, top + 17),
        "EPISODE COMPLETE",
        font=_font(22, bold=True),
        fill=GREEN,
    )
    draw.text(
        (left + 82, top + 48),
        f"OUTCOME  ·  {_end_outcome(row)}",
        font=_font(11, bold=True),
        fill=MUTED,
    )
    draw.line((left + 18, top + 82, right - 18, top + 82), fill=(39, 75, 69))
    draw.text(
        (left + 18, top + 97),
        "FINAL OBSERVED SUBTASK",
        font=_font(11, bold=True),
        fill=CYAN,
    )
    values = _prediction_texts(row, profile=profile, prediction_index=0)
    content_top = top + 124
    content_bottom = bottom - 42
    section_height = max(1, (content_bottom - content_top) // max(len(values), 1))
    for index, (unit, caption, progress) in enumerate(values):
        section_top = content_top + index * section_height
        if index:
            draw.line(
                (left + 18, section_top, right - 18, section_top),
                fill=(34, 59, 59),
            )
        _draw_unit_progress(
            draw,
            (left + 18, section_top + 7, right - 18, section_top + 21),
            unit=unit,
            progress=progress,
        )
        _draw_text_box(
            draw,
            caption,
            (
                left + 18,
                section_top + 27,
                right - 18,
                section_top + section_height - 5,
            ),
            color=WHITE,
            max_size=max_size,
            min_size=min_size,
        )
    draw.text(
        (left + 18, bottom - 29),
        "TERMINAL DECISION  ·  NO NEXT SUBTASK",
        font=_font(10, bold=True),
        fill=GREEN,
    )


def _display_context_label(
    row: Mapping[str, Any] | None,
    variant: str,
) -> str:
    if row is not None and row.get("execution_context_policy") == "observations_only":
        return "OBSERVATIONS ONLY · DISPLAY MEMORY"
    return VARIANT_LABELS[variant]


def _draw_context_status(
    draw: Any,
    x: int,
    y: int,
    variant: str,
    row: Mapping[str, Any] | None = None,
) -> None:
    if row is not None and row.get("execution_context_policy") == "observations_only":
        label = "MODEL INPUT · OBSERVATIONS ONLY"
    else:
        label = (
            "CONTEXT · INITIAL PLAN ACTIVE"
            if variant == "with_memory_with_initial"
            else "CONTEXT · NO INITIAL PLAN"
        )
    _badge(
        draw,
        x,
        y,
        label,
        color=BLUE if variant == "no_memory_no_initial" else VIOLET,
        fill=PANEL_ALT,
        size=10,
    )


def _draw_focus_memory(
    draw: Any,
    box: tuple[int, int, int, int],
    *,
    row: Mapping[str, Any] | None,
    variant: str,
) -> None:
    left, top, right, bottom = box
    if variant == "no_memory_no_initial":
        _panel(draw, box)
        _section_title(draw, left + 18, top + 17, "MEMORY", "CONTEXT POLICY")
        _badge(draw, left + 18, top + 60, "DISABLED", color=DIM, fill=PANEL_ALT, size=13)
        _draw_text_box(
            draw,
            "Memory is disabled by experiment design. Every anchor is inferred from the current synchronized observations and task instruction only.",
            (left + 18, top + 112, left + 720, bottom - 25),
            color=MUTED,
            max_size=21,
            min_size=16,
            max_lines=4,
        )
        draw.line((left + 760, top + 30, left + 760, bottom - 30), fill=BORDER_SOFT)
        _section_title(draw, left + 795, top + 17, "CURRENT ANCHOR", "MODEL TELEMETRY")
        state, color = _row_state(row)
        _badge(draw, left + 795, top + 60, state, color=color, fill=PANEL_ALT, size=13)
        _draw_text_box(
            draw,
            _technical_line(row),
            (left + 795, top + 112, right - 24, bottom - 25),
            color=TEXT,
            max_size=20,
            min_size=15,
            max_lines=3,
        )
        return

    short_right = left + 480
    _panel(draw, (left, top, short_right, bottom))
    _section_title(draw, left + 18, top + 17, "SHORT MEMORY", "PREVIOUS COMMITTED SUBTASK")
    _draw_text_box(
        draw,
        _short_memory_text(row),
        (left + 18, top + 63, short_right - 18, bottom - 20),
        color=TEXT,
        max_size=19,
        min_size=14,
    )
    long_left = short_right + 18
    _panel(draw, (long_left, top, right, bottom))
    entries = _long_memory(row)
    _section_title(draw, long_left + 18, top + 17, "LONG MEMORY", "COMPLETED TASK HISTORY")
    draw.text((right - 112, top + 18), f"{len(entries)}/8 ENTRIES", font=_font(11, bold=True), fill=MUTED)
    if not entries:
        _draw_text_box(
            draw,
            "Long memory is empty at this anchor.",
            (long_left + 18, top + 66, right - 18, bottom - 20),
            color=MUTED,
            max_size=19,
            min_size=14,
        )
        return
    grid_top = top + 55
    grid_bottom = bottom - 13
    column_gap = 10
    cell_width = (right - long_left - 36 - column_gap) // 2
    row_height = (grid_bottom - grid_top) // 4
    for index, entry in enumerate(entries[-8:]):
        column = index // 4
        row_index = index % 4
        cell_left = long_left + 18 + column * (cell_width + column_gap)
        cell_top = grid_top + row_index * row_height
        cell_box = (
            cell_left,
            cell_top + 3,
            cell_left + cell_width,
            cell_top + row_height - 3,
        )
        draw.rounded_rectangle(cell_box, radius=9, fill=PANEL_ALT, outline=BORDER_SOFT)
        memory_index = entry.get("index", index + 1)
        draw.text((cell_left + 10, cell_top + 10), f"{int(memory_index):02d}", font=_font(10, bold=True), fill=CYAN)
        _draw_text_box(
            draw,
            _memory_entry_text(entry),
            (cell_left + 38, cell_top + 8, cell_left + cell_width - 9, cell_top + row_height - 7),
            color=MUTED,
            max_size=14,
            min_size=11,
            max_lines=3,
        )


def _draw_comparison_memory(
    draw: Any,
    box: tuple[int, int, int, int],
    *,
    row: Mapping[str, Any] | None,
    variant: str,
) -> None:
    left, top, right, bottom = box
    _panel(draw, box, fill=PANEL_ALT, outline=BORDER_SOFT, radius=11)
    if variant == "no_memory_no_initial":
        draw.text((left + 13, top + 11), "MEMORY DISABLED", font=_font(11, bold=True), fill=DIM)
        draw.text((left + 13, top + 35), "Clean current-context baseline", font=_font(12), fill=MUTED)
        return
    entries = _long_memory(row)
    draw.text((left + 13, top + 10), f"MEMORY ACTIVE  ·  {len(entries)}/8 LONG", font=_font(11, bold=True), fill=GREEN)
    latest = entries[-2:]
    text = "  |  ".join(_memory_entry_text(entry) for entry in latest)
    if not text:
        text = _short_memory_text(row).replace("\n", "  ")
    _draw_text_box(
        draw,
        text,
        (left + 13, top + 33, right - 13, bottom - 8),
        color=MUTED,
        max_size=13,
        min_size=10,
        max_lines=2,
    )


def _plan_item_copy(item: Mapping[str, Any]) -> tuple[str, str]:
    action = item.get("action")
    if not isinstance(action, Mapping):
        return "Action unavailable", ""
    action_caption = str(action.get("caption") or "Action unavailable")
    raw_segments = action.get("segments")
    if isinstance(raw_segments, (str, bytes)) or not isinstance(raw_segments, Sequence):
        return action_caption, ""
    captions: list[str] = []
    for raw in raw_segments:
        if not isinstance(raw, Mapping):
            continue
        segment = raw.get("segment")
        if isinstance(segment, Mapping) and str(segment.get("caption") or "").strip():
            captions.append(str(segment["caption"]).strip())
    return action_caption, "  →  ".join(captions)


def _initial_plan_canvas(
    *,
    images: Sequence[Any],
    views: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    profile: str,
    page_index: int,
    page_count: int,
) -> Any:
    from PIL import ImageDraw

    canvas = _background().copy()
    _draw_topbar(
        canvas,
        task=_task_instruction(rows),
        frame_index=0,
        total_frames=1,
        fps=1.0,
        mode=f"{PROFILE_LABELS[profile]}  /  INITIAL PLAN",
        state_override="PLAN",
    )
    _draw_observations(
        canvas,
        images=images,
        views=views,
        box=(24, 104, 1896, 434),
        frame_index=0,
        total_frames=1,
        compact=True,
    )
    draw = ImageDraw.Draw(canvas)
    _panel(draw, (24, 454, 1896, 1056))
    row = _initial_plan_row(rows, profile)
    state, color = _initial_plan_state(rows, profile)
    _section_title(
        draw,
        44,
        471,
        "INITIAL PLAN",
        "MODEL-PREDICTED GLOBAL TASK DECOMPOSITION",
    )
    _badge(draw, 1518, 467, state, color=color, fill=PANEL_ALT, size=11)
    if page_count > 1:
        draw.text(
            (1752, 475),
            f"{page_index + 1}/{page_count}",
            font=_font(11, bold=True),
            fill=MUTED,
        )
    items = _initial_plan_items(row)
    if not items:
        error = _error_text(row) if row is not None else None
        _draw_text_box(
            draw,
            error or "A valid model-predicted initial plan is unavailable.",
            (44, 540, 1876, 1018),
            color=RED,
            max_size=28,
            min_size=18,
            max_lines=8,
        )
        return canvas

    page = items[
        page_index * INITIAL_PLAN_PAGE_SIZE:
        (page_index + 1) * INITIAL_PLAN_PAGE_SIZE
    ]
    gap_x = 14
    gap_y = 12
    grid_left, grid_top, grid_right, grid_bottom = 44, 525, 1876, 1035
    card_width = (grid_right - grid_left - gap_x) // 2
    card_height = (grid_bottom - grid_top - 2 * gap_y) // 3
    for offset, item in enumerate(page):
        column = offset % 2
        row_index = offset // 2
        left = grid_left + column * (card_width + gap_x)
        top = grid_top + row_index * (card_height + gap_y)
        right = left + card_width
        bottom = top + card_height
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=11,
            fill=PANEL_ALT,
            outline=BORDER_SOFT,
        )
        plan_index = int(item.get("index", page_index * INITIAL_PLAN_PAGE_SIZE + offset + 1))
        draw.text(
            (left + 14, top + 12),
            f"{plan_index:02d}",
            font=_font(13, bold=True),
            fill=CYAN,
        )
        draw.text(
            (left + 54, top + 13),
            "ACTION",
            font=_font(10, bold=True),
            fill=BLUE,
        )
        action_caption, segment_copy = _plan_item_copy(item)
        _draw_text_box(
            draw,
            action_caption,
            (left + 54, top + 34, right - 14, top + 93),
            color=WHITE,
            max_size=17,
            min_size=12,
            max_lines=3,
            bold=True,
        )
        if segment_copy:
            draw.text(
                (left + 54, top + 101),
                "SEGMENTS",
                font=_font(9, bold=True),
                fill=VIOLET,
            )
            _draw_text_box(
                draw,
                segment_copy,
                (left + 132, top + 99, right - 14, bottom - 10),
                color=MUTED,
                max_size=13,
                min_size=10,
                max_lines=3,
            )
    return canvas


def _focus_canvas(
    *,
    images: Sequence[Any],
    views: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    active: Mapping[str, Mapping[str, Any]],
    profile: str,
    variant: str,
    frame_index: int,
    total_frames: int,
    fps: float,
    current_unit_progress: Mapping[str, Mapping[str, Any]] | None = None,
) -> Any:
    from PIL import ImageDraw

    row = active.get(variant)
    canvas = _background().copy()
    _draw_topbar(
        canvas,
        task=_task_instruction(rows),
        frame_index=frame_index,
        total_frames=total_frames,
        fps=fps,
        mode=f"{PROFILE_LABELS[profile]}  /  {_display_context_label(row, variant)}",
    )
    _draw_observations(
        canvas,
        images=images,
        views=views,
        box=(24, 104, 1228, 684),
        frame_index=frame_index,
        total_frames=total_frames,
    )
    draw = ImageDraw.Draw(canvas)
    _panel(draw, (1248, 104, 1896, 684))
    if _is_end(row):
        _section_title(draw, 1266, 120, "FINAL OUTCOME", "TERMINAL EPISODE STATE")
        assert row is not None
        _badge(
            draw,
            1550,
            116,
            f"OUTCOME · {_end_outcome(row)}",
            color=GREEN,
            fill=(11, 29, 35),
            size=10,
        )
    elif _is_early_end(row):
        _section_title(
            draw,
            1266,
            120,
            "MODEL END SIGNAL",
            "EPISODE MEDIA CONTINUES",
        )
        _badge(
            draw,
            1550,
            116,
            "PREMATURE END",
            color=ORANGE,
            fill=(47, 36, 20),
            size=10,
        )
    else:
        _section_title(
            draw,
            1266,
            120,
            "ROLLING PREDICTION",
            "CONTINUOUS TWO-SUBTASK FORECAST",
        )
        _draw_context_status(draw, 1510, 116, variant, row)
    accent = BLUE if variant == "no_memory_no_initial" else VIOLET
    _draw_branch_header(draw, (1266, 165, 1878, 235), row=row, variant=variant, accent=accent)
    if _is_end(row):
        assert row is not None
        _draw_end_card(
            draw,
            (1266, 244, 1878, 668),
            row=row,
            profile=profile,
            max_size=22,
            min_size=15,
        )
    else:
        _draw_prediction_card(
            draw,
            (1266, 244, 1878, 444),
            row=row,
            profile=profile,
            prediction_index=0,
            title="CURRENT SUBTASK",
            accent=accent,
            max_size=21,
            min_size=14,
            unit_progress_overrides=current_unit_progress,
        )
        draw.text((1566, 447), "↓", font=_font(18, bold=True), fill=DIM)
        _draw_prediction_card(
            draw,
            (1266, 468, 1878, 668),
            row=row,
            profile=profile,
            prediction_index=1,
            title="NEXT SUBTASK",
            accent=VIOLET,
            max_size=21,
            min_size=14,
        )
    _draw_focus_memory(draw, (24, 704, 1896, 1056), row=row, variant=variant)
    return canvas


def _comparison_canvas(
    *,
    images: Sequence[Any],
    views: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    active: Mapping[str, Mapping[str, Any]],
    profile: str,
    frame_index: int,
    total_frames: int,
    fps: float,
) -> Any:
    from PIL import ImageDraw

    canvas = _background().copy()
    _draw_topbar(
        canvas,
        task=_task_instruction(rows),
        frame_index=frame_index,
        total_frames=total_frames,
        fps=fps,
        mode=f"{PROFILE_LABELS[profile]}  /  MEMORY COMPARISON",
    )
    _draw_observations(
        canvas,
        images=images,
        views=views,
        box=(24, 104, 1896, 434),
        frame_index=frame_index,
        total_frames=total_frames,
        compact=True,
    )
    draw = ImageDraw.Draw(canvas)
    gap = 18
    column_width = (1872 - gap) // 2
    for index, variant in enumerate(FOCUS_VARIANTS):
        left = 24 + index * (column_width + gap)
        right = left + column_width
        _panel(draw, (left, 454, right, 1056))
        row = active.get(variant)
        accent = BLUE if variant == "no_memory_no_initial" else VIOLET
        _draw_branch_header(draw, (left + 18, 471, right - 18, 548), row=row, variant=variant, accent=accent)
        if index == 1 and not _is_end(row):
            _draw_context_status(draw, right - 285, 470, variant, row)
        if _is_end(row):
            assert row is not None
            _draw_end_card(
                draw,
                (left + 18, 557, right - 18, 968),
                row=row,
                profile=profile,
                max_size=20,
                min_size=13,
            )
        else:
            _draw_prediction_card(
                draw,
                (left + 18, 557, right - 18, 757),
                row=row,
                profile=profile,
                prediction_index=0,
                title="CURRENT SUBTASK",
                accent=accent,
                max_size=19,
                min_size=13,
            )
            _draw_prediction_card(
                draw,
                (left + 18, 768, right - 18, 968),
                row=row,
                profile=profile,
                prediction_index=1,
                title="NEXT SUBTASK",
                accent=VIOLET,
                max_size=19,
                min_size=13,
            )
        _draw_comparison_memory(draw, (left + 18, 979, right - 18, 1038), row=row, variant=variant)
    return canvas


def _compact_prediction_text(
    row: Mapping[str, Any] | None,
    *,
    profile: str,
    prediction_index: int,
) -> str:
    error = _error_text(row)
    if error is not None:
        return error.splitlines()[0]
    values = _prediction_texts(row, profile=profile, prediction_index=prediction_index)
    if not values and prediction_index > 0:
        return "No next subtask in output contract"
    return "  /  ".join(
        f"{unit} {progress if progress is not None else '--'}% · {caption}"
        for unit, caption, progress in values
    ) or "Prediction unavailable"


def _draw_overview_branch(
    draw: Any,
    box: tuple[int, int, int, int],
    *,
    row: Mapping[str, Any] | None,
    profile: str,
    variant: str,
) -> None:
    left, top, right, bottom = box
    accent = BLUE if variant == "no_memory_no_initial" else VIOLET
    _panel(draw, box, fill=PANEL_ALT, outline=BORDER_SOFT, radius=11)
    draw.text(
        (left + 12, top + 9),
        _display_context_label(row, variant),
        font=_font(11, bold=True),
        fill=accent,
    )
    state, state_color = _row_state(row)
    draw.text((left + 140, top + 9), state, font=_font(10, bold=True), fill=state_color)
    progress = _task_progress(row)
    draw.text(
        (right - 105, top + 8),
        f"TASK {progress if progress is not None else '--'}%",
        font=_font(11, bold=True),
        fill=WHITE,
    )
    _draw_text_box(
        draw,
        (
            "FINAL  "
            if _is_end(row)
            else "END SIGNAL  "
            if _is_early_end(row)
            else "NOW  "
        )
        + _compact_prediction_text(row, profile=profile, prediction_index=0),
        (left + 12, top + 33, right - 12, top + 82),
        color=WHITE if _is_end(row) else TEXT,
        max_size=16,
        min_size=12,
        max_lines=2,
    )
    current_values = _prediction_texts(
        row,
        profile=profile,
        prediction_index=0,
    )
    progress_top = top + 85
    for index, (unit, _caption, progress_value) in enumerate(current_values):
        unit_top = progress_top + index * 15
        _draw_unit_progress(
            draw,
            (left + 12, unit_top, right - 12, unit_top + 13),
            unit=unit,
            progress=progress_value,
        )
    detail_top = progress_top + len(current_values) * 15 + 5
    if _is_end(row):
        assert row is not None
        _draw_text_box(
            draw,
            f"EPISODE COMPLETE  ·  OUTCOME {_end_outcome(row)}  ·  NO NEXT SUBTASK",
            (left + 12, detail_top, right - 12, bottom - 25),
            color=GREEN,
            max_size=15,
            min_size=11,
            max_lines=2,
            bold=True,
        )
    else:
        _draw_text_box(
            draw,
            "NEXT  " + _compact_prediction_text(row, profile=profile, prediction_index=1),
            (left + 12, detail_top, right - 12, bottom - 25),
            color=MUTED,
            max_size=15,
            min_size=11,
            max_lines=2,
        )
    if variant == "with_memory_no_initial":
        count = len(_long_memory(row))
        draw.text((right - 118, bottom - 20), f"MEM {count}/8", font=_font(9, bold=True), fill=GREEN)


def _overview_canvas(
    *,
    images: Sequence[Any],
    views: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    active_by_profile: Mapping[str, Mapping[str, Mapping[str, Any]]],
    frame_index: int,
    total_frames: int,
    fps: float,
) -> Any:
    from PIL import ImageDraw

    canvas = _background().copy()
    _draw_topbar(
        canvas,
        task=_task_instruction(rows),
        frame_index=frame_index,
        total_frames=total_frames,
        fps=fps,
        mode="ALL PROFILES  /  MEMORY COMPARISON",
    )
    _draw_observations(
        canvas,
        images=images,
        views=views,
        box=(24, 104, 1896, 414),
        frame_index=frame_index,
        total_frames=total_frames,
        compact=True,
    )
    draw = ImageDraw.Draw(canvas)
    panel_top = 434
    panel_bottom = 1056
    _panel(draw, (24, panel_top, 1896, panel_bottom))
    row_height = (panel_bottom - panel_top - 36) // len(PROFILES)
    label_width = 204
    card_gap = 12
    card_width = (1872 - 36 - label_width - card_gap) // 2
    for index, profile in enumerate(PROFILES):
        top = panel_top + 18 + index * row_height
        if index:
            draw.line((42, top - 7, 1878, top - 7), fill=BORDER_SOFT)
        draw.text((44, top + 8), PROFILE_LABELS[profile], font=_font(15, bold=True), fill=ORANGE)
        _draw_text_box(
            draw,
            "DISPLAY CONTEXT\nNO INITIAL PLAN",
            (44, top + 42, 44 + label_width - 10, top + 92),
            color=DIM,
            max_size=11,
            min_size=9,
            max_lines=2,
            bold=True,
        )
        active = active_by_profile[profile]
        for branch_index, variant in enumerate(FOCUS_VARIANTS):
            left = 44 + label_width + branch_index * (card_width + card_gap)
            _draw_overview_branch(
                draw,
                (left, top, left + card_width, top + row_height - 15),
                row=active.get(variant),
                profile=profile,
                variant=variant,
            )
    return canvas


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_one(
    *,
    spec: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    profile: str | None,
    variant: str | None = None,
    end_hold_seconds: float = 2.0,
    initial_plan_page_seconds: float = 0.0,
    encoder_preset: str = "medium",
    source_frame_step: int = 1,
) -> dict[str, Any]:
    import av

    if variant is not None and profile is None:
        raise ValueError("a focus variant requires a profile")
    if profile is not None and profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    if variant is not None and variant not in SUPPORTED_FOCUS_VARIANTS:
        raise ValueError(f"unsupported focus variant: {variant}")
    if not math.isfinite(end_hold_seconds) or end_hold_seconds < 0:
        raise ValueError("end_hold_seconds must be finite and non-negative")
    if (
        not math.isfinite(initial_plan_page_seconds)
        or initial_plan_page_seconds < 0
    ):
        raise ValueError(
            "initial_plan_page_seconds must be finite and non-negative"
        )
    if initial_plan_page_seconds and profile is None:
        raise ValueError("initial-plan prelude requires a focus profile")
    if encoder_preset not in {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
        "placebo",
    }:
        raise ValueError(f"unsupported libx264 preset: {encoder_preset}")
    if not isinstance(source_frame_step, int) or source_frame_step < 1:
        raise ValueError("source_frame_step must be a positive integer")
    videos = dict(spec["videos"])
    views = tuple(sorted(videos))
    containers = [av.open(videos[view], mode="r") for view in views]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.partial{output_path.suffix}")
    total_frames = int(spec["total_frames"])
    expected_fps = float(spec["fps"])
    if profile is None:
        active_timeline: Any = {
            name: [
                _active_rows(rows, profile=name, frame_index=frame_index)
                for frame_index in range(total_frames)
            ]
            for name in PROFILES
        }
    else:
        active_timeline = [
            _active_rows(rows, profile=profile, frame_index=frame_index)
            for frame_index in range(total_frames)
        ]
    current_unit_progress_timeline = (
        _current_unit_progress_timeline(
            active_timeline,
            profile=profile,
            variant=variant,
        )
        if profile is not None and variant is not None
        else None
    )
    if profile is None:
        has_terminal_end = any(
            _is_end(active_timeline[name][-1].get(context_variant))
            for name in PROFILES
            for context_variant in FOCUS_VARIANTS
        )
    elif variant is None:
        has_terminal_end = any(
            _is_end(active_timeline[-1].get(context_variant))
            for context_variant in FOCUS_VARIANTS
        )
    else:
        has_terminal_end = _is_end(active_timeline[-1].get(variant))
    end_hold_frames = 0
    initial_plan_hold_frames = 0
    initial_plan_pages = 0
    try:
        streams = [container.streams.video[0] for container in containers]
        fps = _view_rate(streams[0])
        output_fps = fps / source_frame_step
        if abs(float(fps) - expected_fps) > 1e-6:
            raise ValueError(f"source FPS {float(fps)} differs from spec {expected_fps}")
        for stream in streams[1:]:
            if abs(float(_view_rate(stream)) - float(fps)) > 1e-6:
                raise ValueError("source view FPS values differ")
        if has_terminal_end:
            end_hold_frames = round(float(output_fps) * end_hold_seconds)
        plan_items = _initial_plan_items(
            _initial_plan_row(rows, profile) if profile is not None else None
        )
        if initial_plan_page_seconds:
            if not plan_items:
                raise ValueError(
                    "initial-plan prelude requires a schema-valid model initial plan"
                )
            initial_plan_pages = math.ceil(
                len(plan_items) / INITIAL_PLAN_PAGE_SIZE
            )
            initial_plan_hold_frames = (
                initial_plan_pages
                * round(float(output_fps) * initial_plan_page_seconds)
            )
        decoders = [
            iter(container.decode(stream))
            for container, stream in zip(containers, streams)
        ]
        output = av.open(
            str(temporary), mode="w", format="mp4", options={"movflags": "+faststart"}
        )
        try:
            stream_out = output.add_stream("libx264", rate=output_fps)
            stream_out.width = WIDTH
            stream_out.height = HEIGHT
            stream_out.pix_fmt = "yuv420p"
            stream_out.options = {"crf": "22", "preset": encoder_preset}
            last_canvas: Any | None = None
            first_frames: list[Any] | None = None
            if initial_plan_hold_frames:
                try:
                    first_frames = [next(decoder) for decoder in decoders]
                except StopIteration as exc:
                    raise ValueError("a source view has no first frame") from exc
                first_images = [
                    frame.to_image().convert("RGB") for frame in first_frames
                ]
                frames_per_page = round(
                    float(output_fps) * initial_plan_page_seconds
                )
                for page_index in range(initial_plan_pages):
                    canvas = _initial_plan_canvas(
                        images=first_images,
                        views=views,
                        rows=rows,
                        profile=profile,
                        page_index=page_index,
                        page_count=initial_plan_pages,
                    )
                    for page_frame in range(frames_per_page):
                        encoded = av.VideoFrame.from_image(canvas)
                        encoded.pts = page_index * frames_per_page + page_frame
                        encoded.time_base = Fraction(1, 1) / output_fps
                        for packet in stream_out.encode(encoded):
                            output.mux(packet)
            selected_frame_sequence = list(
                range(0, total_frames, source_frame_step)
            )
            if selected_frame_sequence[-1] != total_frames - 1:
                selected_frame_sequence[-1] = total_frames - 1
            selected_frame_indices = set(selected_frame_sequence)
            rendered_source_frames = 0
            for frame_index in range(total_frames):
                if frame_index == 0 and first_frames is not None:
                    frames = first_frames
                else:
                    try:
                        frames = [next(decoder) for decoder in decoders]
                    except StopIteration as exc:
                        raise ValueError(
                            f"a source view ended before frame {total_frames}"
                        ) from exc
                if frame_index not in selected_frame_indices:
                    continue
                images = [frame.to_image().convert("RGB") for frame in frames]
                if profile is None:
                    canvas = _overview_canvas(
                        images=images,
                        views=views,
                        rows=rows,
                        active_by_profile={
                            name: active_timeline[name][frame_index] for name in PROFILES
                        },
                        frame_index=frame_index,
                        total_frames=total_frames,
                        fps=float(fps),
                    )
                elif variant is None:
                    canvas = _comparison_canvas(
                        images=images,
                        views=views,
                        rows=rows,
                        active=active_timeline[frame_index],
                        profile=profile,
                        frame_index=frame_index,
                        total_frames=total_frames,
                        fps=float(fps),
                    )
                else:
                    canvas = _focus_canvas(
                        images=images,
                        views=views,
                        rows=rows,
                        active=active_timeline[frame_index],
                        profile=profile,
                        variant=variant,
                        frame_index=frame_index,
                        total_frames=total_frames,
                        fps=float(fps),
                        current_unit_progress=(
                            current_unit_progress_timeline[frame_index]
                            if current_unit_progress_timeline is not None
                            else None
                        ),
                    )
                encoded = av.VideoFrame.from_image(canvas)
                last_canvas = canvas
                encoded.pts = initial_plan_hold_frames + rendered_source_frames
                encoded.time_base = Fraction(1, 1) / output_fps
                for packet in stream_out.encode(encoded):
                    output.mux(packet)
                rendered_source_frames += 1
            for decoder in decoders:
                try:
                    next(decoder)
                except StopIteration:
                    pass
                else:
                    raise ValueError("a source view contains frames beyond total_frames")
            if end_hold_frames:
                assert last_canvas is not None
                for hold_index in range(end_hold_frames):
                    encoded = av.VideoFrame.from_image(last_canvas)
                    encoded.pts = (
                        initial_plan_hold_frames
                        + len(selected_frame_indices)
                        + hold_index
                    )
                    encoded.time_base = Fraction(1, 1) / output_fps
                    for packet in stream_out.encode(encoded):
                        output.mux(packet)
            for packet in stream_out.encode():
                output.mux(packet)
        finally:
            output.close()
    except BaseException:
        if temporary.is_file():
            temporary.unlink()
        raise
    finally:
        for container in containers:
            container.close()
    os.replace(temporary, output_path)
    expected_rendered_frames = (
        initial_plan_hold_frames + len(selected_frame_sequence) + end_hold_frames
    )
    with av.open(str(output_path), mode="r") as verification:
        stream = verification.streams.video[0]
        frames = sum(1 for _frame in verification.decode(stream))
        if frames != expected_rendered_frames:
            raise RuntimeError(
                f"rendered frame count {frames} != {expected_rendered_frames}"
            )
        verified_fps = float(_view_rate(stream))
        report = {
            "output": str(output_path),
            "layout_version": LAYOUT_VERSION,
            "profile": profile or "combined",
            "context_variant": variant or ("comparison" if profile else "all"),
            "ground_truth_visible": False,
            "current_unit_progress_policy": (
                CURRENT_UNIT_PROGRESS_POLICY
                if current_unit_progress_timeline is not None
                else "model_reported_v1"
            ),
            "frames": frames,
            "source_frames": total_frames,
            "rendered_source_frames": len(selected_frame_sequence),
            "source_frame_step": source_frame_step,
            "source_fps": float(fps),
            "initial_plan_pages": initial_plan_pages,
            "initial_plan_hold_frames": initial_plan_hold_frames,
            "initial_plan_page_seconds": initial_plan_page_seconds,
            "initial_plan_hold_seconds": initial_plan_hold_frames / verified_fps,
            "end_hold_frames": end_hold_frames,
            "end_hold_seconds": end_hold_frames / verified_fps,
            "duration_seconds": frames / verified_fps,
            "fps": verified_fps,
            "width": stream.width,
            "height": stream.height,
            "codec": stream.codec_context.name,
            "encoder_crf": 22,
            "encoder_preset": encoder_preset,
            "size": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        }
    write_json(output_path.with_suffix(".video.json"), report)
    return report


def render_focus_video(
    *,
    spec: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    profile: str,
    context_variant: str,
    initial_plan_page_seconds: float = 2.0,
    end_hold_seconds: float = 2.0,
    encoder_preset: str = "medium",
    source_frame_step: int = 1,
) -> dict[str, Any]:
    """Render one explicitly selected rollout branch with an optional plan prelude."""

    return _render_one(
        spec=spec,
        rows=rows,
        output_path=output_path,
        profile=profile,
        variant=context_variant,
        initial_plan_page_seconds=initial_plan_page_seconds,
        end_hold_seconds=end_hold_seconds,
        encoder_preset=encoder_preset,
        source_frame_step=source_frame_step,
    )


def render_all_videos(
    *,
    spec: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    video_set: str = "full",
    source_predictions_sha256: str | None = None,
    end_hold_seconds: float = 2.0,
) -> dict[str, Any]:
    if video_set not in {"core", "full"}:
        raise ValueError("video_set must be 'core' or 'full'")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for profile in PROFILES:
        reports[profile] = _render_one(
            spec=spec,
            rows=rows,
            output_path=output_dir / f"{profile}.mp4",
            profile=profile,
            end_hold_seconds=end_hold_seconds,
        )
    reports["combined"] = _render_one(
        spec=spec,
        rows=rows,
        output_path=output_dir / "combined.mp4",
        profile=None,
        end_hold_seconds=end_hold_seconds,
    )
    if video_set == "full":
        for profile in PROFILES:
            for variant, suffix in (
                ("no_memory_no_initial", "no_memory"),
                ("with_memory_no_initial", "with_memory"),
            ):
                name = f"{profile}_{suffix}"
                reports[name] = _render_one(
                    spec=spec,
                    rows=rows,
                    output_path=output_dir / f"{name}.mp4",
                    profile=profile,
                    variant=variant,
                    end_hold_seconds=end_hold_seconds,
                )
    manifest = {
        "schema_version": "v5_3_video_render_v5",
        "layout_version": LAYOUT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video_set": video_set,
        "ground_truth_visible": False,
        "requested_end_hold_seconds": end_hold_seconds,
        "source_predictions_sha256": source_predictions_sha256,
        "source_total_rows": len(rows),
        "renderer_sha256": _sha256(Path(__file__).resolve()),
        "presentation_policy": {
            "early_end": "show_structurally_valid_ongoing_end_as_orange_warning",
            "early_end_rows": sum(1 for row in rows if _is_early_end(row)),
            "schema_validity_unchanged": True,
        },
        "outputs": reports,
    }
    write_json(output_dir / "render_result.json", manifest)
    return reports


def _load_artifact(artifact_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    artifact_dir = artifact_dir.resolve()
    manifest_path = artifact_dir / "run_manifest.json"
    predictions_path = artifact_dir / "predictions.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episode_spec = Path(str(manifest.get("episode_spec") or ""))
    if not episode_spec.is_absolute() or not episode_spec.is_file():
        raise FileNotFoundError(f"missing episode spec from run manifest: {episode_spec}")
    from .rollout import load_episode_spec

    spec = load_episode_spec(episode_spec, verify_sources=True)
    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = int(manifest.get("slot_count", len(rows)))
    if len(rows) != expected:
        raise ValueError(f"prediction row count {len(rows)} != manifest slot count {expected}")
    return spec, rows, predictions_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--video-set", choices=("core", "full"), default="full")
    parser.add_argument("--end-hold-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)
    artifact_dir = args.artifact_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else artifact_dir / "videos_console_v4"
    )
    spec, rows, predictions_path = _load_artifact(artifact_dir)
    reports = render_all_videos(
        spec=spec,
        rows=rows,
        output_dir=output_dir,
        video_set=args.video_set,
        source_predictions_sha256=_sha256(predictions_path),
        end_hold_seconds=args.end_hold_seconds,
    )
    result = {
        "artifact_dir": str(artifact_dir),
        "output_dir": str(output_dir),
        "layout_version": LAYOUT_VERSION,
        "video_count": len(reports),
        "videos": reports,
        "render_result": str(output_dir / "render_result.json"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["LAYOUT_VERSION", "render_all_videos", "render_focus_video"]
