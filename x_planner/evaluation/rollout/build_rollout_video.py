"""Render one-screen 4K video for an Initial-Plan + streaming V4 rollout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

import av
from PIL import Image, ImageDraw, ImageFont


WIDTH = 3840
HEIGHT = 2160
FPS = 20
INTRO_SECONDS = 4
PLAYBACK_REPEAT = 2
BACKGROUND = "#07101f"
PANEL = "#101a2e"
WHITE = "#f8fafc"
MUTED = "#aab6cc"
CYAN = "#67e8f9"
GREEN = "#86efac"
ORANGE = "#fbbf24"
RED = "#fda4af"
FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
VIEW_BOXES = {
    "head": (40, 140, 2190, 1232),
    "left_wrist": (40, 1430, 1070, 690),
    "right_wrist": (1160, 1430, 1070, 690),
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD_PATH if bold else FONT_PATH), size)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: Any, width: int) -> list[str]:
    result: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        words = paragraph.split()
        if not words:
            result.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if draw.textbbox((0, 0), candidate, font=font)[2] <= width:
                line = candidate
            else:
                result.append(line)
                line = word
        result.append(line)
    return result


def _fit_image(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    x, y, width, height = box
    prepared = image.convert("RGB").copy()
    prepared.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas.paste(prepared, (x + (width - prepared.width) // 2, y + (height - prepared.height) // 2))


def _plan_lines(target: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for item in target["initial_plan"]:
        subtask = item.get("subtask")
        if subtask is not None:
            lines.append(f"L2 #{item['index']}: {subtask['caption']}")
            for action_entry in subtask.get("actions", []):
                action = action_entry["action"]
                lines.append(f"  L1 #{action_entry['index']}: {action['caption']}")
                for segment_entry in action.get("segments", []):
                    segment = segment_entry["l0"]
                    lines.append(
                        f"    L0 #{segment_entry['index']} ({segment['source']}): {segment['caption']}"
                    )
            continue
        action = item.get("action")
        if action is not None:
            lines.append(f"L1 #{item['index']}: {action['caption']}")
            for segment_entry in action.get("segments", []):
                segment = segment_entry["l0"]
                lines.append(
                    f"  L0 #{segment_entry['index']} ({segment['source']}): {segment['caption']}"
                )
            continue
        segment = item["l0"]
        lines.append(f"L0 #{item['index']} ({segment['source']}): {segment['caption']}")
    return lines


def _continuous_lines(target: dict[str, Any]) -> list[str]:
    lines = [f"Task progress: {target['task_progress_percent']}%"]
    for prediction in target["predictions"]:
        lines.append(f"Prediction {prediction['index']}:")
        for field, label in (("subtask", "L2"), ("action", "L1"), ("l0", "L0")):
            if field not in prediction:
                continue
            unit = prediction[field]
            source = f", {unit['source']}" if field == "l0" else ""
            lines.append(
                f"  {label}{source} [{unit['progress_percent']}%]: {unit['caption']}"
            )
    return lines


def _memory_lines(record: dict[str, Any]) -> list[str]:
    long_memory = record["input_long_memory"]
    short_memory = record["input_short_memory"]
    lines = ["Long Memory: [none]" if not long_memory else "Long Memory:"]
    lines.extend(f"  {value}" for value in long_memory)
    if not short_memory:
        lines.append("Short Memory: [none] (first continuous anchor)")
    else:
        lines.append("Short Memory (previous MODEL Prediction 1):")
        lines.extend(
            f"  [{value['progress_percent']}%] {value['caption']}" for value in short_memory
        )
    return lines


Block = tuple[str, list[str], str]


def _panel_height(
    draw: ImageDraw.ImageDraw, blocks: Iterable[Block], body_size: int, width: int
) -> int:
    body = _font(body_size)
    total = 0
    for title, lines, _ in blocks:
        total += body_size + 12
        for line in lines:
            total += len(_wrap(draw, line, body, width)) * (body_size + 5)
        total += 14
    return total


def _draw_text_panel(
    canvas: Image.Image,
    blocks: list[Block],
    *,
    preferred_body_size: int = 24,
    minimum_body_size: int = 13,
) -> None:
    if preferred_body_size < minimum_body_size or minimum_body_size <= 0:
        raise ValueError("invalid text panel font-size bounds")
    draw = ImageDraw.Draw(canvas)
    x, y, width, bottom = 2290, 135, 1495, 2120
    draw.rounded_rectangle((2260, 115, 3815, 2135), radius=20, fill=PANEL, outline="#334155", width=3)
    body_size = preferred_body_size
    while body_size > minimum_body_size and _panel_height(draw, blocks, body_size, width) > bottom - y:
        body_size -= 1
    if _panel_height(draw, blocks, body_size, width) > bottom - y:
        raise RuntimeError("semantic text does not fit the 4K review panel")
    body = _font(body_size)
    title_font = _font(body_size + 4, bold=True)
    for title, lines, color in blocks:
        draw.text((x, y), title, font=title_font, fill=color)
        y += body_size + 12
        for text in lines:
            for line in _wrap(draw, text, body, width):
                draw.text((x, y), line, font=body, fill=WHITE)
                y += body_size + 5
        y += 14


def _draw_views(canvas: Image.Image, frames: dict[str, Image.Image]) -> None:
    draw = ImageDraw.Draw(canvas)
    for view, box in VIEW_BOXES.items():
        x, y, width, height = box
        draw.rounded_rectangle((x, y, x + width, y + height), radius=16, fill="#020617", outline="#334155", width=3)
        _fit_image(canvas, frames[view], box)
        label = view.replace("_", " ").upper()
        draw.rounded_rectangle((x + 14, y + 14, x + 275, y + 58), radius=8, fill="#020617")
        draw.text((x + 26, y + 21), label, font=_font(24, bold=True), fill=CYAN)


def _render(
    frames: dict[str, Image.Image],
    *,
    instruction: str,
    status: str,
    blocks: list[Block],
    preferred_body_size: int = 24,
    minimum_body_size: int = 13,
) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, WIDTH, 105), fill="#030b18")
    draw.text((40, 20), f'Task instruction (L3): "{instruction}"', font=_font(34, bold=True), fill=WHITE)
    status_width = draw.textbbox((0, 0), status, font=_font(27, bold=True))[2]
    draw.text((WIDTH - status_width - 45, 28), status, font=_font(27, bold=True), fill=CYAN)
    _draw_views(canvas, frames)
    _draw_text_panel(
        canvas,
        blocks,
        preferred_body_size=preferred_body_size,
        minimum_body_size=minimum_body_size,
    )
    return canvas


def _decode_video(path: str) -> tuple[list[Image.Image], float]:
    frames: list[Image.Image] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else float(FPS)
        for frame in container.decode(stream):
            frames.append(frame.to_image().convert("RGB"))
    if not frames:
        raise RuntimeError(f"decoded no frames from {path}")
    return frames, fps


def _view_paths(initial: dict[str, Any]) -> dict[str, str]:
    paths = {str(item["view"]): str(item["video"]) for item in initial["images"]}
    missing = set(VIEW_BOXES) - set(paths)
    if missing:
        raise RuntimeError(f"missing source views: {sorted(missing)}")
    return paths


def _encode_video(
    output: Path,
    intro: Image.Image,
    source_frames: list[Image.Image],
) -> None:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-threads", "16", "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        intro_bytes = intro.tobytes()
        for _ in range(INTRO_SECONDS * FPS):
            process.stdin.write(intro_bytes)
        for frame in source_frames:
            payload = frame.tobytes()
            for _ in range(PLAYBACK_REPEAT):
                process.stdin.write(payload)
    finally:
        process.stdin.close()
    assert process.stderr is not None
    error = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg failed with code {return_code}: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rollout_dir = args.rollout_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    summary = _read_json(rollout_dir / "summary.json")
    initial = _read_json(rollout_dir / "initial_plan.json")
    anchors = sorted(
        (_read_json(path) for path in rollout_dir.glob("anchor_*.json")),
        key=lambda record: int(record["anchor_frame"]),
    )
    paths = _view_paths(initial)
    decoded: dict[str, list[Image.Image]] = {}
    source_fps: dict[str, float] = {}
    for view, path in paths.items():
        decoded[view], source_fps[view] = _decode_video(path)
    frame_count = min(len(frames) for frames in decoded.values())
    if frame_count <= int(anchors[-1]["anchor_frame"]):
        raise RuntimeError("source video ends before the final inference anchor")

    frame_zero = {view: decoded[view][0] for view in VIEW_BOXES}
    intro_blocks: list[Block] = [
        ("INITIAL PLAN — MODEL (generated once)", _plan_lines(initial["prediction"]), GREEN),
        ("INITIAL PLAN — GROUND TRUTH", _plan_lines(initial["gt"]), ORANGE),
        ("MEMORY STATE", ["Long Memory: [none]", "Short Memory: [none]"], CYAN),
    ]
    intro = _render(
        frame_zero,
        instruction=summary["task_instruction"],
        status="INITIAL PLAN / EPISODE START",
        blocks=intro_blocks,
    )
    intro.save(output_dir / "poster_initial_plan.png", optimize=True)

    rendered_source: list[Image.Image] = []
    active_index = 0
    posters: list[str] = []
    for frame_index in range(frame_count):
        while active_index + 1 < len(anchors) and int(anchors[active_index + 1]["anchor_frame"]) <= frame_index:
            active_index += 1
        record = anchors[active_index]
        model_terminal = "yes" if record["predicted_terminal"] else "no"
        gt_terminal = "yes" if record["is_terminal_window"] else "no"
        blocks = [
            ("INITIAL PLAN — MODEL (fixed for Episode)", _plan_lines(initial["prediction"]), GREEN),
            ("STREAMING MEMORY INPUT", _memory_lines(record), CYAN),
            (
                f"CONTINUOUS — MODEL (terminal={model_terminal})",
                _continuous_lines(record["prediction"]),
                GREEN,
            ),
            (
                f"CONTINUOUS — GT (terminal={gt_terminal})",
                _continuous_lines(record["gt"]),
                ORANGE,
            ),
        ]
        frames = {view: decoded[view][frame_index] for view in VIEW_BOXES}
        rendered = _render(
            frames,
            instruction=summary["task_instruction"],
            status=(
                f"STREAMING / frame={frame_index}/{frame_count - 1} / "
                f"anchor={record['anchor_frame']} / 20 Hz"
            ),
            blocks=blocks,
        )
        rendered_source.append(rendered)
        if frame_index == int(record["anchor_frame"]):
            name = f"poster_anchor_{frame_index:06d}.png"
            rendered.save(output_dir / name, optimize=True)
            posters.append(name)

    video = output_dir / "full_episode_streaming_demo_4k.mp4"
    _encode_video(video, intro, rendered_source)
    manifest = {
        "schema_version": "memory_v4_full_episode_video_demo_v1",
        "rollout_dir": str(rollout_dir),
        "checkpoint": summary["checkpoint"],
        "snapshot": summary["snapshot"],
        "task_instruction": summary["task_instruction"],
        "source_videos": paths,
        "source_fps": source_fps,
        "source_frame_count": frame_count,
        "anchors": [int(record["anchor_frame"]) for record in anchors],
        "intro_seconds": INTRO_SECONDS,
        "playback_repeat": PLAYBACK_REPEAT,
        "output_fps": FPS,
        "output_seconds": INTRO_SECONDS + frame_count * PLAYBACK_REPEAT / FPS,
        "video": str(video),
        "posters": ["poster_initial_plan.png", *posters],
        "short_memory_mode": summary["short_memory_mode"],
        "gt_short_memory_used_as_input": summary["gt_short_memory_used_as_input"],
        "initial_plan_fed_to_continuous_prompt": summary["initial_plan_fed_to_continuous_prompt"],
        "predicted_terminal_count": summary["predicted_terminal_count"],
        "gt_terminal_count": summary["gt_terminal_count"],
    }
    _write_json(output_dir / "video_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
