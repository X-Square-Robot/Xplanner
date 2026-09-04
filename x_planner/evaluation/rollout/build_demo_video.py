"""Build reviewable MP4 demos from Memory V4 inference JSON artifacts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont
from x2robot_dataset_v2.readers.multimodal_jsonl_reader import load_indexed_jsonl_item

from x_planner.data.context.dataset import MemoryV3VisionProcessor


WIDTH = 1920
HEIGHT = 1080
BACKGROUND = "#0b1020"
PANEL = "#111a2e"
WHITE = "#f8fafc"
MUTED = "#aab6cc"
CYAN = "#5eead4"
GREEN = "#86efac"
ORANGE = "#fbbf24"
RED = "#fda4af"
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_BOLD = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT), size)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: Any, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if draw.textbbox((0, 0), candidate, font=font)[2] <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _draw_lines(
    draw: ImageDraw.ImageDraw,
    lines: Iterable[str],
    *,
    x: int,
    y: int,
    width: int,
    bottom: int,
    color: str = WHITE,
    size: int = 22,
    gap: int = 8,
) -> int:
    font = _font(size)
    line_height = size + 7
    for item in lines:
        wrapped = _wrap(draw, item, font, width)
        for line in wrapped:
            if y + line_height > bottom:
                draw.text((x, bottom - line_height), "...", font=font, fill=MUTED)
                return bottom
            draw.text((x, y), line, font=font, fill=color)
            y += line_height
        y += gap
    return y


def _fit_image(canvas: Image.Image, image: Image.Image, box: tuple[int, int, int, int]) -> None:
    x, y, width, height = box
    prepared = image.convert("RGB").copy()
    prepared.thumbnail((width, height), Image.Resampling.LANCZOS)
    px = x + (width - prepared.width) // 2
    py = y + (height - prepared.height) // 2
    canvas.paste(prepared, (px, py))


def _draw_media(
    canvas: Image.Image,
    images: dict[str, Image.Image],
    *,
    frame_offset: int,
) -> None:
    draw = ImageDraw.Draw(canvas)
    boxes = {
        "head": (35, 145, 1045, 555),
        "left_wrist": (35, 720, 510, 320),
        "right_wrist": (570, 720, 510, 320),
    }
    for view, box in boxes.items():
        x, y, width, height = box
        draw.rounded_rectangle(
            (x, y, x + width, y + height), radius=14, fill="#030712", outline="#334155", width=2
        )
        if view in images:
            _fit_image(canvas, images[view], box)
        label = f"{view}  [frame_offset={frame_offset}]"
        label_width = draw.textbbox((0, 0), label, font=_font(18, bold=True))[2]
        draw.rounded_rectangle(
            (x + 10, y + 10, x + 28 + label_width, y + 42), radius=8, fill="#020617"
        )
        draw.text((x + 19, y + 15), label, font=_font(18, bold=True), fill=CYAN)


def _base_slide(record: dict[str, Any], title: str) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, WIDTH, 120), fill="#07101f")
    draw.text((36, 20), title, font=_font(38, bold=True), fill=WHITE)
    instruction = f'Task instruction (L3): "{record["task_instruction"]}"'
    _draw_lines(draw, [instruction], x=36, y=70, width=1810, bottom=118, color=CYAN, size=23, gap=0)
    draw.rounded_rectangle((1110, 145, 1885, 1040), radius=18, fill=PANEL, outline="#334155", width=2)
    return canvas


def _input_slide(
    record: dict[str, Any], images: dict[str, Image.Image], offset: int
) -> Image.Image:
    title = f'MEMORY V4 / CHECKPOINT-5000 / {record["task"].upper()} / MODEL INPUT'
    canvas = _base_slide(record, title)
    _draw_media(canvas, images, frame_offset=offset)
    draw = ImageDraw.Draw(canvas)
    x, y, width = 1145, 180, 700
    draw.text((x, y), "WHAT THE MODEL SEES", font=_font(28, bold=True), fill=CYAN)
    y += 55
    lines = [
        f"Task: {record['task']}",
        f"Source: {record['source_id']}   Profile: {record['profile']}",
        f"Synchronized views: {record['image_count'] // (3 if record['image_count'] == 9 else 1)} views shown at this offset",
        f"Current input page: frame_offset={offset}",
        "Hierarchy: Task (L3) > Subtask (L2) > Action (L1) > Segment (L0)",
        "The human Task instruction is input context. It is not an Assistant output field.",
        "Only the sampled frames in this demo were passed to the model; no future frames are added.",
    ]
    _draw_lines(draw, lines, x=x, y=y, width=width, bottom=990, color=WHITE, size=23)
    return canvas


def _unit_lines(target: dict[str, Any], index: int) -> list[str]:
    prediction = target["predictions"][index]
    lines = []
    for field, label in (("subtask", "L2"), ("action", "L1"), ("l0", "L0")):
        value = prediction[field]
        lines.append(f"{label} [{value['progress_percent']}%]: {value['caption']}")
    return lines


def _comparison_slide(
    record: dict[str, Any],
    images: dict[str, Image.Image],
    *,
    prediction_index: int,
) -> Image.Image:
    phase = "CURRENT UNIT" if prediction_index == 0 else "NEXT UNIT / TERMINAL"
    title = f'MEMORY V4 / CHECKPOINT-5000 / {record["task"].upper()} / {phase}'
    canvas = _base_slide(record, title)
    _draw_media(canvas, images, frame_offset=0)
    draw = ImageDraw.Draw(canvas)
    pred = record["prediction"]
    gt = record["gt"]
    x, y, width = 1145, 170, 700
    draw.text((x, y), "MODEL PREDICTION", font=_font(27, bold=True), fill=GREEN)
    y += 48
    pred_lines = _unit_lines(pred, prediction_index)
    if prediction_index == 0:
        pred_lines.insert(0, f"Task progress: {pred['task_progress_percent']}%")
    y = _draw_lines(draw, pred_lines, x=x, y=y, width=width, bottom=600, color=WHITE, size=20, gap=5)
    y = max(y + 12, 610)
    draw.line((x, y, x + width, y), fill="#475569", width=2)
    y += 20
    draw.text((x, y), "GROUND TRUTH", font=_font(27, bold=True), fill=ORANGE)
    y += 48
    gt_lines = _unit_lines(gt, prediction_index)
    if prediction_index == 0:
        gt_lines.insert(0, f"Task progress: {gt['task_progress_percent']}%")
    _draw_lines(draw, gt_lines, x=x, y=y, width=width, bottom=1010, color=WHITE, size=20, gap=5)
    return canvas


def _plan_counts(plan: dict[str, Any]) -> tuple[int, int, int]:
    l2 = len(plan["initial_plan"])
    l1 = 0
    l0 = 0
    for item in plan["initial_plan"]:
        actions = item.get("subtask", {}).get("actions", [])
        l1 += len(actions)
        for action in actions:
            l0 += len(action.get("action", {}).get("segments", []))
    return l2, l1, l0


def _plan_lines(plan: dict[str, Any]) -> list[str]:
    counts = _plan_counts(plan)
    lines = [f"Hierarchy counts: L2={counts[0]}, L1={counts[1]}, L0={counts[2]}"]
    for item in plan["initial_plan"][:3]:
        subtask = item.get("subtask", {})
        lines.append(f"L2 #{item['index']}: {subtask.get('caption', '')}")
        for action in subtask.get("actions", [])[:3]:
            value = action.get("action", {})
            lines.append(f"  L1 #{action['index']}: {value.get('caption', '')}")
    return lines


def _plan_comparison_slide(
    record: dict[str, Any], images: dict[str, Image.Image]
) -> Image.Image:
    title = "MEMORY V4 / CHECKPOINT-5000 / INITIAL PLAN / PREDICTION VS GT"
    canvas = _base_slide(record, title)
    _draw_media(canvas, images, frame_offset=0)
    draw = ImageDraw.Draw(canvas)
    x, y, width = 1145, 170, 700
    draw.text((x, y), "MODEL PREDICTION", font=_font(27, bold=True), fill=GREEN)
    y = _draw_lines(
        draw, _plan_lines(record["prediction"]), x=x, y=y + 48, width=width, bottom=585, color=WHITE, size=19, gap=4
    )
    y = max(y + 10, 600)
    draw.line((x, y, x + width, y), fill="#475569", width=2)
    y += 20
    draw.text((x, y), "GROUND TRUTH", font=_font(27, bold=True), fill=ORANGE)
    _draw_lines(
        draw, _plan_lines(record["gt"]), x=x, y=y + 48, width=width, bottom=1010, color=WHITE, size=19, gap=4
    )
    return canvas


def _load_row(snapshot: Path, task: str, row_index: int) -> dict[str, Any]:
    root = snapshot / "datasets" / task / "validation"
    return load_indexed_jsonl_item(str(root), row_index)


def _decode_groups(row: dict[str, Any]) -> dict[int, dict[str, Image.Image]]:
    sample = row["v4_sample"]
    refs = list(row["image"])
    vision = MemoryV3VisionProcessor(
        image_factor=32,
        min_pixels=1024,
        max_pixels=589824,
        pixel_cap=589824,
        target_long_edge=640,
        decoder_backend="av",
    )
    images = vision.load_image_refs(refs, str(refs[0]["video"]))
    grouped: dict[int, dict[str, Image.Image]] = defaultdict(dict)
    for ref, spec, image in zip(refs, sample["images"], images):
        offset = int(spec.get("relative_frame", 0))
        grouped[offset][str(ref.get("view") or spec["view"])] = image
    return dict(grouped)


def _encode(slides: list[tuple[Path, float]], output: Path) -> None:
    concat = output.with_suffix(".concat.txt")
    with concat.open("w", encoding="utf-8") as handle:
        for image, duration in slides:
            handle.write(f"file '{image}'\n")
            handle.write(f"duration {duration:.3f}\n")
        handle.write(f"file '{slides[-1][0]}'\n")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat),
            "-vf", "fps=24,format=yuv420p", "-c:v", "libx264",
            "-preset", "medium", "-crf", "18", "-movflags", "+faststart",
            str(output),
        ],
        check=True,
    )


def _combine(videos: list[Path], output: Path) -> None:
    concat = output.with_suffix(".concat.txt")
    with concat.open("w", encoding="utf-8") as handle:
        for video in videos:
            handle.write(f"file '{video}'\n")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat),
            "-c", "copy", "-movflags", "+faststart", str(output),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inference_dir = args.inference_dir.resolve()
    snapshot = args.snapshot.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    slide_dir = output_dir / "slides"
    slide_dir.mkdir(parents=True)
    summary = json.loads((inference_dir / "summary.json").read_text(encoding="utf-8"))
    outputs: list[Path] = []
    manifest: dict[str, Any] = {
        "schema_version": "memory_v4_video_demo_v1",
        "checkpoint": summary["checkpoint"],
        "snapshot": str(snapshot),
        "inference_dir": str(inference_dir),
        "tasks": {},
    }
    for task in ("continuous", "initial_plan", "terminal"):
        record = json.loads((inference_dir / f"{task}.json").read_text(encoding="utf-8"))
        row = _load_row(snapshot, task, int(record["validation_row"]))
        groups = _decode_groups(row)
        slides: list[tuple[Path, float]] = []
        for position, offset in enumerate(sorted(groups)):
            image = _input_slide(record, groups[offset], offset)
            path = slide_dir / f"{task}_{position:02d}_input_{offset:+d}.png"
            image.save(path, optimize=True)
            slides.append((path, 1.8 if len(groups) > 1 else 3.0))
        current_images = groups[max(groups)]
        if task == "initial_plan":
            image = _plan_comparison_slide(record, current_images)
            path = slide_dir / f"{task}_{len(slides):02d}_prediction_vs_gt.png"
            image.save(path, optimize=True)
            slides.append((path, 7.0))
        else:
            for prediction_index, duration in ((0, 5.0), (1, 5.0)):
                image = _comparison_slide(
                    record, current_images, prediction_index=prediction_index
                )
                path = slide_dir / f"{task}_{len(slides):02d}_prediction_{prediction_index + 1}.png"
                image.save(path, optimize=True)
                slides.append((path, duration))
        output = output_dir / f"{task}_demo.mp4"
        _encode(slides, output)
        outputs.append(output)
        manifest["tasks"][task] = {
            "video": str(output),
            "sample_key": record["sample_key"],
            "validation_row": record["validation_row"],
            "slides": [{"path": str(path), "duration": duration} for path, duration in slides],
            "source_videos": sorted({str(ref["video"]) for ref in row["image"]}),
            "model_input_offsets": sorted(groups),
        }
    combined = output_dir / "memory_v4_three_task_demo.mp4"
    _combine(outputs, combined)
    manifest["combined_video"] = str(combined)
    _write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
