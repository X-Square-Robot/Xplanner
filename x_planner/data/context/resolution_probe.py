"""Stratified, no-copy resolution probe for Memory V3 frame references."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from x_planner.data.video_frames import VideoFrameVisionProcessor
from x2robot_dataset_v2.processors.vision.base import smart_resize

from .common import iter_jsonl, write_json, write_jsonl_atomic
from .dataset import auto_near_640_dimensions


def _rows(snapshot: Path, tasks: tuple[str, ...]) -> Iterator[tuple[str, dict[str, Any]]]:
    import struct

    for task in tasks:
        path = snapshot / "datasets" / task / "train" / "data.jsonl"
        index_path = snapshot / "datasets" / task / "train" / "data.index"
        if not path.is_file() or not index_path.is_file():
            continue
        with path.open("rb") as data, index_path.open("rb") as index:
            while True:
                raw = index.read(8)
                if not raw:
                    break
                if len(raw) != 8:
                    raise EOFError(index_path)
                data.seek(struct.unpack("<Q", raw)[0])
                wrapped = json.loads(data.readline())
                sample = wrapped.get("v3_sample")
                if isinstance(sample, dict):
                    yield task, sample


def _mode_dimensions(height: int, width: int, images_per_sample: int) -> dict[str, tuple[int, int]]:
    split_cap = max(1024, 589824 // max(1, images_per_sample))
    current = smart_resize(height, width, factor=32, min_pixels=1024, max_pixels=split_cap)
    near_640 = auto_near_640_dimensions(height, width)
    original_capped = auto_near_640_dimensions(
        height, width, target_long_edge=max(height, width), factor=32, pixel_cap=589824
    )
    return {"A_current": current, "B_auto_near_640": near_640, "C_original_capped": original_capped}


def probe(snapshot: Path, output: Path, *, limit: int, seed: int) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    rng = random.Random(seed)
    candidates: dict[tuple[str, str, str], list[tuple[str, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    max_candidates = max(limit * 12, 2000)
    per_stratum_cap = max(16, min(256, limit))
    scanned = 0
    for task, sample in _rows(snapshot, ("continuous", "initial_plan", "terminal")):
        images = sample.get("images") or ()
        for image in images:
            key = (str(sample.get("source_id")), str(image.get("view")), task)
            bucket = candidates[key]
            value = (task, sample, image)
            if len(bucket) < per_stratum_cap:
                bucket.append(value)
            else:
                position = rng.randrange(scanned + 1)
                if position < len(bucket):
                    bucket[position] = value
            scanned += 1
        if scanned >= max_candidates and sum(map(len, candidates.values())) >= limit:
            break
    selected = []
    keys = sorted(candidates)
    while len(selected) < limit and keys:
        next_keys = []
        for key in keys:
            bucket = candidates[key]
            if bucket:
                selected.append(bucket.pop(rng.randrange(len(bucket))))
                if len(selected) == limit:
                    break
            if bucket:
                next_keys.append(key)
        keys = next_keys
    decoder = VideoFrameVisionProcessor(
        image_factor=32,
        min_pixels=1024,
        max_pixels=589824,
        max_pixels_split_by_images=True,
        decoder_backend="av",
    )
    records = []
    failures: Counter[str] = Counter()
    for task, sample, image_ref in selected:
        try:
            image = decoder.load_image_refs([image_ref], str(snapshot / "manifest.json"))[0]
            modes = _mode_dimensions(image.height, image.width, len(sample["images"]))
            records.append({
                "sample_key": sample["sample_key"],
                "source_id": sample["source_id"],
                "task_type": task,
                "view": image_ref["view"],
                "video": image_ref["video"],
                "frame": image_ref["frame"],
                "original_width": image.width,
                "original_height": image.height,
                "aspect_ratio": image.width / image.height,
                "modes": {
                    name: {
                        "width": width,
                        "height": height,
                        "estimated_visual_tokens": (width // 32) * (height // 32),
                    }
                    for name, (height, width) in modes.items()
                },
            })
            image.close()
        except Exception as exc:
            failures[f"{type(exc).__name__}: {str(exc)[:160]}"] += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    records_path = output.with_suffix(".frames.jsonl")
    write_jsonl_atomic(records_path, records)
    mode_summary = {}
    for mode in ("A_current", "B_auto_near_640", "C_original_capped"):
        tokens = [row["modes"][mode]["estimated_visual_tokens"] for row in records]
        pixels = [row["modes"][mode]["width"] * row["modes"][mode]["height"] for row in records]
        mode_summary[mode] = {
            "frames": len(tokens),
            "mean_visual_tokens": sum(tokens) / len(tokens) if tokens else 0.0,
            "max_visual_tokens": max(tokens, default=0),
            "mean_pixels": sum(pixels) / len(pixels) if pixels else 0.0,
            "max_pixels": max(pixels, default=0),
            "batch_6_peak_memory_bytes": None,
            "batch_6_step_time_seconds": None,
            "gpu_status": "pending measured smoke; theoretical values are not substituted",
        }
    summary = {
        "schema_version": "memory_v3_resolution_probe_v1",
        "snapshot": str(snapshot),
        "requested_frames": limit,
        "decoded_frames": len(records),
        "failures": dict(failures),
        "strata": dict(Counter(
            f"{row['source_id']}|{row['view']}|{row['task_type']}" for row in records
        )),
        "modes": mode_summary,
        "records": str(records_path),
        "selected_policy": None,
        "acceptance": "B requires a real single-GPU batch=6 smoke without OOM",
    }
    write_json(str(output), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = probe(args.snapshot, args.output, limit=args.limit, seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
