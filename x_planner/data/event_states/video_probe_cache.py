"""Load stat-bound Takeover video metadata caches."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any


def load_video_probe_cache(
    path: Path,
) -> dict[Path, tuple[int, int, Fraction, int]]:
    """Load path -> size, mtime_ns, fps, frame_count from a complete cache."""

    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "v5_takeover_video_probe_cache_v1"
        or payload.get("complete") is not True
        or not isinstance(payload.get("entries"), dict)
    ):
        raise ValueError(f"invalid or incomplete Takeover video probe cache: {path}")
    result: dict[Path, tuple[int, int, Fraction, int]] = {}
    for raw_path, raw in payload["entries"].items():
        if not isinstance(raw_path, str) or not isinstance(raw, dict):
            raise ValueError(f"invalid Takeover video probe cache entry: {raw_path!r}")
        value = Path(raw_path)
        if not value.is_absolute() or ".." in value.parts:
            raise ValueError(f"unsafe Takeover video probe cache path: {raw_path!r}")
        fps = Fraction(int(raw["fps_numerator"]), int(raw["fps_denominator"]))
        size = int(raw["size"])
        mtime_ns = int(raw["mtime_ns"])
        frame_count = int(raw["frame_count"])
        if fps <= 0 or size <= 0 or frame_count <= 0:
            raise ValueError(f"invalid Takeover video probe cache values: {raw_path!r}")
        result[value] = (size, mtime_ns, fps, frame_count)
    if len(result) != int(payload.get("num_entries") or -1):
        raise ValueError(f"Takeover video probe cache count mismatch: {path}")
    return result
