"""Strict full-decode video validation used by scanner workers."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import av

from .constants import DEFAULT_MAX_CAMERA_VIEWS, VIEW_PRIORITY


class VideoValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class VideoProbe:
    path: str
    size: int
    mtime_ns: int
    decoded_frames: int
    declared_frames: int | None
    average_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_video(path: str, expected_frames: int) -> VideoProbe:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    stat = os.stat(path)
    decoded_frames = 0
    declared_frames: int | None = None
    average_rate: float | None = None
    try:
        with av.open(path, mode="r") as container:
            if not container.streams.video:
                raise VideoValidationError("no video stream")
            stream = container.streams.video[0]
            declared_frames = int(stream.frames) if stream.frames else None
            if stream.average_rate is not None:
                average_rate = float(stream.average_rate)
            for frame in container.decode(stream):
                if frame.width <= 0 or frame.height <= 0:
                    raise VideoValidationError(f"invalid decoded frame {decoded_frames}")
                decoded_frames += 1
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise VideoValidationError(f"decode failed: {type(exc).__name__}: {exc}") from exc
    if decoded_frames != expected_frames:
        raise VideoValidationError(
            f"frame count mismatch: decoded={decoded_frames} expected={expected_frames} declared={declared_frames}"
        )
    return VideoProbe(
        path=os.path.abspath(path),
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        decoded_frames=decoded_frames,
        declared_frames=declared_frames,
        average_rate=average_rate,
    )


def validate_views(
    candidates: dict[str, str],
    expected_frames: int,
    *,
    max_camera_views: int = DEFAULT_MAX_CAMERA_VIEWS,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, Any]]]:
    valid: dict[str, str] = {}
    failures: dict[str, str] = {}
    probes: dict[str, dict[str, Any]] = {}
    for view in VIEW_PRIORITY:
        path = candidates.get(view)
        if not path:
            continue
        try:
            probe = probe_video(path, expected_frames)
        except Exception as exc:
            failures[view] = f"{type(exc).__name__}: {exc}"
            continue
        if len(valid) < max_camera_views:
            valid[view] = probe.path
            probes[view] = probe.to_dict()
    if not valid:
        detail = "; ".join(f"{view}={reason}" for view, reason in failures.items())
        raise VideoValidationError(f"all views invalid: {detail or 'no candidate videos'}")
    return valid, failures, probes
