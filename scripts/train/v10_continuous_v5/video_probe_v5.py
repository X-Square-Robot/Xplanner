"""PyAV metadata fallback for V5 hosts without the ffprobe executable."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path


def probe_video_pyav(path: Path) -> tuple[Fraction, int]:
    """Return fps and frame count from container metadata without decoding."""

    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"PyAV returned no video stream for {path}")
        stream = container.streams.video[0]
        raw_rate = stream.average_rate or stream.base_rate or stream.guessed_rate
        if raw_rate is None:
            raise ValueError(f"PyAV returned no video frame rate for {path}")
        fps = Fraction(raw_rate.numerator, raw_rate.denominator)
        frame_count = int(stream.frames or 0)
        if frame_count <= 0:
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration / av.time_base)
            else:
                raise ValueError(f"PyAV returned no duration or frame count for {path}")
            frame_count = int(round(duration * float(fps)))
    if fps <= 0 or frame_count <= 0:
        raise ValueError(
            f"invalid PyAV video metadata for {path}: fps={fps}, frames={frame_count}"
        )
    return fps, frame_count
