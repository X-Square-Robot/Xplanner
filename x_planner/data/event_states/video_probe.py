"""Zero-dependency MP4 video frame-count probe.

Only ISO-BMFF box headers and the video track's sample table are read.  Large
``mdat`` payloads are skipped with seeks, which keeps label-first scanning fast
on remote storage and avoids a PyAV/ffprobe dependency in smoke environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import BinaryIO, Iterator


@dataclass(frozen=True, slots=True)
class Box:
    kind: bytes
    payload_start: int
    end: int


def _boxes(handle: BinaryIO, start: int, end: int) -> Iterator[Box]:
    position = start
    while position + 8 <= end:
        handle.seek(position)
        header = handle.read(8)
        if len(header) != 8:
            return
        size, kind = struct.unpack(">I4s", header)
        header_size = 8
        if size == 1:
            extended = handle.read(8)
            if len(extended) != 8:
                return
            size = struct.unpack(">Q", extended)[0]
            header_size = 16
        elif size == 0:
            size = end - position
        if size < header_size or position + size > end:
            raise ValueError(
                f"invalid MP4 box {kind!r} at {position}: size={size}, end={end}"
            )
        yield Box(kind=kind, payload_start=position + header_size, end=position + size)
        position += size


def _child(handle: BinaryIO, parent: Box, kind: bytes) -> Box | None:
    return next(
        (box for box in _boxes(handle, parent.payload_start, parent.end) if box.kind == kind),
        None,
    )


def _video_track(handle: BinaryIO, moov: Box) -> Box:
    for track in _boxes(handle, moov.payload_start, moov.end):
        if track.kind != b"trak":
            continue
        mdia = _child(handle, track, b"mdia")
        hdlr = _child(handle, mdia, b"hdlr") if mdia else None
        if hdlr is None or hdlr.payload_start + 12 > hdlr.end:
            continue
        handle.seek(hdlr.payload_start + 8)
        if handle.read(4) == b"vide":
            return track
    raise ValueError("MP4 has no video track")


def _sample_count(handle: BinaryIO, track: Box) -> int:
    mdia = _child(handle, track, b"mdia")
    minf = _child(handle, mdia, b"minf") if mdia else None
    stbl = _child(handle, minf, b"stbl") if minf else None
    if stbl is None:
        raise ValueError("MP4 video track has no sample table")
    stsz = _child(handle, stbl, b"stsz")
    if stsz is not None and stsz.payload_start + 12 <= stsz.end:
        handle.seek(stsz.payload_start + 8)
        count = struct.unpack(">I", handle.read(4))[0]
        if count > 0:
            return count
    # Compact sample-size tables still carry the same sample count.
    stz2 = _child(handle, stbl, b"stz2")
    if stz2 is not None and stz2.payload_start + 12 <= stz2.end:
        handle.seek(stz2.payload_start + 8)
        count = struct.unpack(">I", handle.read(4))[0]
        if count > 0:
            return count
    # ``stts`` is a safe fallback: each entry gives a sample count and delta.
    stts = _child(handle, stbl, b"stts")
    if stts is not None and stts.payload_start + 8 <= stts.end:
        handle.seek(stts.payload_start + 4)
        entries = struct.unpack(">I", handle.read(4))[0]
        if stts.payload_start + 8 + entries * 8 > stts.end:
            raise ValueError("truncated MP4 time-to-sample table")
        count = 0
        for _ in range(entries):
            count += struct.unpack(">I", handle.read(4))[0]
            handle.seek(4, 1)
        if count > 0:
            return count
    raise ValueError("MP4 video sample count is unavailable")


def video_frame_count(path: Path | str) -> int:
    value = Path(path)
    size = value.stat().st_size
    if size < 8:
        raise ValueError(f"not an MP4 container: {value}")
    with value.open("rb") as handle:
        moov = next((box for box in _boxes(handle, 0, size) if box.kind == b"moov"), None)
        if moov is None:
            raise ValueError(f"MP4 has no moov box: {value}")
        count = _sample_count(handle, _video_track(handle, moov))
    if count <= 0:
        raise ValueError(f"invalid MP4 frame count for {value}: {count}")
    return count


def synchronized_frame_count(paths: list[str]) -> tuple[int, dict[str, int]]:
    if not paths:
        raise ValueError("no synchronized videos")
    counts = {path: video_frame_count(path) for path in paths}
    return min(counts.values()), counts


__all__ = ["synchronized_frame_count", "video_frame_count"]
