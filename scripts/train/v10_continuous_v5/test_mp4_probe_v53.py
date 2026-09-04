from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from .mp4_probe_v53 import video_frame_count


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


class Mp4ProbeV53Test(unittest.TestCase):
    def test_reads_video_stsz_without_media_decode(self) -> None:
        hdlr = _box(b"hdlr", b"\0\0\0\0" + b"\0\0\0\0" + b"vide")
        stsz = _box(
            b"stsz",
            b"\0\0\0\0" + struct.pack(">II", 0, 321),
        )
        stbl = _box(b"stbl", stsz)
        minf = _box(b"minf", stbl)
        mdia = _box(b"mdia", hdlr + minf)
        trak = _box(b"trak", mdia)
        moov = _box(b"moov", trak)
        # A large-looking mdat box is skipped by seek; its bytes are never
        # interpreted as video data.
        payload = _box(b"ftyp", b"isom") + _box(b"mdat", b"fixture") + moov
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "video.mp4"
            path.write_bytes(payload)
            self.assertEqual(video_frame_count(path), 321)


if __name__ == "__main__":
    unittest.main()
