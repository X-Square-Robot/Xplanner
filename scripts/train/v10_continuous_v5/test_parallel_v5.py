from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from scripts.train.v10_continuous_v5.parallel_v5 import (
    HARD_MAX_WORKERS,
    bounded_ordered_map,
    plan_jsonl_chunks,
    resolve_workers,
)


def _square(value: int) -> int:
    return value * value


class ParallelV5Test(unittest.TestCase):
    def test_auto_workers_is_affinity_aware_and_hard_capped(self) -> None:
        affinity = len(os.sched_getaffinity(0))
        self.assertEqual(resolve_workers(0), min(32, max(4, affinity // 4)))
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            resolve_workers(HARD_MAX_WORKERS + 1)

    def test_chunks_are_newline_aligned_and_conserve_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl"
            rows = [f'{{"index":{index}}}\n'.encode() for index in range(20)]
            path.write_bytes(b"".join(rows))
            chunks = plan_jsonl_chunks((path,), chunk_bytes=37)
            self.assertGreater(len(chunks), 1)
            self.assertEqual(sum(chunk.num_lines for chunk in chunks), len(rows))
            self.assertEqual(chunks[0].start, 0)
            self.assertEqual(chunks[-1].end, path.stat().st_size)
            with path.open("rb") as handle:
                for chunk in chunks:
                    handle.seek(chunk.start)
                    payload = handle.read(chunk.end - chunk.start)
                    self.assertTrue(payload.endswith(b"\n"))
                    self.assertEqual(payload.count(b"\n"), chunk.num_lines)

    def test_single_and_multi_worker_results_are_identical_and_ordered(self) -> None:
        values = list(range(40))
        single = list(bounded_ordered_map(_square, values, workers=1))
        multi = list(bounded_ordered_map(_square, values, workers=4))
        self.assertEqual(single, multi)
        self.assertEqual(multi, [value * value for value in values])


if __name__ == "__main__":
    unittest.main()
