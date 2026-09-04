from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

from .holdout_v5 import Benchmark3Holdout
from .index_materialize_v53 import infer_task_instruction, materialize
from .merge_index_materialize_shards_v53 import merge_shards
from .task_instruction_v53 import TaskInstructionError


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def _video(path: Path, frames: int) -> None:
    hdlr = _box(b"hdlr", b"\0" * 8 + b"vide")
    stsz = _box(b"stsz", b"\0" * 8 + struct.pack(">I", frames))
    path.write_bytes(_box(b"ftyp", b"isom") + _box(b"moov", _box(
        b"trak", _box(b"mdia", hdlr + _box(b"minf", _box(b"stbl", stsz)))
    )))


class IndexMaterializeV53Test(unittest.TestCase):
    def test_authoritative_instruction_required_and_slug_rejected(self) -> None:
        text, source = infer_task_instruction({
            "task_instruction": "Sort and fold the cloth.",
            "task_instruction_status": "resolved",
            "task_instruction_source": "media_task_instruction_json.episode.instruction",
            "task_instruction_source_path": "/fixture/instruction.json",
        })
        self.assertEqual(text, "Sort and fold the cloth.")
        self.assertEqual(source, "media_task_instruction_json.episode.instruction")
        with self.assertRaises(TaskInstructionError):
            infer_task_instruction({
                "episode_key": "20260203-day-10254-sort_and_fold_cloth_infer@MODE@time",
                "task_hint": "sort_and_fold_cloth",
                "task_instruction": "Sort and fold cloth",
                "task_instruction_status": "resolved",
                "task_instruction_source": "episode_slug",
                "task_instruction_source_path": "/fixture/path",
            })

    def test_index_paths_materialize_all_three_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = []
            for name, frames in (("faceImg.mp4", 100), ("leftImg.mp4", 103), ("rightImg.mp4", 101)):
                path = root / name
                _video(path, frames)
                videos.append(str(path))
            intervals = [
                {"start_frame": index * 20, "end_frame": (index + 1) * 20,
                 "caption": f"Move fixture object in action {index + 1}"}
                for index in range(4)
            ]
            segments = [
                {"start_frame": index * 20, "end_frame": (index + 1) * 20,
                 "caption": f"Execute fixture motion segment {index + 1}"}
                for index in range(4)
            ]
            index = root / "instruction_index.jsonl"
            row = {
                "episode_key": "20260827-day-1-place_items_in_box@mode@time",
                "source_group": "zhengwei",
                "task_hint": "place_items_in_box",
                "task_instruction": "Place the items in the box.",
                "task_instruction_status": "resolved",
                "task_instruction_source": "media_task_instruction_json.episode.instruction",
                "task_instruction_source_path": str(root / "instruction.json"),
                "task_instruction_source_field": "instruction",
                "eligible_profiles": ["action_only", "segment_only", "action_segment_joint"],
                "actions": intervals,
                "segments": segments,
                "camera_videos": videos,
                "resolved_episode_path": str(root),
                "instruction_relative": "1/body/place_items/instruction.json",
                "benchmark3_excluded": False,
            }
            index.write_text(json.dumps(row) + "\n", encoding="utf-8")
            benchmark = root / "b3.jsonl"
            benchmark.write_text(json.dumps({"uid": "held-out"}) + "\n", encoding="utf-8")
            holdout = Benchmark3Holdout.load(benchmark, expected_sha256=None)
            report = materialize(
                instruction_index=index,
                output_root=root / "out",
                holdout=holdout,
            )
            self.assertEqual(set(report["buckets"]), {"initial_plan", "ongoing", "end"})
            end_rows = [
                json.loads(line)["v5_sample"]
                for line in (root / "out/buckets/end/train.jsonl").read_text().splitlines()
            ]
            self.assertTrue(end_rows)
            self.assertTrue(all(row["images"][0]["frame"] == 99 for row in end_rows))
            self.assertTrue(all(
                row["task_instruction"] == "Place the items in the box."
                for row in end_rows
            ))

    def test_parallel_frame_prefetch_is_output_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = []
            for name, frames in (("faceImg.mp4", 100), ("leftImg.mp4", 103), ("rightImg.mp4", 101)):
                path = root / name
                _video(path, frames)
                videos.append(str(path))
            actions = [
                {
                    "start_frame": index * 20,
                    "end_frame": (index + 1) * 20,
                    "caption": f"Move fixture object in action {index + 1}",
                }
                for index in range(4)
            ]
            rows = []
            for index in range(3):
                rows.append({
                    "episode_key": f"20260827-day-{index}-place_items_in_box@mode@time",
                    "source_group": "zhengwei",
                    "task_instruction": "Place the items in the box.",
                    "task_instruction_status": "resolved",
                    "task_instruction_source": "media_task_instruction_json.episode.instruction",
                    "task_instruction_source_path": str(root / "instruction.json"),
                    "task_instruction_source_field": "instruction",
                    "eligible_profiles": ["action_only"],
                    "actions": actions,
                    "segments": [],
                    "camera_videos": videos,
                    "resolved_episode_path": str(root),
                    "instruction_relative": f"{index}/instruction.json",
                    "benchmark3_excluded": False,
                })
            bad = dict(rows[-1])
            bad["episode_key"] = "20260827-day-bad-place_items_in_box@mode@time"
            bad["camera_videos"] = [str(root / "missing.mp4")]
            rows.insert(1, bad)
            instruction_index = root / "instruction_index.jsonl"
            instruction_index.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            benchmark = root / "b3.jsonl"
            benchmark.write_text(json.dumps({"uid": "held-out"}) + "\n", encoding="utf-8")
            holdout_serial = Benchmark3Holdout.load(benchmark, expected_sha256=None)
            holdout_parallel = Benchmark3Holdout.load(benchmark, expected_sha256=None)
            serial_root = root / "serial"
            parallel_root = root / "parallel"
            serial = materialize(
                instruction_index=instruction_index,
                output_root=serial_root,
                holdout=holdout_serial,
                frame_probe_workers=1,
                materialize_batch_size=2,
                progress_every_rows=0,
            )
            parallel = materialize(
                instruction_index=instruction_index,
                output_root=parallel_root,
                holdout=holdout_parallel,
                frame_probe_workers=4,
                materialize_batch_size=2,
                progress_every_rows=0,
            )
            self.assertEqual(serial["total_records"], parallel["total_records"])
            self.assertEqual(serial["buckets"], parallel["buckets"])
            for relative in (
                "buckets/initial_plan/train.jsonl",
                "buckets/initial_plan/contracts.jsonl",
                "buckets/ongoing/train.jsonl",
                "buckets/ongoing/contracts.jsonl",
                "buckets/end/train.jsonl",
                "buckets/end/contracts.jsonl",
                "quarantine.jsonl",
                "media_frame_cache.jsonl",
            ):
                self.assertEqual(
                    (serial_root / relative).read_bytes(),
                    (parallel_root / relative).read_bytes(),
                    relative,
                )

    def test_contiguous_shards_merge_to_serial_core_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            videos = []
            for name, frames in (
                ("faceImg.mp4", 100),
                ("leftImg.mp4", 103),
                ("rightImg.mp4", 101),
            ):
                path = root / name
                _video(path, frames)
                videos.append(str(path))
            actions = [
                {
                    "start_frame": index * 20,
                    "end_frame": (index + 1) * 20,
                    "caption": f"Move fixture object in action {index + 1}",
                }
                for index in range(4)
            ]
            rows = []
            for index in range(6):
                rows.append({
                    "episode_key": f"20260827-day-{index}-place_items_in_box@mode@time",
                    "source_group": "zhengwei",
                    "task_instruction": "Place the items in the box.",
                    "task_instruction_status": "resolved",
                    "task_instruction_source": "media_task_instruction_json.episode.instruction",
                    "task_instruction_source_path": str(root / "instruction.json"),
                    "task_instruction_source_field": "instruction",
                    "eligible_profiles": ["action_only"],
                    "actions": actions,
                    "segments": [],
                    "camera_videos": videos,
                    "resolved_episode_path": str(root),
                    "instruction_relative": f"{index}/instruction.json",
                    "benchmark3_excluded": False,
                })
            bad = dict(rows[2])
            bad["episode_key"] = "20260827-day-bad-place_items_in_box@mode@time"
            bad["camera_videos"] = [str(root / "missing.mp4")]
            rows.insert(3, bad)
            instruction_index = root / "instruction_index.jsonl"
            instruction_index.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            benchmark = root / "b3.jsonl"
            benchmark.write_text(json.dumps({"uid": "held-out"}) + "\n", encoding="utf-8")
            serial_root = root / "serial"
            materialize(
                instruction_index=instruction_index,
                output_root=serial_root,
                holdout=Benchmark3Holdout.load(benchmark, expected_sha256=None),
                frame_probe_workers=1,
                progress_every_rows=0,
            )
            shard_a = root / "shard-a"
            shard_b = root / "shard-b"
            materialize(
                instruction_index=instruction_index,
                output_root=shard_a,
                holdout=Benchmark3Holdout.load(benchmark, expected_sha256=None),
                frame_probe_workers=2,
                materialize_batch_size=2,
                progress_every_rows=0,
                line_start=1,
                line_end=3,
            )
            materialize(
                instruction_index=instruction_index,
                output_root=shard_b,
                holdout=Benchmark3Holdout.load(benchmark, expected_sha256=None),
                frame_probe_workers=2,
                materialize_batch_size=2,
                progress_every_rows=0,
                line_start=4,
                line_end=7,
            )
            merged_root = root / "merged"
            merged = merge_shards(
                shard_roots=[shard_a, shard_b],
                output_root=merged_root,
                instruction_index=instruction_index,
                expected_index_rows=7,
                holdout=Benchmark3Holdout.load(benchmark, expected_sha256=None),
            )
            self.assertTrue(merged["complete"])
            self.assertFalse(merged["provenance"]["partial"])
            self.assertEqual(merged["provenance"]["read_rows"], 7)
            self.assertEqual(merged["provenance"]["materialize_shard_count"], 2)
            for relative in (
                "buckets/initial_plan/train.jsonl",
                "buckets/initial_plan/contracts.jsonl",
                "buckets/ongoing/train.jsonl",
                "buckets/ongoing/contracts.jsonl",
                "buckets/end/train.jsonl",
                "buckets/end/contracts.jsonl",
                "quarantine.jsonl",
                "media_frame_cache.jsonl",
            ):
                self.assertEqual(
                    (serial_root / relative).read_bytes(),
                    (merged_root / relative).read_bytes(),
                    relative,
                )


if __name__ == "__main__":
    unittest.main()
