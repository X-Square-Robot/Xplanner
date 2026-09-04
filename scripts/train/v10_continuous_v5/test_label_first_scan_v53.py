from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from .label_first_scan_v53 import build_collection_device_index, scan


class _NoHoldout:
    def match_sample(self, _sample):
        return []

    def metadata(self):
        return {"manifest_sha256": "fixture"}


def _captions(count: int, prefix: str) -> dict[str, str]:
    return {f"{index * 10} {(index + 1) * 10}": f"{prefix} step {index}" for index in range(count)}


class LabelFirstScanV53Test(unittest.TestCase):
    def test_profile_filter_path_resolution_and_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = root / "labels"
            instruction = labels / "10000" / "pick-cup" / "instruction.json"
            instruction.parent.mkdir(parents=True)
            instruction.write_text(json.dumps({
                "episode-a": {
                    "action_caption": _captions(4, "Action"),
                    "human_segment_caption": _captions(4, "Segment"),
                },
                "episode-b": {
                    "action_caption": _captions(3, "Action"),
                    "human_segment_caption": None,
                },
            }), encoding="utf-8")
            zhengwei = root / "zhengwei"
            media = zhengwei / "10000" / "pick-cup" / "episode-a"
            media.mkdir(parents=True)
            (media.parent / "instruction.json").write_text(json.dumps({
                "episode-a": {"instruction": "Pick up the cup."},
            }), encoding="utf-8")
            for name in ("faceImg.mp4", "leftImg.mp4", "rightImg.mp4"):
                (media / name).write_bytes(b"fixture")
            output = root / "out"
            report = scan(
                label_root=labels,
                output_root=output,
                zhengwei_root=zhengwei,
                collection_root=root / "collection",
                open_action_root=root / "open",
                robodojo_media_root=root / "dojo",
                workers=2,
                holdout=_NoHoldout(),
            )
            self.assertTrue(report["complete"])
            rows = [json.loads(line) for line in (output / "instruction_index.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["instruction_relative"], "10000/pick-cup/instruction.json")
            self.assertEqual(rows[0]["eligible_profiles"], [
                "action_only", "segment_only", "action_segment_joint"
            ])
            self.assertEqual(rows[0]["task_instruction"], "Pick up the cup.")
            self.assertEqual(
                rows[0]["task_instruction_source"],
                "media_task_instruction_json.episode.instruction",
            )
            self.assertEqual(
                rows[0]["task_instruction_source_path"],
                str((media.parent / "instruction.json").resolve()),
            )
            self.assertEqual(rows[1]["eligible_profiles"], [])
            self.assertEqual(rows[1]["excluded_profiles"]["action_only"], "subtask_count_le_3")
            self.assertEqual(rows[1]["task_instruction_status"], "missing_task_instruction")
            self.assertEqual(report["missing_task_instruction_rows"], 1)
            reused = scan(
                label_root=labels,
                output_root=output,
                zhengwei_root=zhengwei,
                collection_root=root / "collection",
                open_action_root=root / "open",
                robodojo_media_root=root / "dojo",
                workers=2,
                reuse=True,
                holdout=_NoHoldout(),
            )
            self.assertTrue(reused["cache_reused"])

    def test_collection_index_is_shallow_and_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            device = root / "11" / "RD001" / "XRRD11ZD2604070071"
            device.mkdir(parents=True)
            self.assertEqual(
                build_collection_device_index(root),
                {device.name: str(device.resolve())},
            )

    def test_collection_index_retains_duplicate_device_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            name = "XRRD11ZD2604070085"
            left = root / "11" / "RD001" / name
            right = root / "12" / "RD002" / name
            left.mkdir(parents=True)
            right.mkdir(parents=True)
            index = build_collection_device_index(root)
            self.assertEqual(index[name], tuple(sorted((str(left.resolve()), str(right.resolve())))))


if __name__ == "__main__":
    unittest.main()
