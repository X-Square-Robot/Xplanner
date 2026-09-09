from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from x_planner.data.analysis_subset import parse_path_maps
from x_planner.data.evaluation_release import (
    audit_manifest,
    export_release,
    portable_id,
)


class EvaluationReleaseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        source = self.root / "mounted"
        episode = source / "episodes" / "ep-one"
        episode.mkdir(parents=True)
        (episode / "head.mp4").write_bytes(b"fake-video")
        (episode / "instruction.json").write_text(json.dumps({
            "ep-one": {"instruction": "Move the cup.", "task": "move-cup"}
        }), encoding="utf-8")
        labels = source / "labels.json"
        labels.write_text(json.dumps({
            "ep-one": {"action_caption": {"0 10": "Move the cup."}}
        }), encoding="utf-8")
        self.manifest = self.root / "manifest.jsonl"
        self.manifest.write_text(json.dumps({
            "uid": "fixture/task/ep-one",
            "dataset": "fixture",
            "source_group": "public_named",
            "episode_key": "ep-one",
            "instruction": "Move the cup.",
            "task": "move-cup",
            "anchor_frame": 0,
            "resolved_episode_path": "/private/episodes/ep-one",
            "instruction_path": "/private/labels.json",
            "camera_videos": [{
                "logical_view": "head",
                "mp4_path": "/private/episodes/ep-one/head.mp4",
                "duration_s": 1.0,
            }],
            "target_subtask_1": "Move the cup.",
        }) + "\n", encoding="utf-8")
        self.mappings = parse_path_maps((f"/private={source}",))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_audit_and_release_are_path_portable(self) -> None:
        report, inspected = audit_manifest(self.manifest, self.mappings)
        self.assertTrue(report["release_ready"])
        self.assertEqual(report["available_video_count"], 1)
        output = self.root / "release"
        export_release(self.manifest, output, report, inspected)
        item = json.loads((output / "items.jsonl").read_text())
        self.assertFalse(item["media"][0]["path"].startswith("/"))
        self.assertNotIn("/private", json.dumps(item))
        self.assertTrue((output / item["media"][0]["path"]).is_file())
        self.assertTrue((output / "checksums.sha256").is_file())

    def test_missing_video_fails_closed(self) -> None:
        (self.root / "mounted/episodes/ep-one/head.mp4").unlink()
        report, inspected = audit_manifest(self.manifest, self.mappings)
        self.assertFalse(report["release_ready"])
        self.assertEqual(report["missing_video_count"], 1)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            export_release(self.manifest, self.root / "release", report, inspected)

    def test_portable_id_is_stable_and_bounded(self) -> None:
        uid = "dataset/" + "very-long-name/" * 30
        self.assertEqual(portable_id(uid), portable_id(uid))
        self.assertLessEqual(len(portable_id(uid)), 109)


if __name__ == "__main__":
    unittest.main()
