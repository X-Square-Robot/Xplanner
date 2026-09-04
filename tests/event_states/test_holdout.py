from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import x_planner.data.event_states.holdout as holdout_module
from x_planner.data.event_states.holdout import EvaluationHoldout, HoldoutFilter, main, normalize_path


class EvaluationHoldoutTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[EvaluationHoldout, Path]:
        episode = root / "media" / "task_001" / "episode_001"
        episode.mkdir(parents=True)
        video = episode / "faceImg.mp4"
        video.write_bytes(b"fixture")
        manifest = root / "benchmark.jsonl"
        row = {
            "uid": "task_001/episode_001",
            "episode_key": "episode_001",
            "logical_episode_path": "/dataset/task_001/episode_001",
            "resolved_episode_path": "/dataset/task_001/episode_001",
            "existing_episode_path": str(episode),
            "path_candidates": [str(episode)],
            "camera_videos": [
                {
                    "mp4_path": (
                        "/dataset/task_001/episode_001/head.mp4"
                    )
                },
                {"mp4_path": str(video)},
            ],
        }
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return EvaluationHoldout.load(manifest, expected_sha256=None), video

    def test_mount_aliases_normalize_to_one_logical_path(self) -> None:
        expected = "/dataset/task_001/episode_001/head.mp4"
        aliases = (("/mount/primary", "/dataset"), ("/mount/secondary", "/dataset"))
        with patch.object(holdout_module, "_PATH_ALIASES", aliases):
            for prefix in ("/mount/primary", "/mount/secondary"):
                self.assertEqual(
                    normalize_path(prefix + "/task_001/episode_001/head.mp4"),
                    expected,
                )

    def test_matches_episode_video_and_inode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            holdout, video = self._fixture(Path(temporary))
            logical = {
                "sample_id": "sample",
                "base_sample_id": "base",
                "source": "baseline",
                "provenance": {"episode_key": "task_001/episode_001"},
                "images": [{
                    "video": "/dataset/task_001/episode_001/head.mp4"
                }],
            }
            modes = {item["mode"] for item in holdout.match_sample(logical)}
            self.assertIn("episode_identity", modes)
            self.assertIn("normalized_video_path", modes)
            self.assertIn("normalized_episode_path", modes)

            string_video_sample = {
                **logical,
                "provenance": {"episode_key": "unrelated"},
                "images": [
                    "/dataset/task_001/episode_001/head.mp4"
                ],
            }
            modes = {
                item["mode"]
                for item in holdout.match_sample(string_video_sample)
            }
            self.assertIn("normalized_video_path", modes)

            inode_sample = {
                **logical,
                "provenance": {"episode_key": "unrelated"},
                "images": [{"video": str(video)}],
            }
            modes = {item["mode"] for item in holdout.match_sample(inode_sample)}
            self.assertIn("device_inode", modes)

    def test_filter_keeps_unrelated_and_reports_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            holdout, _video = self._fixture(Path(temporary))
            checker = HoldoutFilter(holdout)
            held = {
                "sample_id": "held",
                "base_sample_id": "base-held",
                "source": "baseline",
                "provenance": {"episode_key": "episode_001"},
                "images": [],
            }
            clean = {
                "sample_id": "clean",
                "base_sample_id": "base-clean",
                "source": "baseline",
                "provenance": {"episode_key": "ep2"},
                "images": [],
            }
            self.assertFalse(checker.keep(held))
            self.assertTrue(checker.keep(clean))
            report = checker.report()
            self.assertEqual(report["checked_samples"], 2)
            self.assertEqual(report["excluded_samples"], 1)
            self.assertFalse(report["passed"])

    def test_cli_fail_closed_persists_negative_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            holdout, _video = self._fixture(root)
            artifact = root / "artifact"
            artifact.mkdir()
            (artifact / "data.jsonl").write_text(
                json.dumps({
                    "event_sample": {
                        "sample_id": "held",
                        "base_sample_id": "held",
                        "source": "negative_control",
                        "provenance": {"episode_key": "episode_001"},
                        "images": [],
                    }
                }) + "\n",
                encoding="utf-8",
            )
            output = root / "negative.json"
            status = main([
                "--manifest", str(holdout.manifest_path),
                "--expected-sha256", holdout.manifest_sha256,
                "--artifact-root", str(artifact),
                "--output", str(output),
                "--fail-on-match",
            ])
            self.assertEqual(status, 3)
            self.assertFalse(json.loads(output.read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
