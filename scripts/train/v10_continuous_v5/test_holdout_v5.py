from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .holdout_v5 import Benchmark3Holdout, HoldoutFilter, main, normalize_path


class Benchmark3HoldoutTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Benchmark3Holdout, Path]:
        episode = root / "x2robot_data" / "zhengwei" / "10000" / "body" / "ep1"
        episode.mkdir(parents=True)
        video = episode / "faceImg.mp4"
        video.write_bytes(b"fixture")
        manifest = root / "benchmark.jsonl"
        row = {
            "uid": "10000/body/ep1",
            "episode_key": "ep1",
            "logical_episode_path": "/x2robot_data/zhengwei/10000/body/ep1",
            "resolved_episode_path": "/x2robot_data/zhengwei/10000/body/ep1",
            "existing_episode_path": str(episode),
            "path_candidates": [str(episode)],
            "camera_videos": [
                {
                    "mp4_path": (
                        "/x2robot_data/zhengwei/10000/body/ep1/faceImg.mp4"
                    )
                },
                {"mp4_path": str(video)},
            ],
        }
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return Benchmark3Holdout.load(manifest, expected_sha256=None), video

    def test_mount_aliases_normalize_to_one_logical_path(self) -> None:
        expected = "/x2robot_data/zhengwei/10000/body/ep1/faceImg.mp4"
        for prefix in (
            "/mnt/cpfs/zbl-cpfs-new",
            "/mnt/cpfs",
            "/mnt/data",
            "",
        ):
            self.assertEqual(
                normalize_path(prefix + expected),
                expected,
            )
        self.assertEqual(
            normalize_path(
                "/mnt/cpfs/zbl-cpfs-new/open_data/"
                "Open_Action_datasets_as_mp4/ds/body/ep/cam.mp4"
            ),
            "/open_data/video/ds/body/ep/cam.mp4",
        )
        self.assertEqual(
            normalize_path(
                "/mnt/oss/zbl-open-data/AgiBotWorld-Alpha/327/ep/faceImg.mp4"
            ),
            "/open_data/video/AgiBotWorld-Alpha-v2/327/ep/faceImg.mp4",
        )

    def test_matches_episode_video_and_inode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            holdout, video = self._fixture(Path(temporary))
            logical = {
                "sample_id": "sample",
                "base_sample_id": "base",
                "source": "baseline",
                "provenance": {"episode_key": "10000/body/ep1"},
                "images": [{
                    "video": "/mnt/data/x2robot_data/zhengwei/10000/body/ep1/faceImg.mp4"
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
                    "/mnt/cpfs/zbl-cpfs-new/open_data/x2robot_data/zhengwei/"
                    "10000/body/ep1/faceImg.mp4"
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
                "provenance": {"episode_key": "ep1"},
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
                    "v5_sample": {
                        "sample_id": "held",
                        "base_sample_id": "held",
                        "source": "negative_control",
                        "provenance": {"episode_key": "ep1"},
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
