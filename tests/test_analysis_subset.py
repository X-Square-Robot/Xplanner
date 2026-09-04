from __future__ import annotations

import unittest

from x_planner.data.analysis_subset import (
    parse_path_maps,
    portable_row,
    remap_path,
    safe_component,
)


class AnalysisSubsetExportTest(unittest.TestCase):
    def test_path_mapping_uses_longest_prefix(self) -> None:
        mappings = parse_path_maps(("/old=/new", "/old/special=/other"))
        self.assertEqual(
            str(remap_path("/old/special/video.mp4", mappings)),
            "/other/video.mp4",
        )

    def test_public_row_omits_private_paths(self) -> None:
        source = {
            "uid": "sample/one",
            "instruction": "Move the cup.",
            "task": "tabletop",
            "task_class": ["planning"],
            "dataset": "fixture",
            "anchor_frame": 12,
            "target_subtask_1": "reach",
            "resolved_episode_path": "/private/storage/episode",
        }
        row = portable_row(
            source,
            sample_id=safe_component(source["uid"]),
            media=[{"view": "head", "path": "media/sample-one/head.jpg", "sha256": "0" * 64}],
        )
        self.assertEqual(row["id"], "sample-one")
        self.assertEqual(row["split"], "analysis")
        self.assertNotIn("resolved_episode_path", row)

    def test_row_includes_analysis_labels(self) -> None:
        source = {
            "instruction": "Move the cup.",
            "task": "tabletop",
            "dataset": "fixture",
            "target_subtask_1": "reach",
        }
        row = portable_row(
            source,
            sample_id="sample-one",
            media=[{"view": "head", "path": "media/sample-one/head.jpg", "sha256": "0" * 64}],
        )
        self.assertEqual(row["labels"]["target_subtask_1"], "reach")


if __name__ == "__main__":
    unittest.main()
