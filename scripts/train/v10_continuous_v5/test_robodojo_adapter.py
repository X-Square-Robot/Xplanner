from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from robodojo_adapter import (  # noqa: E402
    ROBODOJO_TASKS,
    RobodojoAdapterError,
    build_canonical_records,
    canonical_episode_id,
    scan_robodojo,
)


TEST_TASKS = ("build_tower", "make_toast", "pack_objects_into_box")


class RobodojoAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.media_root = root / "robotwin30_x2" / "arx_x5"
        self.label_root = root / "captions" / "robotwin30_x2" / "arx_x5"
        self.split_path = root / "split.json"
        self.split_path.write_text(
            json.dumps(
                {
                    "meta": {"seed": 0},
                    "train": {
                        "build_tower": [0, 1],
                        "make_toast": [72, 73],
                    },
                    "holdout_traj": {"build_tower": [2]},
                    "holdout_task": {"pack_objects_into_box": [0]},
                }
            ),
            encoding="utf-8",
        )
        self._write_task(
            "build_tower",
            {
                "trajectory_0": ("Build a tower.", ("Approach a block", "Place the block")),
                "trajectory_1": ("Build a tower.", ("Lift the board", "Place the board")),
                "trajectory_2": ("Build a tower.", ("Grasp a block", "Stack the block")),
            },
        )
        self._write_task(
            "make_toast",
            {
                "trajectory_72": ("Make toast.", ("Pick up the bread", "Insert the bread")),
                "trajectory_73": ("Make toast.", ("Pick up the bread", "Insert the bread")),
            },
        )
        self._write_task(
            "pack_objects_into_box",
            {
                "trajectory_0": ("Pack the objects into the box.", ("Pick up an object", "Place it in the box")),
            },
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_task(
        self,
        task_name: str,
        episodes: dict[str, tuple[str, tuple[str, ...]]],
    ) -> None:
        media_task = self.media_root / task_name
        label_task = self.label_root / task_name
        media_task.mkdir(parents=True)
        label_task.mkdir(parents=True)
        media_annotations = {}
        action_annotations = {}
        for trajectory_name, (instruction, captions) in episodes.items():
            total = 10 * len(captions)
            media_annotations[trajectory_name] = {
                "instruction": instruction,
                "total": total,
                "ignored_reference": "/not/copied/by/the/adapter",
            }
            action_annotations[trajectory_name] = {
                "action_caption": {
                    f"{offset * 10} {(offset + 1) * 10}": caption
                    for offset, caption in enumerate(captions)
                },
                "action_caption_zh": {"0 10": "不应读取"},
            }
            episode_dir = media_task / trajectory_name
            episode_dir.mkdir()
            for file_name in ("faceImg.mp4", "leftImg.mp4", "rightImg.mp4"):
                (episode_dir / file_name).touch()
        (media_task / "instruction.json").write_text(
            json.dumps(media_annotations), encoding="utf-8"
        )
        (label_task / "instruction.json").write_text(
            json.dumps(action_annotations, ensure_ascii=False), encoding="utf-8"
        )

    def _scan(self, **kwargs):
        return scan_robodojo(
            self.media_root,
            self.label_root,
            self.split_path,
            tasks=TEST_TASKS,
            expected_episodes_per_task=None,
            **kwargs,
        )

    def test_canonical_task_inventory_and_episode_id(self) -> None:
        self.assertEqual(len(ROBODOJO_TASKS), 35)
        self.assertEqual(len(set(ROBODOJO_TASKS)), 35)
        self.assertEqual(
            canonical_episode_id("build_tower", "trajectory_7"),
            "robotwin30_x2/arx_x5/build_tower/trajectory_7",
        )
        with self.assertRaises(RobodojoAdapterError):
            canonical_episode_id("../build_tower", "trajectory_7")

    def test_default_scan_is_train_only_and_excludes_quarantine(self) -> None:
        scan = self._scan()
        self.assertEqual(
            [episode.canonical_episode_id for episode in scan.episodes],
            [
                "robotwin30_x2/arx_x5/build_tower/trajectory_0",
                "robotwin30_x2/arx_x5/build_tower/trajectory_1",
                "robotwin30_x2/arx_x5/make_toast/trajectory_73",
            ],
        )
        self.assertEqual(scan.excluded_by_quarantine, 1)
        self.assertEqual(scan.excluded_by_split, 2)
        self.assertFalse(scan.issues)
        self.assertTrue(all(episode.split == "train" for episode in scan.episodes))

    def test_holdout_requires_explicit_double_opt_in(self) -> None:
        with self.assertRaises(PermissionError):
            self._scan(include_splits=("holdout_traj",))
        scan = self._scan(
            include_splits=("holdout_traj",), allow_holdouts=True
        )
        self.assertEqual(len(scan.episodes), 1)
        self.assertEqual(scan.episodes[0].trajectory_name, "trajectory_2")
        self.assertEqual(scan.episodes[0].split, "holdout_traj")

    def test_action_only_records_have_two_predictions_and_mask_segments(self) -> None:
        episode = self._scan().episodes[0]
        records = build_canonical_records(episode)
        self.assertEqual([record["category"] for record in records], [
            "initial_plan", "ongoing", "ongoing", "end"
        ])

        initial = records[0]
        self.assertEqual(len(initial["supervision"]["initial_plan"]), 2)
        for step in initial["supervision"]["initial_plan"]:
            self.assertTrue(step["action"]["label_available"])
            self.assertFalse(step["segment"]["label_available"])

        first_ongoing, last_ongoing = records[1], records[2]
        for ongoing in (first_ongoing, last_ongoing):
            predictions = ongoing["supervision"]["predictions"]
            self.assertEqual(len(predictions), 2)
            self.assertEqual(
                [(item["index"], item["role"]) for item in predictions],
                [(1, "current"), (2, "next")],
            )
            self.assertTrue(all(
                item["segment"] == {"label_available": False, "caption": ""}
                for item in predictions
            ))
            self.assertFalse(
                ongoing["supervision"]["execution_decision"]["label_available"]
            )
        self.assertTrue(
            first_ongoing["supervision"]["predictions"][1]["action"]["label_available"]
        )
        self.assertFalse(
            last_ongoing["supervision"]["predictions"][1]["action"]["label_available"]
        )

        end = records[-1]
        self.assertEqual(end["anchor_frame"], episode.total_frames - 1)
        self.assertFalse(end["supervision"]["end_outcome"]["label_available"])
        self.assertFalse(
            end["supervision"]["execution_decision"]["label_available"]
        )

        serialized = json.dumps(records).casefold()
        for forbidden in (
            '"profile"',
            '"raw_levels"',
            '"hierarchy"',
            '"subtask"',
            '"l0"',
            '"l1"',
            '"l2"',
            '"l3"',
        ):
            self.assertNotIn(forbidden, serialized)

    def test_end_candidate_anchor_stays_inside_last_action(self) -> None:
        media_file = self.media_root / "build_tower" / "instruction.json"
        media = json.loads(media_file.read_text(encoding="utf-8"))
        media["trajectory_0"]["total"] = 25
        media_file.write_text(json.dumps(media), encoding="utf-8")
        episode = self._scan().episodes[0]
        end = build_canonical_records(episode)[-1]
        self.assertEqual(end["anchor_frame"], 19)

    def test_non_english_action_is_rejected_without_failing_other_episodes(self) -> None:
        label_file = self.label_root / "build_tower" / "instruction.json"
        labels = json.loads(label_file.read_text(encoding="utf-8"))
        labels["trajectory_1"]["action_caption"]["0 10"] = "抓取木板"
        label_file.write_text(
            json.dumps(labels, ensure_ascii=False), encoding="utf-8"
        )
        scan = self._scan()
        self.assertEqual(len(scan.issues), 1)
        self.assertEqual(scan.issues[0].reason, "episode_validation_failed")
        self.assertIn("not valid English", scan.issues[0].detail)
        self.assertNotIn(
            "trajectory_1",
            [episode.trajectory_name for episode in scan.episodes],
        )

    def test_robot_action_tree_is_rejected(self) -> None:
        blocked = Path(self.temp_dir.name) / "open_action_dataset" / "robodojo_ee_v2"
        with self.assertRaises(RobodojoAdapterError):
            scan_robodojo(
                blocked,
                self.label_root,
                self.split_path,
                tasks=TEST_TASKS,
                expected_episodes_per_task=None,
            )


if __name__ == "__main__":
    unittest.main()
