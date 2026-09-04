from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from x_planner.data.event_states.replan_adapter import (
    _images as replan_images,
    _task_instruction as replan_task_instruction,
)
from x_planner.data.event_states.task_instruction import (
    TaskInstructionError,
    resolve_episode_task_instruction,
    select_record_instruction,
)


class TaskInstructionV53Test(unittest.TestCase):
    def test_task_caption_has_priority_over_instruction(self) -> None:
        text, source, path, field = select_record_instruction(
            {
                "task_caption": "Place the cup on the tray.",
                "instruction": "This lower-priority text must not be selected.",
            },
            source_prefix="fixture",
            source_path="/fixture/instruction.json",
        )
        self.assertEqual(text, "Place the cup on the tray.")
        self.assertEqual(source, "fixture.task_caption")
        self.assertEqual(path, "/fixture/instruction.json")
        self.assertEqual(field, "task_caption")

    def test_exact_media_episode_mapping_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_root = Path(temporary) / "task"
            episode = task_root / "episode-a"
            episode.mkdir(parents=True)
            metadata = task_root / "instruction.json"
            metadata.write_text(json.dumps({
                "episode-a": {"instruction": "Open the drawer."},
                "episode-b": {"instruction": "Close the drawer."},
            }), encoding="utf-8")
            value = resolve_episode_task_instruction(
                annotation={"action_caption": {}},
                annotation_path="/fixture/compact-label.json",
                episode_key="episode-a",
                resolved_episode_path=episode,
            )
            self.assertEqual(value.text, "Open the drawer.")
            self.assertEqual(
                value.source,
                "media_task_instruction_json.episode.instruction",
            )
            self.assertEqual(value.source_path, str(metadata.resolve()))

    def test_missing_instruction_never_falls_back_to_task_slug(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_path = Path(temporary) / "samples.json"
            source_path.write_text("[]", encoding="utf-8")
            with self.assertRaises(TaskInstructionError):
                replan_task_instruction(
                    {"task": "20260520-day-10439-take_bowl"},
                    source_path=source_path,
                )

    def test_replan_static_frames_have_view_aware_media_references(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = root / "frames" / "frame_001.jpg"
            frame.parent.mkdir()
            frame.write_bytes(b"fixture")
            values = replan_images({
                "frames": {
                    "cot_frames": [{"path": "frames/frame_001.jpg", "frame": 17}]
                }
            }, root)
            self.assertEqual(values, [{
                "path": str(frame.resolve()),
                "view": "head",
                "source_frame": 17,
            }])


if __name__ == "__main__":
    unittest.main()
