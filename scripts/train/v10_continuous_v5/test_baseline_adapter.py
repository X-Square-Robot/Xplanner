from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    from .baseline_adapter import BaselineAdapter, BaselineAdapterError
except ImportError:
    from baseline_adapter import BaselineAdapter, BaselineAdapterError


def _unit(caption: str, progress: int, *, kind: str) -> dict[str, object]:
    result: dict[str, object] = {
        "level": "legacy",
        "caption": caption,
        "progress_percent": progress,
    }
    if kind == "segment":
        result["source"] = "segment"
    return result


def _row(
    *,
    sample_id: str,
    episode: str,
    source_shape: str,
    anchor_kind: str,
    unit_index: int,
    frame: int,
    history: list[str],
    action_captions: tuple[str, ...] = (),
    segment_captions: tuple[str, ...] = (),
) -> dict[str, object]:
    predictions: list[dict[str, object]] = []
    count = max(len(action_captions), len(segment_captions))
    for offset in range(count):
        prediction: dict[str, object] = {"index": offset + 1}
        if offset < len(action_captions):
            prediction["action"] = _unit(
                action_captions[offset], 40 if offset == 0 else 0, kind="action"
            )
        if offset < len(segment_captions):
            prediction["l0"] = _unit(
                segment_captions[offset], 70 if offset == 0 else 0, kind="segment"
            )
        predictions.append(prediction)
    sample = {
        "sample_id": sample_id,
        "episode_key": episode,
        "split": "train",
        "profile": source_shape,
        "unit_type": anchor_kind,
        "unit_index": unit_index,
        "current_frame": frame,
        "task_caption": "Place the blocks in the tray",
        "long_memory": history,
        "images": [
            {"view": "head", "video": "/fixture/head.mp4", "frame": frame},
            {"view": "left_wrist", "video": "/fixture/left.mp4", "frame": frame},
        ],
        "target": {
            "task": {
                "level": "legacy",
                "caption": "Place the blocks in the tray",
                "progress_percent": 50,
            },
            "predictions": predictions,
        },
    }
    return {
        "data_id": sample_id,
        "episode_key": episode,
        "profile": source_shape,
        "unit_type": anchor_kind,
        "unit_index": unit_index,
        "current_frame": frame,
        "v10_sample": sample,
    }


class BaselineAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "manifest.json").write_text(
            json.dumps({
                "complete": True,
                "content_digest": "fixture-digest",
                "version": "fixture-version",
            }),
            encoding="utf-8",
        )
        self.source = self.root / "rows.jsonl"
        self.adapter = BaselineAdapter(
            snapshot_root=self.root,
            expected_content_digest=None,
            expected_version=None,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, rows: list[dict[str, object]]) -> None:
        self.source.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def test_clean_rows_become_explicit_action_segment_supervision(self) -> None:
        rows = [
            _row(
                sample_id="action-only",
                episode="episode-action",
                source_shape="L3L1",
                anchor_kind="action",
                unit_index=0,
                frame=10,
                history=[],
                action_captions=("Approach the blue block", "Grasp the blue block"),
            ),
            _row(
                sample_id="segment-only",
                episode="episode-segment",
                source_shape="L3L0",
                anchor_kind="segment",
                unit_index=0,
                frame=20,
                history=[],
                segment_captions=("Move above the blue block", "Close the gripper"),
            ),
            _row(
                sample_id="both",
                episode="episode-both",
                source_shape="L3L1L0",
                anchor_kind="action",
                unit_index=0,
                frame=30,
                history=[],
                action_captions=("Approach the red block", "Lift the red block"),
                segment_captions=("Move above the red block", "Raise the gripper"),
            ),
        ]
        self._write(rows)
        records = list(self.adapter.iter_ongoing(source_paths=[self.source]))
        self.assertEqual(len(records), 3)
        action, segment, both = records
        self.assertTrue(
            action["supervision"]["predictions"][0]["action"]["label_available"]
        )
        self.assertFalse(
            action["supervision"]["predictions"][0]["segment"]["label_available"]
        )
        self.assertFalse(
            segment["supervision"]["predictions"][0]["action"]["label_available"]
        )
        self.assertTrue(
            segment["supervision"]["predictions"][0]["segment"]["label_available"]
        )
        self.assertTrue(
            both["supervision"]["predictions"][0]["action"]["label_available"]
        )
        self.assertTrue(
            both["supervision"]["predictions"][0]["segment"]["label_available"]
        )

    def test_invalid_higher_anchor_is_excluded_without_nested_projection(self) -> None:
        invalid = _row(
            sample_id="invalid-higher-anchor",
            episode="episode-invalid",
            source_shape="full",
            anchor_kind="subtask",
            unit_index=0,
            frame=10,
            history=[],
            action_captions=("This nested action must never be used",),
            segment_captions=("This nested segment must never be used",),
        )
        invalid["v10_sample"]["target"]["predictions"][0]["subtask"] = _unit(
            "Invalid higher label", 10, kind="action"
        )
        self._write([invalid])
        self.assertEqual(list(self.adapter.iter_ongoing(source_paths=[self.source])), [])
        self.assertEqual(
            self.adapter.statistics.excluded_rows["invalid_anchor_kind"], 1
        )

    def test_last_row_stays_ongoing_and_does_not_invent_end(self) -> None:
        last = _row(
            sample_id="last",
            episode="episode-last",
            source_shape="L3L1",
            anchor_kind="action",
            unit_index=1,
            frame=99,
            history=["Approach the object"],
            action_captions=("Release the object",),
        )
        self._write([last])
        record = next(self.adapter.iter_ongoing(source_paths=[self.source]))
        self.assertEqual(record["category"], "ongoing")
        self.assertFalse(
            record["supervision"]["predictions"][1]["action"]["label_available"]
        )
        self.assertEqual(
            record["supervision"]["execution_decision"],
            {"label_available": False, "value": ""},
        )
        self.assertNotIn("End", json.dumps(record))
        self.assertTrue(record["history_material"]["with_memory_eligible"])
        self.assertEqual(
            record["history_material"]["short"]["progress_percent"], 100
        )

    def test_model_visible_dirty_term_is_reasoned_exclusion(self) -> None:
        row = _row(
            sample_id="dirty-text",
            episode="episode-dirty-text",
            source_shape="L3L1",
            anchor_kind="action",
            unit_index=1,
            frame=99,
            history=["Complete the previous subtask"],
            action_captions=("Release the object",),
        )
        self._write([row])
        self.assertEqual(
            list(self.adapter.iter_ongoing(source_paths=[self.source])), []
        )
        self.assertEqual(
            self.adapter.statistics.excluded_rows[
                "model_visible_forbidden_v5_term"
            ],
            1,
        )

    def test_natural_recover_verb_is_not_the_removed_decision(self) -> None:
        row = _row(
            sample_id="natural-recover",
            episode="episode-natural-recover",
            source_shape="L3L1",
            anchor_kind="action",
            unit_index=1,
            frame=99,
            history=["Retract upward to recover the robotic arm"],
            action_captions=("Approach the next object",),
        )
        self._write([row])
        records = list(self.adapter.iter_ongoing(source_paths=[self.source]))
        self.assertEqual(len(records), 1)

    def test_streaming_collector_handles_shuffled_rows_and_builds_plan(self) -> None:
        later = _row(
            sample_id="later",
            episode="episode-plan",
            source_shape="L3L1L0",
            anchor_kind="action",
            unit_index=1,
            frame=20,
            history=["Approach the cup"],
            action_captions=("Grasp the cup",),
            segment_captions=("Close the gripper",),
        )
        earlier = _row(
            sample_id="earlier",
            episode="episode-plan",
            source_shape="L3L1L0",
            anchor_kind="action",
            unit_index=0,
            frame=10,
            history=[],
            action_captions=("Approach the cup", "Grasp the cup"),
            segment_captions=("Move above the cup", "Close the gripper"),
        )
        self._write([later, earlier])
        plans = list(self.adapter.iter_initial_plans(source_paths=[self.source]))
        self.assertEqual(len(plans), 1)
        plan = plans[0]
        self.assertEqual(plan["category"], "initial_plan")
        self.assertEqual(plan["anchor_frame"], 10)
        self.assertEqual(
            [item["action"]["caption"] for item in plan["supervision"]["initial_plan"]],
            ["Approach the cup", "Grasp the cup"],
        )
        self.assertFalse(plan["history_material"]["with_memory_eligible"])

    def test_no_legacy_hierarchy_or_shape_name_leaks_to_canonical_record(self) -> None:
        row = _row(
            sample_id="clean",
            episode="episode-clean",
            source_shape="L3L1L0",
            anchor_kind="action",
            unit_index=0,
            frame=10,
            history=[],
            action_captions=("Approach the object",),
            segment_captions=("Move forward",),
        )
        self._write([row])
        encoded = json.dumps(
            next(self.adapter.iter_ongoing(source_paths=[self.source])),
            sort_keys=True,
        )
        self.assertNotRegex(encoded, r'(?i)subtask|"profile"|"l0"|\bL[0-3]\b')

    def test_incomplete_or_unpinned_snapshot_is_rejected(self) -> None:
        (self.root / "manifest.json").write_text(
            json.dumps({"complete": False, "content_digest": "wrong", "version": "wrong"}),
            encoding="utf-8",
        )
        with self.assertRaises(BaselineAdapterError):
            BaselineAdapter(snapshot_root=self.root)


if __name__ == "__main__":
    unittest.main()
