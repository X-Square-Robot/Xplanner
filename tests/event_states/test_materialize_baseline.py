from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from x_planner.data.event_states.baseline_adapter import (
    BaselineAdapter,
    EpisodeActionPlanCollector,
)
from x_planner.data.event_states.materialize_baseline import (
    build_samples,
    convert_initial_plan,
    convert_ongoing,
)
from tests.event_states.test_baseline_adapter import _row


class BaselineMaterializeV5Test(unittest.TestCase):
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

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def records(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
        self.source.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        adapter = BaselineAdapter(
            snapshot_root=self.root,
            expected_content_digest=None,
            expected_version=None,
        )
        return list(adapter.iter_ongoing(source_paths=(self.source,)))

    def test_action_segment_target_pairs_memory_without_current_target_leak(self) -> None:
        rows = [
            _row(
                sample_id="first",
                episode="episode-pair",
                source_shape="L3L1L0",
                anchor_kind="action",
                unit_index=0,
                frame=10,
                history=[],
                action_captions=("Approach the red block", "Grasp the red block"),
                segment_captions=("Move above the red block", "Close the gripper"),
            ),
            _row(
                sample_id="second",
                episode="episode-pair",
                source_shape="L3L1L0",
                anchor_kind="action",
                unit_index=1,
                frame=20,
                history=["Approach the red block"],
                action_captions=("Grasp the red block", "Lift the red block"),
                segment_captions=("Close the gripper", "Raise the gripper"),
            ),
        ]
        canonical = self.records(rows)
        collector = EpisodeActionPlanCollector(
            snapshot_version="fixture-version",
            snapshot_content_digest="fixture-digest",
        )
        collector.extend(canonical)
        initial_record = next(collector.iter_records())
        initial_sample, plan = convert_initial_plan(
            initial_record, mode="action_segment"
        )
        self.assertEqual(initial_sample["category"], "initial_plan")
        self.assertEqual(initial_sample["memory_variant"], "no_memory")

        samples = convert_ongoing(canonical[1], initial_plan=plan)
        self.assertEqual([sample["memory_variant"] for sample in samples], [
            "no_memory", "with_memory"
        ])
        self.assertEqual(samples[0]["base_sample_id"], samples[1]["base_sample_id"])
        self.assertEqual(samples[0]["target"], samples[1]["target"])
        short = samples[1]["prompt_context"]["short_memory"]
        self.assertEqual(short["prediction1"]["action"]["caption"], "Approach the red block")
        self.assertEqual(short["prediction1"]["action"]["progress_percent"], 100)
        self.assertNotIn("segment", short["prediction1"])
        self.assertNotEqual(
            short["prediction1"]["action"]["caption"],
            samples[1]["target"]["predictions"][0]["action"]["caption"],
        )

    def test_segment_only_is_no_memory_and_masks_unknown_decision(self) -> None:
        row = _row(
            sample_id="segment-only",
            episode="episode-segment",
            source_shape="L3L0",
            anchor_kind="segment",
            unit_index=0,
            frame=20,
            history=[],
            segment_captions=("Move above the blue block", "Close the gripper"),
        )
        canonical = self.records([row])[0]
        samples = convert_ongoing(canonical, initial_plan=None)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["memory_variant"], "no_memory")
        self.assertFalse(sample["provenance"]["memory_pair_eligible"])
        self.assertIn("/execution_decision", sample["supervision"]["loss_mask_paths"])
        self.assertNotIn("action", sample["target"]["predictions"][0])
        self.assertTrue(sample["target"]["predictions"][0]["segment"]["available"])
        self.assertEqual(sample["output_spec"]["prediction1_units"], ["segment"])

    def test_both_split_build_combines_streams_and_reports_deterministically(self) -> None:
        reports = {
            "train": {
                "split": "train",
                "source_paths": ["train.jsonl"],
                "pass1_canonical_rows": 7,
                "initial_plan_records": 2,
                "collector_conflicts": 1,
                "collector_incomplete": 0,
                "_episode_keys": {"episode-shared", "episode-train"},
            },
            "validation": {
                "split": "validation",
                "source_paths": ["validation.jsonl"],
                "pass1_canonical_rows": 3,
                "initial_plan_records": 1,
                "collector_conflicts": 0,
                "collector_incomplete": 1,
                "_episode_keys": {"episode-shared", "episode-validation"},
            },
        }

        def fake_single(
            *,
            split: str,
            max_rows_per_path: int | None,
            **_parallel: object,
        ):
            self.assertEqual(max_rows_per_path, 5)
            if split == "train":
                rows = (
                    {"split": split, "provenance": {"episode_key": "episode-shared"}},
                    {"split": split, "provenance": {"episode_key": "episode-train"}},
                )
            else:
                rows = (
                    {"split": split, "provenance": {"episode_key": "episode-shared"}},
                )
            return iter(rows), reports[split]

        with mock.patch(
            "x_planner.data.event_states.materialize_baseline."
            "_build_single_split_samples",
            side_effect=fake_single,
        ):
            samples, report = build_samples(split="both", max_rows_per_path=5)

        self.assertEqual(list(samples), [
            {"split": "train", "provenance": {"episode_key": "episode-train"}},
            {"split": "validation", "provenance": {"episode_key": "episode-shared"}},
        ])
        self.assertEqual(report["split"], "both")
        self.assertEqual(report["source_paths"], [
            "train.jsonl", "validation.jsonl"
        ])
        self.assertEqual(report["pass1_canonical_rows"], 10)
        self.assertEqual(report["initial_plan_records"], 3)
        self.assertEqual(report["collector_conflicts"], 1)
        self.assertEqual(report["collector_incomplete"], 1)
        self.assertEqual(set(report["splits"]), {"train", "validation"})
        self.assertEqual(report["split_protection"], {
            "policy": "validation_episode_precedence",
            "protected_validation_episodes": 2,
            "excluded_train_samples": 1,
        })


if __name__ == "__main__":
    unittest.main()
