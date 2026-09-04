from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.train.v10_continuous_v5.materialize_v5 import _sample, materialize_dataset
from scripts.train.v10_continuous_v5 import validate_v5 as validate_module
from scripts.train.v10_continuous_v5.snapshot_v5 import (
    clone_incremental_snapshot,
    compare_snapshots,
    compose_snapshot,
)
from scripts.train.v10_continuous_v5.validate_v5 import validate_dataset
from scripts.train.v10_continuous_v5.schema_v5 import FAILURE_TYPES


def _fixture_sample(
    source: str,
    *,
    episode_key: str | None = None,
    task_name: str = "place_blue_block",
) -> dict[str, object]:
    if source == "takeover_q":
        return _sample(
            sample_id="takeover_q_sample",
            base_sample_id="takeover_q_base",
            source=source,
            category="takeover",
            memory_variant="no_memory",
            output_spec={
                "prediction1_units": ["action", "segment"],
                "prediction2_units": ["action"],
                "plan_units": ["action"],
            },
            task_instruction="Place the blue block in the tray",
            images=["/fixture/frame.jpg"],
            prompt_context={},
            target={
                "execution_decision": "Takeover",
                "decision_detail": {
                    "failure_analysis": {
                        "failed_action_context": "Approach the blue block",
                        "expected_action": "Grasp the blue block",
                        "observed_failure": "The gripper misses the blue block",
                        "failure_type": FAILURE_TYPES[0],
                    },
                    "recovery_plan": [{
                        "index": 1,
                        "action": {"caption": "Realign and grasp the blue block"},
                    }],
                },
            },
            loss_mask_paths=(),
            provenance={
                "episode_key": episode_key or "takeover_q_episode",
                "task_name": task_name,
                "split": "train",
                "canonical_source": "fixture/takeover_q",
                "memory_pair_eligible": False,
                "label_sources": {"failed_action_context": "q4.caption"},
                "future_takeover_frames_used": False,
            },
        )
    return _sample(
        sample_id=f"{source}_sample",
        base_sample_id=f"{source}_base",
        source=source,
        category="initial_plan",
        memory_variant="no_memory",
        output_spec={
            "prediction1_units": [],
            "prediction2_units": [],
            "plan_units": ["action"],
        },
        task_instruction="Place the blue block in the tray",
        images=["/fixture/frame.jpg"],
        prompt_context={},
        target={
            "initial_plan": [
                {
                    "index": 1,
                    "action": {
                        "caption": "Place the blue block in the tray",
                    },
                }
            ]
        },
        loss_mask_paths=(),
        provenance={
            "episode_key": episode_key or f"{source}_episode",
            "task_name": task_name,
            "split": "train",
            "canonical_source": f"fixture/{source}",
            "memory_pair_eligible": False,
        },
    )


class SnapshotV5Test(unittest.TestCase):
    @staticmethod
    def _compose_formal_fixture(
        root: Path,
        *,
        robodojo_episode: str,
        robodojo_task: str,
        split_marker: str = "fixture-a",
    ) -> tuple[Path, Path]:
        root.mkdir(parents=True, exist_ok=True)
        official_split = root / "official_split.json"
        official_split.write_text(
            json.dumps(
                {
                    "meta": {"fixture_marker": split_marker},
                    "train": {"train_task": [0]},
                    "holdout_traj": {"train_task": [1]},
                    "holdout_task": {"heldout_task": [0]},
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        sources = []
        for source in ("baseline", "robodojo", "takeover_q"):
            output = root / f"source_{source}"
            sample = (
                _fixture_sample(
                    source,
                    episode_key=robodojo_episode,
                    task_name=robodojo_task,
                )
                if source == "robodojo"
                else _fixture_sample(source)
            )
            materialize_dataset(
                [sample],
                output,
                source=source,
                partial=False,
                limit=None,
                robodojo_official_split_path=(
                    official_split if source == "robodojo" else None
                ),
                plan_cache_report=(
                    {
                        "schema_version": "v5_takeover_q_exclusion_report_v1",
                        "scan_complete": True,
                        "num_exclusions": 0,
                        "exclusion_reason_counts": {},
                        "exclusions": [],
                    }
                    if source == "takeover_q"
                    else (
                        {"schema_version": "fixture_baseline_plan_report_v1"}
                        if source == "baseline"
                        else None
                    )
                ),
            )
            sources.append(output)
        snapshot = root / "formal"
        compose_snapshot(sources, snapshot, snapshot_id="formal-fixture")
        return snapshot, official_split

    def test_atomic_compose_and_incremental_clone_are_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = []
            for source in ("baseline", "robodojo", "takeover_q"):
                output = root / f"source_{source}"
                materialize_dataset(
                    [_fixture_sample(source)],
                    output,
                    source=source,
                    partial=True,
                    limit=1,
                )
                sources.append(output)
            full = root / "full"
            current = root / "current.json"
            manifest = compose_snapshot(
                sources,
                full,
                snapshot_id="fixture-full",
                current_path=current,
            )
            self.assertTrue(manifest["complete"])
            self.assertTrue(manifest["partial"])
            self.assertEqual(manifest["num_samples"], 3)
            self.assertTrue(current.is_file())
            audit = validate_dataset(
                full, expect_composed=True, require_full=False
            )
            self.assertTrue(audit["passed"])
            self.assertEqual(audit["sample_count"], 3)

            incremental = root / "incremental"
            cloned = clone_incremental_snapshot(
                full,
                incremental,
                snapshot_id="fixture-incremental",
            )
            self.assertEqual(cloned["build_mode"], "incremental_no_delta")
            self.assertTrue(cloned["equivalence"]["equivalent"])
            self.assertTrue(compare_snapshots(full, incremental)["equivalent"])

    def test_formal_validation_uses_35_across_splits_but_train_only_tasks(self) -> None:
        assignments = {
            "robotwin30_x2/arx_x5/train_task/trajectory_0": "train",
            "robotwin30_x2/arx_x5/train_task/trajectory_1": "holdout_traj",
            "robotwin30_x2/arx_x5/heldout_task/trajectory_0": "holdout_task",
        }
        with tempfile.TemporaryDirectory() as temporary:
            snapshot, official_split = self._compose_formal_fixture(
                Path(temporary),
                robodojo_episode=(
                    "robotwin30_x2/arx_x5/train_task/trajectory_0"
                ),
                robodojo_task="train_task",
            )
            with (
                mock.patch.object(
                    validate_module,
                    "ROBODOJO_TASKS",
                    ("train_task", "heldout_task"),
                ),
                mock.patch.object(
                    validate_module,
                    "load_official_split_assignments",
                    return_value=assignments,
                ),
                mock.patch.object(
                    validate_module, "FAILURE_TYPES", (FAILURE_TYPES[0],)
                ),
            ):
                audit = validate_dataset(
                    snapshot,
                    expect_composed=True,
                    require_full=True,
                    official_split_path=official_split,
                )
                partial_gate_audit = validate_dataset(
                    snapshot,
                    expect_composed=True,
                    require_full=False,
                    official_split_path=official_split,
                )
            snapshot_manifest = json.loads(
                (snapshot / "manifest.json").read_text(encoding="utf-8")
            )
            split_metadata = snapshot_manifest["sources"]["robodojo"][
                "robodojo_official_split"
            ]
            snapshot_split_copy_exists = (
                snapshot / split_metadata["snapshot_relative_path"]
            ).is_file()
        self.assertEqual(audit["robodojo_task_count"], 1)
        self.assertEqual(audit["robodojo_official_train_task_count"], 1)
        self.assertEqual(audit["robodojo_official_task_count"], 2)
        self.assertEqual(len(audit["robodojo_official_split_sha256"]), 64)
        self.assertEqual(
            split_metadata["sha256"],
            audit["robodojo_official_split_sha256"],
        )
        self.assertTrue(snapshot_split_copy_exists)
        self.assertEqual(
            partial_gate_audit["robodojo_official_split_sha256"],
            audit["robodojo_official_split_sha256"],
        )
        self.assertEqual(partial_gate_audit["robodojo_task_count"], 1)
        self.assertEqual(audit["takeover_q_exclusion_count"], 0)
        self.assertEqual(
            len(snapshot_manifest["sources"]["baseline"][
                "plan_cache_report_sha256"
            ]),
            64,
        )

    def test_formal_validation_rejects_tampered_takeover_exclusion_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot, official_split = self._compose_formal_fixture(
                Path(temporary),
                robodojo_episode=(
                    "robotwin30_x2/arx_x5/train_task/trajectory_0"
                ),
                robodojo_task="train_task",
            )
            report = (
                snapshot
                / "metadata"
                / "sources"
                / "takeover_q"
                / "plan_cache_report.json"
            )
            report.write_text('{"tampered":true}\n', encoding="utf-8")
            assignments = {
                "robotwin30_x2/arx_x5/train_task/trajectory_0": "train",
                "robotwin30_x2/arx_x5/train_task/trajectory_1": "holdout_traj",
                "robotwin30_x2/arx_x5/heldout_task/trajectory_0": "holdout_task",
            }
            with (
                mock.patch.object(
                    validate_module,
                    "ROBODOJO_TASKS",
                    ("train_task", "heldout_task"),
                ),
                mock.patch.object(
                    validate_module,
                    "load_official_split_assignments",
                    return_value=assignments,
                ),
                mock.patch.object(validate_module, "FAILURE_TYPES", ()),
            ):
                with self.assertRaisesRegex(ValueError, "copy/digest mismatch"):
                    validate_dataset(
                        snapshot,
                        expect_composed=True,
                        require_full=True,
                        official_split_path=official_split,
                    )

    def test_formal_validation_rejects_task_and_trajectory_holdouts(self) -> None:
        assignments = {
            "robotwin30_x2/arx_x5/train_task/trajectory_0": "train",
            "robotwin30_x2/arx_x5/train_task/trajectory_1": "holdout_traj",
            "robotwin30_x2/arx_x5/heldout_task/trajectory_0": "holdout_task",
        }
        cases = (
            (
                "robotwin30_x2/arx_x5/train_task/trajectory_1",
                "train_task",
            ),
            (
                "robotwin30_x2/arx_x5/heldout_task/trajectory_0",
                "heldout_task",
            ),
        )
        for episode_key, task_name in cases:
            with self.subTest(episode_key=episode_key):
                with tempfile.TemporaryDirectory() as temporary:
                    snapshot, official_split = self._compose_formal_fixture(
                        Path(temporary),
                        robodojo_episode=episode_key,
                        robodojo_task=task_name,
                    )
                    with (
                        mock.patch.object(
                            validate_module,
                            "ROBODOJO_TASKS",
                            ("train_task", "heldout_task"),
                        ),
                        mock.patch.object(
                            validate_module,
                            "load_official_split_assignments",
                            return_value=assignments,
                        ),
                        mock.patch.object(validate_module, "FAILURE_TYPES", ()),
                    ):
                        with self.assertRaisesRegex(
                            ValueError, "task/trajectory holdout"
                        ):
                            validate_dataset(
                                snapshot,
                                expect_composed=True,
                                require_full=True,
                                official_split_path=official_split,
                            )

    def test_formal_validation_rejects_official_split_digest_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot, official_split = self._compose_formal_fixture(
                Path(temporary),
                robodojo_episode=(
                    "robotwin30_x2/arx_x5/train_task/trajectory_0"
                ),
                robodojo_task="train_task",
            )
            official_split.write_text('{"changed":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "digest disagrees with the snapshot"
            ):
                validate_dataset(
                    snapshot,
                    expect_composed=True,
                    require_full=True,
                    official_split_path=official_split,
                )

    def test_split_digest_participates_in_snapshot_input_cache_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, _ = self._compose_formal_fixture(
                root / "first",
                robodojo_episode=(
                    "robotwin30_x2/arx_x5/train_task/trajectory_0"
                ),
                robodojo_task="train_task",
                split_marker="first",
            )
            second, _ = self._compose_formal_fixture(
                root / "second",
                robodojo_episode=(
                    "robotwin30_x2/arx_x5/train_task/trajectory_0"
                ),
                robodojo_task="train_task",
                split_marker="second",
            )
            first_manifest = json.loads(
                (first / "manifest.json").read_text(encoding="utf-8")
            )
            second_manifest = json.loads(
                (second / "manifest.json").read_text(encoding="utf-8")
            )
        self.assertEqual(
            first_manifest["content_digest"],
            second_manifest["content_digest"],
        )
        self.assertNotEqual(
            first_manifest["input_cache_key"],
            second_manifest["input_cache_key"],
        )


if __name__ == "__main__":
    unittest.main()
