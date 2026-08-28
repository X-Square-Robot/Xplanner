from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from scripts.train.v10_continuous_v5.indexed_io_v5 import (
    IndexedLeafWriter,
    LeafKey,
    LEAF_SCHEMA_VERSION,
    read_indexed_item,
)
from scripts.train.v10_continuous_v5.materialize_v5 import (
    _iter_takeover_records_with_report,
    _smoke_all_takeover_failure_types,
    _smoke_one_per_train_task,
    _takeover_cache_row_to_record,
    convert_robodojo_episode_records,
    main as materialize_main,
    materialize_dataset,
    takeover_record_to_sample,
)
from scripts.train.v10_continuous_v5.robodojo_adapter import (
    DEFAULT_OFFICIAL_SPLIT,
    ActionLabel,
    RobodojoEpisode,
    build_canonical_records,
)
from scripts.train.v10_continuous_v5.schema_v5 import (
    FAILURE_TYPE_BY_SOURCE_CODE,
    validate_sample,
)


def _dataset_discoverer_contract() -> tuple[str, Any]:
    """Load the actual dependency-free discovery nodes from dataset_v5."""

    path = Path(__file__).with_name("dataset_v5.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "LEAF_SCHEMA_VERSION"
            for target in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "discover_leaf_datasets":
            nodes.append(node)
    if len(nodes) != 2:
        raise AssertionError("dataset_v5 discovery contract nodes are missing")

    def file_sha256(value: Path) -> str:
        return hashlib.sha256(value.read_bytes()).hexdigest()

    namespace = {
        "Any": Any,
        "Path": Path,
        "file_sha256": file_sha256,
        "json": json,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["LEAF_SCHEMA_VERSION"], namespace["discover_leaf_datasets"]


DATASET_LEAF_SCHEMA_VERSION, discover_leaf_datasets = _dataset_discoverer_contract()


def _robodojo_records(action_count: int = 10) -> tuple[dict, ...]:
    actions = tuple(
        ActionLabel(
            start_frame=index * 10,
            end_frame=(index + 1) * 10,
            caption=f"Move object number {index + 1}",
        )
        for index in range(action_count)
    )
    episode = RobodojoEpisode(
        canonical_episode_id="robotwin30_x2/arx_x5/build_tower/trajectory_0",
        task_name="build_tower",
        trajectory_name="trajectory_0",
        split="train",
        task_instruction="Arrange ten objects in order",
        total_frames=action_count * 10,
        videos=(
            ("face_view", "/data/face.mp4"),
            ("left_wrist_view", "/data/left.mp4"),
            ("right_wrist_view", "/data/right.mp4"),
        ),
        actions=actions,
        media_instruction_file="/labels/task/instruction.json",
        action_annotation_file="/labels/action/instruction.json",
    )
    return build_canonical_records(episode)


def _takeover_record(decision: str) -> dict:
    labels = {
        "execution_decision": decision,
        "current_action": "Move toward the green block",
    }
    label_sources = {
        "execution_decision": "q1 normal-execution membership",
        "current_action": "q1.caption",
    }
    provenance = {
        "episode_key": "takeover_episode_001",
        "case_id": "case_001",
        "snapshot_id": "fixture",
    }
    if decision == "Takeover":
        labels = {
            "execution_decision": "Takeover",
            "expected_action": "Grasp the green block",
            "observed_failure": "The gripper closes without securing the block",
            "failure_type": "adapter-local text is not authoritative",
            "failed_action_context": "Approach and align with the green block",
            "recovery_action": "Realign the gripper and grasp the green block",
        }
        label_sources = {
            "execution_decision": "takeover case membership",
            "expected_action": "q2q3.q2_caption expected side",
            "observed_failure": "q2q3.q2_caption actual side",
            "failure_type": "q2q3.q3_type normalized offline",
            "failed_action_context": "q4.caption",
            "recovery_action": "takeover.caption",
        }
        provenance["raw_failure_source_key"] = "1.1"
        provenance["raw_q3_type"] = "1.1 offline provenance"
    return {
        "schema_version": "v5_takeover_adapter_v1",
        "sample_id": f"takeover-fixture-{decision.casefold()}",
        "task_type": "ongoing",
        "decision_class": decision,
        "memory_variant": "no_memory",
        "conditioning": {
            "task_instruction": "Place the green block in the tray"
        },
        "images": [
            {"video": "/data/face.mp4", "frame": 20, "view": "head"},
            {"video": "/data/left.mp4", "frame": 20, "view": "left_wrist"},
            {"video": "/data/right.mp4", "frame": 20, "view": "right_wrist"},
        ],
        "anchor_frame": 20,
        "context_frames": [20],
        "labels": labels,
        "label_timing_sec": {},
        "supervision": {
            "status": "direct_or_reviewed_derived",
            "label_sources": label_sources,
        },
        "provenance": provenance,
    }


class MaterializeV5Test(unittest.TestCase):
    def test_takeover_cache_projection_ignores_continue_and_rerenders_takeover(self) -> None:
        sample = takeover_record_to_sample(_takeover_record("Takeover"))
        projected = _takeover_cache_row_to_record({"v5_sample": sample})
        self.assertIsNotNone(projected)
        rebuilt = takeover_record_to_sample(projected)
        self.assertEqual(
            set(rebuilt["target"]), {"execution_decision", "decision_detail"}
        )
        self.assertEqual(rebuilt["target"]["execution_decision"], "Takeover")

        continue_sample = dict(sample)
        continue_sample["category"] = "ongoing"
        self.assertIsNone(
            _takeover_cache_row_to_record({"v5_sample": continue_sample})
        )

    def test_takeover_smoke_selector_covers_all_failure_types_deterministically(self) -> None:
        failure_order = tuple(FAILURE_TYPE_BY_SOURCE_CODE.values())
        records = list(
            {
                "sample_id": f"failure-{index}",
                "decision_class": "Takeover",
                "labels": {"failure_type": failure_type},
            }
            for index, failure_type in enumerate(reversed(failure_order))
        )
        records.append({
            "sample_id": "duplicate",
            "decision_class": "Takeover",
            "labels": {"failure_type": failure_order[0]},
        })

        selected = _smoke_all_takeover_failure_types(records)

        self.assertEqual(len(selected), 15)
        self.assertEqual(
            [record["labels"]["failure_type"] for record in selected],
            list(failure_order),
        )
        self.assertNotEqual(selected[0]["sample_id"], "duplicate")
        with self.assertRaisesRegex(ValueError, "missing failure types"):
            _smoke_all_takeover_failure_types(records[:2])

    def test_robodojo_cli_requests_train_only_and_rejects_holdout_scan(self) -> None:
        unsafe_scan = SimpleNamespace(
            episodes=(SimpleNamespace(split="holdout_traj"),)
        )
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch(
                "scripts.train.v10_continuous_v5.materialize_v5.scan_robodojo",
                return_value=unsafe_scan,
            ) as scanner:
                with self.assertRaisesRegex(
                    RuntimeError, "received a holdout episode"
                ):
                    materialize_main([
                        "--source",
                        "robodojo",
                        "--output",
                        str(Path(temporary) / "output"),
                    ])
        scanner.assert_called_once_with(
            official_split_path=DEFAULT_OFFICIAL_SPLIT.resolve(),
            include_splits=("train",),
            allow_holdouts=False,
        )

    def test_smoke_selector_is_deterministic_and_cli_records_it(self) -> None:
        def episode(task: str, trajectory: int) -> SimpleNamespace:
            trajectory_name = f"trajectory_{trajectory}"
            return SimpleNamespace(
                canonical_episode_id=(
                    f"robotwin30_x2/arx_x5/{task}/{trajectory_name}"
                ),
                task_name=task,
                trajectory_name=trajectory_name,
                split="train",
            )

        episodes = (
            episode("build_tower", 4),
            episode("arrange_largest_number", 10),
            episode("arrange_largest_number", 2),
            episode("build_tower", 1),
        )
        assignments = {
            item.canonical_episode_id: "train" for item in episodes
        }
        assignments[
            "robotwin30_x2/arx_x5/build_tower/trajectory_99"
        ] = "holdout_traj"
        selected = _smoke_one_per_train_task(episodes, assignments)
        self.assertEqual(
            [item.canonical_episode_id for item in selected],
            [
                "robotwin30_x2/arx_x5/arrange_largest_number/trajectory_2",
                "robotwin30_x2/arx_x5/build_tower/trajectory_1",
            ],
        )

        with tempfile.TemporaryDirectory() as temporary:
            split_path = Path(temporary) / "split.json"
            split_path.write_text("{}\n", encoding="utf-8")
            with (
                mock.patch(
                    "scripts.train.v10_continuous_v5.materialize_v5.scan_robodojo",
                    return_value=SimpleNamespace(episodes=episodes),
                ),
                mock.patch(
                    "scripts.train.v10_continuous_v5.materialize_v5."
                    "load_official_split_assignments",
                    return_value=assignments,
                ),
                mock.patch(
                    "scripts.train.v10_continuous_v5.materialize_v5."
                    "materialize_dataset",
                    return_value={"complete": True},
                ) as publisher,
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    materialize_main([
                        "--source",
                        "robodojo",
                        "--output",
                        str(Path(temporary) / "output"),
                        "--robodojo-official-split",
                        str(split_path),
                        "--smoke-one-per-train-task",
                    ])
        published = publisher.call_args.kwargs
        self.assertTrue(published["partial"])
        self.assertIsNone(published["limit"])
        self.assertEqual(
            published["selector"]["mode"],
            "first_valid_episode_per_official_train_task",
        )
        self.assertEqual(published["selector"]["selected_train_task_count"], 2)
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            materialize_main([
                "--source",
                "robodojo",
                "--output",
                "/tmp/not_published",
                "--limit",
                "1",
                "--smoke-one-per-train-task",
            ])

    def test_robodojo_strict_memory_pairs_and_end_review_fixture(self) -> None:
        samples, review = convert_robodojo_episode_records(_robodojo_records())
        self.assertEqual(len(samples), 21)
        self.assertEqual(len(review), 1)
        self.assertEqual(review[0]["category"], "end")
        self.assertNotIn("end", {sample["category"] for sample in samples})

        initial = [sample for sample in samples if sample["category"] == "initial_plan"]
        self.assertEqual(len(initial), 1)
        self.assertEqual(initial[0]["memory_variant"], "no_memory")
        self.assertEqual(len(initial[0]["target"]["initial_plan"]), 10)
        self.assertFalse(initial[0]["provenance"]["memory_pair_eligible"])

        by_base: dict[str, list[dict]] = defaultdict(list)
        for sample in samples:
            validate_sample(sample)
            if sample["category"] == "ongoing":
                by_base[sample["base_sample_id"]].append(sample)
        self.assertEqual(len(by_base), 10)
        for pair in by_base.values():
            self.assertEqual(
                {sample["memory_variant"] for sample in pair},
                {"no_memory", "with_memory"},
            )
            no_memory = next(
                sample for sample in pair if sample["memory_variant"] == "no_memory"
            )
            with_memory = next(
                sample for sample in pair if sample["memory_variant"] == "with_memory"
            )
            self.assertEqual(no_memory["target"], with_memory["target"])
            self.assertEqual(no_memory["images"], with_memory["images"])
            self.assertEqual(no_memory["provenance"], with_memory["provenance"])
            self.assertEqual(no_memory["prompt_context"], {})
            self.assertTrue(
                no_memory["provenance"]["memory_pair_eligible"]
            )
            context = with_memory["prompt_context"]
            self.assertEqual(len(context["initial_plan_memory"]), 10)
            self.assertLessEqual(len(context["long_memory"]), 8)
            short = context["short_memory"]
            action_index = with_memory["provenance"]["action_index"]
            if action_index == 1:
                self.assertIsNone(short)
            else:
                self.assertLess(
                    short["task_progress_percent"],
                    with_memory["target"]["task_progress_percent"],
                )
                self.assertEqual(
                    short["prediction1"]["action"]["caption"],
                    f"Move object number {action_index - 1}",
                )
                self.assertTrue(short["prediction1"]["action"]["available"])
                self.assertNotIn("segment", short["prediction1"])
            masks = set(with_memory["supervision"]["loss_mask_paths"])
            self.assertNotIn("/predictions/0/segment", masks)
            self.assertNotIn("/predictions/1/segment", masks)
            self.assertNotIn("segment", with_memory["target"]["predictions"][0])
        last_with_memory = next(
            sample
            for sample in samples
            if sample["category"] == "ongoing"
            and sample["memory_variant"] == "with_memory"
            and sample["provenance"]["action_index"] == 10
        )
        self.assertEqual(
            len(last_with_memory["prompt_context"]["long_memory"]), 8
        )
        self.assertNotIn("action", last_with_memory["target"]["predictions"][1])

    def test_takeover_q_continue_is_rejected_and_failure_uses_q4_taxonomy(self) -> None:
        with self.assertRaisesRegex(ValueError, "Continue is excluded"):
            takeover_record_to_sample(_takeover_record("Continue"))
        failure = takeover_record_to_sample(_takeover_record("Takeover"))

        self.assertEqual(failure["category"], "takeover")
        self.assertEqual(failure["memory_variant"], "no_memory")
        self.assertEqual(failure["prompt_context"], {})
        self.assertEqual(
            failure["provenance"]["canonical_source"],
            "takeover_q/final_reviewed_bilingual",
        )
        detail = failure["target"]["decision_detail"]
        self.assertEqual(
            detail["failure_analysis"]["failed_action_context"],
            "Approach and align with the green block",
        )
        self.assertEqual(
            failure["provenance"]["label_sources"]["failed_action_context"],
            "q4.caption",
        )
        self.assertEqual(
            detail["failure_analysis"]["failure_type"],
            FAILURE_TYPE_BY_SOURCE_CODE["1.1"],
        )
        recovery = "Realign the gripper and grasp the green block"
        self.assertEqual(
            detail["recovery_plan"][0]["action"]["caption"], recovery
        )
        self.assertEqual(
            tuple(failure["target"]), ("execution_decision", "decision_detail")
        )
        self.assertEqual(failure["output_spec"], {
            "prediction1_units": ["action"],
            "prediction2_units": ["action"],
            "plan_units": ["action"],
        })
        self.assertEqual(
            failure["output_profile_id"], "p1-action__p2-action__plan-action"
        )
        self.assertEqual(failure["supervision"]["loss_mask_paths"], [])

    def test_atomic_indexed_leaves_support_random_access_and_review_sidecar(self) -> None:
        samples, review = convert_robodojo_episode_records(_robodojo_records(3))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "published"
            plan_cache_report = {"schema_version": "fixture", "split": "both"}
            manifest = materialize_dataset(
                samples,
                output,
                source="robodojo",
                partial=True,
                limit=1,
                review_fixtures=review,
                selector={"mode": "fixture_selector"},
                plan_cache_report=plan_cache_report,
            )
            self.assertTrue(output.is_dir())
            self.assertEqual(
                json.loads((output / "plan_cache_report.json").read_text(
                    encoding="utf-8"
                )),
                plan_cache_report,
            )
            self.assertEqual(manifest["num_samples"], 7)
            self.assertEqual(manifest["num_leaves"], 5)
            self.assertEqual(manifest["num_review_fixtures"], 1)
            self.assertEqual(
                manifest["canonical_raw_sources"], ["robotwin30_x2/arx_x5"]
            )
            self.assertTrue(manifest["partial"])
            self.assertEqual(
                manifest["selector"], {"mode": "fixture_selector"}
            )
            self.assertEqual(LEAF_SCHEMA_VERSION, DATASET_LEAF_SCHEMA_VERSION)

            for leaf in manifest["leaves"]:
                root = output / leaf["path"]
                self.assertEqual(
                    {path.name for path in root.iterdir()},
                    {
                        "data.jsonl",
                        "data.index",
                        "episodes.jsonl",
                        "manifest.json",
                    },
                )
                count = leaf["num_samples"]

                leaf_manifest = json.loads(
                    (root / "manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    leaf_manifest["canonical_raw_sources"],
                    ["robotwin30_x2/arx_x5"],
                )
                expected_pair_eligible = (
                    leaf["leaf"]["category"] == "ongoing"
                )
                self.assertIs(
                    leaf_manifest["memory_pair_eligible"],
                    expected_pair_eligible,
                )
                order = list(dict.fromkeys((count - 1, 0, count // 2)))
                for index in order:
                    outer = read_indexed_item(root, index)
                    self.assertEqual(
                        tuple(outer), ("data_id", "v5_sample", "image")
                    )
                    self.assertEqual(
                        outer["data_id"], outer["v5_sample"]["sample_id"]
                    )
                    self.assertEqual(
                        outer["image"], outer["v5_sample"]["images"]
                    )
                with self.assertRaises(IndexError):
                    read_indexed_item(root, count)
                episode_rows = [
                    json.loads(line)
                    for line in (root / "episodes.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                self.assertEqual(len(episode_rows), 1)
                self.assertEqual(episode_rows[0]["num_samples"], count)

            discovered = discover_leaf_datasets(output, split="train")
            self.assertEqual(len(discovered), 5)
            self.assertEqual(
                {item["path"] for item in discovered},
                {
                    str(output / leaf["path"])
                    for leaf in manifest["leaves"]
                },
            )
            self.assertEqual(
                {item["source"] for item in discovered}, {"robodojo"}
            )
            self.assertEqual(
                {
                    (item["category"], item["memory_pair_eligible"])
                    for item in discovered
                },
                {("initial_plan", False), ("ongoing", True)},
            )

            self.assertFalse(any(
                leaf["leaf"]["category"] == "end"
                for leaf in manifest["leaves"]
            ))
            fixture_path = (
                output / "review_fixtures" / "end_candidates.jsonl"
            )
            fixtures = fixture_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(fixtures), 1)
            self.assertEqual(json.loads(fixtures[0])["category"], "end")

            with self.assertRaises(FileExistsError):
                materialize_dataset(
                    samples,
                    output,
                    source="robodojo",
                    partial=True,
                    limit=1,
                    review_fixtures=review,
                )

    def test_takeover_exclusions_are_closed_into_the_atomic_sidecar(self) -> None:
        class Adapter:
            exclusions = (
                {"reason": "q2_contains_forbidden_v5_term", "case_id": "a"},
                {"reason": "q2_contains_forbidden_v5_term", "case_id": "b"},
                {"reason": "q4_contains_cjk", "case_id": "c"},
            )

            @staticmethod
            def iter_records(*, decisions: tuple[str, ...]) -> Any:
                assert decisions == ("Takeover",)
                yield {"sample_id": "one"}
                yield {"sample_id": "two"}

        report: dict[str, Any] = {}
        rows = list(_iter_takeover_records_with_report(Adapter(), report))

        self.assertEqual([row["sample_id"] for row in rows], ["one", "two"])
        self.assertTrue(report["scan_complete"])
        self.assertEqual(report["num_exclusions"], 3)
        self.assertEqual(report["exclusion_reason_counts"], {
            "q2_contains_forbidden_v5_term": 2,
            "q4_contains_cjk": 1,
        })
        self.assertEqual(len(report["exclusions"]), 3)

    def test_takeover_samples_create_takeover_leaves_only(self) -> None:
        samples = [takeover_record_to_sample(_takeover_record("Takeover"))]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "takeover"
            manifest = materialize_dataset(
                samples,
                output,
                source="takeover_q",
                partial=True,
                limit=1,
            )
            self.assertEqual(manifest["num_leaves"], 1)
            self.assertEqual(
                manifest["canonical_raw_sources"],
                ["takeover_q/final_reviewed_bilingual"],
            )
            self.assertEqual(
                {leaf["leaf"]["category"] for leaf in manifest["leaves"]},
                {"takeover"},
            )
            for leaf in manifest["leaves"]:
                self.assertEqual(leaf["leaf"]["source"], "takeover_q")
                self.assertEqual(leaf["leaf"]["memory_variant"], "no_memory")
                leaf_manifest = json.loads(
                    (output / leaf["path"] / "manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertFalse(leaf_manifest["memory_pair_eligible"])
                outer = read_indexed_item(output / leaf["path"], 0)
                validate_sample(outer["v5_sample"])

    def test_indexed_leaf_rejects_mixed_memory_pair_eligibility(self) -> None:
        samples, _review = convert_robodojo_episode_records(
            _robodojo_records(2)
        )
        no_memory = [
            sample
            for sample in samples
            if sample["category"] == "ongoing"
            and sample["memory_variant"] == "no_memory"
        ]
        conflicting = json.loads(json.dumps(no_memory[0]))
        conflicting["sample_id"] += "_conflict"
        conflicting["provenance"]["memory_pair_eligible"] = False
        with tempfile.TemporaryDirectory() as temporary:
            writer = IndexedLeafWriter(
                Path(temporary) / "leaf",
                LeafKey.from_sample(no_memory[0]),
            )
            try:
                writer.write(no_memory[0])
                with self.assertRaisesRegex(
                    ValueError, "identical within an indexed leaf"
                ):
                    writer.write(conflicting)
            finally:
                writer.abort()


if __name__ == "__main__":
    unittest.main()
