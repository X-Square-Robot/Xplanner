from __future__ import annotations

import copy
import unittest

from x_planner.data.event_states.context_variants import noisy_context
from x_planner.data.event_states.prompt import render_user
from x_planner.data.event_states.schema import (
    FAILURE_TYPES,
    SCHEMA_VERSION_V53,
    V5ValidationError,
    output_profile_id,
    validate_sample,
)
from x_planner.data.event_states.build_snapshot import validate_baseline_context_contract


SPEC = {
    "prediction1_units": ["action"],
    "prediction2_units": ["action"],
    "plan_units": ["action"],
}


def _unit(caption: str, progress: int) -> dict:
    return {"available": True, "caption": caption, "progress_percent": progress}


def _plan(caption: str = "Move the gripper toward the cup") -> list[dict]:
    return [{"index": 1, "action": {"caption": caption}}]


def _target(category: str) -> dict:
    if category == "replan":
        return {
            "execution_decision": "Replan",
            "decision_detail": {
                "reason": "The gripper is misaligned with the cup",
                "updated_plan": _plan("Realign with the cup and grasp it"),
            },
        }
    if category == "takeover":
        return {
            "execution_decision": "Takeover",
            "decision_detail": {
                "failure_analysis": {
                    "failed_action_context": "The arm attempted to grasp the cup",
                    "expected_action": "The gripper should close around the cup",
                    "observed_failure": "The gripper closed beside the cup",
                    "failure_type": FAILURE_TYPES[8],
                },
                "recovery_plan": _plan("Realign with the cup and grasp it"),
            },
        }
    decision = "End" if category == "end" else "Continue"
    return {
        "task_progress_percent": 100 if category == "end" else 40,
        "predictions": [
            {"index": 1, "role": "current", "action": _unit("Approach the cup", 60)},
            {"index": 2, "role": "next", "action": _unit("Grasp the cup", 0)},
        ],
        "execution_decision": decision,
        "decision_detail": {"outcome": "completed"} if decision == "End" else None,
    }


def _sample(category: str, bucket: str, *, context_variant: str = "no_memory_no_initial") -> dict:
    context = {}
    if context_variant.startswith("with_memory"):
        context = {
            "long_memory": [
                {"index": 1, "action": "Move toward the table"},
                {"index": 2, "action": "Open the gripper"},
            ],
            "short_memory": {
                "task_progress_percent": 25,
                "prediction1": {"action": _unit("Open the gripper", 80)},
            },
        }
        if "with_initial" in context_variant:
            context = {"initial_plan_memory": _plan(), **context}
    provenance = {
        "episode_key": f"episode-{category}",
        "task_name": "pick_cup",
        "split": "train",
        "memory_pair_eligible": bool(context),
    }
    if bucket in {"replan_self", "replan_open"}:
        provenance.update({
            "task_instruction_source": "samples_json.instruction",
            "task_instruction_source_path": "/fixture/samples.json",
        })
    sample = {
        "schema_version": SCHEMA_VERSION_V53,
        "sample_id": f"sample-{category}-{context_variant}",
        "base_sample_id": f"sample-{category}",
        "source": bucket,
        "training_bucket": bucket,
        "category": category,
        "context_variant": context_variant,
        "split": "train",
        "output_spec": copy.deepcopy(SPEC),
        "output_profile_id": output_profile_id(SPEC),
        "task_instruction": "Pick up the cup",
        "images": ["/tmp/frame.jpg"],
        "prompt_context": context,
        "target": _target(category),
        "supervision": {"loss_mask_paths": []},
        "provenance": provenance,
    }
    return sample


class V53ContractTest(unittest.TestCase):
    def test_baseline_snapshot_requires_requested_context_variants(self) -> None:
        execution = {
            "no_memory_no_initial": 10,
            "with_memory_no_initial": 8,
            "with_memory_no_initial_noisy": 8,
            "with_memory_with_initial": 6,
            "with_memory_with_initial_noisy": 6,
        }
        report = validate_baseline_context_contract([
            {
                "training_bucket": "initial_plan",
                "context_variant_counts": {"no_memory_no_initial": 4},
            },
            {"training_bucket": "ongoing", "context_variant_counts": execution},
            {"training_bucket": "end", "context_variant_counts": execution},
        ])
        self.assertTrue(report["passed"])

    def test_baseline_snapshot_rejects_missing_noisy_with_initial(self) -> None:
        with self.assertRaisesRegex(ValueError, "with_memory_with_initial_noisy"):
            validate_baseline_context_contract([
                {
                    "training_bucket": "initial_plan",
                    "context_variant_counts": {"no_memory_no_initial": 4},
                },
                {
                    "training_bucket": "ongoing",
                    "context_variant_counts": {
                        "no_memory_no_initial": 10,
                        "with_memory_with_initial": 6,
                    },
                },
                {
                    "training_bucket": "end",
                    "context_variant_counts": {
                        "no_memory_no_initial": 2,
                        "with_memory_with_initial": 2,
                        "with_memory_with_initial_noisy": 2,
                    },
                },
            ])

    def test_production_source_rejects_slug_instruction_provenance(self) -> None:
        sample = _sample("ongoing", "ongoing")
        sample["source"] = "baseline_v2v3umi"
        sample["provenance"].update({
            "task_instruction_source": "episode_slug",
            "task_instruction_source_path": "/fixture/episode",
        })
        with self.assertRaises(V5ValidationError):
            validate_sample(sample)

        sample["provenance"].update({
            "task_instruction_source": "media_task_instruction_json.episode.instruction",
            "task_instruction_source_path": "/fixture/instruction.json",
        })
        validate_sample(sample)

    def test_takeover_requires_exact_instruction_json_provenance(self) -> None:
        sample = _sample("takeover", "takeover")
        sample["source"] = "takeover_q"
        sample["provenance"].update({
            "task_instruction_source": "bilingual.instruction",
            "task_instruction_source_path": "/fixture/reviewed_episode.json",
        })
        with self.assertRaisesRegex(V5ValidationError, "untrusted"):
            validate_sample(sample)

        sample["provenance"].update({
            "task_instruction_source": (
                "media_task_instruction_json.episode.detailed_instruction"
            ),
            "task_instruction_source_path": "/fixture/instruction.json",
            "task_instruction_source_field": "detailed_instruction",
            "task_instruction_source_sha256": "a" * 64,
            "task_instruction_checked_paths": ["/fixture/instruction.json"],
            "task_instruction_policy": (
                "exact_episode_instruction_json_instruction_then_detailed_"
                "with_placeholder_rejection_v1"
            ),
        })
        validate_sample(sample)

        sample["provenance"]["task_instruction_source_path"] = ""
        with self.assertRaisesRegex(V5ValidationError, "source JSON path"):
            validate_sample(sample)

    def test_replan_is_decision_only(self) -> None:
        sample = validate_sample(_sample("replan", "replan_self"))
        self.assertEqual(tuple(sample["target"]), ("execution_decision", "decision_detail"))

    def test_execution_instruction_is_category_independent(self) -> None:
        prompts = [
            render_user(validate_sample(_sample(category, bucket))).rsplit("\n", 1)[-1]
            for category, bucket in (
                ("ongoing", "ongoing"),
                ("end", "end"),
                ("takeover", "takeover"),
                ("replan", "replan_open"),
            )
        ]
        self.assertEqual(len(set(prompts)), 1)
        self.assertIn("For Replan", prompts[0])

    def test_memory_without_initial_is_valid(self) -> None:
        value = validate_sample(_sample("ongoing", "ongoing", context_variant="with_memory_no_initial"))
        self.assertNotIn("initial_plan_memory", value["prompt_context"])
        self.assertIn("Long memory:", render_user(value))

    def test_noisy_context_is_deterministic_and_audited(self) -> None:
        clean = _sample("ongoing", "ongoing", context_variant="with_memory_with_initial")
        kwargs = {
            "context_variant": "with_memory_with_initial_noisy",
            "sample_id": clean["sample_id"],
            "episode_key": clean["provenance"]["episode_key"],
            "output_profile_id": clean["output_profile_id"],
        }
        left, left_meta = noisy_context(clean["prompt_context"], **kwargs)
        right, right_meta = noisy_context(clean["prompt_context"], **kwargs)
        self.assertEqual((left, left_meta), (right, right_meta))
        self.assertNotEqual(left, clean["prompt_context"])
        noisy = _sample("ongoing", "ongoing", context_variant="with_memory_with_initial_noisy")
        noisy["prompt_context"] = left
        noisy["provenance"]["context_noise"] = left_meta
        validate_sample(noisy)

    def test_clean_context_rejects_noise_metadata(self) -> None:
        sample = _sample("ongoing", "ongoing")
        sample["provenance"]["context_noise"] = {
            "changed": True,
            "seed": 1,
            "operations": [{"kind": "bad"}],
        }
        with self.assertRaises(V5ValidationError):
            validate_sample(sample)


if __name__ == "__main__":
    unittest.main()
