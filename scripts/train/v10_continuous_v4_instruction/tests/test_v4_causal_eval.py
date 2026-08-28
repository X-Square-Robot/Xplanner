from __future__ import annotations

import copy
import unittest

from ..infer_three_tasks_v4 import _has_complete_root_json
from ..causal_eval_v4 import (
    _invalid_output_memory_update,
    _plan_memory,
    _prediction_state,
    _strict_continuous_prompt,
    _strict_initial_prompt,
)


def _sample() -> dict:
    return {
        "task_instruction": "Put the objects into the matching containers",
        "source_id": "collection",
        "profile": "L3L0",
        "unit_type": "segment",
        "long_memory": ["GT completed segment must not leak"],
        "short_memory": [{"caption": "GT current segment must not leak", "progress_percent": 42}],
        "images": [
            {"view": "head", "frame": 0, "relative_frame": 0, "video": "/tmp/head.mp4"},
        ],
        "target": {
            "task_progress_percent": 42,
            "predictions": [{"l0": {"caption": "GT target must not leak"}}],
        },
    }


class StrictCausalPromptTest(unittest.TestCase):
    def test_json_root_stop_accepts_only_one_complete_object(self) -> None:
        self.assertTrue(_has_complete_root_json('{"a":1}'))
        self.assertTrue(_has_complete_root_json(' {"a":{"b":2}}\n'))
        self.assertFalse(_has_complete_root_json('{"a":1'))
        self.assertFalse(_has_complete_root_json('{"a":1}"'))
        self.assertFalse(_has_complete_root_json('{"a":1}{"b":2}'))
        self.assertFalse(_has_complete_root_json('[1,2]'))

    def test_strict_prompt_uses_only_explicit_causal_memory(self) -> None:
        sample = _sample()
        prompt = _strict_continuous_prompt(
            sample,
            long_memory=["model completed subtask"],
            short_memory=[{"caption": "model current subtask", "progress_percent": 21}],
            plan=None,
        )
        self.assertIn("Put the objects into the matching containers", prompt)
        self.assertIn("model completed subtask", prompt)
        self.assertIn("model current subtask", prompt)
        self.assertNotIn("GT completed segment", prompt)
        self.assertNotIn("GT current segment", prompt)
        self.assertNotIn("GT target", prompt)
        self.assertNotIn("Profile:", prompt)
        self.assertNotIn("L3L0", prompt)
        self.assertNotIn("segment scale", prompt.lower())
        self.assertIn('"subtask":{"level":"L2"', prompt)
        self.assertIn('"action":{"level":"L1"', prompt)
        self.assertIn('"l0":{"level":"L0","source":"segment"', prompt)

    def test_strict_prompt_is_invariant_to_gt_metadata(self) -> None:
        first = _sample()
        second = copy.deepcopy(first)
        second["profile"] = "full"
        second["unit_type"] = "subtask"
        second["long_memory"] = ["different GT memory"]
        second["short_memory"] = []
        second["target"] = {"completely": "different"}
        arguments = {
            "long_memory": ["model memory"],
            "short_memory": [{"caption": "model state", "progress_percent": 7}],
            "plan": None,
        }
        self.assertEqual(
            _strict_continuous_prompt(first, **arguments),
            _strict_continuous_prompt(second, **arguments),
        )

    def test_initial_prompt_has_no_memory_or_hidden_scale(self) -> None:
        prompt = _strict_initial_prompt(_sample())
        self.assertNotIn("long_memory", prompt)
        self.assertNotIn("short_memory", prompt)
        self.assertNotIn("Profile:", prompt)
        self.assertNotIn("L3L0", prompt)
        self.assertIn('"initial_plan"', prompt)
        self.assertIn('"actions"', prompt)
        self.assertIn('"segments"', prompt)

    def test_prediction_state_uses_fixed_l2_prediction_one(self) -> None:
        prediction = {
            "predictions": [
                {"subtask": {"caption": "current L2", "progress_percent": 31}},
                {"subtask": {"caption": "next L2", "progress_percent": 0}},
            ]
        }
        short, observation = _prediction_state(prediction, field="subtask")
        self.assertEqual(short, [{"caption": "current L2", "progress_percent": 31}])
        self.assertEqual(observation.caption, "current L2")
        self.assertEqual(observation.next_caption, "next L2")

    def test_plan_memory_contains_only_model_plan(self) -> None:
        plan = {
            "initial_plan": [
                {"index": 1, "subtask": {"caption": "first model L2"}},
                {"index": 2, "subtask": {"caption": "second model L2"}},
            ]
        }
        memory = _plan_memory(plan, ["first model L2"])
        self.assertEqual(memory["completed_l2"], ["first model L2"])
        self.assertEqual(memory["remaining_l2"], ["second model L2"])

    def test_invalid_output_holds_last_valid_memory(self) -> None:
        class Bank:
            long_memory = ["completed model subtask"]

        update = _invalid_output_memory_update(Bank())
        self.assertEqual(update["reason"], "invalid_output_hold_last_valid")
        self.assertEqual(update["long_memory"], ["completed model subtask"])
        self.assertFalse(update["transitioned"])


if __name__ == "__main__":
    unittest.main()
