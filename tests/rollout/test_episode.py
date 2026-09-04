from __future__ import annotations

import json
import unittest

from x_planner.evaluation.rollout.rollout_episode import (
    loads_inference_continuous,
    predicted_short_memory,
)


class RolloutShortMemoryTest(unittest.TestCase):
    def _continuous_text(self, second_caption: str) -> str:
        return json.dumps({
            "task_progress_percent": 25,
            "predictions": [
                {
                    "index": 1,
                    "subtask": {
                        "level": "L2",
                        "caption": "press the button",
                        "progress_percent": 25,
                    },
                },
                {
                    "index": 2,
                    "subtask": {
                        "level": "L2",
                        "caption": second_caption,
                        "progress_percent": 0,
                    },
                },
            ],
        })

    def test_inference_parser_accepts_model_terminal_choice(self) -> None:
        _, predicted_terminal = loads_inference_continuous(
            self._continuous_text("the task is complete"),
            "L3L2",
            "Press the black button",
        )
        self.assertTrue(predicted_terminal)

    def test_inference_parser_accepts_model_nonterminal_choice(self) -> None:
        _, predicted_terminal = loads_inference_continuous(
            self._continuous_text("return the hand to the starting position"),
            "L3L2",
            "Press the black button",
        )
        self.assertFalse(predicted_terminal)

    def test_uses_prediction_one_at_active_scale(self) -> None:
        prediction = {
            "predictions": [
                {
                    "subtask": {"caption": "predicted L2", "progress_percent": 37},
                    "action": {"caption": "predicted L1", "progress_percent": 52},
                    "l0": {"caption": "predicted L0", "progress_percent": 81},
                },
                {
                    "subtask": {"caption": "future L2", "progress_percent": 0},
                    "action": {"caption": "future L1", "progress_percent": 0},
                    "l0": {"caption": "future L0", "progress_percent": 0},
                },
            ]
        }
        self.assertEqual(
            predicted_short_memory(prediction, "subtask"),
            [{"caption": "predicted L2", "progress_percent": 37}],
        )
        self.assertEqual(
            predicted_short_memory(prediction, "action"),
            [{"caption": "predicted L1", "progress_percent": 52}],
        )
        self.assertEqual(
            predicted_short_memory(prediction, "segment"),
            [{"caption": "predicted L0", "progress_percent": 81}],
        )

    def test_rejects_unknown_unit_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown rollout unit_type"):
            predicted_short_memory({"predictions": []}, "task")


if __name__ == "__main__":
    unittest.main()
