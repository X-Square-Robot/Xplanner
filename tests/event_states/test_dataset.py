from __future__ import annotations

import unittest

from x_planner.data.event_states.dataset import dialogues_from_sample
from x_planner.data.event_states.schema import SCHEMA_VERSION, output_profile_id


class DatasetV5Test(unittest.TestCase):
    def test_dialogue_carries_exact_mask_spans(self) -> None:
        output_spec = {
            "prediction1_units": ["action"],
            "prediction2_units": [],
            "plan_units": ["action"],
        }
        sample = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": "sample-1",
            "base_sample_id": "base-1",
            "source": "fixture",
            "category": "ongoing",
            "memory_variant": "no_memory",
            "output_spec": output_spec,
            "output_profile_id": output_profile_id(output_spec),
            "task_instruction": "Place the green block in the tray",
            "images": [{"video": "/tmp/head.mp4", "frame": 10}],
            "prompt_context": {},
            "target": {
                "task_progress_percent": 0,
                "predictions": [
                    {
                        "index": 1,
                        "role": "current",
                        "action": {"available": True, "caption": "Move toward the block", "progress_percent": 10},
                    },
                    {
                        "index": 2,
                        "role": "next",
                    },
                ],
                "execution_decision": "Continue",
                "decision_detail": None,
            },
            "supervision": {"loss_mask_paths": ["/task_progress_percent", "/predictions/0/action"]},
            "provenance": {},
        }
        dialogues = dialogues_from_sample(sample)
        self.assertEqual([turn["role"] for turn in dialogues], ["user", "assistant"])
        spans = dialogues[1]["loss_mask_char_spans"]
        self.assertEqual([span[2] for span in spans], ["/task_progress_percent", "/predictions/0/action"])
        self.assertEqual(dialogues[0]["text"].count("<image>"), 1)


if __name__ == "__main__":
    unittest.main()
