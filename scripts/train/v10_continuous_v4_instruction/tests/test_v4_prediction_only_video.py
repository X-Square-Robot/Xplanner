from __future__ import annotations

import unittest

from PIL import Image

from ..build_full_episode_rollout_video_v4 import _draw_text_panel
from ..build_prediction_only_episode_video_v4 import (
    _intro_blocks,
    _memory_output_lines,
    _render_initial_fullscreen,
    _stream_blocks,
)


class PredictionOnlyVideoTest(unittest.TestCase):
    def _initial(self) -> dict:
        return {
            "prediction_schema_valid": True,
            "prediction": {
                "initial_plan": [
                    {
                        "index": 1,
                        "subtask": {"caption": "open the case", "actions": []},
                    }
                ]
            },
        }

    def _record(self) -> dict:
        return {
            "prediction_schema_valid": True,
            "predicted_no_next_same_scale_unit": False,
            "prediction": {
                "task_progress_percent": 25,
                "predictions": [
                    {
                        "index": 1,
                        "subtask": {
                            "caption": "open the case",
                            "progress_percent": 25,
                        },
                    },
                    {
                        "index": 2,
                        "subtask": {
                            "caption": "place the pen",
                            "progress_percent": 0,
                        },
                    },
                ],
            },
            "input_long_memory": [],
            "input_short_memory": [
                {"caption": "reach the case", "progress_percent": 50}
            ],
            "memory_update": {
                "transitioned": True,
                "reason": "transition_committed",
                "committed": "reach the case",
                "long_memory": ["reach the case"],
            },
            "output_short_memory_for_next_anchor": [
                {"caption": "open the case", "progress_percent": 25}
            ],
        }

    def test_display_blocks_never_include_ground_truth(self) -> None:
        titles = [title for title, _, _ in _intro_blocks(self._initial())]
        titles.extend(title for title, _, _ in _stream_blocks(self._record()))
        joined = " ".join(titles).lower()
        self.assertNotIn("ground truth", joined)
        self.assertNotIn("review only", joined)
        self.assertNotIn("— gt", joined)

    def test_memory_output_displays_commit_and_next_short_memory(self) -> None:
        lines = _memory_output_lines(self._record())
        joined = "\n".join(lines)
        self.assertIn("Committed to Long Memory: reach the case", joined)
        self.assertIn("Long Memory after update:", joined)
        self.assertIn("Short Memory for next anchor:", joined)
        self.assertIn("[25%] open the case", joined)

    def test_rejects_invalid_font_size_bounds(self) -> None:
        with self.assertRaisesRegex(ValueError, "font-size bounds"):
            _draw_text_panel(
                Image.new("RGB", (3840, 2160)),
                [],
                preferred_body_size=20,
                minimum_body_size=24,
            )

    def test_long_initial_plan_fits_one_4k_frame_at_readable_size(self) -> None:
        initial = self._initial()
        initial["prediction"]["initial_plan"] = [
            {
                "index": index,
                "subtask": {
                    "caption": f"complete distinct ordered subtask {index} using the visible objects",
                    "actions": [],
                },
            }
            for index in range(1, 48)
        ]
        image, body_size = _render_initial_fullscreen(
            instruction="complete the demonstrated household task",
            status="MODEL ONLY / INITIAL PLAN / STRICT CAUSAL",
            blocks=_intro_blocks(initial),
        )
        self.assertEqual(image.size, (3840, 2160))
        self.assertGreaterEqual(body_size, 24)

    def test_fullscreen_initial_plan_rejects_unreadable_overflow(self) -> None:
        initial = self._initial()
        initial["prediction"]["initial_plan"] = [
            {
                "index": index,
                "subtask": {"caption": "an intentionally long subtask", "actions": []},
            }
            for index in range(1, 500)
        ]
        with self.assertRaisesRegex(RuntimeError, "full-screen 4K"):
            _render_initial_fullscreen(
                instruction="complete the demonstrated household task",
                status="MODEL ONLY / INITIAL PLAN / STRICT CAUSAL",
                blocks=_intro_blocks(initial),
            )


if __name__ == "__main__":
    unittest.main()
