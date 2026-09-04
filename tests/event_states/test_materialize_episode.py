from __future__ import annotations

import unittest

from x_planner.data.event_states.materialize_episode import materialize_episode
from x_planner.data.event_states.prompt import render_user


def _interval(start: int, end: int, caption: str) -> dict:
    return {"start_frame": start, "end_frame": end, "caption": caption}


class EpisodeMaterializeV53Test(unittest.TestCase):
    def test_profiles_contexts_and_exact_end(self) -> None:
        actions = [
            _interval(index * 20, (index + 1) * 20, f"Perform action number {index + 1}")
            for index in range(5)
        ]
        segments = [
            _interval(index * 20, (index + 1) * 20, f"Execute motion segment number {index + 1}")
            for index in range(5)
        ]
        samples, missing = materialize_episode(
            episode_key="episode-one",
            source="baseline_v2v3umi",
            source_group="zhengwei",
            task_instruction="Complete the ordered robot manipulation task",
            actions=actions,
            segments=segments,
            videos={"head": "/tmp/faceImg.mp4"},
            total_frames=120,
            profiles=("action_only", "segment_only", "action_segment_joint"),
            provenance={
                "fixture": True,
                "task_instruction_source": "media_task_instruction_json.episode.instruction",
                "task_instruction_source_path": "/fixture/instruction.json",
            },
        )
        self.assertTrue(samples)
        self.assertTrue(missing)  # first no-initial memory has nothing causal to corrupt
        by_profile = {}
        for sample in samples:
            by_profile.setdefault(sample["provenance"]["label_profile"], []).append(sample)
        self.assertEqual(set(by_profile), {
            "action_only", "segment_only", "action_segment_joint"
        })

        action = by_profile["action_only"]
        segment = by_profile["segment_only"]
        joint = by_profile["action_segment_joint"]
        self.assertEqual(
            next(row for row in action if row["category"] == "ongoing")["output_spec"]["prediction1_units"],
            ["action"],
        )
        self.assertEqual(
            next(row for row in segment if row["category"] == "ongoing")["output_spec"]["prediction1_units"],
            ["segment"],
        )
        self.assertEqual(
            next(row for row in joint if row["category"] == "ongoing")["output_spec"]["prediction1_units"],
            ["action", "segment"],
        )
        self.assertFalse(any(row["category"] == "initial_plan" for row in segment))

        end_rows = [row for row in samples if row["category"] == "end"]
        self.assertTrue(end_rows)
        for row in end_rows:
            self.assertEqual(row["images"][0]["frame"], 119)
            self.assertEqual(row["target"]["task_progress_percent"], 100)
            self.assertEqual(row["target"]["execution_decision"], "End")
            self.assertTrue(row["provenance"]["video_end_exact"])

        noisy = [row for row in samples if row["context_variant"].endswith("_noisy")]
        self.assertTrue(noisy)
        self.assertTrue(all(row["provenance"]["context_noise"]["changed"] for row in noisy))

    def test_execution_prompt_is_not_category_conditioned(self) -> None:
        actions = [
            _interval(index * 10, (index + 1) * 10, f"Move object in action {index + 1}")
            for index in range(4)
        ]
        samples, _ = materialize_episode(
            episode_key="episode-two",
            source="robodojo",
            source_group="robodojo",
            task_instruction="Move all objects into the container",
            actions=actions,
            segments=(),
            videos=["/tmp/faceImg.mp4"],
            total_frames=50,
            profiles=("action_only",),
            provenance={
                "task_instruction_source": "media_instruction_json.episode.instruction",
                "task_instruction_source_path": "/fixture/instruction.json",
            },
        )
        ongoing = next(
            row for row in samples
            if row["category"] == "ongoing"
            and row["context_variant"] == "no_memory_no_initial"
            and row["output_spec"]["prediction2_units"] == []
        )
        end = next(
            row for row in samples
            if row["category"] == "end" and row["context_variant"] == "no_memory_no_initial"
        )
        self.assertEqual(render_user(ongoing).rsplit("\n", 1)[-1], render_user(end).rsplit("\n", 1)[-1])


if __name__ == "__main__":
    unittest.main()
