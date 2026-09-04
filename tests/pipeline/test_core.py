from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from x_planner.data.pipeline.adapters import (
    AnnotationValidationError,
    EpisodeJob,
    adapt_job,
)
from x_planner.data.pipeline.captions import (
    is_valid_english_caption,
    normalize_caption,
    select_l3,
)
from x_planner.data.pipeline.hierarchy import (
    EpisodeValidationError,
    build_samples,
    build_target,
    canonicalize_episode,
    frame_indices,
    progress_percent,
)
from x_planner.data.pipeline.memory import (
    MemoryAugmentor,
    MemoryBank,
    MemoryCodec,
    UnitObservation,
)
from x_planner.data.pipeline.models import TemporalUnit
from x_planner.data.pipeline.metrics import score_target_text
from x_planner.data.pipeline.prompt import sample_to_indexed_jsonl
from x_planner.data.pipeline.schema import (
    TargetValidationError,
    dumps_target,
    loads_target,
    validate_target,
)
from x_planner.data.pipeline.validation_runtime import (
    OracleGenerator,
    render_generation_prompt,
    run_validation,
)
from x_planner.data.pipeline.vision import (
    decode_refs_by_complete_view,
    load_quarantined_sample_ids,
)


def unit(level: str, index: int, start: int, end: int, *, source: str | None = None):
    return TemporalUnit(
        unit_id=f"{level}-{index}",
        level=level,
        caption=f"perform {level.lower()} unit {index}",
        start_frame=start,
        end_frame=end,
        source=source,
    )


def full_episode():
    levels = {
        "L2": (unit("L2", 0, 0, 30), unit("L2", 1, 30, 60)),
        "L1": tuple(unit("L1", i, i * 10, (i + 1) * 10) for i in range(6)),
        "L0": tuple(
            unit("L0", i, i * 5, (i + 1) * 5, source="human_segment")
            for i in range(12)
        ),
    }
    return canonicalize_episode(
        source="fixture",
        episode_key="fixture/topic/episode",
        episode_name="episode",
        split="train",
        num_frames=60,
        task_caption="organize the objects",
        raw_levels=levels,
        videos={
            "head": "/tmp/head.mp4",
            "left_wrist": "/tmp/left.mp4",
            "right_wrist": "/tmp/right.mp4",
            "side": "/tmp/side.mp4",
        },
    )


class CaptionTests(unittest.TestCase):
    def test_l3_precedence_and_english(self):
        annotation = {
            "task_caption": "整理桌面",
            "instruction": "Put the objects into the tray",
            "detailed_instruction": "Use both arms",
        }
        self.assertEqual(select_l3(annotation), "Put the objects into the tray")
        self.assertTrue(is_valid_english_caption("Put the cup down"))
        self.assertFalse(is_valid_english_caption("放下杯子"))

    def test_normalization(self):
        self.assertEqual(normalize_caption("  Pick   Up the Cup!!!  "), "pick up the cup")

    def test_flat_annotation_uses_local_fallback_for_empty_external_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = root / "episode_000001"
            episode.mkdir()
            external = root / "external.json"
            local = root / "instruction.json"
            metadata = episode / "episode_000001.json"
            external.write_text(json.dumps({"episode_000001": {}}), encoding="utf-8")
            local.write_text(json.dumps({
                "episode_000001": {"detailed_instruction": "Arrange the sofa cushions"}
            }), encoding="utf-8")
            metadata.write_text(json.dumps({"total": 30}), encoding="utf-8")
            job = EpisodeJob(
                source="fixture",
                kind="flat",
                episode_key="fixture/topic/episode_000001",
                episode_name="episode_000001",
                topic=str(root),
                episode_dir=str(episode),
                annotation_paths=(str(external), str(local)),
                metadata_path=str(metadata),
            )
            self.assertEqual(adapt_job(job).task_caption, "Arrange the sofa cushions")
            local_empty = root / "instruction_empty.json"
            local_empty.write_text(json.dumps({"episode_000001": {}}), encoding="utf-8")
            empty_job = EpisodeJob(
                source=job.source,
                kind=job.kind,
                episode_key=job.episode_key,
                episode_name=job.episode_name,
                topic=job.topic,
                episode_dir=job.episode_dir,
                annotation_paths=(str(external), str(local_empty)),
                metadata_path=job.metadata_path,
            )
            with self.assertRaises(AnnotationValidationError):
                adapt_job(empty_job)


class HierarchyTests(unittest.TestCase):
    def test_full_profile_target_and_samples(self):
        episode = full_episode()
        self.assertEqual(episode.profile, "full")
        self.assertEqual(tuple(episode.videos), ("head", "left_wrist", "right_wrist", "side"))
        target, index = build_target(episode, 15)
        self.assertEqual(index, 0)
        self.assertEqual(tuple(target["predictions"][0]), ("index", "subtask", "action", "l0"))
        self.assertEqual(target["predictions"][0]["action"]["caption"], "perform l1 unit 1")
        self.assertEqual(target["predictions"][0]["l0"]["caption"], "perform l0 unit 3")
        self.assertEqual(target["predictions"][1]["subtask"]["progress_percent"], 0)
        self.assertEqual(target["predictions"][1]["action"]["caption"], "perform l1 unit 3")
        self.assertEqual(target["predictions"][1]["l0"]["caption"], "perform l0 unit 6")
        samples = build_samples(episode)
        self.assertEqual(len(samples), 6)
        self.assertTrue(all(len(sample.images) <= 9 for sample in samples))
        self.assertEqual(samples[0].long_memory, ())
        self.assertEqual(samples[-1].long_memory, ("perform l2 unit 0",))
        self.assertEqual(tuple(image.view for image in samples[0].images[-3:]), (
            "head", "left_wrist", "right_wrist"
        ))

    def test_last_unit_has_one_prediction(self):
        target, index = build_target(full_episode(), 45)
        self.assertEqual(index, 1)
        self.assertEqual(len(target["predictions"]), 1)

    def test_profile_can_skip_invalid_l1(self):
        levels = {
            "L2": (unit("L2", 0, 0, 30),),
            "L1": (unit("L1", 0, 0, 2),),
            "L0": tuple(
                unit("L0", i, i * 10, (i + 1) * 10, source="segment")
                for i in range(3)
            ),
        }
        episode = canonicalize_episode(
            source="fixture",
            episode_key="fixture/l3l2l0",
            episode_name="l3l2l0",
            split="validation",
            num_frames=30,
            task_caption="sort the objects",
            raw_levels=levels,
            videos={"head": "/tmp/head.mp4"},
        )
        self.assertEqual(episode.profile, "L3L2L0")

    def test_overlap_invalidates_layer(self):
        with self.assertRaises(EpisodeValidationError):
            canonicalize_episode(
                source="fixture",
                episode_key="bad",
                episode_name="bad",
                split="train",
                num_frames=20,
                task_caption="perform a task",
                raw_levels={
                    "L1": (
                        unit("L1", 0, 0, 12),
                        unit("L1", 1, 10, 18),
                        unit("L1", 2, 18, 20),
                    )
                },
                videos={"head": "/tmp/head.mp4"},
            )

    def test_half_open_progress_and_history_only_frames(self):
        self.assertEqual(progress_percent(0, 0, 10), 0)
        self.assertEqual(progress_percent(5, 0, 10), 50)
        self.assertEqual(frame_indices(5), (5,))
        self.assertEqual(frame_indices(20), (0, 10, 20))


class SchemaPromptTests(unittest.TestCase):
    def test_schema_and_row_contract(self):
        sample = build_samples(full_episode())[0]
        encoded = dumps_target(sample.target, sample.profile)
        self.assertEqual(loads_target(encoded, sample.profile), sample.target)
        row = sample_to_indexed_jsonl(sample)
        self.assertEqual(len(row["image"]), row["text"][0]["text"].count("<image>"))
        self.assertIn("Actual camera views: head, left_wrist, right_wrist", row["text"][0]["text"])
        json.loads(row["text"][1]["text"])

    def test_schema_rejects_extra_and_future_progress(self):
        target, _ = build_target(full_episode(), 10)
        target["predictions"][0]["extra"] = 1
        with self.assertRaises(TargetValidationError):
            validate_target(target, "full")
        target, _ = build_target(full_episode(), 10)
        target["predictions"][1]["action"]["progress_percent"] = 1
        with self.assertRaises(TargetValidationError):
            validate_target(target, "full")


class MemoryTests(unittest.TestCase):
    def test_codec_visibility_and_short_tail(self):
        codec = MemoryCodec(short_memory_k=2, visible_long_memory_limit=3)
        long_memory = tuple(f"Unit {index}." for index in range(5))
        self.assertEqual(codec.short_from_long(long_memory), ("unit 3", "unit 4"))
        rendered = codec.render_long(long_memory)
        self.assertTrue(rendered.startswith("[earlier steps omitted]"))
        self.assertNotIn("unit 1", rendered)

    def test_noise_is_deterministic_and_short_comes_after_long(self):
        codec = MemoryCodec()
        augmentor = MemoryAugmentor(codec, seed=7, max_probability=1.0, ramp_steps=1)
        history = ("pick up the cup", "place the cup on the tray")
        first = augmentor.augment(history, sample_id="sample", global_step=10)
        second = augmentor.augment(history, sample_id="sample", global_step=10)
        self.assertEqual(first, second)
        self.assertEqual(first.short_memory, first.long_memory[-1:] if first.long_memory else ())
        self.assertNotIn("future unit", first.long_memory)

    def test_memory_bank_requires_stable_transition(self):
        bank = MemoryBank(done_threshold=80, stable_steps=2)
        bank.step("ep", UnitObservation("pick up cup", 85, "place cup"))
        high = bank.step("ep", UnitObservation("pick up cup", 95, "place cup"))
        self.assertEqual(high.long_memory, ())
        candidate = bank.step("ep", UnitObservation("place cup", 0, None))
        self.assertFalse(candidate.transitioned)
        committed = bank.step("ep", UnitObservation("place cup", 5, None))
        self.assertTrue(committed.transitioned)
        self.assertEqual(committed.committed, "pick up cup")
        self.assertEqual(committed.long_memory, ("pick up cup",))

    def test_prediction_two_alone_never_commits_and_episode_resets(self):
        bank = MemoryBank(done_threshold=80, stable_steps=2)
        bank.step("one", UnitObservation("first", 10, "second"))
        update = bank.step("one", UnitObservation("first", 20, "second"))
        self.assertEqual(update.long_memory, ())
        reset = bank.step("two", UnitObservation("other", 0, None))
        self.assertEqual(reset.long_memory, ())
        self.assertEqual(reset.episode_id, "two")

    def test_embedding_similarity_and_closed_vocabulary_mapping(self):
        codec = MemoryCodec(similarity_threshold=0.5)
        self.assertGreater(codec.similarity("pick up the cup", "pick up cup"), 0.5)
        self.assertTrue(codec.same("pick up the cup", "pick up cup"))
        self.assertFalse(codec.same("pick up the cup", "open the drawer"))
        self.assertEqual(
            codec.closest("pick up the cup", ("pick up cup", "open the drawer")),
            "pick up cup",
        )


class ValidationTests(unittest.TestCase):
    def test_generation_prompt_disables_open_thinking_block(self):
        class FakeProcessor:
            def __init__(self):
                self.kwargs = None

            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return "<|im_start|>assistant\n<think>\n\n</think>\n\n"

        processor = FakeProcessor()
        rendered = render_generation_prompt(
            processor,
            [{"role": "user", "content": "return JSON"}],
        )
        self.assertFalse(processor.kwargs["enable_thinking"])
        self.assertTrue(rendered.endswith("<think>\n\n</think>\n\n"))

    def test_metrics_and_resumable_oracle_rollout(self):
        samples = build_samples(full_episode())
        encoded = dumps_target(samples[0].target, samples[0].profile)
        score = score_target_text(encoded, samples[0].target, samples[0].profile)
        self.assertTrue(score.exact_match)
        self.assertEqual(score.score, 1.0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train").mkdir()
            with (root / "train" / "data.jsonl").open("w", encoding="utf-8") as handle:
                for sample in samples:
                    handle.write(json.dumps(sample_to_indexed_jsonl(sample)) + "\n")
            output = root / "rollout.jsonl"
            first = run_validation(
                mode="rollout",
                snapshot=root,
                split="train",
                output_path=output,
                generator=OracleGenerator(),
            )
            second = run_validation(
                mode="rollout",
                snapshot=root,
                split="train",
                output_path=output,
                generator=OracleGenerator(),
            )
            self.assertEqual(first["rollout_score"], 1.0)
            self.assertEqual(second["rollout_score"], 1.0)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), len(samples))


class ResilientVisionTests(unittest.TestCase):
    def test_quarantine_loader_accepts_records_and_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "quarantine.json"
            path.write_text(
                json.dumps({
                    "sample_ids": [
                        "v10-known-bad",
                        {"sample_id": "v10-second-bad", "reason": "decode"},
                    ]
                }),
                encoding="utf-8",
            )
            self.assertEqual(
                load_quarantined_sample_ids(path),
                frozenset({"v10-known-bad", "v10-second-bad"}),
            )

    def test_failed_view_is_removed_as_a_complete_temporal_group(self):
        refs = [
            {"video": "/tmp/head.mp4", "frame": 0, "view": "head"},
            {"video": "/tmp/left.mp4", "frame": 0, "view": "left_wrist"},
            {"video": "/tmp/head.mp4", "frame": 10, "view": "head"},
            {"video": "/tmp/left.mp4", "frame": 10, "view": "left_wrist"},
        ]

        def load_fn(view_refs, _jsonl_path):
            view_refs = list(view_refs)
            if view_refs[0]["view"] == "left_wrist":
                raise RuntimeError("synthetic decode failure")
            return [Image.new("RGB", (8, 8), color="white") for _ in view_refs]

        positions, images, failures = decode_refs_by_complete_view(
            refs, "/tmp/data.jsonl", load_fn
        )
        self.assertEqual(positions, [0, 2])
        self.assertEqual(len(images), 2)
        self.assertEqual(tuple(failures), ("left_wrist",))


if __name__ == "__main__":
    unittest.main()
