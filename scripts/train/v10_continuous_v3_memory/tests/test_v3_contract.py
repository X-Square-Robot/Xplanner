from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ...v10_continuous.models import CanonicalEpisode, TemporalUnit
from ..build_json_v3 import _build_shard, _candidate_v1_keys, _check_episode_contract
from ..build_snapshot_v3 import (
    _materialize_task,
    _materialize_task_parallel,
    _materialize_terminal_zero_copy_parallel,
)
from ..dataset_v3 import (
    allocate_mix_counts,
    auto_near_640_dimensions,
    exact_mix_budget,
    resize_policy_identifier,
)
from ..checkpoint_v3 import _read_parent_metadata, branch_selected, complete_generation
from ..memory_v3 import MemoryCodecV3, StreamingMemoryV3
from ..merge_lists_v3 import _merge_shard_part
from ..metrics_v3 import evaluate_jsonl, terminal_metrics
from ..prompt_v3 import render_continuous_user, render_initial_plan_user
from ..sampling_v3 import anchor_specs, continuous_samples, initial_plan_sample
from ..schema_v3 import (
    TERMINAL_CAPTION,
    active_field,
    validate_initial_plan,
    validate_short_memory,
)
from ..train_smoke_report_v3 import summarize


def _unit(level: str, index: int, start: int, end: int) -> TemporalUnit:
    return TemporalUnit(
        unit_id=f"{level}-{index}",
        level=level,
        caption=f"perform {level.lower()} unit {index}",
        start_frame=start,
        end_frame=end,
        source="segment" if level == "L0" else None,
    )


def _episode(source: str = "collection") -> CanonicalEpisode:
    return CanonicalEpisode(
        source=source,
        episode_key="demo/episode",
        episode_name="episode",
        split="train",
        num_frames=56,
        task_caption="assemble the demonstration object",
        profile="full",
        unit_type="segment",
        levels={
            "L2": (_unit("L2", 0, 0, 45), _unit("L2", 1, 45, 56)),
            "L1": (
                _unit("L1", 0, 0, 20), _unit("L1", 1, 20, 45),
                _unit("L1", 2, 45, 50), _unit("L1", 3, 50, 56),
            ),
            "L0": (
                _unit("L0", 0, 0, 20), _unit("L0", 1, 20, 45),
                _unit("L0", 2, 45, 50), _unit("L0", 3, 50, 56),
            ),
        },
        videos={"head": "/data/head.mp4", "left_wrist": "/data/left.mp4"},
    )


class MemoryV3ContractTest(unittest.TestCase):
    def test_missing_parent_metadata_is_explicitly_reconstructed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metadata, status = _read_parent_metadata(
                Path(temporary) / "missing-v10-checkpoint-meta.json"
            )
            self.assertEqual(metadata, {})
            self.assertEqual(status["status"], "missing")

    def test_resize_policy_uses_canonical_snapshot_key(self) -> None:
        self.assertEqual(
            resize_policy_identifier({"policy_id": "auto_near_640_no_upscale_v1"}),
            "auto_near_640_no_upscale_v1",
        )
        with self.assertRaises(ValueError):
            resize_policy_identifier({"id": "legacy-wrong-key"})

    def test_parallel_snapshot_matches_sequential_bytes_and_terminal_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = continuous_samples(
                _episode(), source_id="collection",
                global_episode_key="collection:demo/episode",
            )
            shard = root / "continuous.jsonl"
            with shard.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            refs = root / "continuous.list"
            terminal_refs = root / "terminal.list"
            with refs.open("w") as all_handle, terminal_refs.open("w") as terminal_handle:
                for line_number, row in enumerate(rows, 1):
                    ref = {
                        "sample_key": row["sample_key"],
                        "shard_path": str(shard),
                        "line_number": line_number,
                        "global_episode_key": row["global_episode_key"],
                        "split": row["split"],
                        "profile": row["profile"],
                        "task_type": "terminal" if row["is_terminal_window"] else "continuous",
                    }
                    all_handle.write(json.dumps(ref) + "\n")
                    if row["is_terminal_window"]:
                        terminal_handle.write(json.dumps(ref) + "\n")
            sequential = root / "sequential"
            parallel = root / "parallel"
            sequential_result = _materialize_task(
                refs, sequential, expected_task="continuous"
            )
            parallel_result = _materialize_task_parallel(
                refs, parallel, expected_task="continuous", workers=2
            )
            self.assertEqual(sequential_result["samples"], parallel_result["samples"])
            for name in ("data.jsonl", "data.index", "episodes.jsonl"):
                self.assertEqual((sequential / name).read_bytes(), (parallel / name).read_bytes())
            terminal_result = _materialize_terminal_zero_copy_parallel(
                terminal_refs, parallel, root / "terminal", workers=2
            )
            self.assertEqual(
                terminal_result["samples"],
                sum(row["is_terminal_window"] for row in rows),
            )
            self.assertEqual(
                os.stat(parallel / "data.jsonl").st_ino,
                os.stat(root / "terminal" / "data.jsonl").st_ino,
            )

    def test_parallel_merge_part_keeps_zero_copy_terminal_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = root / "shard-00000"
            shard.mkdir()
            continuous = {
                "sample_key": "continuous-1", "global_episode_key": "source:episode",
                "split": "train", "profile": "L3L0",
            }
            initial = {
                "sample_key": "initial-1", "global_episode_key": "source:episode",
                "split": "train", "profile": "L3L0",
            }
            (shard / "continuous.jsonl").write_text(json.dumps(continuous) + "\n")
            (shard / "initial_plan.jsonl").write_text(json.dumps(initial) + "\n")
            terminal = {
                "sample_key": "continuous-1", "shard_path": str(shard / "continuous.jsonl"),
                "line_number": 1, "global_episode_key": "source:episode",
                "split": "train", "profile": "L3L0", "task_type": "terminal",
            }
            (shard / "terminal_refs.jsonl").write_text(json.dumps(terminal) + "\n")
            (shard / "initial_plan_oversize.jsonl").write_text("")
            result = _merge_shard_part({
                "order": 0, "shard": str(shard), "part_root": str(root / "part"),
            })
            self.assertEqual(result["counts"], {
                "continuous": 1, "initial_plan": 1, "terminal": 1,
            })
            terminal_out = json.loads((root / "part" / "terminal.list").read_text())
            self.assertEqual(terminal_out, terminal)
    def test_failed_shard_never_receives_success_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / "index.jsonl"
            index.write_text(
                json.dumps({
                    "kind": "invalid",
                    "global_episode_key": "collection:invalid",
                    "contract": {},
                }) + "\n"
            )
            shard = root / "shard"
            result = _build_shard({
                "shard_id": 0,
                "shard_root": str(shard),
                "index_path": str(index),
                "config": {
                    "anchor_stride_frames": 20,
                    "history_offsets_frames": [-20, -10, 0],
                    "initial_plan": {},
                    "resize": {"policy_id": "test"},
                    "terminal_caption": TERMINAL_CAPTION,
                },
            })
            self.assertEqual(result["failure_count"], 1)
            self.assertEqual(result["schema_version"], "memory_v3_shard_failed_v1")
            self.assertFalse((shard / "_SUCCESS").exists())

    def test_v1_legacy_source_is_normalized_from_formal_contract(self) -> None:
        episode = _episode(source="v2v3umi")
        normalized = _check_episode_contract(episode, {
            "kind": "v1_episode_shard",
            "contract": {
                "source_id": "open_action",
                "profile": episode.profile,
                "views": list(episode.videos),
                "split": episode.split,
            },
        })
        self.assertEqual(normalized.source, "open_action")
        self.assertEqual(normalized.metadata["legacy_source_id"], "v2v3umi")

    def test_v1_open_action_key_reuses_v2_legacy_root(self) -> None:
        row = {
            "source": "v2v3umi",
            "episode_key": (
                "v2v3umi/cpfs/zbl-cpfs-new/open_data/"
                "Open_Action_datasets_as_mp4/10Kh-demo/topic/episode"
            ),
        }
        self.assertIn(
            "open_action:10Kh-demo/topic/episode",
            _candidate_v1_keys(row),
        )

    def test_stride_history_short_memory_and_terminal(self) -> None:
        episode = _episode()
        self.assertEqual(anchor_specs(episode), ((0, False), (20, False), (40, False), (45, True)))
        rows = continuous_samples(
            episode, source_id="collection", global_episode_key="collection:demo/episode"
        )
        self.assertEqual([row["anchor_frame"] for row in rows], [0, 20, 40, 45])
        self.assertEqual({image["frame"] for image in rows[0]["images"]}, {0})
        self.assertEqual(rows[0]["short_memory"], [])
        for previous, current in zip(rows, rows[1:]):
            field = active_field(episode.profile)
            expected = previous["target"]["predictions"][0][field]
            self.assertEqual(current["short_memory"], [{
                "caption": expected["caption"],
                "progress_percent": expected["progress_percent"],
            }])
        terminal = rows[-1]
        self.assertTrue(terminal["is_terminal_window"])
        self.assertTrue(terminal["forced_terminal_anchor"])
        second = terminal["target"]["predictions"][1]
        for field in ("subtask", "action", "l0"):
            self.assertEqual(second[field]["caption"], TERMINAL_CAPTION)
            self.assertEqual(second[field]["progress_percent"], 0)

    def test_prompt_is_frame_only_and_source_rate_is_exact(self) -> None:
        row = continuous_samples(
            _episode(), source_id="collection", global_episode_key="collection:demo/episode"
        )[1]
        prompt = render_continuous_user(row)
        self.assertIn("[frame_offset=-20][view=head]", prompt)
        self.assertIn("[frame_offset=-10][view=head]", prompt)
        self.assertIn("[frame_offset=0][view=head]", prompt)
        self.assertNotIn("seconds ago", prompt)
        self.assertNotIn("20 Hz", prompt)
        zhengwei = dict(row, source_id="zhengwei")
        self.assertIn("Source frame rate: 20 Hz.", render_continuous_user(zhengwei))
        legacy = dict(row, source_id="", images=[
            {**image, "video": "/mnt/cpfs/zbl-cpfs-new/x2robot_data/zhengwei/demo.mp4"}
            for image in row["images"]
        ])
        self.assertIn("Source frame rate: 20 Hz.", render_continuous_user(legacy))
        self.assertNotIn("20 Hz", render_continuous_user(dict(legacy, source_id="collection")))

    def test_unlabelled_unit_gaps_skip_grid_points_without_fabricating_gt(self) -> None:
        episode = CanonicalEpisode(
            source="collection", episode_key="demo/gapped", episode_name="gapped",
            split="train", num_frames=70,
            task_caption="perform a gapped demonstration", profile="L3L0",
            unit_type="segment",
            levels={"L0": (_unit("L0", 0, 0, 25), _unit("L0", 1, 45, 70))},
            videos={"head": "/data/head.mp4"},
        )
        self.assertEqual(anchor_specs(episode), ((0, False), (20, False), (60, False)))
        rows = continuous_samples(
            episode, source_id="collection", global_episode_key="collection:demo/gapped"
        )
        self.assertEqual([row["anchor_frame"] for row in rows], [0, 20, 60])
        self.assertEqual([row["anchor_grid_index"] for row in rows], [0, 1, 3])
        self.assertTrue(rows[-1]["is_terminal_window"])

    def test_lower_level_gaps_are_skipped_and_terminal_is_forced_to_valid_frame(self) -> None:
        episode = CanonicalEpisode(
            source="collection", episode_key="demo/lower-gap", episode_name="lower-gap",
            split="train", num_frames=60, task_caption="perform the complete task",
            profile="L3L2L0", unit_type="subtask",
            levels={
                "L2": (_unit("L2", 0, 0, 40), _unit("L2", 1, 40, 60)),
                "L0": (_unit("L0", 0, 0, 30), _unit("L0", 1, 48, 58)),
            },
            videos={"head": "/data/head.mp4"},
        )
        self.assertEqual(anchor_specs(episode), ((0, False), (20, False), (48, True)))
        rows = continuous_samples(
            episode, source_id="collection", global_episode_key="collection:demo/lower-gap"
        )
        self.assertEqual([row["anchor_frame"] for row in rows], [0, 20, 48])
        self.assertTrue(rows[-1]["forced_terminal_anchor"])
        self.assertTrue(rows[-1]["is_terminal_window"])

    def test_long_memory_contains_only_deduplicated_completed_units(self) -> None:
        units = (
            TemporalUnit("L0-0", "L0", "repeat the same motion", 0, 20, "segment"),
            TemporalUnit("L0-1", "L0", "repeat the same motion", 20, 40, "segment"),
            TemporalUnit("L0-2", "L0", "finish with a distinct motion", 40, 60, "segment"),
        )
        episode = CanonicalEpisode(
            source="collection", episode_key="demo/repeated", episode_name="repeated",
            split="train", num_frames=60, task_caption="perform repeated motions",
            profile="L3L0", unit_type="segment", levels={"L0": units},
            videos={"head": "/data/head.mp4"},
        )
        rows = continuous_samples(
            episode, source_id="collection", global_episode_key="collection:demo/repeated"
        )
        self.assertEqual(rows[0]["long_memory"], [])
        self.assertEqual(rows[1]["long_memory"], ["repeat the same motion"])
        self.assertEqual(rows[2]["long_memory"], ["repeat the same motion"])

    def test_exact_checkpoint_branch_and_quoted_smoke_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "checkpoint-7"
            source.mkdir(parents=True)
            (source / "trainer_state.json").write_text('{"global_step":7}\n')
            (source / "v10_checkpoint_meta.json").write_text(
                '{"schema_version":"v10_checkpoint_meta_v1"}\n'
            )
            for name in ("optimizer.pt", "scheduler.pt", "model.safetensors", "rng_state_0.pth"):
                (source / name).write_bytes(name.encode())
            (source / "x2_sampler_state.json").write_text('{"consumed":123}\n')
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "manifest.json").write_text('{"content_digest":"demo"}\n')
            data_config = root / "data.yml"
            data_config.write_text("dataset: {}\n")
            result = branch_selected(source, root / "target", snapshot, data_config)
            branch = Path(result["branch_checkpoint"])
            self.assertFalse((branch / "x2_sampler_state.json").exists())
            self.assertEqual(
                os.stat(source / "optimizer.pt").st_ino,
                os.stat(branch / "optimizer.pt").st_ino,
            )
            run_state = root / "current_generation.json"
            run_state.write_text(json.dumps({
                "schema_version": "memory_v3_training_generation_v1",
                "current_train_dir": str(root / "target"),
                "snapshot": str(snapshot),
                "global_step": 0,
            }) + "\n")
            completed = complete_generation(
                root / "target", run_state, root / "current_run.json"
            )
            self.assertEqual(completed["global_step"], 7)
            self.assertEqual(
                json.loads(run_state.read_text())["current_checkpoint"], str(branch)
            )
            log = root / "launch.log"
            log.write_text(
                "[v10-first-batch] {}\n"
                "{'loss': '1.25', 'grad_norm': '2.5'}\n"
                "{'train_runtime': '3.0'}\n"
            )
            monitor = root / "gpu.csv"
            monitor.write_text("1024,50\n")
            report = summarize(
                log, monitor, root / "report.json", task="continuous",
                resize_mode="B_auto_near_640", required_steps=1, exit_code=0,
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["losses"], [1.25])

    def test_first_v3_branch_resets_only_scheduler_stage(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source" / "checkpoint-9"
            source.mkdir(parents=True)
            (source / "trainer_state.json").write_text('{"global_step":9}\n')
            (source / "v10_checkpoint_meta.json").write_text(
                '{"schema_version":"v10_checkpoint_meta_v1"}\n'
            )
            (source / "optimizer.pt").write_bytes(b"optimizer-moments")
            (source / "model.safetensors").write_bytes(b"model")
            (source / "rng_state.pth").write_bytes(b"rng")
            scheduler = {
                "base_lrs": [3e-6, 3e-6], "last_epoch": 9, "_step_count": 10,
                "_is_initial": False, "_get_lr_called_within_step": False,
                "_last_lr": [1e-9, 1e-9], "lr_lambdas": [{}, {}],
            }
            torch.save(scheduler, source / "scheduler.pt")
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "manifest.json").write_text('{"content_digest":"demo"}\n')
            data_config = root / "data.yml"
            data_config.write_text("dataset: {}\n")
            result = branch_selected(
                source, root / "target", snapshot, data_config,
                reset_scheduler_stage=True,
            )
            branch = Path(result["branch_checkpoint"])
            self.assertTrue(result["scheduler_stage_reset"])
            self.assertFalse(result["scheduler_preserved"])
            self.assertEqual(
                os.stat(source / "optimizer.pt").st_ino,
                os.stat(branch / "optimizer.pt").st_ino,
            )
            self.assertNotEqual(
                os.stat(source / "scheduler.pt").st_ino,
                os.stat(branch / "scheduler.pt").st_ino,
            )
            self.assertEqual(torch.load(source / "scheduler.pt")["last_epoch"], 9)
            reset = torch.load(branch / "scheduler.pt")
            self.assertEqual(reset["last_epoch"], 0)
            self.assertEqual(reset["_step_count"], 1)
            self.assertEqual(reset["_last_lr"], [0.0, 0.0])

    def test_initial_plan_is_complete_memory_free_and_progress_free(self) -> None:
        episode = _episode()
        continuous = continuous_samples(
            episode, source_id="collection", global_episode_key="collection:demo/episode"
        )
        sample = initial_plan_sample(
            episode,
            source_id="collection",
            global_episode_key="collection:demo/episode",
            images=continuous[0]["images"],
        )
        validate_initial_plan(sample["target"], "full", expected_top_level_units=2)
        encoded = json.dumps(sample["target"])
        self.assertNotIn("progress_percent", encoded)
        self.assertNotIn(TERMINAL_CAPTION, encoded)
        prompt = render_initial_plan_user(sample)
        self.assertNotIn("Long Memory:", prompt)
        self.assertNotIn("Short Memory:", prompt)
        self.assertEqual(sample["anchor_frame"], 0)

    def test_legacy_short_memory_and_streaming_reset(self) -> None:
        self.assertEqual(
            validate_short_memory(["place the cup into the box"]),
            ({"caption": "place the cup into the box", "progress_percent": 100},),
        )
        self.assertEqual(MemoryCodecV3().render_short([]), "[none]")
        row = continuous_samples(
            _episode(), source_id="collection", global_episode_key="collection:demo/episode"
        )[0]
        memory = StreamingMemoryV3()
        memory.reset("one")
        memory.observe_model_target("one", row["target"], row["profile"])
        self.assertEqual(len(memory.prompt_memory("one")[1]), 1)
        self.assertEqual(memory.prompt_memory("two"), ((), ()))

    def test_terminal_metrics_report_guardrails(self) -> None:
        rows = continuous_samples(
            _episode(), source_id="collection", global_episode_key="collection:demo/episode"
        )
        values = terminal_metrics([
            (rows[-1]["target"], rows[-1]["target"], "full"),
            (rows[-1]["target"], rows[0]["target"], "full"),
            (rows[0]["target"], rows[-1]["target"], "full"),
        ])
        self.assertEqual(values["terminal_recall"], 0.5)
        self.assertEqual(values["premature_terminal_rate"], 1.0)
        self.assertEqual(values["missed_terminal_rate"], 0.5)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_text("\n".join([
                json.dumps({
                    "profile": "full", "prediction": rows[-1]["target"],
                    "target": rows[-1]["target"],
                }),
                json.dumps({
                    "profile": "full", "assistant_json": "not-json",
                    "target": rows[-1]["target"],
                }),
            ]) + "\n", encoding="utf-8")
            report = evaluate_jsonl(path)
        self.assertEqual(report["evaluated_rows"], 2)
        self.assertEqual(report["invalid_prediction_json"], 1)
        self.assertEqual(report["terminal_recall"], 0.5)
        self.assertEqual(report["missed_terminal_rate"], 0.5)
        self.assertIn("premature_terminal_rate", report)

    def test_resize_and_mix_contracts(self) -> None:
        self.assertEqual(auto_near_640_dimensions(1080, 1920), (352, 640))
        self.assertEqual(auto_near_640_dimensions(480, 640), (480, 640))
        h, w = auto_near_640_dimensions(720, 1280)
        self.assertLessEqual(h, 720)
        self.assertLessEqual(w, 1280)
        self.assertEqual(h % 32, 0)
        self.assertEqual(w % 32, 0)
        budget = exact_mix_budget(
            {"continuous": 700, "initial_plan": 150, "terminal": 150},
            {"continuous": 0.70, "initial_plan": 0.15, "terminal": 0.15},
        )
        self.assertEqual(budget, 1000)
        self.assertEqual(
            allocate_mix_counts(100, {
                "continuous": 0.70, "initial_plan": 0.15, "terminal": 0.15,
            }),
            {"continuous": 70, "initial_plan": 15, "terminal": 15},
        )


if __name__ == "__main__":
    unittest.main()
