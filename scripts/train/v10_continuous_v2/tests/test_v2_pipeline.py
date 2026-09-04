from __future__ import annotations

import json
import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.train.v10_continuous.adapters import AdaptedEpisode
from scripts.train.v10_continuous.models import CanonicalEpisode, TemporalUnit
from scripts.train.v10_continuous.validate_schema import validate_snapshot
from scripts.train.v10_continuous_v2.artifact_ledger import register_artifacts
from scripts.train.v10_continuous_v2.audit_samples_v2 import main as audit_main
from scripts.train.v10_continuous_v2.build_dataset_manifests import build_manifests
from scripts.train.v10_continuous_v2.build_snapshot_v2 import build_snapshot
from scripts.train.v10_continuous_v2.common.atomic import iter_jsonl
from scripts.train.v10_continuous_v2.common.hashing import sampling_config_hash
from scripts.train.v10_continuous_v2.merge_catalogs_v2 import (
    _completion_state,
    iter_legacy_samples,
    merge_catalogs,
    resolve_legacy_catalogs,
)
from scripts.train.v10_continuous_v2.sampling_v2 import build_samples_v2, validate_sampling_config
from scripts.train.v10_continuous_v2.scan_v10_v2 import _default_run_id, _inventory_for_command
from scripts.train.v10_continuous_v2.scan_status_v2 import collect_status
from scripts.train.v10_continuous_v2.scanner_v2 import (
    _replace_episode_samples,
    _run_shard,
    run_inventory,
)
from scripts.train.v10_continuous_v2.shard_state import (
    build_inventory,
    inventory_matches,
    iter_plan,
    resolve_discovery_inputs,
    version_root,
)
from scripts.train.v10_continuous_v2.source_discovery import discover_episodes
from scripts.train.v10_continuous_v2.validate_episode import _resolve_l3_caption, process_episode
from scripts.train.v10_continuous_v2.validate_views import resolve_view_candidates


def _metadata(path: Path, frames: int = 40) -> None:
    path.write_text(json.dumps({"name": path.stem, "total": frames, "data": []}), encoding="utf-8")


def _hierarchy_episode(root: Path, relative: str, name: str = "episode") -> Path:
    directory = root / relative
    directory.mkdir(parents=True)
    (directory / f"{name}_hierarchy.json").write_text(json.dumps({
        "subtasks": [{
            "caption_en": "complete the task", "start_frame": 0, "end_frame": 40,
            "actions": [
                {"caption_en": "first action", "start_frame": 0, "end_frame": 20, "segment_details": []},
                {"caption_en": "second action", "start_frame": 20, "end_frame": 40, "segment_details": []},
                {"caption_en": "third action", "start_frame": 30, "end_frame": 40, "segment_details": []},
            ],
        }],
    }), encoding="utf-8")
    (directory / "instruction.json").write_text(json.dumps({name: {"task_caption": "complete the task"}}), encoding="utf-8")
    _metadata(directory / f"{name}.json")
    (directory / "faceImg.mp4").write_bytes(b"not-a-real-video")
    return directory


def _config(root_a: Path, root_b: Path | None = None) -> dict:
    sources = [{"source_id": "collection", "kind": "hierarchy_root", "roots": [str(root_a)]}]
    if root_b is not None:
        sources.append({"source_id": "zhengwei", "kind": "hierarchy_root", "roots": [str(root_b)]})
    return {
        "version": 2,
        "sources": sources,
        "sampling": {
            "anchor_quantiles": [0.25, 0.5, 0.75], "anchor_stride": 10,
            "max_anchors_per_episode": None, "visual_stride": 10,
            "visual_timesteps": 3, "max_camera_views": 3, "max_visual_inputs": 9,
        },
    }


def _fake_success(item, **kwargs):
    del kwargs
    target = {
        "task": {"level": "L3", "caption": "complete the task", "progress_percent": 50},
        "predictions": [{
            "index": 1,
            "subtask": {
                "level": "L2", "caption": "first action", "progress_percent": 50,
            },
        }],
    }
    sample = {
        "sample_key": f"sample-{item.global_episode_key}",
        "global_episode_key": item.global_episode_key,
        "source_id": item.source_id,
        "episode_key": item.episode_key,
        "dataset_name": item.dataset_name,
        "profile": "L3L2",
        "unit_level": "L2",
        "views": ["head"],
        "sample_id": f"v10-{item.global_episode_key}",
        "split": "train",
        "unit_type": "subtask",
        "unit_index": 0,
        "current_frame": 10,
        "task_caption": "complete the task",
        "long_memory": [],
        "images": [{
            "view": "head", "video": f"{item.job.episode_dir}/faceImg.mp4",
            "frame": 10, "relative_frame": 0,
        }],
        "anchor_index": 10,
        "target": target,
        "label": target,
        "input_paths": [item.job.episode_dir],
        "media_realpath": str(Path(item.job.episode_dir).resolve()),
        "metadata": {"sampling_config_hash": "test"},
    }
    return {
        "status": "success", "run_id": "smoke", "source_id": item.source_id,
        "episode_key": item.episode_key, "global_episode_key": item.global_episode_key,
        "dataset_name": item.dataset_name, "profile": "L3L2", "unit_level": "L2",
        "views": ["head"], "input_paths": [item.job.episode_dir],
        "media_realpath": str(Path(item.job.episode_dir).resolve()),
        "sample_count": 1, "samples": [sample], "attempts": 1,
    }


class DiscoveryTests(unittest.TestCase):
    def test_auto_uses_fast_only_when_all_requested_sources_are_covered(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            collection = base / "collection"
            zhengwei = base / "zhengwei"
            _hierarchy_episode(collection, "dataset/topic/episode-a", "episode-a")
            _hierarchy_episode(zhengwei, "dataset/topic/episode-b", "episode-b")
            config = _config(collection, zhengwei)
            source_inventory = build_inventory(
                config, base / "source-run", num_shards=2,
                source_filter={"collection"}, max_episodes=0,
                run_id="source", ledger_path=None,
            )
            config["discovery"] = {
                "default_mode": "auto",
                "fast_inventories": {"collection": source_inventory.path},
            }

            mode, inputs = resolve_discovery_inputs(
                config, source_filter={"collection"}, requested_mode=None,
            )
            self.assertEqual(mode, "fast")
            self.assertEqual(inputs[0][1].inventory_hash, source_inventory.inventory_hash)
            self.assertEqual(
                resolve_discovery_inputs(
                    config, source_filter=None, requested_mode=None,
                ),
                ("root", ()),
            )
            with self.assertRaisesRegex(ValueError, "no immutable inventory.*zhengwei"):
                resolve_discovery_inputs(
                    config, source_filter=None, requested_mode="fast",
                )

    def test_fast_inventory_preserves_rows_and_never_modifies_input_list(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data = base / "data"
            _hierarchy_episode(data, "dataset/topic/episode-a", "episode-a")
            _hierarchy_episode(data, "dataset/topic/episode-b", "episode-b")
            config = _config(data)
            source_inventory = build_inventory(
                config, base / "source-run", num_shards=2,
                source_filter={"collection"}, max_episodes=0,
                run_id="source", ledger_path=None,
            )
            protected = [Path(source_inventory.path), *(
                Path(plan.path) for plan in source_inventory.shards
            )]
            before = {path: path.read_bytes() for path in protected}
            fast_config = json.loads(json.dumps(config))
            fast_config["inventory_shard_key"] = "topic"
            fast_config["discovery"] = {
                "default_mode": "fast",
                "fast_inventories": {"collection": source_inventory.path},
            }
            mode, inputs = resolve_discovery_inputs(
                fast_config, source_filter={"collection"}, requested_mode=None,
            )
            target_inventory = build_inventory(
                fast_config, base / "target-run", num_shards=3,
                source_filter={"collection"}, max_episodes=0,
                run_id="target", ledger_path=None,
                discovery_mode=mode, input_inventories=inputs,
            )

            source_rows = {
                item.global_episode_key: item.to_dict()
                for plan in source_inventory.shards for item in iter_plan(plan)
            }
            target_rows = {
                item.global_episode_key: item.to_dict()
                for plan in target_inventory.shards for item in iter_plan(plan)
            }
            self.assertEqual(target_rows, source_rows)
            self.assertEqual(target_inventory.discovery_mode, "fast")
            self.assertEqual(
                target_inventory.input_inventory_hashes,
                {"collection": source_inventory.inventory_hash},
            )
            self.assertTrue(inventory_matches(
                target_inventory, fast_config, num_shards=3,
                source_filter={"collection"}, max_episodes=0,
                discovery_mode="fast",
                input_inventory_hashes={"collection": source_inventory.inventory_hash},
            ))
            self.assertFalse(inventory_matches(
                target_inventory, fast_config, num_shards=3,
                source_filter={"collection"}, max_episodes=0,
                discovery_mode="root",
            ))
            self.assertEqual({path: path.read_bytes() for path in protected}, before)

    def test_source_filter_skips_unselected_annotation_mirror_walk(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data = base / "data"
            _hierarchy_episode(data, "dataset/topic/episode", "episode")
            config = _config(data)
            config["annotation_root"] = str(base / "missing-annotations")
            config["sources"].append({
                "source_id": "annotated_extra", "kind": "annotation_mirror",
                "dir_pattern": ".*", "media_roots": [str(data)],
            })
            items = list(discover_episodes(config, source_filter={"collection"}))
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].source_id, "collection")

    def test_topic_sharding_colocates_episode_annotation_reads(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data = base / "data"
            _hierarchy_episode(data, "dataset/topic/episode-a", "episode-a")
            _hierarchy_episode(data, "dataset/topic/episode-b", "episode-b")
            config = _config(data)
            config["inventory_shard_key"] = "topic"
            inventory = build_inventory(
                config, base / "run", num_shards=8, source_filter=None,
                max_episodes=0, run_id="topic-shards", ledger_path=None,
            )
            occupied = [plan.shard_id for plan in inventory.shards if plan.episode_count]
            self.assertEqual(len(occupied), 1)
            self.assertEqual(inventory.total_episodes, 2)

    def test_fixed_depth_includes_empty_directory_and_excludes_deeper_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "zhengwei"
            episode = root / "10000" / "task" / "episode"
            episode.mkdir(parents=True)
            (episode / "deeper").mkdir()
            config = {
                "annotation_root": str(base / "annotations"),
                "l3_roots": [str(base / "l3")],
                "sources": [{
                    "source_id": "zhengwei", "kind": "media_sweep",
                    "roots": [str(root)], "annotation_relative_strip": 0,
                    "episode_depth": 3,
                }],
            }
            items = list(discover_episodes(config))
            self.assertEqual([item.episode_key for item in items], ["10000/task/episode"])
            self.assertEqual(items[0].media_realpath, str(episode.resolve()))
            self.assertIn("no_metadata_json", items[0].discovery_warnings)

    def test_annotation_mirror_uses_first_matching_source_and_realpath(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            annotations = base / "annotations"
            media = base / "media"
            topic = Path("10000") / "task"
            (annotations / topic).mkdir(parents=True)
            (annotations / topic / "instruction.json").write_text(
                json.dumps({"episode-a": {"task_caption": "do the task"}}),
                encoding="utf-8",
            )
            episode = media / topic / "episode-a"
            episode.mkdir(parents=True)
            _metadata(episode / "episode-a.json")
            config = {
                "annotation_root": str(annotations),
                "topic_max_depth": 4,
                "sources": [
                    {"source_id": "specific", "kind": "annotation_mirror",
                     "dir_pattern": "^[0-9]+$", "media_roots": [str(media)]},
                    {"source_id": "catch_all", "kind": "annotation_mirror",
                     "dir_pattern": ".*", "media_roots": [str(media)]},
                ],
            }
            items = list(discover_episodes(config))
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].source_id, "specific")
            self.assertEqual(items[0].media_realpath, str(episode.resolve()))
            self.assertEqual(
                list(discover_episodes(config, source_filter={"catch_all"})), []
            )

    def test_source_scoped_identity_and_root_complete_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / "collection"
            second = root / "zhengwei"
            _hierarchy_episode(first, "dataset/topic/same", "same")
            _hierarchy_episode(second, "dataset/topic/same", "same")
            items = list(discover_episodes(_config(first, second)))
            self.assertEqual(len(items), 2)
            self.assertEqual({item.global_episode_key for item in items}, {
                "collection:dataset/topic/same", "zhengwei:dataset/topic/same",
            })

    def test_flat_root_finds_episode_not_in_datalist(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "open"
            episode = root / "dataset" / "topic" / "trajectory"
            episode.mkdir(parents=True)
            _metadata(episode / "trajectory.json")
            (episode / "move1Img.mp4").write_bytes(b"video")
            config = {"sources": [{
                "source_id": "open_action", "kind": "flat_root", "roots": [str(root)],
                "instruction_templates": ["{topic_path}/instruction.json"],
            }]}
            item = next(discover_episodes(config))
            self.assertEqual(item.global_episode_key, "open_action:dataset/topic/trajectory")
            self.assertIn("not_in_datalist", item.discovery_warnings)


class SamplingTests(unittest.TestCase):
    def test_frozen_visual_sampling_rejects_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "visual_stride is frozen"):
            validate_sampling_config({"visual_stride": 11})

    def test_view_priority_resolves_nonfatal_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            episode = Path(temp)
            primary = episode / "faceImg.mp4"
            secondary = episode / "move1Img.mp4"
            primary.write_bytes(b"primary")
            secondary.write_bytes(b"secondary")
            selected, warnings = resolve_view_candidates(
                str(episode), {}, {
                    "views": {
                        "aliases": {"faceImg": "head", "move1Img": "head"},
                        "order": ["head"], "stem_priority": ["faceImg", "move1Img"],
                        "duplicate_is_fatal": False, "case_sensitive": True,
                    },
                },
            )
            self.assertEqual(selected, {"head": str(primary.resolve())})
            self.assertTrue(any(value.startswith("duplicate_view:head") for value in warnings))

    def test_stride_is_superset_and_history_never_uses_future(self) -> None:
        episode = CanonicalEpisode(
            source="collection", episode_key="dataset/episode", episode_name="episode",
            split="train", num_frames=40, task_caption="complete the task",
            profile="L3L2", unit_type="subtask",
            levels={"L2": (
                TemporalUnit("L2-0", "L2", "first task", 0, 20),
                TemporalUnit("L2-1", "L2", "second task", 20, 40),
            )},
            videos={"head": "/tmp/head.mp4"},
        )
        sampling = {
            "anchor_quantiles": [0.25, 0.5, 0.75], "anchor_stride": 10,
            "max_anchors_per_episode": None,
        }
        rows = build_samples_v2(
            episode, source_id="collection", global_episode_key="collection:dataset/episode",
            dataset_name="dataset", sampling=sampling,
            sampling_hash=sampling_config_hash(sampling), input_paths=("/tmp/head.mp4",),
        )
        anchors = {row["anchor_index"] for row in rows}
        self.assertTrue({0, 10, 20, 30}.issubset(anchors))
        self.assertEqual(len({row["sample_key"] for row in rows}), len(rows))
        for row in rows:
            self.assertTrue(all(image["frame"] <= row["anchor_index"] for image in row["images"]))


class PipelineTests(unittest.TestCase):
    def test_episode_sample_replacement_is_indexed_and_scoped(self) -> None:
        samples = {
            "old-a": {"sample_key": "old-a", "global_episode_key": "episode-a"},
            "old-b": {"sample_key": "old-b", "global_episode_key": "episode-b"},
        }
        index = {"episode-a": {"old-a"}, "episode-b": {"old-b"}}
        replacement = [{
            "sample_key": "new-a", "global_episode_key": "episode-a",
        }]
        _replace_episode_samples(samples, index, "episode-a", replacement)
        self.assertEqual(set(samples), {"new-a", "old-b"})
        self.assertEqual(index, {"episode-a": {"new-a"}, "episode-b": {"old-b"}})

    def test_single_resume_exhausts_retry_budget_until_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data = base / "data"
            _hierarchy_episode(data, "dataset/topic/episode", "episode")
            config = _config(data)
            run_root = base / "out" / "runs" / "resume"
            inventory = build_inventory(
                config, run_root, num_shards=1, source_filter=None, max_episodes=0,
                run_id="resume", ledger_path=None,
            )
            item = next(discover_episodes(config))

            def retryable_failure(*_args, **_kwargs):
                return {
                    "status": "failed", "run_id": "resume", "source_id": item.source_id,
                    "episode_key": item.episode_key,
                    "global_episode_key": item.global_episode_key,
                    "dataset_name": item.dataset_name, "media_realpath": item.media_realpath,
                    "input_paths": [item.job.episode_dir], "retryable": True,
                    "error": {"error_type": "video_decode_error"}, "attempts": 1,
                }

            settings = {"sampling": config["sampling"], "max_attempts": 3}
            with mock.patch(
                "scripts.train.v10_continuous_v2.scanner_v2.process_episode",
                side_effect=retryable_failure,
            ):
                run_inventory(
                    inventory, run_root, run_id="resume", stage="scan", mode="scan",
                    settings=settings, view_config={"views": {}}, num_workers=1,
                    shard_id=None, ledger_path=None, fail_fast=True,
                )
            with mock.patch(
                "scripts.train.v10_continuous_v2.scanner_v2.process_episode",
                side_effect=[retryable_failure(), _fake_success(item)],
            ) as process:
                run_inventory(
                    inventory, run_root, run_id="resume", stage="scan", mode="resume",
                    settings=settings, view_config={"views": {}}, num_workers=1,
                    shard_id=None, ledger_path=None, fail_fast=True,
                )
            self.assertEqual(process.call_count, 2)
            attempt = sorted(version_root(run_root, inventory.shards[0]).glob("attempt-*"))[-1]
            success = next(iter_jsonl(str(attempt / "episodes_success.jsonl")))
            self.assertEqual(success["attempts"], 3)
            self.assertEqual(json.loads((attempt / ".done").read_text())["retryable_remaining"], 0)

    def test_resume_marks_exhausted_retry_as_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data = base / "data"
            _hierarchy_episode(data, "dataset/topic/episode", "episode")
            config = _config(data)
            run_root = base / "out" / "runs" / "exhausted"
            inventory = build_inventory(
                config, run_root, num_shards=1, source_filter=None, max_episodes=0,
                run_id="exhausted", ledger_path=None,
            )
            item = next(discover_episodes(config))

            def retryable_failure(*_args, **_kwargs):
                return {
                    "status": "failed", "run_id": "exhausted", "source_id": item.source_id,
                    "episode_key": item.episode_key,
                    "global_episode_key": item.global_episode_key,
                    "dataset_name": item.dataset_name, "media_realpath": item.media_realpath,
                    "input_paths": [item.job.episode_dir], "retryable": True,
                    "error": {"error_type": "video_decode_error", "retryable": True},
                }

            settings = {"sampling": config["sampling"], "max_attempts": 3}
            common = dict(
                inventory=inventory, run_root=run_root, run_id="exhausted",
                stage="scan", settings=settings, view_config={"views": {}},
                num_workers=1, shard_id=None, ledger_path=None, fail_fast=True,
            )
            with mock.patch(
                "scripts.train.v10_continuous_v2.scanner_v2.process_episode",
                side_effect=retryable_failure,
            ):
                run_inventory(mode="scan", **common)
                resumed = run_inventory(mode="resume", **common)

            attempt = Path(resumed["results"][0]["attempt"])
            row = next(iter_jsonl(str(attempt / "episodes_failed.jsonl")))
            self.assertEqual(row["attempts"], 3)
            self.assertFalse(row["retryable"])
            self.assertTrue(row["retry_exhausted"])
            self.assertTrue(row["error"]["retryable"])
            self.assertEqual(resumed["results"][0]["statistics"]["retryable_failures"], 0)
            self.assertEqual(json.loads((attempt / ".done").read_text())["retryable_remaining"], 0)

    def test_complete_recovery_journal_materializes_durable_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data = base / "data"
            _hierarchy_episode(data, "dataset/topic/episode", "episode")
            config = _config(data)
            run_root = base / "out" / "runs" / "journal"
            inventory = build_inventory(
                config, run_root, num_shards=1, source_filter=None, max_episodes=0,
                run_id="journal", ledger_path=None,
            )
            plan = inventory.shards[0]
            item = next(discover_episodes(config))
            result = _fake_success(item)
            root = version_root(run_root, plan)
            root.mkdir(parents=True)
            (root / "journal-recovery.jsonl").write_text(json.dumps({
                "global_episode_key": item.global_episode_key,
                "result": result,
            }) + "\n", encoding="utf-8")
            output = _run_shard({
                "plan": plan.to_dict(), "run_root": str(run_root), "run_id": "journal",
                "stage": "scan", "mode": "scan", "settings": {"sampling": config["sampling"]},
                "view_config": {"views": {}}, "sampling_hash": "sampling",
                "completion_hash": "recovery", "ledger_path": None,
            })
            self.assertFalse(output["skipped"])
            self.assertTrue((Path(output["attempt"]) / ".done").is_file())
            self.assertFalse((root / "journal-recovery.jsonl").exists())

    def test_new_run_can_reference_matching_immutable_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data = base / "data"
            _hierarchy_episode(data, "dataset/topic/episode", "episode")
            config = _config(data)
            source_root = base / "runs" / "source-run"
            inventory = build_inventory(
                config, source_root, num_shards=2, source_filter={"collection"},
                max_episodes=0, run_id="source-run", ledger_path=None,
            )
            target_root = base / "runs" / "target-run"
            target_root.mkdir(parents=True)
            args = SimpleNamespace(
                source=["collection"], max_episodes=0,
                inventory_from_run_id="source-run", cc_ledger=None,
            )
            borrowed = _inventory_for_command(
                "scan", config=config, run_root=target_root, run_id="target-run",
                args=args, settings={"num_shards": 2},
            )
            self.assertEqual(borrowed.inventory_hash, inventory.inventory_hash)
            self.assertEqual(
                json.loads((target_root / "inventory_reference.json").read_text())["source_run_id"],
                "source-run",
            )
            self.assertEqual(
                json.loads((target_root / "current_inventory.json").read_text())["path"],
                inventory.path,
            )

    def test_default_run_id_ignores_execution_and_merge_only_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            config = _config(Path(temp) / "data")
            config["runtime"] = {"num_shards": 8, "num_workers": 2, "checkpoint_episodes": 10}
            args = SimpleNamespace(source=["collection"], max_episodes=0, num_shards=None)
            expected = _default_run_id(config, args)
            changed = json.loads(json.dumps(config))
            changed["runtime"]["num_workers"] = 99
            changed["runtime"]["checkpoint_episodes"] = 1
            changed["merge"] = {"prior_catalogs": ["/different/baseline"]}
            self.assertEqual(_default_run_id(changed, args), expected)
            self.assertNotEqual(
                _default_run_id(config, SimpleNamespace(
                    source=["collection"], max_episodes=0, num_shards=16,
                )),
                expected,
            )

    def test_legacy_training_snapshot_uses_flat_sequential_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            media_root = base / "media"
            episode = media_root / "dataset" / "topic" / "episode"
            episode.mkdir(parents=True)
            video = episode / "faceImg.mp4"
            video.write_bytes(b"video")
            snapshot = base / "snapshot"
            (snapshot / "train").mkdir(parents=True)
            (snapshot / "validation").mkdir()
            (snapshot / "manifest.json").write_text(json.dumps({
                "schema_version": "v10_training_snapshot_v1", "complete": True,
            }), encoding="utf-8")
            sample = {
                "sample_id": "sample", "episode_key": "legacy", "split": "train",
                "profile": "L3L2", "unit_type": "subtask", "unit_index": 0,
                "current_frame": 10, "task_caption": "complete the task",
                "long_memory": [], "images": [{
                    "view": "head", "video": str(video), "frame": 10, "relative_frame": 0,
                }],
                "target": {"task": {"level": "L3", "caption": "complete", "progress_percent": 1},
                           "predictions": []},
            }
            (snapshot / "train" / "data.jsonl").write_text(
                json.dumps({"data_id": "one", "v10_sample": sample}) + "\n", encoding="utf-8"
            )
            (snapshot / "validation" / "data.jsonl").write_text("", encoding="utf-8")
            with mock.patch(
                "scripts.train.v10_continuous_v2.merge_catalogs_v2.ROOTS",
                (("collection", media_root.resolve()),),
            ):
                self.assertEqual(resolve_legacy_catalogs((snapshot,)), [snapshot.resolve()])
                rows = list(iter_legacy_samples(snapshot, sampling_hash="hash"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["origin"], "v1")
            self.assertEqual(rows[0]["media_realpath"], str(episode.resolve()))
            self.assertEqual(rows[0]["global_episode_key"], "collection:dataset/topic/episode")

    def test_fixed_seed_audit_cli_compares_existing_successes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data = base / "data"
            for index in range(2):
                _hierarchy_episode(data, f"dataset/topic/episode-{index}", f"episode-{index}")
            config = _config(data)
            run_root = base / "out" / "runs" / "audit"
            inventory = build_inventory(
                config, run_root, num_shards=2, source_filter=None, max_episodes=0,
                run_id="audit", ledger_path=None,
            )
            with mock.patch(
                "scripts.train.v10_continuous_v2.scanner_v2.process_episode",
                side_effect=_fake_success,
            ):
                run_inventory(
                    inventory, run_root, run_id="audit", stage="scan", mode="scan",
                    settings={"sampling": config["sampling"]}, view_config={"views": {}},
                    num_workers=1, shard_id=None, ledger_path=None, fail_fast=True,
                )
            config_path = base / "sources.yml"
            views_path = base / "views.yml"
            output = base / "audit-output"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            views_path.write_text(json.dumps({"views": {}}), encoding="utf-8")
            argv = [
                "audit_samples_v2", "--run-root", str(run_root),
                "--config", str(config_path), "--views-config", str(views_path),
                "--output-dir", str(output), "--inventory-samples", "2",
                "--successful-samples", "2",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch(
                "scripts.train.v10_continuous_v2.audit_samples_v2.process_episode",
                side_effect=_fake_success,
            ):
                audit_main()
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["uniform"]["status"], {"success": 2})
            self.assertEqual(summary["successful"]["sample_keys_match"], 2)

    def test_inventory_recovers_after_interrupted_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data = base / "data"
            for index in range(3):
                _hierarchy_episode(data, f"dataset/topic/episode-{index}", f"episode-{index}")
            config = _config(data)
            run_root = base / "out" / "runs" / "recover"
            original = list(discover_episodes(config))

            def interrupted(*_args, **_kwargs):
                yield original[0]
                raise RuntimeError("simulated interruption")

            with mock.patch(
                "scripts.train.v10_continuous_v2.shard_state.discover_episodes",
                side_effect=interrupted,
            ):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    build_inventory(
                        config, run_root, num_shards=4, source_filter=None, max_episodes=0,
                        run_id="recover", ledger_path=None,
                    )
            self.assertTrue((run_root / "inventory_build.json").is_file())
            inventory = build_inventory(
                config, run_root, num_shards=4, source_filter=None, max_episodes=0,
                run_id="recover", ledger_path=None,
            )
            self.assertEqual(inventory.total_episodes, 3)
            self.assertFalse((run_root / "inventory_build.json").exists())
            merge_only_change = json.loads(json.dumps(config))
            merge_only_change["merge"] = {"prior_catalogs": ["/different/baseline"]}
            self.assertTrue(inventory_matches(
                inventory, merge_only_change, num_shards=4,
                source_filter=None, max_episodes=0,
            ))
            discovery_change = json.loads(json.dumps(config))
            discovery_change["sources"][0]["roots"] = [str(base / "other-data")]
            self.assertFalse(inventory_matches(
                inventory, discovery_change, num_shards=4,
                source_filter=None, max_episodes=0,
            ))

    def test_status_reports_inventory_build_before_manifest_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run_root = Path(temp) / "run"
            run_root.mkdir()
            (run_root / "inventory_build.json").write_text(json.dumps({
                "status": "building_inventory", "processed": 123,
                "by_source": {"zhengwei": 123}, "updated_at": 1.0,
                "request": {"num_shards": 8},
            }), encoding="utf-8")
            report = collect_status(run_root)
            self.assertEqual(report["phase"], "building_inventory")
            self.assertEqual(report["discovered"], 123)
            self.assertEqual(report["terminal"], 0)
            self.assertEqual(report["shards"]["total"], 8)
            self.assertEqual(report["shards"]["with_episode_failures"], 0)

    def test_annotation_profile_failure_does_not_probe_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data = base / "data"
            _hierarchy_episode(data, "dataset/topic/episode", "episode")
            item = next(discover_episodes(_config(data)))
            adapted = AdaptedEpisode(
                task_caption="complete the task", num_frames=40, raw_levels={},
                video_candidates={}, annotation_sources=(),
                metadata={"adapter": "flat", "l3_candidates": {}},
            )
            with mock.patch(
                "scripts.train.v10_continuous_v2.validate_episode._adapt_or_classify",
                return_value=adapted,
            ), mock.patch(
                "scripts.train.v10_continuous_v2.validate_episode.validate_episode_views"
            ) as probe:
                result = process_episode(
                    item, run_id="test", shard_id=0, worker_id="test",
                    settings={"sampling": {}}, view_config={}, sampling_hash="test",
                )
            self.assertEqual(result["error"]["error_type"], "profile_unavailable")
            self.assertIn(
                "valid_counts=L2:0,L1:0,L0:0; no_valid_temporal_level",
                result["error"]["error_message"],
            )
            probe.assert_not_called()

    def test_l3_conflict_uses_task_supported_detailed_caption(self) -> None:
        selected, resolution = _resolve_l3_caption("Store the fabric", {
            "l3_candidates": {
                "task": "VR_Pour_the_kettle_water_into_the_pot",
                "instruction": "Store the fabric",
                "detailed_instruction": "Pour water from kettle into pot",
            },
        })
        self.assertEqual(selected, "Pour water from kettle into pot")
        self.assertEqual(resolution["selected_source"], "detailed_instruction")

    def test_l3_paraphrases_and_hypernyms_resolve_without_quarantine(self) -> None:
        cases = [
            ("Folding carton.", "Stand up the flat paper box, then fold its bottom.", "fold-item"),
            ("Stack the packaging boxes", "Fold the cardboard into a box.", "fold_box_new_arm"),
            ("Making lemonade", "Pick up the sliced lemon and add water.", "make_lemon-water"),
            ("Move the vegetable", "Pick up chopped scallions from the table.", "pick_green_onion"),
            (
                "Perform the action as described | blue square block | rope | spoon",
                "Randomly pick up an object and move it to another position.",
                "move-random-item",
            ),
        ]
        for instruction, detailed, task in cases:
            with self.subTest(instruction=instruction):
                selected, _resolution = _resolve_l3_caption(instruction, {"l3_candidates": {
                    "instruction": instruction,
                    "detailed_instruction": detailed,
                    "task": task,
                }})
                self.assertIn(selected, {instruction, detailed})

    def test_merge_refuses_incomplete_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data = base / "data"
            _hierarchy_episode(data, "dataset/topic/episode", "episode")
            config = _config(data)
            run_root = base / "out" / "runs" / "partial"
            build_inventory(
                config, run_root, num_shards=2, source_filter=None, max_episodes=0,
                run_id="partial", ledger_path=None,
            )
            with self.assertRaisesRegex(RuntimeError, "refusing incomplete"):
                merge_catalogs(
                    run_root, run_id="partial", sampling=config["sampling"], bucket_count=2,
                )

    def test_completion_state_rejects_newer_in_progress_attempt_and_missing_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data = base / "data"
            _hierarchy_episode(data, "dataset/topic/episode", "episode")
            config = _config(data)
            run_root = base / "out" / "runs" / "completion"
            inventory = build_inventory(
                config, run_root, num_shards=1, source_filter=None, max_episodes=0,
                run_id="completion", ledger_path=None,
            )
            with mock.patch(
                "scripts.train.v10_continuous_v2.scanner_v2.process_episode",
                side_effect=_fake_success,
            ):
                run_inventory(
                    inventory, run_root, run_id="completion", stage="scan", mode="scan",
                    settings={"sampling": config["sampling"]}, view_config={"views": {}},
                    num_workers=1, shard_id=None, ledger_path=None, fail_fast=True,
                )
            state = _completion_state(run_root)
            self.assertEqual(state["completed_shards"], 1)
            attempt = next((run_root / "shards" / "shard-00000").glob("*/attempt-0001"))
            marker = json.loads((attempt / ".done").read_text())
            marker["retryable_remaining"] = 1
            (attempt / ".done").write_text(json.dumps(marker), encoding="utf-8")
            self.assertEqual(_completion_state(run_root)["incomplete_shards"], [0])
            marker["retryable_remaining"] = 0
            marker.pop("completion_hash")
            (attempt / ".done").write_text(json.dumps(marker), encoding="utf-8")
            self.assertEqual(_completion_state(run_root)["incomplete_shards"], [0])
            marker["completion_hash"] = "restored"
            (attempt / ".done").write_text(json.dumps(marker), encoding="utf-8")
            (attempt.parent / "attempt-0002").mkdir()
            self.assertEqual(_completion_state(run_root)["incomplete_shards"], [0])
            with mock.patch(
                "scripts.train.v10_continuous_v2.scanner_v2.process_episode",
                side_effect=_fake_success,
            ):
                run_inventory(
                    inventory, run_root, run_id="completion", stage="scan", mode="scan",
                    settings={"sampling": config["sampling"]}, view_config={"views": {}},
                    num_workers=1, shard_id=None, ledger_path=None, fail_fast=True,
                )
            self.assertEqual(_completion_state(run_root)["completed_shards"], 1)
            self.assertTrue(any((attempt.parent / "aborted_attempts").iterdir()))

    def test_merge_combines_multiple_v2_run_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            roots = []
            for index in range(2):
                data = base / f"data-{index}"
                _hierarchy_episode(data, f"dataset/topic/episode-{index}", f"episode-{index}")
                config = _config(data)
                run_root = base / "out" / "runs" / f"run-{index}"
                inventory = build_inventory(
                    config, run_root, num_shards=2, source_filter=None, max_episodes=0,
                    run_id=f"run-{index}", ledger_path=None,
                )
                with mock.patch(
                    "scripts.train.v10_continuous_v2.scanner_v2.process_episode",
                    side_effect=_fake_success,
                ):
                    run_inventory(
                        inventory, run_root, run_id=f"run-{index}", stage="scan", mode="scan",
                        settings={"sampling": config["sampling"]}, view_config={"views": {}},
                        num_workers=1, shard_id=None, ledger_path=None, fail_fast=True,
                    )
                roots.append((run_root, config))
            merged = merge_catalogs(
                roots[0][0], run_id="aggregate", sampling=roots[0][1]["sampling"],
                input_run_roots=(roots[1][0],), bucket_count=4,
            )
            self.assertEqual(merged["samples_after_dedup"], 2)
            self.assertEqual(len(merged["input_catalogs"]), 2)

    def test_changed_execution_fingerprint_does_not_skip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data = base / "data"
            _hierarchy_episode(data, "dataset/topic/episode", "episode")
            config = _config(data)
            run_root = base / "out" / "runs" / "fingerprint"
            inventory = build_inventory(
                config, run_root, num_shards=2, source_filter=None, max_episodes=0,
                run_id="fingerprint", ledger_path=None,
            )
            common = dict(
                run_id="fingerprint", stage="scan", mode="scan",
                view_config={"views": {}}, num_workers=1, shard_id=None,
                ledger_path=None, fail_fast=True,
            )
            with mock.patch(
                "scripts.train.v10_continuous_v2.scanner_v2.process_episode",
                side_effect=_fake_success,
            ) as process:
                run_inventory(
                    inventory, run_root,
                    settings={"sampling": config["sampling"], "rule_version": "one"},
                    **common,
                )
                second = run_inventory(
                    inventory, run_root,
                    settings={"sampling": config["sampling"], "rule_version": "two"},
                    **common,
                )
            self.assertEqual(process.call_count, 2)
            self.assertFalse(second["results"][0]["skipped"])

    def test_resume_retries_retryable_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data = base / "data"
            _hierarchy_episode(data, "dataset/topic/episode", "episode")
            config = _config(data)
            run_root = base / "out" / "runs" / "retry"
            inventory = build_inventory(
                config, run_root, num_shards=2, source_filter=None, max_episodes=0,
                run_id="retry", ledger_path=None,
            )

            def failing(item, **kwargs):
                del kwargs
                return {
                    "status": "failed", "run_id": "retry", "source_id": item.source_id,
                    "episode_key": item.episode_key,
                    "global_episode_key": item.global_episode_key,
                    "dataset_name": item.dataset_name, "input_paths": [item.job.episode_dir],
                    "error": {"error_type": "video_decode_error", "input_paths": [item.job.episode_dir]},
                    "retryable": True, "attempts": 1,
                }

            common = dict(
                run_id="retry", stage="scan",
                settings={"sampling": config["sampling"], "max_attempts": 3},
                view_config={"views": {}}, num_workers=1, shard_id=None,
                ledger_path=None, fail_fast=True,
            )
            with mock.patch(
                "scripts.train.v10_continuous_v2.scanner_v2.process_episode",
                side_effect=failing,
            ):
                first = run_inventory(inventory, run_root, mode="scan", **common)
            with mock.patch(
                "scripts.train.v10_continuous_v2.scanner_v2.process_episode",
                side_effect=_fake_success,
            ):
                resumed = run_inventory(inventory, run_root, mode="resume", **common)
            self.assertEqual(first["episodes"], {"failed": 1})
            self.assertEqual(resumed["episodes"], {"success": 1})
            success_path = Path(resumed["results"][0]["attempt"]) / "episodes_success.jsonl"
            row = next(iter_jsonl(str(success_path)))
            self.assertEqual(row["attempts"], 2)

    def test_scan_skip_resume_publish_and_ledger_idempotency(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data = base / "data"
            _hierarchy_episode(data, "dataset/topic/episode", "episode")
            _hierarchy_episode(data, "dataset/topic/episode-two", "episode-two")
            config = _config(data)
            run_root = base / "out" / "runs" / "smoke"
            ledger = base / "ledger.md"
            inventory = build_inventory(
                config, run_root, num_shards=4, source_filter=None, max_episodes=0,
                run_id="smoke", ledger_path=str(ledger),
            )
            common = dict(
                run_id="smoke", stage="scan", mode="scan",
                settings={"sampling": config["sampling"], "max_attempts": 3},
                view_config={"views": {}}, num_workers=1, shard_id=None,
                ledger_path=str(ledger), fail_fast=True,
            )
            with mock.patch(
                "scripts.train.v10_continuous_v2.scanner_v2.process_episode",
                side_effect=_fake_success,
            ):
                first = run_inventory(inventory, run_root, **common)
                second = run_inventory(inventory, run_root, **common)
            self.assertEqual(first["episodes"], {"success": 2})
            self.assertTrue(second["results"][0]["skipped"])

            v2_catalog = Path(first["results"][0]["attempt"]) / "catalog.jsonl"
            v2_row = next(iter_jsonl(str(v2_catalog)))
            legacy_sample = {
                key: value for key, value in v2_row.items()
                if key in {
                    "sample_id", "episode_key", "split", "profile", "unit_type",
                    "unit_index", "current_frame", "task_caption", "long_memory",
                    "images", "target",
                }
            }
            legacy_sample["target"] = json.loads(json.dumps(legacy_sample["target"]))
            legacy_sample["target"]["predictions"][0]["subtask"]["caption"] = "legacy action"
            video = Path(legacy_sample["images"][0]["video"])
            legacy_shard = base / "legacy-shard.json"
            legacy_shard.write_text(json.dumps({
                "episode": {
                    "videos": {"head": str(video)},
                    "annotation_sources": [],
                    "profile": "L3L2", "unit_type": "subtask",
                },
                "samples": [legacy_sample],
            }), encoding="utf-8")
            legacy_catalog = base / "accepted.jsonl"
            legacy_catalog.write_text(
                json.dumps({"shard_path": str(legacy_shard)}) + "\n", encoding="utf-8"
            )
            with mock.patch(
                "scripts.train.v10_continuous_v2.merge_catalogs_v2.ROOTS",
                (("collection", data.resolve()),),
            ):
                merged = merge_catalogs(
                    run_root, run_id="smoke", sampling=config["sampling"],
                    legacy_catalogs=(legacy_catalog,), bucket_count=4, ledger_path=str(ledger),
                )
            self.assertEqual(merged["samples_after_dedup"], 2)
            self.assertEqual(merged["duplicate_sample_count"], 1)
            self.assertEqual(merged["conflict_count"], 1)
            manifests = build_manifests(run_root, run_id="smoke", ledger_path=str(ledger))
            self.assertEqual(manifests["sample_count"], 2)
            merged_stat = Path(merged["catalog"]).stat()
            all_stat = (Path(manifests["root"]) / "all.jsonl").stat()
            self.assertEqual((all_stat.st_dev, all_stat.st_ino), (merged_stat.st_dev, merged_stat.st_ino))
            conflict = next(iter_jsonl(str(Path(merged["root"]) / "conflicts.jsonl")))
            self.assertEqual(conflict["winner_origin"], "v2")
            self.assertIn("previous_task_caption", conflict)
            self.assertIn("incoming_content_fingerprint", conflict)
            config_path = base / "sources.yml"
            config_path.write_text("version: 2\n", encoding="utf-8")
            snapshot = build_snapshot(
                run_root, run_id="smoke", config_paths=(config_path,), ledger_path=str(ledger)
            )
            self.assertEqual(snapshot["sample_count"], 2)
            validation = validate_snapshot(Path(snapshot["root"]))
            self.assertTrue(validation["valid"])
            self.assertTrue((Path(snapshot["root"]) / "manifest.json").is_file())
            before = ledger.read_text(encoding="utf-8")
            register_artifacts(
                ledger, [snapshot["root"]], purpose="duplicate", source_id="all", run_id="smoke"
            )
            register_artifacts(
                ledger, [snapshot["root"]], purpose="duplicate", source_id="all", run_id="smoke"
            )
            after = ledger.read_text(encoding="utf-8")
            exact_row = f"| {Path(snapshot['root']).resolve()} | duplicate | all | smoke |"
            self.assertEqual(after.count(exact_row), 1)
            self.assertGreaterEqual(len(after), len(before))
            catalog_rows = list(iter_jsonl(merged["catalog"]))
            self.assertTrue(all(row["origin"] == "v2" for row in catalog_rows))


if __name__ == "__main__":
    unittest.main()
