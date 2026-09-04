from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.train.v10_continuous_v5.baseline_fragment_reduce_v5 import (
    ACCELERATOR_SCHEMA_VERSION,
    _assert_pid_inactive,
    _iter_slice_samples,
    _phase_cache_key,
    _recover_cached_phase,
    _recover_initial_samples_hybrid,
    _recover_initial_samples_from_spool,
    materialize_fragment_shards,
)
from scripts.train.v10_continuous_v5.indexed_io_v5 import read_indexed_item
from scripts.train.v10_continuous_v5.materialize_v5 import _sample, materialize_dataset
from scripts.train.v10_continuous_v5.baseline_materialize_v5 import _task_name
from scripts.train.v10_continuous_v5.baseline_adapter import EpisodeActionPlanCollector
from scripts.train.v10_continuous_v5.baseline_materialize_v5 import (
    convert_initial_plan,
)
from scripts.train.v10_continuous_v5.parallel_v5 import plan_jsonl_chunks
from scripts.train.v10_continuous_v5.validate_v5 import validate_dataset


def _fixture_sample(sample_id: str, episode: str) -> dict:
    return _sample(
        sample_id=sample_id,
        base_sample_id=f"base-{sample_id}",
        source="baseline",
        category="initial_plan",
        memory_variant="no_memory",
        output_spec={
            "prediction1_units": [],
            "prediction2_units": [],
            "plan_units": ["action"],
        },
        task_instruction="Place the red block in the tray",
        images=[{"video": "/fixture/face.mp4", "frame": 0, "view": "face"}],
        prompt_context={},
        target={
            "initial_plan": [
                {"index": 1, "action": {"caption": "Grasp the red block"}}
            ]
        },
        loss_mask_paths=(),
        provenance={
            "episode_key": episode,
            "task_name": "place_red_block_fixture",
            "split": "train",
            "canonical_source": "pinned_complete_baseline",
            "memory_pair_eligible": False,
        },
    )


def _ongoing_sample(sample_id: str, episode: str, split: str) -> dict:
    return _sample(
        sample_id=sample_id,
        base_sample_id=f"base-{sample_id}",
        source="baseline",
        category="ongoing",
        memory_variant="no_memory",
        output_spec={
            "prediction1_units": ["action"],
            "prediction2_units": ["action"],
            "plan_units": ["action"],
        },
        task_instruction="Place the red block in the tray",
        images=[{"video": "/fixture/face.mp4", "frame": 1, "view": "face"}],
        prompt_context={},
        target={
            "task_progress_percent": 50,
            "predictions": [
                {
                    "index": 1,
                    "role": "current",
                    "action": {
                        "available": True,
                        "caption": "Grasp the red block",
                        "progress_percent": 50,
                    },
                },
                {
                    "index": 2,
                    "role": "next",
                    "action": {
                        "available": True,
                        "caption": "Place the red block in the tray",
                        "progress_percent": 0,
                    },
                },
            ],
            "execution_decision": "Continue",
            "decision_detail": None,
        },
        loss_mask_paths=("/execution_decision", "/decision_detail"),
        provenance={
            "episode_key": episode,
            "task_name": "place_red_block_fixture",
            "split": split,
            "canonical_source": "pinned_complete_baseline",
            "memory_pair_eligible": False,
        },
    )


def _write_fragment(path: Path, samples: list[dict]) -> None:
    payload = b"".join(
        json.dumps(sample, separators=(",", ":")).encode() + b"\n"
        for sample in samples
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(".meta.json").write_text(json.dumps({
        "schema_version": "v5_parallel_fragment_v1",
        "cache_key": f"fixture-{path.stem}",
        "phase": "pass2",
        "fragment": str(path),
        "fragment_sha256": digest,
        "output_count": len(samples),
        "read_physical_lines": len(samples),
        "adapter_statistics": {
            "read_rows": len(samples),
            "emitted_ongoing_rows": len(samples),
            "excluded_rows": {},
        },
    }), encoding="utf-8")


def _write_cached_phase(
    raw: Path,
    cache_root: Path,
    *,
    chunk_bytes: int,
) -> list[Path]:
    fragments: list[Path] = []
    for index, chunk in enumerate(plan_jsonl_chunks((raw,), chunk_bytes=chunk_bytes)):
        fragment = cache_root / "pass1" / f"pending-{index}.jsonl"
        fragment.parent.mkdir(parents=True, exist_ok=True)
        fragment.write_text(
            "".join(json.dumps({"row": row}) + "\n" for row in range(chunk.num_lines)),
            encoding="utf-8",
        )
        digest = hashlib.sha256(fragment.read_bytes()).hexdigest()
        key = _phase_cache_key(
            phase="pass1",
            path=raw.resolve(),
            start=chunk.start,
            end=chunk.end,
            first_line=chunk.first_line,
            num_lines=chunk.num_lines,
            plan_digest="none",
        )
        final = fragment.with_name(f"{index:06d}-{key[:16]}.jsonl")
        fragment.rename(final)
        final.with_suffix(".meta.json").write_text(json.dumps({
            "schema_version": "v5_parallel_fragment_v1",
            "cache_key": key,
            "phase": "pass1",
            "fragment": str(final),
            "fragment_sha256": digest,
            "output_count": chunk.num_lines,
            "read_physical_lines": chunk.num_lines,
            "adapter_statistics": {
                "read_rows": chunk.num_lines,
                "emitted_ongoing_rows": chunk.num_lines,
                "excluded_rows": {},
            },
            "reused": False,
        }), encoding="utf-8")
        fragments.append(final)
    return fragments


def _canonical_action_record(episode: str, first: str, second: str) -> dict:
    return {
        "schema_version": "v5_baseline_canonical_v1",
        "record_id": f"record-{episode}",
        "source": "pinned_complete_baseline",
        "category": "ongoing",
        "canonical_episode_id": episode,
        "split": "train",
        "task_instruction": "Place the red block in the tray",
        "anchor_frame": 10,
        "images": [
            {"video": f"/fixture/{episode}.mp4", "frame": 10, "view": "face"}
        ],
        "history_material": {
            "long": [],
            "short": {
                "label_available": False,
                "unit_kind": "action",
                "caption": "",
                "progress_percent": 0,
            },
            "with_memory_eligible": False,
        },
        "supervision": {
            "task_progress_percent": 10,
            "predictions": [
                {
                    "index": 1,
                    "role": "current",
                    "action": {
                        "label_available": True,
                        "caption": first,
                        "progress_percent": 25,
                    },
                    "segment": {
                        "label_available": False,
                        "caption": "",
                        "progress_percent": 0,
                    },
                },
                {
                    "index": 2,
                    "role": "next",
                    "action": {
                        "label_available": True,
                        "caption": second,
                        "progress_percent": 0,
                    },
                    "segment": {
                        "label_available": False,
                        "caption": "",
                        "progress_percent": 0,
                    },
                },
            ],
            "execution_decision": {"label_available": False, "value": ""},
        },
        "provenance": {
            "dataset": "pinned_complete_baseline",
            "snapshot_version": "fixture-version",
            "snapshot_content_digest": "fixture-digest",
            "source_sample_id": f"source-{episode}",
            "episode_key": episode,
            "source_split": "train",
            "source_line": 1,
            "anchor_kind": "action",
            "anchor_sequence_index": 0,
        },
    }


def _initial_sample_and_plan(record: dict) -> tuple[dict, list[dict]]:
    collector = EpisodeActionPlanCollector(
        snapshot_version="fixture-version",
        snapshot_content_digest="fixture-digest",
    )
    collector.add(record)
    initial_record = next(collector.iter_records())
    return convert_initial_plan(initial_record, mode="action")


class BaselineFragmentReduceV5Test(unittest.TestCase):
    def test_slice_reader_disables_buffered_range_readahead(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "slice.jsonl"
            payload = json.dumps({"sample_id": "fixture"}).encode() + b"\n"
            path.write_bytes(payload + b"unrelated bytes after the slice\n")
            original_open = Path.open
            calls = []

            def tracked_open(instance, *args, **kwargs):
                calls.append((args, kwargs))
                return original_open(instance, *args, **kwargs)

            with mock.patch.object(Path, "open", tracked_open):
                values = list(_iter_slice_samples([{
                    "shard": str(path),
                    "offset": 0,
                    "length": len(payload),
                    "count": 1,
                }]))
            self.assertEqual(values, [{"sample_id": "fixture"}])
            self.assertEqual(calls[0][1].get("buffering"), 0)

    def test_hybrid_initial_recovery_fills_only_missing_spool_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spool = root / "spool"
            spool.mkdir()
            existing_record = _canonical_action_record(
                "episode-existing", "Approach the red block", "Grasp the red block"
            )
            missing_record = _canonical_action_record(
                "episode-missing", "Approach the blue block", "Grasp the blue block"
            )
            existing_sample, existing_plan = _initial_sample_and_plan(existing_record)
            _missing_sample, missing_plan = _initial_sample_and_plan(missing_record)
            (spool / "initial.jsonl").write_text(
                json.dumps(existing_sample) + "\n", encoding="utf-8"
            )
            fragment = root / "pass1.jsonl"
            fragment.write_text(
                json.dumps(existing_record) + "\n" + json.dumps(missing_record) + "\n",
                encoding="utf-8",
            )
            recovered, report = _recover_initial_samples_hybrid(
                spool,
                split="train",
                plan_by_episode={
                    "episode-existing": existing_plan,
                    "episode-missing": missing_plan,
                },
                pass1_results=[{
                    "fragment": str(fragment),
                    "cache_only_verified": True,
                }],
                workers=2,
                snapshot_version="fixture-version",
                snapshot_content_digest="fixture-digest",
            )
            self.assertEqual(
                [value["provenance"]["episode_key"] for value in recovered],
                ["episode-existing", "episode-missing"],
            )
            self.assertEqual(report["spool_samples"], 1)
            self.assertEqual(report["supplemented_samples"], 1)
            self.assertEqual(report["complete_samples"], 2)
            self.assertEqual(
                recovered[1]["target"]["initial_plan"], missing_plan
            )

            broken_plan = json.loads(json.dumps(missing_plan))
            broken_plan[0]["action"]["caption"] = "Wrong plan"
            with self.assertRaisesRegex(RuntimeError, "supplemented plan mismatch"):
                _recover_initial_samples_hybrid(
                    spool,
                    split="train",
                    plan_by_episode={
                        "episode-existing": existing_plan,
                        "episode-missing": broken_plan,
                    },
                    pass1_results=[{
                        "fragment": str(fragment),
                        "cache_only_verified": True,
                    }],
                    workers=1,
                    snapshot_version="fixture-version",
                    snapshot_content_digest="fixture-digest",
                )

    def test_cache_only_phase_recovery_is_ordered_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw.jsonl"
            raw.write_text(
                "".join(json.dumps({"value": "x" * size}) + "\n" for size in range(5, 35)),
                encoding="utf-8",
            )
            fragments = _write_cached_phase(raw, root / "cache", chunk_bytes=80)
            recovered = _recover_cached_phase(
                paths=(raw,),
                phase="pass1",
                cache_root=root / "cache",
                workers=2,
                chunk_bytes=80,
            )
            self.assertEqual(
                [Path(value["fragment"]) for value in recovered], fragments
            )
            self.assertTrue(all(value["reused"] for value in recovered))

            metadata = fragments[1].with_suffix(".meta.json")
            value = json.loads(metadata.read_text(encoding="utf-8"))
            value["cache_key"] = "0" * 64
            metadata.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "cache key mismatch"):
                _recover_cached_phase(
                    paths=(raw,),
                    phase="pass1",
                    cache_root=root / "cache",
                    workers=1,
                    chunk_bytes=80,
                )

    def test_initial_samples_recover_from_complete_frozen_spool(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spool = root / "spool"
            spool.mkdir()
            plans = {}
            samples = []
            for episode in ("episode-b", "episode-a"):
                plan = [{"index": 1, "action": {"caption": "Grasp the red block"}}]
                plans[episode] = plan
                payload = "\0".join((episode, "initial_plan", "action")).encode()
                stable = hashlib.sha256(payload).hexdigest()[:24]
                # The canonical collector owns a distinct 20-hex record ID;
                # materialized sample/base IDs use the converter's 24-hex ID.
                record_stable = hashlib.sha256(episode.encode()).hexdigest()[:20]
                samples.append(_sample(
                    sample_id=f"baseline_{stable}_no_memory",
                    base_sample_id=f"baseline_{stable}",
                    source="baseline",
                    category="initial_plan",
                    memory_variant="no_memory",
                    output_spec={
                        "prediction1_units": [],
                        "prediction2_units": [],
                        "plan_units": ["action"],
                    },
                    task_instruction="Place the red block in the tray",
                    images=[{"video": "/fixture/face.mp4", "frame": 0, "view": "face"}],
                    prompt_context={},
                    target={"initial_plan": plan},
                    loss_mask_paths=(),
                    provenance={
                        "episode_key": episode,
                        "task_name": _task_name(
                            "Place the red block in the tray", "action"
                        ),
                        "split": "train",
                        "source_split": "train",
                        "canonical_source": "pinned_complete_baseline",
                        "canonical_record_id": f"v5_baseline_plan_{record_stable}",
                        "memory_pair_eligible": False,
                        "clean_label_mode": "action",
                        "snapshot_version": "fixture-version",
                        "snapshot_content_digest": "fixture-digest",
                        "source_sample_count": 2,
                    },
                ))
            (spool / "initial.jsonl").write_text(
                "".join(json.dumps(value) + "\n" for value in samples),
                encoding="utf-8",
            )
            (spool / "interrupted-empty-leaf.jsonl").write_text("", encoding="utf-8")
            recovered = _recover_initial_samples_from_spool(
                spool,
                split="train",
                plan_by_episode=plans,
                snapshot_version="fixture-version",
                snapshot_content_digest="fixture-digest",
            )
            self.assertEqual(
                [value["provenance"]["episode_key"] for value in recovered],
                ["episode-a", "episode-b"],
            )
            plans["episode-a"][0]["action"]["caption"] = "Changed plan"
            with self.assertRaisesRegex(RuntimeError, "initial plan mismatch"):
                _recover_initial_samples_from_spool(
                    spool,
                    split="train",
                    plan_by_episode=plans,
                    snapshot_version="fixture-version",
                    snapshot_content_digest="fixture-digest",
                )

    def test_deterministic_fragment_order_filter_reuse_and_atomic_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "pass2" / "000000.jsonl"
            second = root / "pass2" / "000001.jsonl"
            _write_fragment(first, [
                _fixture_sample("sample-a", "episode-a"),
                _fixture_sample("sample-drop", "episode-validation"),
            ])
            _write_fragment(second, [
                _fixture_sample("sample-b", "episode-b"),
            ])
            original = {path: path.read_bytes() for path in (first, second)}
            output = root / "published"
            report = {
                "schema_version": "fixture",
                "split_protection": {
                    "policy": "validation_episode_precedence",
                    "excluded_train_samples": None,
                },
            }
            manifest = materialize_fragment_shards(
                [first, second],
                output,
                shard_cache_root=root / "shards",
                workers=2,
                excluded_episodes_by_fragment=[{"episode-validation"}, set()],
                plan_cache_report=report,
            )
            self.assertEqual(manifest["num_samples"], 2)
            self.assertEqual(manifest["accelerator"]["schema_version"], ACCELERATOR_SCHEMA_VERSION)
            self.assertEqual(manifest["accelerator"]["num_reused_shards"], 0)
            leaf = output / manifest["leaves"][0]["path"]
            self.assertEqual(
                [read_indexed_item(leaf, index)["data_id"] for index in range(2)],
                ["sample-a", "sample-b"],
            )
            stored_report = json.loads(
                (output / "plan_cache_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                stored_report["split_protection"]["excluded_train_samples"], 1
            )
            self.assertEqual({path: path.read_bytes() for path in (first, second)}, original)
            self.assertFalse(any(root.glob(".published.fragment-reduce.tmp-*")))

            second_output = root / "published-reuse"
            reused = materialize_fragment_shards(
                [first, second],
                second_output,
                shard_cache_root=root / "shards",
                workers=1,
                excluded_episodes_by_fragment=[{"episode-validation"}, set()],
            )
            self.assertEqual(reused["accelerator"]["num_reused_shards"], 2)
            self.assertEqual(
                (output / manifest["leaves"][0]["path"] / "data.jsonl").read_bytes(),
                (second_output / reused["leaves"][0]["path"] / "data.jsonl").read_bytes(),
            )

    def test_matches_frozen_materializer_for_both_split_stream_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_initial = _fixture_sample("train-initial", "episode-train")
            train_a = _ongoing_sample("train-a", "episode-train", "train")
            train_drop = _ongoing_sample(
                "train-drop", "episode-validation", "train"
            )
            train_b = _ongoing_sample("train-b", "episode-train", "train")
            validation_initial = _fixture_sample(
                "validation-initial", "episode-validation"
            )
            validation_initial["provenance"]["split"] = "validation"
            validation_a = _ongoing_sample(
                "validation-a", "episode-validation", "validation"
            )
            fragments = [
                root / "00-train-initial.jsonl",
                root / "01-train-pass2-a.jsonl",
                root / "02-train-pass2-b.jsonl",
                root / "03-validation-initial.jsonl",
                root / "04-validation-pass2.jsonl",
            ]
            values = [
                [train_initial],
                [train_a, train_drop],
                [train_b],
                [validation_initial],
                [validation_a],
            ]
            for path, samples in zip(fragments, values):
                _write_fragment(path, samples)
            accelerated = root / "accelerated"
            accelerated_manifest = materialize_fragment_shards(
                fragments,
                accelerated,
                shard_cache_root=root / "shards",
                workers=2,
                excluded_episodes_by_fragment=[
                    {"episode-validation"},
                    {"episode-validation"},
                    {"episode-validation"},
                    set(),
                    set(),
                ],
            )
            frozen = root / "frozen"
            frozen_manifest = materialize_dataset(
                [train_initial, train_a, train_b, validation_initial, validation_a],
                frozen,
                source="baseline",
                partial=False,
                limit=None,
                leaf_workers=2,
            )
            self.assertEqual(
                accelerated_manifest["num_samples"], frozen_manifest["num_samples"]
            )
            self.assertEqual(
                accelerated_manifest["leaves"], frozen_manifest["leaves"]
            )
            for leaf in frozen_manifest["leaves"]:
                for name in (
                    "data.jsonl",
                    "data.index",
                    "episodes.jsonl",
                    "manifest.json",
                ):
                    self.assertEqual(
                        (accelerated / leaf["path"] / name).read_bytes(),
                        (frozen / leaf["path"] / name).read_bytes(),
                    )

    def test_mixed_memory_eligibility_is_audited_and_partitioned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ineligible = _ongoing_sample(
                "sample-ineligible", "episode-ineligible", "train"
            )
            eligible_no = _ongoing_sample(
                "sample-eligible-no", "episode-eligible", "train"
            )
            eligible_no["provenance"]["memory_pair_eligible"] = True
            eligible_no["base_sample_id"] = "base-eligible"
            eligible_with = json.loads(json.dumps(eligible_no))
            eligible_with["sample_id"] = "sample-eligible-with"
            eligible_with["memory_variant"] = "with_memory"
            eligible_with["prompt_context"] = {
                "initial_plan_memory": [{
                    "index": 1,
                    "action": {"caption": "Grasp the red block"},
                }],
                "long_memory": [],
                "short_memory": None,
            }
            fragment = root / "mixed.jsonl"
            _write_fragment(fragment, [ineligible, eligible_no, eligible_with])

            output = root / "published"
            manifest = materialize_fragment_shards(
                [fragment],
                output,
                shard_cache_root=root / "shards",
                workers=2,
            )

            partition = manifest["accelerator"]["memory_pair_eligibility_partition"]
            self.assertEqual(
                manifest["accelerator"]["eligibility_audit_mode"],
                "single_sequential_pass_per_shard_plus_mixed_leaf_examples",
            )
            self.assertEqual(partition["mixed_leaf_count"], 1)
            self.assertEqual(partition["retagged_samples"], 1)
            self.assertEqual(len(partition["conflicts"]), 1)
            conflict = partition["conflicts"][0]
            self.assertEqual(
                conflict["counts"], {"eligible": 1, "ineligible": 1}
            )
            self.assertEqual(
                conflict["examples"]["ineligible"]["sample_id"],
                "sample-ineligible",
            )
            self.assertEqual(manifest["num_samples"], 3)
            self.assertEqual(manifest["num_leaves"], 3)

            rows = []
            leaf_manifests = []
            for leaf in manifest["leaves"]:
                leaf_root = output / leaf["path"]
                leaf_manifest = json.loads(
                    (leaf_root / "manifest.json").read_text(encoding="utf-8")
                )
                leaf_manifests.append(leaf_manifest)
                rows.extend(
                    read_indexed_item(leaf_root, index)["v5_sample"]
                    for index in range(leaf_manifest["num_samples"])
                )
            by_id = {row["sample_id"]: row for row in rows}
            rewritten = by_id["sample-ineligible"]
            self.assertEqual(rewritten["target"], ineligible["target"])
            self.assertEqual(rewritten["prompt_context"], ineligible["prompt_context"])
            self.assertEqual(rewritten["images"], ineligible["images"])
            self.assertEqual(rewritten["sample_id"], ineligible["sample_id"])
            self.assertEqual(rewritten["base_sample_id"], ineligible["base_sample_id"])
            self.assertEqual(
                rewritten["provenance"]["original_task_name"],
                ineligible["provenance"]["task_name"],
            )
            self.assertEqual(
                rewritten["provenance"]["materialization_partition_reason"],
                "mixed_memory_pair_eligibility_within_original_leaf",
            )
            self.assertNotEqual(
                rewritten["provenance"]["task_name"],
                ineligible["provenance"]["task_name"],
            )
            self.assertTrue(all(
                isinstance(value["memory_pair_eligible"], bool)
                for value in leaf_manifests
            ))
            formal = validate_dataset(
                output, expect_composed=False, require_full=False
            )
            self.assertTrue(formal["passed"])
            self.assertEqual(formal["memory_pair_count"], 1)

    def test_non_boolean_memory_eligibility_fails_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            invalid = _ongoing_sample("sample-invalid", "episode-invalid", "train")
            invalid["provenance"]["memory_pair_eligible"] = "false"
            fragment = root / "invalid.jsonl"
            _write_fragment(fragment, [invalid])
            output = root / "published"
            with self.assertRaisesRegex(ValueError, "memory_pair_eligible"):
                materialize_fragment_shards(
                    [fragment],
                    output,
                    shard_cache_root=root / "shards",
                    workers=1,
                )
            self.assertFalse(output.exists())

    def test_corrupt_or_incomplete_owned_shard_cache_fails_closed(self) -> None:
        for mode in ("corrupt", "incomplete"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fragment = root / "one.jsonl"
                _write_fragment(
                    fragment, [_fixture_sample("sample-a", "episode-a")]
                )
                materialize_fragment_shards(
                    [fragment],
                    root / "first",
                    shard_cache_root=root / "shards",
                    workers=1,
                )
                manifest_path = next((root / "shards").glob("*.manifest.json"))
                shard_manifest = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                )
                if mode == "corrupt":
                    Path(shard_manifest["shard"]).write_bytes(b"corrupt\n")
                else:
                    shard_manifest["shard"] = str(root / "missing-shard.jsonl")
                    manifest_path.write_text(
                        json.dumps(shard_manifest), encoding="utf-8"
                    )
                with self.assertRaisesRegex(RuntimeError, "owned shard cache is incomplete"):
                    materialize_fragment_shards(
                        [fragment],
                        root / "second",
                        shard_cache_root=root / "shards",
                        workers=1,
                    )

    def test_partition_rejects_fragment_output_count_metadata_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fragment = root / "one.jsonl"
            _write_fragment(fragment, [_fixture_sample("sample-a", "episode-a")])
            metadata_path = fragment.with_suffix(".meta.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["output_count"] = 2
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "output count mismatch"):
                materialize_fragment_shards(
                    [fragment],
                    root / "published",
                    shard_cache_root=root / "shards",
                    workers=1,
                )

    def test_refuses_live_pid_and_existing_output(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "cannot hot-switch"):
            _assert_pid_inactive(__import__("os").getpid())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fragment = root / "one.jsonl"
            _write_fragment(fragment, [_fixture_sample("sample-a", "episode-a")])
            output = root / "published"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                materialize_fragment_shards(
                    [fragment],
                    output,
                    shard_cache_root=root / "shards",
                    workers=1,
                )


if __name__ == "__main__":
    unittest.main()
