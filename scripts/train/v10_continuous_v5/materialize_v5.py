"""Convert V5 adapter records into validated atomic indexed datasets."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import shutil
import uuid
from collections import Counter, OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

from .indexed_io_v5 import LeafKey, write_indexed_leaf
from .holdout_v5 import (
    Benchmark3Holdout,
    DEFAULT_BENCHMARK3_MANIFEST,
    DEFAULT_BENCHMARK3_SHA256,
    HoldoutFilter,
)
from .robodojo_adapter import (
    DEFAULT_OFFICIAL_SPLIT,
    ROBODOJO_TASKS,
    build_canonical_records,
    load_official_split_assignments,
    scan_robodojo,
)
from .schema_v5 import (
    FAILURE_TYPE_BY_SOURCE_CODE,
    SCHEMA_VERSION,
    output_profile_id,
    validate_sample,
)
from .takeover_adapter import TakeoverQAdapter, TakeoverQDataError, VideoMetadata
from .video_probe_cache_v5 import load_video_probe_cache
from .parallel_v5 import bounded_ordered_map, resolve_workers


MATERIALIZATION_SCHEMA_VERSION = "v10_action_segment_v5_materialization_v3"
END_FIXTURE_SCHEMA_VERSION = "v5_robodojo_end_review_fixture_v1"


def _stable_id(*parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=False
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _task_slug(instruction: str) -> str:
    words = re.sub(r"[^A-Za-z0-9]+", "_", instruction.casefold()).strip("_")
    prefix = words[:48].rstrip("_") or "task"
    return f"{prefix}_{_stable_id(instruction)[:8]}"


def _unit(
    caption: str,
    *,
    progress_percent: int = 0,
) -> dict[str, Any]:
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("a present prediction unit requires a caption")
    return {
        "available": True,
        "caption": caption,
        "progress_percent": progress_percent,
    }


def _prediction(
    index: int,
    role: str,
    *,
    action: dict[str, Any] | None = None,
    segment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    value = {"index": index, "role": role}
    if action is not None:
        value["action"] = action
    if segment is not None:
        value["segment"] = segment
    return value


def _sample(
    *,
    sample_id: str,
    base_sample_id: str,
    source: str,
    category: str,
    memory_variant: str,
    output_spec: dict[str, list[str]],
    task_instruction: str,
    images: list[Any],
    prompt_context: dict[str, Any],
    target: dict[str, Any],
    loss_mask_paths: Sequence[str],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "base_sample_id": base_sample_id,
        "source": source,
        "category": category,
        "memory_variant": memory_variant,
        "output_spec": output_spec,
        "output_profile_id": output_profile_id(output_spec),
        "task_instruction": task_instruction,
        "images": images,
        "prompt_context": prompt_context,
        "target": target,
        "supervision": {"loss_mask_paths": list(loss_mask_paths)},
        "provenance": provenance,
    }
    return validate_sample(value)


def takeover_record_to_sample(record: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one failed Takeover-Q canonical record to a V5 sample."""

    decision = str(record.get("decision_class") or "")
    if decision != "Takeover":
        raise ValueError(
            "Takeover-Q Continue is excluded from V5 training; expected Takeover"
        )
    if record.get("memory_variant") != "no_memory":
        raise ValueError("Takeover-Q materialization requires no_memory")
    conditioning = record.get("conditioning")
    labels = record.get("labels")
    provenance_raw = record.get("provenance")
    if not isinstance(conditioning, Mapping) or not isinstance(labels, Mapping):
        raise ValueError("Takeover-Q record is missing conditioning/labels")
    if not isinstance(provenance_raw, Mapping):
        raise ValueError("Takeover-Q record is missing provenance")
    instruction = str(conditioning.get("task_instruction") or "")
    task_name = _task_slug(instruction)
    split = str(provenance_raw.get("split") or "train")
    episode_key = str(provenance_raw.get("episode_key") or "")
    if not episode_key:
        raise ValueError("Takeover-Q provenance is missing episode_key")
    base_id = f"takeover_q_{_stable_id(record.get('sample_id'), episode_key)}"
    common_provenance = {
        **dict(provenance_raw),
        "episode_key": episode_key,
        "task_name": task_name,
        "split": split,
        "canonical_sample_id": str(record.get("sample_id") or ""),
        "canonical_source": "takeover_q/final_reviewed_bilingual",
        "memory_pair_eligible": False,
        "label_sources": dict(
            (record.get("supervision") or {}).get("label_sources") or {}
        ),
    }

    source_code = str(provenance_raw.get("raw_failure_source_key") or "")
    failure_type = FAILURE_TYPE_BY_SOURCE_CODE.get(source_code)
    if failure_type is None:
        raise ValueError(
            f"Takeover-Q record has unsupported failure source code: {source_code!r}"
        )
    expected_action = str(labels.get("expected_action") or "")
    observed_failure = str(labels.get("observed_failure") or "")
    failed_action_context = str(labels.get("failed_action_context") or "")
    recovery_action = str(labels.get("recovery_action") or "")
    target = {
        "execution_decision": "Takeover",
        "decision_detail": {
            "failure_analysis": {
                "failed_action_context": failed_action_context,
                "expected_action": expected_action,
                "observed_failure": observed_failure,
                "failure_type": failure_type,
            },
            "recovery_plan": [
                {
                    "index": 1,
                    "action": {"caption": recovery_action},
                }
            ],
        },
    }
    masks: tuple[str, ...] = ()
    category = "takeover"

    return _sample(
        sample_id=str(record.get("sample_id") or base_id),
        base_sample_id=base_id,
        source="takeover_q",
        category=category,
        memory_variant="no_memory",
        output_spec={
            "prediction1_units": ["action"],
            "prediction2_units": ["action"],
            "plan_units": ["action"],
        },
        task_instruction=instruction,
        images=list(record.get("images") or ()),
        prompt_context={},
        target=target,
        loss_mask_paths=masks,
        provenance=common_provenance,
    )


def iter_takeover_samples(
    records: Iterable[Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    for record in records:
        yield takeover_record_to_sample(record)


def _takeover_cache_row_to_record(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Project one verified V5.0 cache row back to the reviewed canonical seam.

    This is a smoke-only acceleration path.  It accepts only Takeover rows and
    reconstructs labels from the old decision detail; Q1/Continue rows are
    deliberately ignored before V5.1 rendering.
    """

    sample = row.get("v5_sample")
    if not isinstance(sample, Mapping):
        raise ValueError("Takeover cache row is missing v5_sample")
    if sample.get("source") != "takeover_q":
        return None
    if sample.get("category") != "takeover":
        return None
    if sample.get("memory_variant") != "no_memory":
        raise ValueError("cached Takeover-Q row is not no_memory")
    target = sample.get("target")
    provenance = sample.get("provenance")
    if not isinstance(target, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("cached Takeover-Q row lacks target/provenance")
    if target.get("execution_decision") != "Takeover":
        raise ValueError("cached takeover category has a non-Takeover target")
    detail = target.get("decision_detail")
    if not isinstance(detail, Mapping):
        raise ValueError("cached Takeover target lacks decision_detail")
    analysis = detail.get("failure_analysis")
    recovery = detail.get("recovery_plan")
    if not isinstance(analysis, Mapping) or not isinstance(recovery, list) or not recovery:
        raise ValueError("cached Takeover target lacks failure analysis/recovery plan")
    first = recovery[0]
    action = first.get("action") if isinstance(first, Mapping) else None
    if not isinstance(action, Mapping) or not isinstance(action.get("caption"), str):
        raise ValueError("cached Takeover recovery plan lacks its first Action")
    label_sources = provenance.get("label_sources")
    if not isinstance(label_sources, Mapping):
        raise ValueError("cached Takeover provenance lacks label_sources")
    return {
        "schema_version": "v5_takeover_cache_projection_v1",
        "sample_id": str(sample.get("sample_id") or ""),
        "decision_class": "Takeover",
        "memory_variant": "no_memory",
        "conditioning": {"task_instruction": str(sample.get("task_instruction") or "")},
        "labels": {
            "execution_decision": "Takeover",
            "expected_action": str(analysis.get("expected_action") or ""),
            "observed_failure": str(analysis.get("observed_failure") or ""),
            "failure_type": str(analysis.get("failure_type") or ""),
            "failed_action_context": str(analysis.get("failed_action_context") or ""),
            "recovery_action": str(action.get("caption") or ""),
        },
        "images": list(sample.get("images") or ()),
        "supervision": {"label_sources": dict(label_sources)},
        "provenance": dict(provenance),
    }


def _iter_takeover_cache_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"Takeover cache line {line_number} is not an object")
            projected = _takeover_cache_row_to_record(row)
            if projected is not None:
                yield projected


def _convert_takeover_group(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [takeover_record_to_sample(record) for record in records]


def _group_takeover_records(
    records: Iterable[Mapping[str, Any]],
) -> Iterator[tuple[Mapping[str, Any], ...]]:
    def episode_key(record: Mapping[str, Any]) -> str:
        provenance = record.get("provenance")
        return str(provenance.get("episode_key") if isinstance(provenance, Mapping) else "")

    for key, group in itertools.groupby(records, key=episode_key):
        if not key:
            raise ValueError("Takeover-Q parallel group is missing episode_key")
        yield tuple(group)


def _robodojo_plan(initial_record: Mapping[str, Any]) -> list[dict[str, Any]]:
    supervision = initial_record.get("supervision")
    if not isinstance(supervision, Mapping):
        raise ValueError("RoboDojo initial record is missing supervision")
    raw_plan = supervision.get("initial_plan")
    if not isinstance(raw_plan, list) or not raw_plan:
        raise ValueError("RoboDojo initial record has no Action plan")
    plan: list[dict[str, Any]] = []
    for index, raw_step in enumerate(raw_plan, 1):
        action = raw_step.get("action") if isinstance(raw_step, Mapping) else None
        if not isinstance(action, Mapping) or action.get("label_available") is not True:
            raise ValueError("RoboDojo initial Action label is unavailable")
        plan.append({
            "index": index,
            "action": {
                "caption": str(action.get("caption") or ""),
            },
        })
    return plan


def _robodojo_images(record: Mapping[str, Any], frame: int) -> list[dict[str, Any]]:
    videos = record.get("videos")
    if not isinstance(videos, Mapping) or not videos:
        raise ValueError("RoboDojo canonical record is missing videos")
    return [
        {"video": str(path), "frame": frame, "view": str(view)}
        for view, path in sorted(videos.items())
    ]


def _progress(frame: int, start: int, end: int) -> int:
    denominator = max(end - start - 1, 1)
    return max(0, min(100, round((frame - start) * 100 / denominator)))


def _task_progress(frame: int, total_frames: int) -> int:
    return max(0, min(100, round(frame * 100 / max(total_frames - 1, 1))))


def _long_memory(plan: Sequence[Mapping[str, Any]], current_offset: int) -> list[dict[str, Any]]:
    captions: list[str] = []
    seen: set[str] = set()
    for step in plan[:current_offset]:
        caption = str(step["action"]["caption"])
        identity = caption.casefold()
        if identity in seen:
            continue
        seen.add(identity)
        captions.append(caption)
    captions = captions[-8:]
    return [
        {"index": index, "action": caption}
        for index, caption in enumerate(captions, 1)
    ]


def convert_robodojo_episode_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert one episode's canonical records and isolate End review fixtures."""

    if not records:
        raise ValueError("RoboDojo episode records cannot be empty")
    episode_ids = {str(record.get("canonical_episode_id") or "") for record in records}
    if len(episode_ids) != 1 or "" in episode_ids:
        raise ValueError("RoboDojo records must belong to exactly one episode")
    initial_records = [
        record for record in records if record.get("category") == "initial_plan"
    ]
    if len(initial_records) != 1:
        raise ValueError("RoboDojo episode requires exactly one initial plan")
    initial = initial_records[0]
    plan = _robodojo_plan(initial)
    episode_key = next(iter(episode_ids))
    task_name = str(initial.get("task_name") or "")
    split = str(initial.get("split") or "")
    instruction = str(initial.get("task_instruction") or "")
    total_frames = int(initial.get("total_frames") or 0)
    base_provenance = {
        "episode_key": episode_key,
        "task_name": task_name,
        "split": split,
        "canonical_source": str(initial.get("source") or ""),
    }
    samples: list[dict[str, Any]] = []
    review_fixtures: list[dict[str, Any]] = []

    initial_base_id = f"robodojo_{_stable_id(episode_key, 'initial_plan')}"
    samples.append(_sample(
        sample_id=f"{initial_base_id}_no_memory",
        base_sample_id=initial_base_id,
        source="robodojo",
        category="initial_plan",
        memory_variant="no_memory",
        output_spec={
            "prediction1_units": [],
            "prediction2_units": [],
            "plan_units": ["action"],
        },
        task_instruction=instruction,
        images=_robodojo_images(initial, 0),
        prompt_context={},
        target={"initial_plan": plan},
        loss_mask_paths=(),
        provenance={
            **base_provenance,
            "memory_pair_eligible": False,
            "canonical_record_id": str(initial.get("record_id") or ""),
        },
    ))

    ongoing_records = [
        record for record in records if record.get("category") == "ongoing"
    ]
    previous_short_memory: dict[str, Any] | None = None
    for current_offset, record in enumerate(ongoing_records):
        interval = record.get("anchor_interval")
        supervision = record.get("supervision")
        if not isinstance(interval, Mapping) or not isinstance(supervision, Mapping):
            raise ValueError("RoboDojo ongoing record is incomplete")
        raw_predictions = supervision.get("predictions")
        if not isinstance(raw_predictions, list) or len(raw_predictions) != 2:
            raise ValueError("RoboDojo ongoing record must contain two predictions")
        current_raw, next_raw = raw_predictions
        current_action = current_raw.get("action")
        next_action = next_raw.get("action")
        if (
            not isinstance(current_action, Mapping)
            or current_action.get("label_available") is not True
            or not isinstance(next_action, Mapping)
        ):
            raise ValueError("RoboDojo current Action supervision is unavailable")
        start = int(interval["start_frame"])
        end = int(interval["end_frame"])
        anchor = (start + end - 1) // 2
        action_progress = _progress(anchor, start, end)
        task_progress = _task_progress(anchor, total_frames)
        next_caption = (
            str(next_action.get("caption") or "")
            if next_action.get("label_available") is True
            else None
        )
        prediction2 = _prediction(2, "next")
        prediction2_units: list[str] = []
        if next_caption is not None:
            prediction2["action"] = _unit(next_caption)
            prediction2_units.append("action")
        target = {
            "task_progress_percent": task_progress,
            "predictions": [
                _prediction(
                    1,
                    "current",
                    action=_unit(
                        str(current_action.get("caption") or ""),
                        progress_percent=action_progress,
                    ),
                ),
                prediction2,
            ],
            "execution_decision": "Continue",
            "decision_detail": None,
        }
        masks: list[str] = []
        output_spec = {
            "prediction1_units": ["action"],
            "prediction2_units": prediction2_units,
            "plan_units": ["action"],
        }
        base_id = f"robodojo_{_stable_id(record.get('record_id'), anchor)}"
        images = _robodojo_images(record, anchor)
        provenance = {
            **base_provenance,
            "memory_pair_eligible": True,
            "canonical_record_id": str(record.get("record_id") or ""),
            "anchor_frame": anchor,
            "action_index": current_offset + 1,
        }
        no_memory = _sample(
            sample_id=f"{base_id}_no_memory",
            base_sample_id=base_id,
            source="robodojo",
            category="ongoing",
            memory_variant="no_memory",
            output_spec=output_spec,
            task_instruction=instruction,
            images=images,
            prompt_context={},
            target=target,
            loss_mask_paths=masks,
            provenance=provenance,
        )
        with_memory = _sample(
            sample_id=f"{base_id}_with_memory",
            base_sample_id=base_id,
            source="robodojo",
            category="ongoing",
            memory_variant="with_memory",
            output_spec=output_spec,
            task_instruction=instruction,
            images=images,
            prompt_context={
                "initial_plan_memory": plan,
                "long_memory": _long_memory(plan, current_offset),
                "short_memory": previous_short_memory,
            },
            target=target,
            loss_mask_paths=masks,
            provenance=provenance,
        )
        samples.extend((no_memory, with_memory))
        previous_short_memory = {
            "task_progress_percent": task_progress,
            "prediction1": {
                "action": {
                    "available": True,
                    "caption": str(current_action.get("caption") or ""),
                    "progress_percent": action_progress,
                },
            },
        }

    for record in records:
        if record.get("category") != "end":
            continue
        review_fixtures.append({
            "schema_version": END_FIXTURE_SCHEMA_VERSION,
            "source": "robodojo",
            "category": "end",
            "canonical_episode_id": episode_key,
            "task_name": task_name,
            "split": split,
            "reason": "End lacks independent completion evidence",
            "canonical_record": dict(record),
        })
    return samples, review_fixtures


def _convert_robodojo_episode(episode: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return convert_robodojo_episode_records(build_canonical_records(episode))


class _SpoolPool:
    def __init__(self, root: Path, max_open: int = 4096) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=False)
        self.max_open = max_open
        self.paths: dict[LeafKey, Path] = {}
        self.handles: OrderedDict[LeafKey, BinaryIO] = OrderedDict()
        self.counts: dict[LeafKey, int] = {}
        self.closed = False

    def write(self, sample: Mapping[str, Any]) -> None:
        leaf = LeafKey.from_sample(sample)
        path = self.paths.get(leaf)
        if path is None:
            path = self.root / f"{_stable_id(*leaf.relative_path().parts)}.jsonl"
            self.paths[leaf] = path
            self.counts[leaf] = 0
        handle = self.handles.pop(leaf, None)
        if handle is None:
            handle = path.open("ab")
        self.handles[leaf] = handle
        handle.write(_json_bytes(sample) + b"\n")
        self.counts[leaf] += 1
        if len(self.handles) > self.max_open:
            _old_leaf, old_handle = self.handles.popitem(last=False)
            # The spool is disposable staging state and is read before the
            # atomic dataset publish.  An fsync on every LRU eviction turns a
            # many-leaf build into one synchronous PFS transaction per row.
            # close() still flushes Python buffers; durable fsync remains in
            # IndexedLeafWriter for every file that can reach the final tree.
            old_handle.close()

    def close(self) -> None:
        if self.closed:
            return
        for handle in self.handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        self.handles.clear()
        if self.root.is_dir():
            _fsync_directory(self.root)
        self.closed = True


def _iter_spool(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"spool row is not an object: {path}")
                yield value


def _write_spooled_leaf(task: Mapping[str, Any]) -> dict[str, Any]:
    """Build one independent indexed leaf in a spawn-safe worker."""
    leaf = task["leaf"]
    if not isinstance(leaf, LeafKey):
        raise TypeError("parallel leaf task is missing LeafKey")
    manifest = write_indexed_leaf(
        Path(str(task["leaf_root"])),
        leaf,
        _iter_spool(Path(str(task["spool_path"]))),
    )
    return {
        "path": leaf.relative_path().as_posix(),
        "num_samples": manifest["num_samples"],
        "num_episodes": manifest["num_episodes"],
        "leaf": manifest["leaf"],
    }


def materialize_dataset(
    samples: Iterable[Mapping[str, Any]],
    output_root: Path | str,
    *,
    source: str,
    partial: bool,
    limit: int | None,
    review_fixtures: Iterable[Mapping[str, Any]] = (),
    selector: Mapping[str, Any] | None = None,
    robodojo_official_split_path: Path | str | None = None,
    expected_robodojo_official_split_sha256: str | None = None,
    plan_cache_report: Mapping[str, Any] | None = None,
    leaf_workers: int = 1,
    benchmark3_holdout: Benchmark3Holdout | None = None,
) -> dict[str, Any]:
    """Atomically publish validated samples grouped into indexed leaves."""

    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"atomic output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(
        f".{output_root.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir()
    spool = _SpoolPool(staging / ".spool")
    sample_count = 0
    review_count = 0
    leaf_manifests: list[dict[str, Any]] = []
    canonical_sources: set[str] = set()
    holdout_filter = (
        HoldoutFilter(benchmark3_holdout)
        if benchmark3_holdout is not None
        else None
    )
    try:
        official_split_metadata: dict[str, Any] | None = None
        if robodojo_official_split_path is not None:
            if source != "robodojo":
                raise ValueError(
                    "robodojo_official_split_path is valid only for RoboDojo"
                )
            split_path = Path(robodojo_official_split_path).resolve()
            observed_digest = _file_sha256(split_path)
            expected_digest = (
                expected_robodojo_official_split_sha256 or observed_digest
            )
            if observed_digest != expected_digest:
                raise RuntimeError(
                    "RoboDojo official split changed after scanner resolution"
                )
            copied_relative = Path("metadata", "robodojo_official_split.json")
            copied_path = staging / copied_relative
            copied_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(split_path, copied_path)
            if (
                _file_sha256(copied_path) != expected_digest
                or _file_sha256(split_path) != expected_digest
            ):
                raise RuntimeError(
                    "RoboDojo official split changed while it was being captured"
                )
            official_split_metadata = {
                "schema_version": "v5_robodojo_official_split_provenance_v1",
                "source_path": str(split_path),
                "copied_relative_path": copied_relative.as_posix(),
                "sha256": expected_digest,
            }

        for sample in samples:
            validated = validate_sample(sample)
            if validated["source"] != source:
                raise ValueError(
                    f"materialization source mismatch: {validated['source']} != {source}"
                )
            if holdout_filter is not None and not holdout_filter.keep(validated):
                continue
            provenance = validated["provenance"]
            canonical_sources.add(
                str(provenance.get("canonical_source") or validated["source"])
            )
            spool.write(validated)
            sample_count += 1
        spool.close()
        if sample_count <= 0:
            raise ValueError("materialization produced no trainable V5 samples")

        worker_count = resolve_workers(leaf_workers)
        leaf_tasks = [
            {
                "leaf": leaf,
                "leaf_root": str(staging / leaf.relative_path()),
                "spool_path": str(spool.paths[leaf]),
            }
            for leaf in sorted(spool.paths)
        ]
        leaf_manifests.extend(bounded_ordered_map(
            _write_spooled_leaf,
            leaf_tasks,
            workers=worker_count,
            max_in_flight=worker_count * 2,
        ))

        fixture_path = staging / "review_fixtures" / "end_candidates.jsonl"
        fixture_handle: BinaryIO | None = None
        try:
            for fixture in review_fixtures:
                if fixture_handle is None:
                    fixture_path.parent.mkdir(parents=True)
                    fixture_handle = fixture_path.open("wb")
                fixture_handle.write(_json_bytes(fixture) + b"\n")
                review_count += 1
            if fixture_handle is not None:
                fixture_handle.flush()
                os.fsync(fixture_handle.fileno())
        finally:
            if fixture_handle is not None:
                fixture_handle.close()

        shutil.rmtree(spool.root)
        selector_value = dict(selector or {
            "mode": (
                "first_n_source_units"
                if limit is not None
                else ("unspecified_partial" if partial else "all_source_units")
            )
        })
        manifest = {
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "complete": True,
            "source": source,
            "partial": bool(partial),
            "limit": limit,
            "selector": selector_value,
            "num_samples": sample_count,
            "num_leaves": len(leaf_manifests),
            "num_review_fixtures": review_count,
            "canonical_raw_sources": sorted(canonical_sources),
            "leaves": leaf_manifests,
        }
        if official_split_metadata is not None:
            manifest["robodojo_official_split"] = official_split_metadata
        if plan_cache_report is not None:
            with (staging / "plan_cache_report.json").open("wb") as handle:
                handle.write(_json_bytes(plan_cache_report) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        if holdout_filter is not None:
            holdout_report = holdout_filter.report()
            holdout_report.update({
                "passed": True,
                "policy": "exclude_before_materialization_spool",
                "input_overlap_samples": holdout_filter.excluded_samples,
                "published_samples_are_holdout_free": True,
                "published_samples": sample_count,
            })
            holdout_relative = Path("metadata", "benchmark3_holdout_report.json")
            holdout_path = staging / holdout_relative
            holdout_path.parent.mkdir(parents=True, exist_ok=True)
            with holdout_path.open("wb") as handle:
                handle.write(_json_bytes(holdout_report) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            manifest["benchmark3_holdout"] = {
                **benchmark3_holdout.metadata(),
                "policy": "exclude_before_materialization_spool",
                "checked_input_samples": holdout_filter.checked_samples,
                "excluded_input_samples": holdout_filter.excluded_samples,
                "published_overlap_samples": 0,
                "report_relative_path": holdout_relative.as_posix(),
                "report_sha256": _file_sha256(holdout_path),
            }
        with (staging / "manifest.json").open("wb") as handle:
            handle.write(_json_bytes(manifest) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(staging)
        os.replace(staging, output_root)
        _fsync_directory(output_root.parent)
        return {**manifest, "output_root": str(output_root)}
    except BaseException:
        spool.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--limit must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=("takeover_q", "robodojo"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--benchmark3-manifest",
        type=Path,
        default=DEFAULT_BENCHMARK3_MANIFEST,
        help="Frozen Benchmark3 JSONL holdout; official CLI builds always enforce it.",
    )
    parser.add_argument(
        "--benchmark3-expected-sha256",
        default=DEFAULT_BENCHMARK3_SHA256,
        help="Fail if the frozen Benchmark3 manifest changed.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Partial source-unit limit; omit for a full build.",
    )
    selection.add_argument(
        "--smoke-one-per-train-task",
        action="store_true",
        help=(
            "RoboDojo only: select the first valid train episode for every "
            "official train task."
        ),
    )
    selection.add_argument(
        "--smoke-all-failure-types",
        action="store_true",
        help=(
            "Takeover-Q only: scan the reviewed source deterministically and "
            "select the first Takeover record for each of the 15 failure types."
        ),
    )
    parser.add_argument(
        "--robodojo-official-split",
        type=Path,
        default=DEFAULT_OFFICIAL_SPLIT,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="CPU workers; zero selects min(32, max(4, affinity/4)).",
    )
    parser.add_argument(
        "--takeover-cache-jsonl",
        type=Path,
        help=(
            "Smoke-only acceleration: re-render verified cached Takeover rows; "
            "requires --source takeover_q --smoke-all-failure-types."
        ),
    )
    parser.add_argument(
        "--takeover-cache-report",
        type=Path,
        help=(
            "Required with --takeover-cache-jsonl: scan-complete exclusion report "
            "from the raw reviewed-source scan that produced the cached rows."
        ),
    )
    parser.add_argument(
        "--takeover-snapshot-id",
        help=(
            "Takeover-Q only: immutable reviewed snapshot identifier recorded in "
            "sample IDs and provenance; must be supplied with both Takeover roots."
        ),
    )
    parser.add_argument(
        "--takeover-snapshot-root",
        type=Path,
        help=(
            "Takeover-Q only: filesystem root prepended to absolute raw video "
            "paths. Use / when those paths are directly mounted on this node."
        ),
    )
    parser.add_argument(
        "--takeover-video-probe-cache",
        type=Path,
        help=(
            "Takeover-Q only: complete stat-bound fps/frame-count cache from "
            "v5_takeover_fast_scan.py; every path is re-statted before use."
        ),
    )
    parser.add_argument(
        "--takeover-reviewed-root",
        type=Path,
        help=(
            "Takeover-Q only: pinned new_completed directory containing "
            "episodes.jsonl and episodes/*.json; do not pass a moving symlink."
        ),
    )
    return parser


def _smoke_one_per_train_task(
    episodes: Sequence[Any],
    assignments: Mapping[str, str],
) -> tuple[Any, ...]:
    official_train_tasks = {
        episode_key.rsplit("/", 2)[-2]
        for episode_key, split in assignments.items()
        if split == "train"
    }
    candidates: dict[str, list[Any]] = {
        task: [] for task in official_train_tasks
    }
    for episode in episodes:
        episode_key = str(episode.canonical_episode_id)
        task_name = str(episode.task_name)
        if episode.split != "train" or assignments.get(episode_key) != "train":
            raise RuntimeError(
                "RoboDojo smoke selector received a holdout episode"
            )
        if task_name not in candidates:
            raise RuntimeError(
                f"RoboDojo smoke selector received an unknown train task: {task_name}"
            )
        candidates[task_name].append(episode)
    missing = sorted(task for task, values in candidates.items() if not values)
    if missing:
        raise ValueError(
            "RoboDojo smoke selector has no valid episode for official train "
            f"tasks: {missing}"
        )
    task_order = [task for task in ROBODOJO_TASKS if task in official_train_tasks]

    def episode_order(episode: Any) -> tuple[int, str]:
        trajectory_name = str(episode.trajectory_name)
        try:
            trajectory_index = int(trajectory_name.rsplit("_", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"invalid RoboDojo trajectory name: {trajectory_name!r}"
            ) from exc
        return trajectory_index, str(episode.canonical_episode_id)

    return tuple(
        min(
            candidates[task],
            key=episode_order,
        )
        for task in task_order
    )


def _smoke_all_takeover_failure_types(
    records: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Select real reviewed records with deterministic complete class coverage."""

    failure_order = tuple(FAILURE_TYPE_BY_SOURCE_CODE.values())
    expected = set(failure_order)
    first_by_failure: dict[str, Mapping[str, Any]] = {}
    for record in records:
        decision = str(record.get("decision_class") or "")
        if decision != "Takeover":
            raise ValueError(
                f"unexpected Takeover-Q decision in smoke selector: {decision!r}"
            )
        labels = record.get("labels")
        if not isinstance(labels, Mapping):
            raise ValueError("Takeover-Q smoke record is missing labels")
        failure_type = str(labels.get("failure_type") or "")
        if failure_type not in expected:
            raise ValueError(
                "Takeover-Q smoke record has an unknown failure type: "
                f"{failure_type!r}"
            )
        first_by_failure.setdefault(failure_type, record)

    missing = [name for name in failure_order if name not in first_by_failure]
    if missing:
        raise ValueError(
            "Takeover-Q smoke selector is missing failure types: "
            f"{missing}"
        )
    return tuple(
        first_by_failure[name] for name in failure_order
    )


def _iter_takeover_records_with_report(
    adapter: TakeoverQAdapter,
    report: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Stream adapter rows and close the audit trail on natural exhaustion."""

    report.update({
        "schema_version": "v5_takeover_q_exclusion_report_v1",
        "scan_complete": False,
        "num_exclusions": None,
        "exclusion_reason_counts": {},
        "exclusions": [],
    })
    completed = False
    try:
        yield from adapter.iter_records(decisions=("Takeover",))
        completed = True
    finally:
        exclusions = [dict(value) for value in adapter.exclusions]
        report.update({
            "scan_complete": completed,
            "num_exclusions": len(exclusions),
            "exclusion_reason_counts": dict(sorted(Counter(
                str(value.get("reason") or "unknown") for value in exclusions
            ).items())),
            "exclusions": exclusions,
        })


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    benchmark3_holdout = Benchmark3Holdout.load(
        args.benchmark3_manifest,
        expected_sha256=args.benchmark3_expected_sha256,
    )
    workers = resolve_workers(args.workers)
    takeover_roots = (
        args.takeover_snapshot_id,
        args.takeover_snapshot_root,
        args.takeover_reviewed_root,
    )
    if args.takeover_video_probe_cache is not None and args.source != "takeover_q":
        raise ValueError("--takeover-video-probe-cache requires --source takeover_q")
    if args.takeover_video_probe_cache is not None and args.takeover_cache_jsonl is not None:
        raise ValueError("video probe cache cannot be combined with smoke row cache")
    if any(value is not None for value in takeover_roots):
        if args.source != "takeover_q":
            raise ValueError("Takeover snapshot options require --source takeover_q")
        if not all(value is not None for value in takeover_roots):
            raise ValueError(
                "--takeover-snapshot-id, --takeover-snapshot-root and "
                "--takeover-reviewed-root must be supplied together"
            )
        if args.takeover_cache_jsonl is not None:
            raise ValueError(
                "Takeover snapshot options cannot be combined with smoke cache input"
            )
    if args.smoke_one_per_train_task and args.source != "robodojo":
        raise ValueError("--smoke-one-per-train-task requires --source robodojo")
    if args.smoke_all_failure_types and args.source != "takeover_q":
        raise ValueError("--smoke-all-failure-types requires --source takeover_q")
    if args.takeover_cache_jsonl is not None and (
        args.source != "takeover_q" or not args.smoke_all_failure_types
    ):
        raise ValueError(
            "--takeover-cache-jsonl requires Takeover-Q smoke-all-failure-types"
        )
    if (args.takeover_cache_jsonl is None) != (args.takeover_cache_report is None):
        raise ValueError(
            "--takeover-cache-jsonl and --takeover-cache-report are required together"
        )
    partial = (
        args.limit is not None
        or args.smoke_one_per_train_task
        or args.smoke_all_failure_types
    )
    review_fixtures: list[dict[str, Any]] = []
    official_split_path: Path | None = None
    official_split_digest: str | None = None
    plan_cache_report: dict[str, Any] | None = None
    if args.source == "takeover_q":
        if args.takeover_cache_jsonl is not None:
            cache_path = args.takeover_cache_jsonl.resolve()
            cache_report_path = args.takeover_cache_report.resolve()
            if not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            if not cache_report_path.is_file():
                raise FileNotFoundError(cache_report_path)
            loaded_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
            if (
                not isinstance(loaded_report, Mapping)
                or loaded_report.get("schema_version")
                != "v5_takeover_q_exclusion_report_v1"
                or loaded_report.get("scan_complete") is not True
            ):
                raise ValueError("Takeover cache report is not a scan-complete V5 report")
            plan_cache_report = {
                **dict(loaded_report),
                "selection_source": "verified_raw_scan_cache_confirmation",
                "selection_cache_path": str(cache_path),
                "selection_cache_sha256": _file_sha256(cache_path),
                "selection_raw_scan_report_path": str(cache_report_path),
                "selection_raw_scan_report_sha256": _file_sha256(cache_report_path),
            }
            records = _iter_takeover_cache_records(cache_path)
        else:
            adapter_kwargs: dict[str, Any] = {}
            if args.takeover_snapshot_id is not None:
                adapter_kwargs = {
                    "snapshot_id": args.takeover_snapshot_id,
                    "snapshot_root": args.takeover_snapshot_root.resolve(),
                    "reviewed_root": args.takeover_reviewed_root.resolve(),
                }
            probe_cache_path: Path | None = None
            probe_cache_payload: Mapping[str, Any] | None = None
            if args.takeover_video_probe_cache is not None:
                probe_cache_path = args.takeover_video_probe_cache.resolve()
                if not probe_cache_path.is_file():
                    raise FileNotFoundError(probe_cache_path)
                probe_cache_entries = load_video_probe_cache(probe_cache_path)
                loaded_probe_payload = json.loads(
                    probe_cache_path.read_text(encoding="utf-8")
                )
                if not isinstance(loaded_probe_payload, Mapping):
                    raise ValueError("Takeover video probe cache is not an object")
                probe_cache_payload = loaded_probe_payload

                def cached_probe(path: Path) -> VideoMetadata:
                    value = probe_cache_entries.get(Path(path))
                    if value is None:
                        raise TakeoverQDataError(
                            f"video is absent from the complete probe cache: {path}"
                        )
                    size, mtime_ns, fps, frame_count = value
                    observed = path.stat()
                    if observed.st_size != size or observed.st_mtime_ns != mtime_ns:
                        raise TakeoverQDataError(
                            f"video changed after probe cache creation: {path}"
                        )
                    return VideoMetadata(fps=fps, frame_count=frame_count)

                adapter_kwargs["video_probe"] = cached_probe
            adapter = TakeoverQAdapter(**adapter_kwargs)
            reviewed_index_sha256 = _file_sha256(adapter.index_path.resolve())
            if probe_cache_payload is not None:
                if (
                    probe_cache_payload.get("snapshot_id") != adapter.snapshot_id
                    or probe_cache_payload.get("reviewed_root")
                    != str(adapter.reviewed_root.resolve())
                    or probe_cache_payload.get("reviewed_index_sha256")
                    != reviewed_index_sha256
                ):
                    raise ValueError(
                        "Takeover video probe cache does not match the pinned snapshot"
                    )
            plan_cache_report = {
                "snapshot_id": adapter.snapshot_id,
                "snapshot_root": str(adapter.snapshot_root.resolve()),
                "reviewed_root": str(adapter.reviewed_root.resolve()),
                "reviewed_index": str(adapter.index_path.resolve()),
                "reviewed_index_sha256": reviewed_index_sha256,
            }
            if probe_cache_path is not None:
                plan_cache_report.update({
                    "video_probe_cache_path": str(probe_cache_path),
                    "video_probe_cache_sha256": _file_sha256(probe_cache_path),
                    "video_probe_cache_entries": len(probe_cache_entries),
                    "video_probe_cache_stat_verified_on_use": True,
                })
            records = _iter_takeover_records_with_report(adapter, plan_cache_report)
        if args.smoke_all_failure_types:
            records = _smoke_all_takeover_failure_types(records)
        elif args.limit is not None:
            records = itertools.islice(records, args.limit)
        def iter_parallel_takeover() -> Iterator[dict[str, Any]]:
            groups = _group_takeover_records(records)
            for converted in bounded_ordered_map(
                _convert_takeover_group,
                groups,
                workers=workers,
                max_in_flight=workers * 2,
            ):
                yield from converted

        samples = iter_parallel_takeover()
        selector = {
            "mode": (
                "first_reviewed_takeover_per_failure_type"
                if args.smoke_all_failure_types
                else (
                    "first_n_canonical_records"
                    if args.limit is not None
                    else "all_canonical_records"
                )
            )
        }
        if args.smoke_all_failure_types:
            selector.update({
                "selected_failure_type_count": len(
                    FAILURE_TYPE_BY_SOURCE_CODE
                ),
                "failure_types": list(FAILURE_TYPE_BY_SOURCE_CODE.values()),
            })
        if args.takeover_cache_jsonl is not None:
            selector.update({
                "input_mode": "verified_corrected_full_cache_confirmation",
                "input_cache_sha256": _file_sha256(args.takeover_cache_jsonl.resolve()),
                "input_raw_scan_report_sha256": _file_sha256(
                    args.takeover_cache_report.resolve()
                ),
            })
        selector["parallel"] = {
            "workers": workers,
            "max_in_flight": workers * 2,
            "grouping": "episode_key",
            "nice": 5,
        }
    else:
        official_split_path = args.robodojo_official_split.resolve()
        official_split_digest = _file_sha256(official_split_path)
        scan = scan_robodojo(
            official_split_path=official_split_path,
            include_splits=("train",),
            allow_holdouts=False,
        )
        if any(episode.split != "train" for episode in scan.episodes):
            raise RuntimeError(
                "RoboDojo training materialization received a holdout episode"
            )
        if _file_sha256(official_split_path) != official_split_digest:
            raise RuntimeError(
                "RoboDojo official split changed during source scanning"
            )
        if args.smoke_one_per_train_task:
            assignments = load_official_split_assignments(official_split_path)
            episodes = _smoke_one_per_train_task(scan.episodes, assignments)
            selector = {
                "mode": "first_valid_episode_per_official_train_task",
                "selected_train_task_count": len(episodes),
                "selected_episode_ids": [
                    episode.canonical_episode_id for episode in episodes
                ],
            }
        else:
            episodes = (
                scan.episodes[: args.limit]
                if args.limit is not None
                else scan.episodes
            )
            selector = {
                "mode": (
                    "first_n_valid_train_episodes"
                    if args.limit is not None
                    else "all_valid_train_episodes"
                )
            }

        def iter_robodojo() -> Iterator[dict[str, Any]]:
            for converted, fixtures in bounded_ordered_map(
                _convert_robodojo_episode,
                episodes,
                workers=workers,
                max_in_flight=workers * 2,
            ):
                review_fixtures.extend(fixtures)
                yield from converted

        samples = iter_robodojo()
        selector["parallel"] = {
            "workers": workers,
            "max_in_flight": workers * 2,
            "grouping": "official_task_episode",
            "nice": 5,
        }
    manifest = materialize_dataset(
        samples,
        args.output,
        source=args.source,
        partial=partial,
        limit=args.limit,
        review_fixtures=review_fixtures,
        selector=selector,
        robodojo_official_split_path=official_split_path,
        expected_robodojo_official_split_sha256=official_split_digest,
        plan_cache_report=plan_cache_report,
        leaf_workers=workers,
        benchmark3_holdout=benchmark3_holdout,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "END_FIXTURE_SCHEMA_VERSION",
    "MATERIALIZATION_SCHEMA_VERSION",
    "convert_robodojo_episode_records",
    "_smoke_all_takeover_failure_types",
    "iter_takeover_samples",
    "materialize_dataset",
    "takeover_record_to_sample",
]
