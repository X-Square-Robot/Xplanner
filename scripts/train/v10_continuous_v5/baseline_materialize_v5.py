"""Materialize the pinned clean baseline into the strict V5 wire schema.

Only canonical records emitted by :mod:`baseline_adapter` are accepted.  Action
and Action+Segment episodes have enough evidence to build a complete Action
plan and therefore produce paired with/no-memory ongoing samples.  Segment-only
episodes retain their direct Segment supervision as no-memory samples; no
Action plan or memory is invented for them.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .baseline_adapter import (
    BaselineAdapter,
    EpisodeActionPlanCollector,
    SNAPSHOT_CONTENT_DIGEST,
    SNAPSHOT_VERSION,
)
from .materialize_v5 import (
    _prediction,
    _sample,
    _task_slug,
    _unit,
    materialize_dataset,
)
from .holdout_v5 import (
    Benchmark3Holdout,
    DEFAULT_BENCHMARK3_MANIFEST,
    DEFAULT_BENCHMARK3_SHA256,
)
from .parallel_v5 import (
    DEFAULT_CHUNK_BYTES,
    bounded_ordered_map,
    plan_jsonl_chunks,
    resolve_workers,
)
from .schema_v5 import SCHEMA_VERSION


SOURCE_NAME = "baseline"
CONVERTER_VERSION = "v5_baseline_materializer_v2"


def _stable_id(*parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _mode(record: Mapping[str, Any]) -> str:
    supervision = record.get("supervision")
    if not isinstance(supervision, Mapping):
        raise ValueError("baseline canonical record is missing supervision")
    predictions = supervision.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 2:
        raise ValueError("baseline canonical record requires two predictions")
    current = predictions[0]
    if not isinstance(current, Mapping):
        raise ValueError("baseline current prediction is invalid")
    action = current.get("action")
    segment = current.get("segment")
    action_available = isinstance(action, Mapping) and action.get("label_available") is True
    segment_available = isinstance(segment, Mapping) and segment.get("label_available") is True
    if action_available and segment_available:
        return "action_segment"
    if action_available:
        return "action"
    if segment_available:
        return "segment"
    raise ValueError("baseline current prediction has neither Action nor Segment")


def _task_name(instruction: str, mode: str) -> str:
    return f"{_task_slug(instruction)}__{mode}"


def _plan_from_canonical(
    record: Mapping[str, Any],
    *,
    mode: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    supervision = record.get("supervision")
    if not isinstance(supervision, Mapping):
        raise ValueError("baseline initial record is missing supervision")
    raw_plan = supervision.get("initial_plan")
    if not isinstance(raw_plan, list) or not raw_plan:
        raise ValueError("baseline initial record has no Action plan")
    plan: list[dict[str, Any]] = []
    masks: list[str] = []
    for offset, raw in enumerate(raw_plan):
        if not isinstance(raw, Mapping):
            raise ValueError("baseline initial plan item is invalid")
        action = raw.get("action")
        if not isinstance(action, Mapping) or action.get("label_available") is not True:
            raise ValueError("baseline initial plan Action is unavailable")
        segments: list[dict[str, Any]] = []
        if mode == "action_segment":
            raw_segments = raw.get("segments")
            if not isinstance(raw_segments, list):
                raise ValueError("baseline initial plan segments are invalid")
            for segment_offset, raw_segment in enumerate(raw_segments, 1):
                segment = raw_segment.get("segment") if isinstance(raw_segment, Mapping) else None
                if not isinstance(segment, Mapping) or segment.get("label_available") is not True:
                    raise ValueError("baseline initial plan Segment is unavailable")
                segments.append({
                    "index": segment_offset,
                    "segment": {"caption": str(segment.get("caption") or "")},
                })
        action_target: dict[str, Any] = {
            "caption": str(action.get("caption") or ""),
        }
        if mode == "action_segment":
            action_target["segments"] = segments
        plan.append({
            "index": offset + 1,
            "action": action_target,
        })
    return plan, masks


def convert_initial_plan(
    record: Mapping[str, Any],
    *,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if mode not in {"action", "action_segment"}:
        raise ValueError("only Action-bearing episodes can create an initial plan")
    plan, masks = _plan_from_canonical(record, mode=mode)
    episode_key = str(record.get("canonical_episode_id") or "")
    instruction = str(record.get("task_instruction") or "")
    split = str(record.get("split") or "")
    task_name = _task_name(instruction, mode)
    base_id = f"baseline_{_stable_id(episode_key, 'initial_plan', mode)}"
    provenance = {
        **dict(record.get("provenance") or {}),
        "episode_key": episode_key,
        "task_name": task_name,
        "split": split,
        "canonical_record_id": str(record.get("record_id") or ""),
        "canonical_source": "pinned_complete_baseline",
        "memory_pair_eligible": False,
        "clean_label_mode": mode,
    }
    sample = _sample(
        sample_id=f"{base_id}_no_memory",
        base_sample_id=base_id,
        source=SOURCE_NAME,
        category="initial_plan",
        memory_variant="no_memory",
        output_spec={
            "prediction1_units": [],
            "prediction2_units": [],
            "plan_units": (
                ["action", "segment"] if mode == "action_segment" else ["action"]
            ),
        },
        task_instruction=instruction,
        images=list(record.get("images") or ()),
        prompt_context={},
        target={"initial_plan": plan},
        loss_mask_paths=masks,
        provenance=provenance,
    )
    return sample, plan


def _target_unit(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping) or raw.get("label_available") is not True:
        return None
    return _unit(
        str(raw.get("caption") or ""),
        progress_percent=int(raw.get("progress_percent") or 0),
    )


def _target_and_masks(
    record: Mapping[str, Any],
    *,
    plan_units: list[str],
) -> tuple[dict[str, Any], list[str], dict[str, list[str]]]:
    supervision = record.get("supervision")
    if not isinstance(supervision, Mapping):
        raise ValueError("baseline ongoing record is missing supervision")
    raw_predictions = supervision.get("predictions")
    if not isinstance(raw_predictions, list) or len(raw_predictions) != 2:
        raise ValueError("baseline ongoing record requires two predictions")
    predictions: list[dict[str, Any]] = []
    masks = ["/execution_decision", "/decision_detail"]
    unit_profiles: list[list[str]] = []
    for offset, raw in enumerate(raw_predictions):
        if not isinstance(raw, Mapping):
            raise ValueError("baseline canonical prediction is invalid")
        action_raw = raw.get("action")
        segment_raw = raw.get("segment")
        action = _target_unit(action_raw)
        segment = _target_unit(segment_raw)
        predictions.append(_prediction(
            offset + 1,
            "current" if offset == 0 else "next",
            action=action,
            segment=segment,
        ))
        unit_profiles.append([
            name for name, value in (("action", action), ("segment", segment))
            if value is not None
        ])
    return {
        "task_progress_percent": int(supervision["task_progress_percent"]),
        "predictions": predictions,
        # The baseline is normal demonstration data, but it has no direct
        # four-way decision annotation.  Continue is only a schema placeholder
        # and both decision fields are removed from the token loss.
        "execution_decision": "Continue",
        "decision_detail": None,
    }, masks, {
        "prediction1_units": unit_profiles[0],
        "prediction2_units": unit_profiles[1],
        "plan_units": plan_units,
    }


def _long_memory(record: Mapping[str, Any], mode: str) -> list[dict[str, Any]]:
    history = record.get("history_material")
    if not isinstance(history, Mapping) or not isinstance(history.get("long"), list):
        raise ValueError("baseline ongoing record is missing long history")
    captions: list[str] = []
    seen: set[str] = set()
    for raw in history["long"]:
        caption = str((raw or {}).get("caption") or "") if isinstance(raw, Mapping) else ""
        identity = caption.casefold()
        if not caption or identity in seen:
            continue
        seen.add(identity)
        captions.append(caption)
    captions = captions[-8:]
    return [
        {
            "index": index,
            ("segment" if mode == "segment" else "action"): caption,
        }
        for index, caption in enumerate(captions, 1)
    ]


def _short_memory(
    record: Mapping[str, Any],
    *,
    plan_length: int,
    mode: str,
) -> dict[str, Any] | None:
    history = record.get("history_material")
    if not isinstance(history, Mapping) or not isinstance(history.get("long"), list):
        raise ValueError("baseline ongoing record is missing history material")
    completed = history["long"]
    if not completed:
        return None
    last = completed[-1]
    if not isinstance(last, Mapping) or not str(last.get("caption") or ""):
        return None
    completed_count = len(completed)
    task_progress = max(0, min(100, round(completed_count * 100 / max(plan_length, 1))))
    return {
        "task_progress_percent": task_progress,
        "prediction1": {
            ("segment" if mode == "segment" else "action"): {
                "available": True,
                "caption": str(last["caption"]),
                "progress_percent": 100,
            },
        },
    }


def convert_ongoing(
    record: Mapping[str, Any],
    *,
    initial_plan: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    mode = _mode(record)
    eligible = mode in {"action", "action_segment"} and bool(initial_plan)
    episode_key = str(record.get("canonical_episode_id") or "")
    instruction = str(record.get("task_instruction") or "")
    split = str(record.get("split") or "")
    task_name = _task_name(instruction, mode)
    plan_units = (
        ["action", "segment"]
        if initial_plan and "segments" in initial_plan[0]["action"]
        else ["action"]
    )
    target, masks, output_spec = _target_and_masks(
        record, plan_units=plan_units
    )
    base_id = f"baseline_{_stable_id(record.get('record_id'), mode)}"
    provenance = {
        **dict(record.get("provenance") or {}),
        "episode_key": episode_key,
        "task_name": task_name,
        "split": split,
        "canonical_record_id": str(record.get("record_id") or ""),
        "canonical_source": "pinned_complete_baseline",
        "memory_pair_eligible": eligible,
        "clean_label_mode": mode,
    }
    common = {
        "base_sample_id": base_id,
        "source": SOURCE_NAME,
        "category": "ongoing",
        "output_spec": output_spec,
        "task_instruction": instruction,
        "images": list(record.get("images") or ()),
        "target": target,
        "loss_mask_paths": masks,
        "provenance": provenance,
    }
    samples = [_sample(
        sample_id=f"{base_id}_no_memory",
        memory_variant="no_memory",
        prompt_context={},
        **common,
    )]
    if eligible:
        assert initial_plan is not None
        samples.append(_sample(
            sample_id=f"{base_id}_with_memory",
            memory_variant="with_memory",
            prompt_context={
                "initial_plan_memory": list(initial_plan),
                "long_memory": _long_memory(record, mode),
                "short_memory": _short_memory(
                    record, plan_length=len(initial_plan), mode=mode
                ),
            },
            **common,
        ))
    return samples


def _default_paths(adapter: BaselineAdapter, split: str) -> tuple[Path, ...]:
    # BaselineAdapter owns and validates the pinned snapshot contract.  This
    # private call is deliberately centralized here rather than duplicating its
    # legacy source directory names into V5 samples or manifests.
    return adapter._default_paths(split)  # noqa: SLF001


_WORKER_PLAN_CACHES: dict[str, dict[str, list[dict[str, Any]]]] = {}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_worker_plan_cache(path: str) -> dict[str, list[dict[str, Any]]]:
    cached = _WORKER_PLAN_CACHES.get(path)
    if cached is None:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("parallel baseline plan cache must be an object")
        cached = value
        _WORKER_PLAN_CACHES[path] = cached
    return cached


def _parallel_chunk_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt one newline-aligned chunk and atomically publish its fragment."""
    fragment = Path(str(task["fragment"]))
    metadata_path = fragment.with_suffix(".meta.json")
    cache_key = str(task["cache_key"])
    if fragment.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("cache_key") == cache_key
            and metadata.get("fragment_sha256") == _file_sha256(fragment)
        ):
            return {**metadata, "reused": True}

    adapter = BaselineAdapter()
    output_count = 0
    read_physical_lines = 0
    phase = str(task["phase"])
    plan_cache = (
        _load_worker_plan_cache(str(task["plan_cache"]))
        if phase == "pass2"
        else {}
    )
    fragment.parent.mkdir(parents=True, exist_ok=True)
    temporary = fragment.with_name(f".{fragment.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with Path(str(task["path"])).open("rb") as source, temporary.open("wb") as output:
        source.seek(int(task["start"]))
        while source.tell() < int(task["end"]):
            payload = source.readline()
            if not payload:
                break
            source_line = int(task["first_line"]) + read_physical_lines
            read_physical_lines += 1
            if not payload.strip():
                continue
            row = json.loads(payload)
            canonical = adapter.adapt_row(row, source_line=source_line)
            if canonical is None:
                continue
            if phase == "pass1":
                if _mode(canonical) == "segment":
                    continue
                values = (canonical,)
            elif phase == "pass2":
                values = convert_ongoing(
                    canonical,
                    initial_plan=plan_cache.get(
                        str(canonical["canonical_episode_id"])
                    ),
                )
            else:
                raise ValueError(f"unknown parallel baseline phase: {phase}")
            for value in values:
                output.write(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    + b"\n"
                )
                output_count += 1
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, fragment)
    metadata = {
        "schema_version": "v5_parallel_fragment_v1",
        "cache_key": cache_key,
        "phase": phase,
        "fragment": str(fragment),
        "fragment_sha256": _file_sha256(fragment),
        "output_count": output_count,
        "read_physical_lines": read_physical_lines,
        "adapter_statistics": adapter.statistics.to_dict(),
        "reused": False,
    }
    metadata_tmp = metadata_path.with_name(
        f".{metadata_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    with metadata_tmp.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(metadata_tmp, metadata_path)
    return metadata


def _aggregate_statistics(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    excluded: Counter[str] = Counter()
    read_rows = 0
    emitted = 0
    for result in results:
        statistics = result["adapter_statistics"]
        read_rows += int(statistics["read_rows"])
        emitted += int(statistics["emitted_ongoing_rows"])
        excluded.update(statistics["excluded_rows"])
    return {
        "read_rows": read_rows,
        "emitted_ongoing_rows": emitted,
        "excluded_rows": dict(sorted(excluded.items())),
    }


def _parallel_phase(
    *,
    paths: Sequence[Path],
    phase: str,
    cache_root: Path,
    workers: int,
    chunk_bytes: int,
    max_rows_per_path: int | None,
    plan_cache: Path | None = None,
) -> list[dict[str, Any]]:
    chunks = plan_jsonl_chunks(
        paths,
        chunk_bytes=chunk_bytes,
        max_rows_per_path=max_rows_per_path,
    )
    phase_root = cache_root / phase
    phase_root.mkdir(parents=True, exist_ok=True)
    plan_digest = _file_sha256(plan_cache) if plan_cache is not None else "none"
    tasks = []
    for index, chunk in enumerate(chunks):
        path = Path(chunk.path)
        stat = path.stat()
        key = hashlib.sha256(json.dumps({
            "converter_version": CONVERTER_VERSION,
            "schema_version": SCHEMA_VERSION,
            "phase": phase,
            "path": chunk.path,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "start": chunk.start,
            "end": chunk.end,
            "first_line": chunk.first_line,
            "num_lines": chunk.num_lines,
            "plan_digest": plan_digest,
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        tasks.append({
            **chunk.to_dict(),
            "phase": phase,
            "plan_cache": str(plan_cache) if plan_cache is not None else "",
            "cache_key": key,
            "fragment": str(phase_root / f"{index:06d}-{key[:16]}.jsonl"),
        })
    return list(bounded_ordered_map(
        _parallel_chunk_worker,
        tasks,
        workers=workers,
        max_in_flight=workers * 2,
    ))


def _iter_fragment_records(results: Sequence[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    for result in results:
        with Path(str(result["fragment"])).open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("parallel fragment row must be an object")
                    yield value


def _build_single_split_samples(
    *,
    split: str,
    max_rows_per_path: int | None,
    workers: int = 1,
    work_cache: Path | None = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> tuple[Iterator[dict[str, Any]], dict[str, Any]]:
    """Make a two-pass baseline stream and its compact plan-cache report."""
    if workers > 1:
        if work_cache is None:
            raise ValueError("parallel baseline build requires work_cache")
        return _build_parallel_single_split_samples(
            split=split,
            max_rows_per_path=max_rows_per_path,
            workers=workers,
            work_cache=work_cache,
            chunk_bytes=chunk_bytes,
        )
    adapter = BaselineAdapter()
    paths = _default_paths(adapter, split)
    collector = EpisodeActionPlanCollector(
        snapshot_version=SNAPSHOT_VERSION,
        snapshot_content_digest=SNAPSHOT_CONTENT_DIGEST,
    )
    episode_modes: dict[str, str] = {}
    episode_keys: set[str] = set()
    pass1_rows = 0
    for path in paths:
        for record in adapter.iter_ongoing(
            split=split,
            source_paths=(path,),
            max_rows=max_rows_per_path,
        ):
            pass1_rows += 1
            episode_keys.add(str(record["canonical_episode_id"]))
            mode = _mode(record)
            if mode != "segment":
                collector.add(record)
                episode = str(record["canonical_episode_id"])
                old = episode_modes.get(episode)
                episode_modes[episode] = (
                    "action_segment" if mode == "action_segment" or old == "action_segment" else "action"
                )

    initial_samples: list[dict[str, Any]] = []
    plan_by_episode: dict[str, list[dict[str, Any]]] = {}
    for record in collector.iter_records():
        episode = str(record["canonical_episode_id"])
        mode = episode_modes[episode]
        sample, plan = convert_initial_plan(record, mode=mode)
        initial_samples.append(sample)
        plan_by_episode[episode] = plan

    def stream() -> Iterator[dict[str, Any]]:
        yield from initial_samples
        second = BaselineAdapter()
        for path in paths:
            for record in second.iter_ongoing(
                split=split,
                source_paths=(path,),
                max_rows=max_rows_per_path,
            ):
                yield from convert_ongoing(
                    record,
                    initial_plan=plan_by_episode.get(
                        str(record["canonical_episode_id"])
                    ),
                )
        report["pass2_adapter_statistics"] = second.statistics.to_dict()

    report = {
        "schema_version": "v5_baseline_plan_cache_report_v2",
        "snapshot_version": SNAPSHOT_VERSION,
        "snapshot_content_digest": SNAPSHOT_CONTENT_DIGEST,
        "split": split,
        "source_paths": [str(path) for path in paths],
        "max_rows_per_path": max_rows_per_path,
        "pass1_canonical_rows": pass1_rows,
        "pass1_adapter_statistics": adapter.statistics.to_dict(),
        "pass2_adapter_statistics": None,
        "initial_plan_records": len(initial_samples),
        "collector_conflicts": collector.excluded_conflicts,
        "collector_incomplete": collector.excluded_incomplete,
        "_episode_keys": episode_keys,
    }
    return stream(), report


def _build_parallel_single_split_samples(
    *,
    split: str,
    max_rows_per_path: int | None,
    workers: int,
    work_cache: Path,
    chunk_bytes: int,
) -> tuple[Iterator[dict[str, Any]], dict[str, Any]]:
    """Run the same two passes with bounded, resumable chunk fragments."""
    adapter = BaselineAdapter()
    paths = _default_paths(adapter, split)
    pass1_paths = tuple(
        path for path in paths
        if not (split == "train" and path.parent.name == "L3L0")
    )
    split_cache = work_cache / split
    pass1_results = _parallel_phase(
        paths=pass1_paths,
        phase="pass1",
        cache_root=split_cache,
        workers=workers,
        chunk_bytes=chunk_bytes,
        max_rows_per_path=max_rows_per_path,
    )
    collector = EpisodeActionPlanCollector(
        snapshot_version=SNAPSHOT_VERSION,
        snapshot_content_digest=SNAPSHOT_CONTENT_DIGEST,
    )
    episode_modes: dict[str, str] = {}
    episode_keys: set[str] = set()
    for record in _iter_fragment_records(pass1_results):
        episode_keys.add(str(record["canonical_episode_id"]))
        collector.add(record)
        mode = _mode(record)
        episode = str(record["canonical_episode_id"])
        old = episode_modes.get(episode)
        episode_modes[episode] = (
            "action_segment"
            if mode == "action_segment" or old == "action_segment"
            else "action"
        )

    initial_samples: list[dict[str, Any]] = []
    plan_by_episode: dict[str, list[dict[str, Any]]] = {}
    for record in collector.iter_records():
        episode = str(record["canonical_episode_id"])
        sample, plan = convert_initial_plan(record, mode=episode_modes[episode])
        initial_samples.append(sample)
        plan_by_episode[episode] = plan

    plan_cache_path = split_cache / "plan_cache.json"
    plan_cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = plan_cache_path.with_name(
        f".{plan_cache_path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            plan_by_episode,
            handle,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, plan_cache_path)

    report = {
        "schema_version": "v5_baseline_plan_cache_report_v2",
        "snapshot_version": SNAPSHOT_VERSION,
        "snapshot_content_digest": SNAPSHOT_CONTENT_DIGEST,
        "split": split,
        "source_paths": [str(path) for path in paths],
        "max_rows_per_path": max_rows_per_path,
        "pass1_canonical_rows": sum(
            int(value["output_count"]) for value in pass1_results
        ),
        "pass1_adapter_statistics": _aggregate_statistics(pass1_results),
        "pass2_adapter_statistics": None,
        "initial_plan_records": len(initial_samples),
        "collector_conflicts": collector.excluded_conflicts,
        "collector_incomplete": collector.excluded_incomplete,
        "_episode_keys": episode_keys,
        "parallel": {
            "workers": workers,
            "chunk_bytes": chunk_bytes,
            "max_in_flight": workers * 2,
            "nice": 5,
            "pass1_segment_only_source_skipped": split == "train",
            "pass1_chunks": len(pass1_results),
            "pass1_reused_chunks": sum(bool(value["reused"]) for value in pass1_results),
            "work_cache": str(split_cache),
        },
    }

    def stream() -> Iterator[dict[str, Any]]:
        yield from initial_samples
        pass2_results = _parallel_phase(
            paths=paths,
            phase="pass2",
            cache_root=split_cache,
            workers=workers,
            chunk_bytes=chunk_bytes,
            max_rows_per_path=max_rows_per_path,
            plan_cache=plan_cache_path,
        )
        report["pass2_adapter_statistics"] = _aggregate_statistics(pass2_results)
        report["parallel"].update({
            "pass2_chunks": len(pass2_results),
            "pass2_reused_chunks": sum(
                bool(value["reused"]) for value in pass2_results
            ),
            "plan_cache_sha256": _file_sha256(plan_cache_path),
        })
        yield from _iter_fragment_records(pass2_results)

    return stream(), report


def build_samples(
    *,
    split: str,
    max_rows_per_path: int | None,
    workers: int = 1,
    work_cache: Path | None = None,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> tuple[Iterator[dict[str, Any]], dict[str, Any]]:
    """Build one split or one atomic train+validation source stream."""

    if split in {"train", "validation"}:
        stream, report = _build_single_split_samples(
            split=split,
            max_rows_per_path=max_rows_per_path,
            workers=workers,
            work_cache=work_cache,
            chunk_bytes=chunk_bytes,
        )
        report.pop("_episode_keys", None)
        return stream, report
    if split != "both":
        raise ValueError("split must be 'train', 'validation', or 'both'")

    validation_stream, validation_report = _build_single_split_samples(
        split="validation",
        max_rows_per_path=max_rows_per_path,
        workers=workers,
        work_cache=work_cache,
        chunk_bytes=chunk_bytes,
    )
    train_stream, train_report = _build_single_split_samples(
        split="train",
        max_rows_per_path=max_rows_per_path,
        workers=workers,
        work_cache=work_cache,
        chunk_bytes=chunk_bytes,
    )
    protected_validation_episodes = set(
        validation_report.pop("_episode_keys", set())
    )
    train_report.pop("_episode_keys", None)
    reports = {"train": train_report, "validation": validation_report}
    report = {
        "schema_version": "v5_baseline_plan_cache_report_v2",
        "snapshot_version": SNAPSHOT_VERSION,
        "snapshot_content_digest": SNAPSHOT_CONTENT_DIGEST,
        "split": "both",
        "source_paths": list(itertools.chain.from_iterable(
            value["source_paths"] for value in reports.values()
        )),
        "max_rows_per_path": max_rows_per_path,
        "pass1_canonical_rows": sum(
            int(value["pass1_canonical_rows"]) for value in reports.values()
        ),
        "initial_plan_records": sum(
            int(value["initial_plan_records"]) for value in reports.values()
        ),
        "collector_conflicts": sum(
            int(value["collector_conflicts"]) for value in reports.values()
        ),
        "collector_incomplete": sum(
            int(value["collector_incomplete"]) for value in reports.values()
        ),
        "split_protection": {
            "policy": "validation_episode_precedence",
            "protected_validation_episodes": len(protected_validation_episodes),
            "excluded_train_samples": 0,
        },
        "splits": reports,
    }

    def protected_train_stream() -> Iterator[dict[str, Any]]:
        for sample in train_stream:
            provenance = sample.get("provenance")
            episode_key = (
                str(provenance.get("episode_key") or "")
                if isinstance(provenance, Mapping)
                else ""
            )
            if episode_key in protected_validation_episodes:
                report["split_protection"]["excluded_train_samples"] += 1
                continue
            yield sample

    return itertools.chain(protected_train_stream(), validation_stream), report


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--benchmark3-manifest",
        type=Path,
        default=DEFAULT_BENCHMARK3_MANIFEST,
    )
    parser.add_argument(
        "--benchmark3-expected-sha256",
        default=DEFAULT_BENCHMARK3_SHA256,
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "both"), default="train"
    )
    parser.add_argument("--max-rows-per-path", type=_positive)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="CPU workers; zero selects min(32, max(4, affinity/4)).",
    )
    parser.add_argument("--chunk-mib", type=_positive, default=256)
    parser.add_argument("--work-cache", type=Path)
    args = parser.parse_args(argv)
    benchmark3_holdout = Benchmark3Holdout.load(
        args.benchmark3_manifest,
        expected_sha256=args.benchmark3_expected_sha256,
    )
    workers = resolve_workers(args.workers)
    work_cache = args.work_cache or args.output.with_name(
        f".{args.output.name}.parallel-cache"
    )
    samples, cache_report = build_samples(
        split=args.split,
        max_rows_per_path=args.max_rows_per_path,
        workers=workers,
        work_cache=work_cache,
        chunk_bytes=args.chunk_mib * 1024 * 1024,
    )
    manifest = materialize_dataset(
        samples,
        args.output,
        source=SOURCE_NAME,
        partial=args.max_rows_per_path is not None,
        limit=args.max_rows_per_path,
        plan_cache_report=cache_report,
        leaf_workers=workers,
        benchmark3_holdout=benchmark3_holdout,
    )
    print(json.dumps({
        "manifest": manifest,
        "plan_cache_report": cache_report,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONVERTER_VERSION",
    "SOURCE_NAME",
    "build_samples",
    "convert_initial_plan",
    "convert_ongoing",
]
