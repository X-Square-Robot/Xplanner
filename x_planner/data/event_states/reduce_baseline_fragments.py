"""Standalone parallel reducer for frozen V5 baseline pass2 fragments.

This module is intentionally outside :func:`snapshot._implementation_digests`.
It does not modify baseline pass1/pass2 fragments.  Instead, each cached JSONL
fragment is deterministically partitioned into one immutable sorted shard whose
manifest records byte ranges by :class:`indexed_io.LeafKey`.  Independent
leaf workers then read those ranges in original fragment order and delegate all
schema validation and indexed-leaf publication to the frozen V5 writer.

The CLI is a replacement *after* the frozen baseline materializer has stopped;
it is not a hot-switch mechanism.  ``--refuse-if-pid-alive`` is mandatory so a
caller must prove the old materializer is no longer using the shared cache.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, BinaryIO

from .baseline_adapter import (
    BaselineAdapter,
    EpisodeActionPlanCollector,
    SNAPSHOT_CONTENT_DIGEST,
    SNAPSHOT_VERSION,
    _stable_id as _canonical_stable_id,
)
from .materialize_baseline import (
    CONVERTER_VERSION,
    SOURCE_NAME,
    _aggregate_statistics,
    _default_paths,
    _iter_fragment_records,
    _mode,
    _parallel_phase,
    _stable_id,
    _task_name,
    convert_initial_plan,
)
from .indexed_io import IndexedLeafWriter, LeafKey, write_indexed_leaf
from .holdout import (
    EvaluationHoldout,
    DEFAULT_EVALUATION_MANIFEST,
    DEFAULT_EVALUATION_SHA256,
)
from .materialize import MATERIALIZATION_SCHEMA_VERSION
from .parallel import DEFAULT_CHUNK_BYTES, bounded_ordered_map, resolve_workers
from .schema import SCHEMA_VERSION, validate_sample


ACCELERATOR_SCHEMA_VERSION = "v5_baseline_fragment_reduce_v1"
SHARD_SCHEMA_VERSION = "v5_baseline_sorted_fragment_shard_v1"
_CACHED_FRAGMENT_RE = re.compile(
    r"^(?P<index>[0-9]{6})-(?P<prefix>[0-9a-f]{16})"
    r"(?:(?P<fragment>\.jsonl)|(?P<metadata>\.meta\.json))$"
)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    with temporary.open("wb") as handle:
        handle.write(_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _leaf_value(leaf: LeafKey) -> dict[str, str]:
    return asdict(leaf)


def _leaf_from_value(value: Mapping[str, Any]) -> LeafKey:
    return LeafKey(
        source=str(value["source"]),
        memory_variant=str(value["memory_variant"]),
        category=str(value["category"]),
        output_profile=str(value["output_profile"]),
        task=str(value["task"]),
        split=str(value["split"]),
    )


def _validate_fragment_metadata(fragment: Path) -> dict[str, Any]:
    metadata_path = fragment.with_suffix(".meta.json")
    if not fragment.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"cached fragment pair is incomplete: {fragment}")
    value = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"fragment metadata must be an object: {metadata_path}")
    if value.get("schema_version") != "v5_parallel_fragment_v1":
        raise ValueError(f"unexpected fragment schema: {metadata_path}")
    digest = value.get("fragment_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"cached fragment digest is invalid: {metadata_path}")
    # Content verification deliberately happens inside _partition_fragment so
    # all 256 MiB fragments are hashed in the bounded worker pool, never in a
    # serial coordinator pre-pass.
    return value


def _phase_cache_key(
    *,
    phase: str,
    path: Path,
    start: int,
    end: int,
    first_line: int,
    num_lines: int,
    plan_digest: str,
) -> str:
    stat = path.stat()
    return hashlib.sha256(json.dumps({
        "converter_version": CONVERTER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "start": start,
        "end": end,
        "first_line": first_line,
        "num_lines": num_lines,
        "plan_digest": plan_digest,
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _fast_chunk_byte_ranges(path: Path, chunk_bytes: int) -> list[tuple[int, int]]:
    """Rebuild newline-aligned ranges with O(chunks) boundary probes.

    Frozen baseline JSONL has no blank physical lines.  That invariant is
    independently checked against each fragment's read/adapter counts by the
    cache recovery path.  A cache containing blank lines fails closed because
    the reconstructed cache key cannot match.
    """

    size = path.stat().st_size
    ranges: list[tuple[int, int]] = []
    start = 0
    with path.open("rb") as handle:
        while start < size:
            threshold = start + chunk_bytes
            if threshold >= size:
                end = size
            else:
                handle.seek(threshold - 1)
                if handle.read(1) == b"\n":
                    end = threshold
                else:
                    handle.seek(threshold)
                    handle.readline()
                    end = handle.tell()
            if end <= start or end > size:
                raise RuntimeError(f"invalid cache-only chunk boundary: {path}")
            ranges.append((start, end))
            start = end
    return ranges


def _verify_cached_result(task: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(task["metadata"])
    fragment = Path(str(value["fragment"]))
    digest = hashlib.sha256()
    lines = 0
    final_byte = b""
    with fragment.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            lines += chunk.count(b"\n")
            final_byte = chunk[-1:]
    if digest.hexdigest() != value["fragment_sha256"]:
        raise RuntimeError(f"cached fragment digest mismatch: {fragment}")
    if fragment.stat().st_size and final_byte != b"\n":
        raise RuntimeError(f"cached fragment lacks final newline: {fragment}")
    if lines != int(value["output_count"]):
        raise RuntimeError(f"cached fragment output count mismatch: {fragment}")
    return {**value, "reused": True, "cache_only_verified": True}


def _recover_cached_phase(
    *,
    paths: Sequence[Path | str],
    phase: str,
    cache_root: Path,
    workers: int,
    chunk_bytes: int,
    plan_cache: Path | None = None,
    verify_fragments: bool = True,
) -> list[dict[str, Any]]:
    """Recover a frozen phase without sequentially enumerating raw JSONL."""

    if phase not in {"pass1", "pass2"}:
        raise ValueError(f"unsupported cached phase: {phase}")
    resolved_paths = tuple(Path(value).resolve() for value in paths)
    phase_root = cache_root / phase
    if not phase_root.is_dir():
        raise FileNotFoundError(f"cached phase is missing: {phase_root}")
    expected_chunks = [
        (path, start, end)
        for path in resolved_paths
        for start, end in _fast_chunk_byte_ranges(path, chunk_bytes)
    ]
    observed_files = [path for path in phase_root.iterdir() if path.is_file()]
    metadata_by_index: dict[int, Path] = {}
    fragments_by_index: dict[int, Path] = {}
    for path in observed_files:
        match = _CACHED_FRAGMENT_RE.fullmatch(path.name)
        if match is None:
            raise RuntimeError(f"unexpected cached phase artifact: {path}")
        index = int(match.group("index"))
        destination = (
            metadata_by_index if match.group("metadata") else fragments_by_index
        )
        if index in destination:
            raise RuntimeError(f"duplicate cached fragment index {index}: {phase_root}")
        destination[index] = path
    expected_indices = set(range(len(expected_chunks)))
    if set(metadata_by_index) != expected_indices or set(fragments_by_index) != expected_indices:
        raise RuntimeError(
            f"cached phase closure mismatch: {phase_root}; "
            f"expected indices 0..{len(expected_chunks) - 1}"
        )

    plan_digest = _sha256(plan_cache) if plan_cache is not None else "none"
    first_line_by_path = {path: 1 for path in resolved_paths}
    results: list[dict[str, Any]] = []
    for index, (path, start, end) in enumerate(expected_chunks):
        metadata_path = metadata_by_index[index]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise RuntimeError(f"cached fragment metadata is not an object: {metadata_path}")
        if metadata.get("schema_version") != "v5_parallel_fragment_v1":
            raise RuntimeError(f"cached fragment schema mismatch: {metadata_path}")
        if metadata.get("phase") != phase:
            raise RuntimeError(f"cached fragment phase mismatch: {metadata_path}")
        statistics = metadata.get("adapter_statistics")
        if not isinstance(statistics, Mapping):
            raise RuntimeError(f"cached fragment statistics missing: {metadata_path}")
        physical_lines = metadata.get("read_physical_lines")
        read_rows = statistics.get("read_rows")
        if (
            isinstance(physical_lines, bool)
            or not isinstance(physical_lines, int)
            or physical_lines <= 0
            or isinstance(read_rows, bool)
            or not isinstance(read_rows, int)
            or read_rows != physical_lines
        ):
            raise RuntimeError(
                "cache-only recovery requires nonblank complete JSONL chunks: "
                f"{metadata_path}"
            )
        first_line = first_line_by_path[path]
        key = _phase_cache_key(
            phase=phase,
            path=path,
            start=start,
            end=end,
            first_line=first_line,
            num_lines=read_rows,
            plan_digest=plan_digest,
        )
        fragment = fragments_by_index[index]
        expected_fragment = phase_root / f"{index:06d}-{key[:16]}.jsonl"
        expected_metadata = expected_fragment.with_suffix(".meta.json")
        if fragment != expected_fragment or metadata_path != expected_metadata:
            raise RuntimeError(f"cached fragment filename/key mismatch: {metadata_path}")
        if metadata.get("cache_key") != key:
            raise RuntimeError(f"cached fragment cache key mismatch: {metadata_path}")
        if Path(str(metadata.get("fragment") or "")) != expected_fragment:
            raise RuntimeError(f"cached fragment path mismatch: {metadata_path}")
        fragment_digest = metadata.get("fragment_sha256")
        if not isinstance(fragment_digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", fragment_digest
        ):
            raise RuntimeError(f"cached fragment digest metadata invalid: {metadata_path}")
        output_count = metadata.get("output_count")
        if isinstance(output_count, bool) or not isinstance(output_count, int) or output_count < 0:
            raise RuntimeError(f"cached fragment output count invalid: {metadata_path}")
        first_line_by_path[path] += physical_lines
        results.append({
            **metadata,
            "reused": True,
            "cache_only_chunk": {
                "source_path": str(path),
                "start": start,
                "end": end,
                "first_line": first_line,
                "num_lines": read_rows,
            },
        })
    if verify_fragments:
        return list(bounded_ordered_map(
            _verify_cached_result,
            ({"metadata": value} for value in results),
            workers=workers,
            max_in_flight=workers * 2,
        ))
    return results


def _recover_initial_samples_from_spool(
    spool_root: Path | str,
    *,
    split: str,
    plan_by_episode: Mapping[str, list[dict[str, Any]]],
    snapshot_version: str,
    snapshot_content_digest: str,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    """Recover and close an initial-plan set from explicit frozen staging."""

    root = Path(spool_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"frozen spool root is missing: {root}")
    by_episode: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            first = next((line for line in handle if line.strip()), None)
            if first is None:
                # An interrupted frozen writer may have opened a leaf spool
                # before emitting its first row.  Empty files carry no sample
                # identity; the exact episode-set closure below still makes a
                # missing initial-plan row fatal.
                continue
            first_value = json.loads(first)
            if not isinstance(first_value, Mapping):
                raise RuntimeError(f"invalid frozen spool row: {path}")
            if first_value.get("category") != "initial_plan":
                continue
            values = itertools.chain((first_value,), (
                json.loads(line) for line in handle if line.strip()
            ))
            for raw in values:
                sample = validate_sample(raw)
                if sample["source"] != SOURCE_NAME or sample["category"] != "initial_plan":
                    raise RuntimeError(f"mixed frozen initial spool leaf: {path}")
                provenance = sample["provenance"]
                episode = str(provenance.get("episode_key") or "")
                plan = plan_by_episode.get(episode)
                if plan is None:
                    raise RuntimeError(f"unexpected frozen initial episode: {episode}")
                if sample["target"].get("initial_plan") != plan:
                    raise RuntimeError(f"frozen initial plan mismatch: {episode}")
                mode = (
                    "action_segment"
                    if plan and "segments" in plan[0]["action"]
                    else "action"
                )
                base_id = f"baseline_{_stable_id(episode, 'initial_plan', mode)}"
                expected = {
                    "sample_id": f"{base_id}_no_memory",
                    "base_sample_id": base_id,
                    "task_name": _task_name(sample["task_instruction"], mode),
                    "canonical_record_id": (
                        f"v5_baseline_plan_{_canonical_stable_id(episode)}"
                    ),
                }
                if sample["sample_id"] != expected["sample_id"] or sample["base_sample_id"] != expected["base_sample_id"]:
                    raise RuntimeError(f"frozen initial deterministic ID mismatch: {episode}")
                if (
                    provenance.get("task_name") != expected["task_name"]
                    or provenance.get("canonical_record_id") != expected["canonical_record_id"]
                    or provenance.get("split") != split
                    or provenance.get("source_split") != split
                    or provenance.get("snapshot_version") != snapshot_version
                    or provenance.get("snapshot_content_digest") != snapshot_content_digest
                    or provenance.get("canonical_source") != "pinned_complete_baseline"
                    or provenance.get("memory_pair_eligible") is not False
                    or provenance.get("clean_label_mode") != mode
                ):
                    raise RuntimeError(f"frozen initial provenance mismatch: {episode}")
                source_count = provenance.get("source_sample_count")
                if isinstance(source_count, bool) or not isinstance(source_count, int) or source_count <= 0:
                    raise RuntimeError(f"frozen initial source count invalid: {episode}")
                if episode in by_episode:
                    raise RuntimeError(f"duplicate frozen initial episode: {episode}")
                by_episode[episode] = sample
    expected_episodes = set(plan_by_episode)
    if require_complete and set(by_episode) != expected_episodes:
        missing = sorted(expected_episodes - set(by_episode))[:5]
        raise RuntimeError(
            "frozen initial spool is not a complete plan-cache closure; "
            f"missing={missing}, observed={len(by_episode)}, expected={len(expected_episodes)}"
        )
    return [by_episode[episode] for episode in sorted(by_episode)]


def _extract_missing_initial_records(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    fragment = Path(str(task["fragment"]))
    missing = frozenset(str(value) for value in task["missing_episodes"])
    records: list[dict[str, Any]] = []
    with fragment.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"pass1 fragment row is not an object: {fragment}:{line_number}"
                )
            episode = str(value.get("canonical_episode_id") or "")
            if episode in missing:
                records.append(value)
    return records


def _recover_initial_samples_hybrid(
    spool_root: Path | str,
    *,
    split: str,
    plan_by_episode: Mapping[str, list[dict[str, Any]]],
    pass1_results: Sequence[Mapping[str, Any]],
    workers: int,
    snapshot_version: str,
    snapshot_content_digest: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Close an interrupted spool by collecting only its missing episodes."""

    spool_samples = _recover_initial_samples_from_spool(
        spool_root,
        split=split,
        plan_by_episode=plan_by_episode,
        snapshot_version=snapshot_version,
        snapshot_content_digest=snapshot_content_digest,
        require_complete=False,
    )
    by_episode = {
        str(value["provenance"]["episode_key"]): value
        for value in spool_samples
    }
    missing = set(plan_by_episode) - set(by_episode)
    if not missing:
        return spool_samples, {
            "schema_version": "v5_hybrid_initial_recovery_v1",
            "spool_samples": len(spool_samples),
            "missing_episodes": 0,
            "matched_pass1_records": 0,
            "supplemented_samples": 0,
            "complete_samples": len(spool_samples),
            "pass1_fragments_scanned": 0,
        }
    if any(value.get("cache_only_verified") is not True for value in pass1_results):
        raise RuntimeError(
            "hybrid initial recovery requires digest/count-verified pass1 fragments"
        )

    collector = EpisodeActionPlanCollector(
        snapshot_version=snapshot_version,
        snapshot_content_digest=snapshot_content_digest,
    )
    episode_modes: dict[str, str] = {}
    matched_records = 0
    missing_values = tuple(sorted(missing))
    tasks = ({
        "fragment": str(value["fragment"]),
        "missing_episodes": missing_values,
    } for value in pass1_results)
    for records in bounded_ordered_map(
        _extract_missing_initial_records,
        tasks,
        workers=workers,
        max_in_flight=workers * 2,
    ):
        for record in records:
            episode = str(record["canonical_episode_id"])
            if episode not in missing:
                raise RuntimeError(
                    f"hybrid pass1 worker returned an unexpected episode: {episode}"
                )
            collector.add(record)
            matched_records += 1
            mode = _mode(record)
            old = episode_modes.get(episode)
            episode_modes[episode] = (
                "action_segment"
                if mode == "action_segment" or old == "action_segment"
                else "action"
            )

    supplemented: dict[str, dict[str, Any]] = {}
    for record in collector.iter_records():
        episode = str(record["canonical_episode_id"])
        sample, plan = convert_initial_plan(record, mode=episode_modes[episode])
        if plan != plan_by_episode[episode]:
            raise RuntimeError(f"hybrid supplemented plan mismatch: {episode}")
        if episode in supplemented:
            raise RuntimeError(f"duplicate hybrid supplemented episode: {episode}")
        supplemented[episode] = sample
    if collector.excluded_conflicts or collector.excluded_incomplete:
        raise RuntimeError(
            "hybrid missing episodes were excluded by the production collector: "
            f"conflicts={collector.excluded_conflicts}, "
            f"incomplete={collector.excluded_incomplete}"
        )
    if set(supplemented) != missing:
        unresolved = sorted(missing - set(supplemented))[:5]
        raise RuntimeError(
            "hybrid initial recovery did not close all missing episodes; "
            f"unresolved={unresolved}, supplemented={len(supplemented)}, "
            f"missing={len(missing)}"
        )
    by_episode.update(supplemented)
    if set(by_episode) != set(plan_by_episode):
        raise RuntimeError("hybrid initial recovery final episode closure mismatch")
    samples = [by_episode[episode] for episode in sorted(by_episode)]
    return samples, {
        "schema_version": "v5_hybrid_initial_recovery_v1",
        "spool_samples": len(spool_samples),
        "missing_episodes": len(missing),
        "matched_pass1_records": matched_records,
        "supplemented_samples": len(supplemented),
        "complete_samples": len(samples),
        "pass1_fragments_scanned": len(pass1_results),
        "collector_conflicts": collector.excluded_conflicts,
        "collector_incomplete": collector.excluded_incomplete,
        "complete_episode_digest": _digest(sorted(by_episode)),
    }


def _shard_cache_key(
    *,
    fragment_sha256: str,
    excluded_episode_digest: str,
    evaluation_manifest_sha256: str | None = None,
) -> str:
    return _digest({
        "schema_version": SHARD_SCHEMA_VERSION,
        "fragment_sha256": fragment_sha256,
        "excluded_episode_digest": excluded_episode_digest,
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
    })


def _load_reusable_shard(
    manifest_path: Path,
    *,
    expected_cache_key: str,
) -> dict[str, Any] | None:
    if not manifest_path.is_file():
        return None
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("cache_key") != expected_cache_key:
        raise RuntimeError(
            f"owned shard cache exists with a different key: {manifest_path}"
        )
    shard = Path(str(value.get("shard") or ""))
    if not shard.is_file() or value.get("shard_sha256") != _sha256(shard):
        raise RuntimeError(f"owned shard cache is incomplete: {manifest_path}")
    return {**value, "reused": True}


def _partition_fragment(task: Mapping[str, Any]) -> dict[str, Any]:
    """Partition one immutable fragment into one sorted shard atomically."""

    fragment = Path(str(task["fragment"]))
    fragment_sha256 = str(task["fragment_sha256"])
    expected_output_count = int(task["expected_output_count"])
    if _sha256(fragment) != fragment_sha256:
        raise ValueError(f"fragment changed after planning: {fragment}")
    excluded = frozenset(str(value) for value in task.get("excluded_episodes", ()))
    excluded_digest = str(task["excluded_episode_digest"])
    if _digest(sorted(excluded)) != excluded_digest:
        raise ValueError("excluded episode set does not match its digest")
    evaluation_holdout_path = task.get("evaluation_manifest")
    evaluation_holdout_sha256 = task.get("evaluation_manifest_sha256")
    evaluation_holdout = (
        EvaluationHoldout.load(
            Path(str(evaluation_holdout_path)),
            expected_sha256=str(evaluation_holdout_sha256),
        )
        if evaluation_holdout_path is not None
        else None
    )
    cache_key = _shard_cache_key(
        fragment_sha256=fragment_sha256,
        excluded_episode_digest=excluded_digest,
        evaluation_manifest_sha256=(
            evaluation_holdout.manifest_sha256
            if evaluation_holdout is not None
            else None
        ),
    )
    cache_root = Path(str(task["shard_cache_root"]))
    fragment_index = int(task["fragment_index"])
    stem = f"{fragment_index:06d}-{cache_key[:16]}"
    shard = cache_root / f"{stem}.jsonl"
    manifest_path = cache_root / f"{stem}.manifest.json"
    reusable = _load_reusable_shard(
        manifest_path, expected_cache_key=cache_key
    )
    if reusable is not None:
        if int(reusable.get("input_samples", -1)) != expected_output_count:
            raise RuntimeError(
                f"cached fragment output count mismatch: {fragment}"
            )
        return reusable
    if shard.exists():
        raise RuntimeError(f"untracked owned shard cache exists: {shard}")

    rows_by_leaf: dict[LeafKey, list[bytes]] = defaultdict(list)
    excluded_samples = 0
    evaluation_holdout_excluded_samples = 0
    input_samples = 0
    with fragment.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            input_samples += 1
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"fragment row must be an object: {fragment}:{line_number}"
                )
            provenance = value.get("provenance")
            episode_key = (
                str(provenance.get("episode_key") or "")
                if isinstance(provenance, Mapping)
                else ""
            )
            if episode_key in excluded:
                excluded_samples += 1
                continue
            if evaluation_holdout is not None and evaluation_holdout.match_sample(value):
                evaluation_holdout_excluded_samples += 1
                continue
            leaf = LeafKey.from_sample(value)
            if leaf.source != SOURCE_NAME:
                raise ValueError(
                    f"baseline reducer received source {leaf.source!r}: {fragment}"
                )
            rows_by_leaf[leaf].append(raw_line.rstrip(b"\r\n") + b"\n")

    if input_samples != expected_output_count:
        raise RuntimeError(f"cached fragment output count mismatch: {fragment}")

    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = shard.with_name(
        f".{shard.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    ranges: list[dict[str, Any]] = []
    with temporary.open("wb") as handle:
        for leaf in sorted(rows_by_leaf):
            offset = handle.tell()
            count = 0
            for row in rows_by_leaf[leaf]:
                handle.write(row)
                count += 1
            ranges.append({
                "leaf": _leaf_value(leaf),
                "offset": offset,
                "length": handle.tell() - offset,
                "count": count,
            })
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, shard)
    _fsync_directory(cache_root)
    manifest = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "cache_key": cache_key,
        "fragment": str(fragment),
        "fragment_sha256": fragment_sha256,
        "fragment_index": fragment_index,
        "excluded_episode_digest": excluded_digest,
        "shard": str(shard),
        "shard_sha256": _sha256(shard),
        "input_samples": input_samples,
        "excluded_samples": excluded_samples,
        "evaluation_holdout_excluded_samples": evaluation_holdout_excluded_samples,
        "evaluation_manifest_sha256": (
            evaluation_holdout.manifest_sha256
            if evaluation_holdout is not None
            else None
        ),
        "output_samples": (
            input_samples - excluded_samples - evaluation_holdout_excluded_samples
        ),
        "ranges": ranges,
        "reused": False,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _iter_exact_jsonl_range(
    handle: BinaryIO,
    *,
    length: int,
    path: Path,
    offset: int,
) -> Iterator[dict[str, Any]]:
    remaining = length
    pending = b""
    while remaining:
        chunk = handle.read(min(remaining, 8 * 1024 * 1024))
        if not chunk:
            raise EOFError(f"truncated sorted shard slice: {path}@{offset}")
        remaining -= len(chunk)
        lines = (pending + chunk).split(b"\n")
        pending = lines.pop()
        for line in lines:
            sample = json.loads(line)
            if not isinstance(sample, dict):
                raise ValueError(f"sorted shard row must be an object: {path}")
            yield sample
    if pending:
        raise ValueError(
            f"sorted shard slice is not newline-aligned: {path}@{offset}"
        )


def _iter_slice_samples(slices: Sequence[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    for value in slices:
        path = Path(str(value["shard"]))
        offset = int(value["offset"])
        length = int(value["length"])
        # Every range is already an exact JSONL byte interval.  BufferedReader
        # would prefetch 8 KiB beyond tiny ranges and amplify the formal shard
        # scan by tens of times because a leaf opens one range per fragment.
        with path.open("rb", buffering=0) as handle:
            handle.seek(offset)
            yield from _iter_exact_jsonl_range(
                handle, length=length, path=path, offset=offset
            )


def _write_leaf(task: Mapping[str, Any]) -> dict[str, Any]:
    leaf = _leaf_from_value(task["leaf"])
    manifest = write_indexed_leaf(
        Path(str(task["root"])) / leaf.relative_path(),
        leaf,
        _iter_slice_samples(task["slices"]),
    )
    return {
        "path": leaf.relative_path().as_posix(),
        "num_samples": manifest["num_samples"],
        "num_episodes": manifest["num_episodes"],
        "leaf": manifest["leaf"],
        "canonical_raw_sources": manifest["canonical_raw_sources"],
    }


def _audit_leaf_memory_pair_eligibility(
    task: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit one original frozen LeafKey before any publication begins."""

    leaf = _leaf_from_value(task["leaf"])
    counts = {"eligible": 0, "ineligible": 0}
    examples: dict[str, dict[str, Any]] = {}
    observed = 0
    for sample in _iter_slice_samples(task["slices"]):
        observed_leaf = LeafKey.from_sample(sample)
        if observed_leaf != leaf:
            raise ValueError(
                f"sorted shard leaf mismatch: expected={leaf}, observed={observed_leaf}"
            )
        provenance = sample.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("V5 sample provenance must be an object")
        eligible = provenance.get("memory_pair_eligible")
        if not isinstance(eligible, bool):
            raise ValueError(
                "sample provenance.memory_pair_eligible must be boolean"
            )
        name = "eligible" if eligible else "ineligible"
        counts[name] += 1
        observed += 1
        if name not in examples:
            examples[name] = {
                "sample_id": str(sample.get("sample_id") or ""),
                "base_sample_id": str(sample.get("base_sample_id") or ""),
                "provenance": dict(provenance),
            }
    expected = sum(int(value["count"]) for value in task["slices"])
    if observed != expected:
        raise RuntimeError(
            f"eligibility audit sample count mismatch for {leaf}: "
            f"expected={expected}, observed={observed}"
        )
    if observed <= 0:
        raise ValueError(f"eligibility audit received empty leaf: {leaf}")
    return {
        "leaf": _leaf_value(leaf),
        "counts": counts,
        "examples": examples,
        "mixed": counts["eligible"] > 0 and counts["ineligible"] > 0,
    }


def _audit_shard_memory_pair_eligibility_counts(
    task: Mapping[str, Any],
) -> dict[str, Any]:
    """Sequentially audit one complete v1 shard, returning range-aligned counts."""

    manifest_path = Path(str(task["manifest_path"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"shard manifest must be an object: {manifest_path}")
    if manifest.get("schema_version") != SHARD_SCHEMA_VERSION:
        raise ValueError(f"unexpected shard schema: {manifest_path}")
    if manifest.get("cache_key") != task["cache_key"]:
        raise RuntimeError(f"shard cache key changed before audit: {manifest_path}")
    shard = Path(str(manifest.get("shard") or ""))
    if shard != Path(str(task["shard"])):
        raise RuntimeError(f"shard path changed before audit: {manifest_path}")
    if manifest.get("shard_sha256") != task["shard_sha256"]:
        raise RuntimeError(f"shard digest metadata changed before audit: {manifest_path}")
    ranges = manifest.get("ranges")
    if not isinstance(ranges, list) or not ranges:
        raise ValueError(f"shard manifest has no ranges: {manifest_path}")

    position = 0
    total_samples = 0
    range_counts: list[list[int]] = []
    # One persistent buffered handle is safe here: ranges must form a complete,
    # contiguous cover, so any read-ahead is consumed by the following range.
    with shard.open("rb", buffering=8 * 1024 * 1024) as handle:
        for value in ranges:
            if not isinstance(value, Mapping):
                raise ValueError(f"invalid shard range: {manifest_path}")
            offset = int(value["offset"])
            length = int(value["length"])
            expected_count = int(value["count"])
            if offset != position or length <= 0 or expected_count <= 0:
                raise RuntimeError(
                    f"shard ranges are not a positive contiguous cover: {manifest_path}"
                )
            leaf = _leaf_from_value(value["leaf"])
            eligible_count = 0
            ineligible_count = 0
            observed = 0
            for sample in _iter_exact_jsonl_range(
                handle, length=length, path=shard, offset=offset
            ):
                if LeafKey.from_sample(sample) != leaf:
                    raise ValueError(
                        f"sorted shard leaf mismatch during audit: {manifest_path}"
                    )
                provenance = sample.get("provenance")
                if not isinstance(provenance, Mapping):
                    raise ValueError("V5 sample provenance must be an object")
                eligible = provenance.get("memory_pair_eligible")
                if not isinstance(eligible, bool):
                    raise ValueError(
                        "sample provenance.memory_pair_eligible must be boolean"
                    )
                if eligible:
                    eligible_count += 1
                else:
                    ineligible_count += 1
                observed += 1
            if observed != expected_count:
                raise RuntimeError(
                    f"shard range sample count mismatch: {manifest_path}; "
                    f"expected={expected_count}, observed={observed}"
                )
            range_counts.append([eligible_count, ineligible_count])
            total_samples += observed
            position += length
        if handle.read(1):
            raise RuntimeError(f"shard ranges do not cover the complete file: {shard}")
    if position != shard.stat().st_size:
        raise RuntimeError(f"shard range byte closure mismatch: {shard}")
    if total_samples != int(manifest.get("output_samples", -1)):
        raise RuntimeError(f"shard audit output count mismatch: {manifest_path}")
    return {
        "fragment_index": int(manifest["fragment_index"]),
        "range_counts": range_counts,
        "total_samples": total_samples,
    }


def _audit_memory_pair_eligibility(
    shard_results: Sequence[Mapping[str, Any]],
    slices_by_leaf: Mapping[LeafKey, Sequence[Mapping[str, Any]]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    """Audit all samples once sequentially per shard, then detail mixed leaves."""

    ordered_leaves = sorted(slices_by_leaf)
    counts_by_leaf = {
        leaf: {"eligible": 0, "ineligible": 0} for leaf in ordered_leaves
    }
    tasks = ({
        "manifest_path": str(Path(str(value["shard"])).with_suffix(".manifest.json")),
        "shard": str(value["shard"]),
        "shard_sha256": str(value["shard_sha256"]),
        "cache_key": str(value["cache_key"]),
    } for value in shard_results)
    audited_shards = 0
    for expected, audit in zip(
        shard_results,
        bounded_ordered_map(
            _audit_shard_memory_pair_eligibility_counts,
            tasks,
            workers=workers,
            max_in_flight=workers * 2,
        ),
    ):
        if int(audit["fragment_index"]) != int(expected["fragment_index"]):
            raise RuntimeError("sequential shard audit order changed")
        ranges = expected["ranges"]
        range_counts = audit["range_counts"]
        if len(range_counts) != len(ranges):
            raise RuntimeError("sequential shard audit range closure mismatch")
        for value, pair in zip(ranges, range_counts):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or any(isinstance(count, bool) or not isinstance(count, int) or count < 0
                       for count in pair)
                or sum(pair) != int(value["count"])
            ):
                raise RuntimeError("invalid sequential shard eligibility counts")
            leaf = _leaf_from_value(value["leaf"])
            if leaf not in counts_by_leaf:
                raise RuntimeError("sequential shard audit produced an unknown leaf")
            counts_by_leaf[leaf]["eligible"] += pair[0]
            counts_by_leaf[leaf]["ineligible"] += pair[1]
        audited_shards += 1
    if audited_shards != len(shard_results):
        raise RuntimeError("sequential shard audit did not close every shard")

    reports = [{
        "leaf": _leaf_value(leaf),
        "counts": counts_by_leaf[leaf],
        "examples": {},
        "mixed": (
            counts_by_leaf[leaf]["eligible"] > 0
            and counts_by_leaf[leaf]["ineligible"] > 0
        ),
    } for leaf in ordered_leaves]
    conflict_tasks = ({
        "leaf": value["leaf"],
        "slices": slices_by_leaf[_leaf_from_value(value["leaf"])],
    } for value in reports if bool(value["mixed"]))
    detailed = list(bounded_ordered_map(
        _audit_leaf_memory_pair_eligibility,
        conflict_tasks,
        workers=workers,
        max_in_flight=workers * 2,
    ))
    detailed_by_leaf = {
        _leaf_from_value(value["leaf"]): value for value in detailed
    }
    for report in reports:
        if not bool(report["mixed"]):
            continue
        leaf = _leaf_from_value(report["leaf"])
        detail = detailed_by_leaf.get(leaf)
        if detail is None or detail["counts"] != report["counts"]:
            raise RuntimeError(
                "mixed leaf detail disagrees with sequential shard audit"
            )
        report["examples"] = detail["examples"]
    if len(detailed_by_leaf) != sum(bool(value["mixed"]) for value in reports):
        raise RuntimeError("mixed leaf detail closure mismatch")
    return reports


def _ineligible_partition_leaf(leaf: LeafKey) -> LeafKey:
    """Return a deterministic schema-safe task suffix for one mixed leaf."""

    suffix = f"__mpe_false_{_digest(_leaf_value(leaf))[:16]}"
    task_name = f"{leaf.task}{suffix}"
    if len(task_name.encode("utf-8")) > 255:
        raise ValueError(
            "memory eligibility partition task exceeds filesystem component limit: "
            f"{leaf.task!r}"
        )
    return LeafKey(
        source=leaf.source,
        memory_variant=leaf.memory_variant,
        category=leaf.category,
        output_profile=leaf.output_profile,
        task=task_name,
        split=leaf.split,
    )


def _leaf_summary(leaf: LeafKey, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": leaf.relative_path().as_posix(),
        "num_samples": manifest["num_samples"],
        "num_episodes": manifest["num_episodes"],
        "leaf": manifest["leaf"],
        "canonical_raw_sources": manifest["canonical_raw_sources"],
    }


def _write_leaf_with_eligibility_partition(
    task: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one pure leaf unchanged or split only an audited mixed leaf."""

    audit = task["eligibility_audit"]
    if not bool(audit["mixed"]):
        value = _write_leaf(task)
        expected = sum(int(count) for count in audit["counts"].values())
        if int(value["num_samples"]) != expected:
            raise RuntimeError("pure leaf count changed after eligibility audit")
        return {"leaves": [value], "retagged_samples": 0}

    original_leaf = _leaf_from_value(task["leaf"])
    ineligible_leaf = _leaf_from_value(task["ineligible_leaf"])
    expected_counts = {
        True: int(audit["counts"]["eligible"]),
        False: int(audit["counts"]["ineligible"]),
    }
    if not all(value > 0 for value in expected_counts.values()):
        raise RuntimeError("mixed leaf audit did not contain both eligibility values")
    writers: dict[bool, IndexedLeafWriter] = {}
    manifests: dict[bool, dict[str, Any]] = {}
    counts = {True: 0, False: 0}
    try:
        for sample in _iter_slice_samples(task["slices"]):
            if LeafKey.from_sample(sample) != original_leaf:
                raise ValueError("sample changed original leaf after eligibility audit")
            provenance = sample.get("provenance")
            if not isinstance(provenance, Mapping):
                raise ValueError("V5 sample provenance must be an object")
            eligible = provenance.get("memory_pair_eligible")
            if not isinstance(eligible, bool):
                raise ValueError(
                    "sample provenance.memory_pair_eligible must be boolean"
                )
            output_leaf = original_leaf if eligible else ineligible_leaf
            output_sample = sample
            if not eligible:
                output_sample = dict(sample)
                output_sample["provenance"] = {
                    **dict(provenance),
                    "task_name": ineligible_leaf.task,
                    "original_task_name": original_leaf.task,
                    "materialization_partition_reason": (
                        "mixed_memory_pair_eligibility_within_original_leaf"
                    ),
                }
            writer = writers.get(eligible)
            if writer is None:
                writer = IndexedLeafWriter(
                    Path(str(task["root"])) / output_leaf.relative_path(),
                    output_leaf,
                )
                writers[eligible] = writer
            writer.write(output_sample)
            counts[eligible] += 1
        if counts != expected_counts:
            raise RuntimeError(
                "memory eligibility changed between audit and leaf write: "
                f"expected={expected_counts}, observed={counts}"
            )
        for eligible in (True, False):
            manifests[eligible] = writers[eligible].close()
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise
    return {
        "leaves": [
            _leaf_summary(original_leaf, manifests[True]),
            _leaf_summary(ineligible_leaf, manifests[False]),
        ],
        "retagged_samples": counts[False],
    }


def materialize_fragment_shards(
    fragments: Sequence[Path | str],
    output_root: Path | str,
    *,
    shard_cache_root: Path | str,
    workers: int,
    excluded_episodes_by_fragment: Sequence[Iterable[str]] | None = None,
    plan_cache_report: Mapping[str, Any] | None = None,
    accelerator_module: Path | str | None = None,
    evaluation_holdout: EvaluationHoldout | None = None,
    partial: bool = False,
    selector: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically publish one baseline source from immutable cached fragments."""

    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"atomic output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    worker_count = resolve_workers(workers)
    paths = tuple(Path(value).resolve() for value in fragments)
    if not paths:
        raise ValueError("at least one cached fragment is required")
    excluded_values = (
        tuple(tuple(sorted(set(map(str, values)))) for values in excluded_episodes_by_fragment)
        if excluded_episodes_by_fragment is not None
        else tuple(() for _ in paths)
    )
    if len(excluded_values) != len(paths):
        raise ValueError("excluded episode sets must align with fragments")

    tasks: list[dict[str, Any]] = []
    for index, (fragment, excluded) in enumerate(zip(paths, excluded_values)):
        metadata = _validate_fragment_metadata(fragment)
        tasks.append({
            "fragment": str(fragment),
            "fragment_sha256": metadata["fragment_sha256"],
            "expected_output_count": metadata["output_count"],
            "fragment_index": index,
            "excluded_episodes": excluded,
            "excluded_episode_digest": _digest(list(excluded)),
            "shard_cache_root": str(Path(shard_cache_root).resolve()),
            "evaluation_manifest": (
                str(evaluation_holdout.manifest_path)
                if evaluation_holdout is not None
                else None
            ),
            "evaluation_manifest_sha256": (
                evaluation_holdout.manifest_sha256
                if evaluation_holdout is not None
                else None
            ),
        })
    shard_results = list(bounded_ordered_map(
        _partition_fragment,
        tasks,
        workers=worker_count,
        max_in_flight=worker_count * 2,
    ))

    slices_by_leaf: dict[LeafKey, list[dict[str, Any]]] = defaultdict(list)
    for shard_result in shard_results:
        shard_path = str(shard_result["shard"])
        for value in shard_result["ranges"]:
            leaf = _leaf_from_value(value["leaf"])
            slices_by_leaf[leaf].append({
                "shard": shard_path,
                "offset": int(value["offset"]),
                "length": int(value["length"]),
                "count": int(value["count"]),
            })
    expected_samples = sum(int(value["output_samples"]) for value in shard_results)
    if expected_samples <= 0:
        raise ValueError("fragment reduction produced no trainable V5 samples")

    ordered_leaves = sorted(slices_by_leaf)
    eligibility_audits = _audit_memory_pair_eligibility(
        shard_results,
        slices_by_leaf,
        workers=worker_count,
    )
    audited_samples = sum(
        int(value["counts"]["eligible"])
        + int(value["counts"]["ineligible"])
        for value in eligibility_audits
    )
    if audited_samples != expected_samples:
        raise RuntimeError(
            "fragment and eligibility-audit sample counts disagree: "
            f"{expected_samples} != {audited_samples}"
        )
    original_leaves = set(ordered_leaves)
    partition_leaf_by_original: dict[LeafKey, LeafKey] = {}
    generated_leaves: set[LeafKey] = set()
    for leaf, audit in zip(ordered_leaves, eligibility_audits):
        if not bool(audit["mixed"]):
            continue
        partition_leaf = _ineligible_partition_leaf(leaf)
        if partition_leaf in original_leaves or partition_leaf in generated_leaves:
            raise RuntimeError(
                "memory eligibility partition leaf collides with an existing leaf: "
                f"{partition_leaf}"
            )
        generated_leaves.add(partition_leaf)
        partition_leaf_by_original[leaf] = partition_leaf
        audit["partitioned_ineligible_leaf"] = _leaf_value(partition_leaf)
    conflicts = [value for value in eligibility_audits if bool(value["mixed"])]
    expected_retagged_samples = sum(
        int(value["counts"]["ineligible"]) for value in conflicts
    )

    staging = output_root.with_name(
        f".{output_root.name}.fragment-reduce.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir()
    try:
        leaf_tasks = ({
            "leaf": _leaf_value(leaf),
            "root": str(staging),
            "slices": slices_by_leaf[leaf],
            "eligibility_audit": audit,
            "ineligible_leaf": _leaf_value(partition_leaf_by_original[leaf])
            if leaf in partition_leaf_by_original
            else None,
        } for leaf, audit in zip(ordered_leaves, eligibility_audits))
        leaf_groups = list(bounded_ordered_map(
            _write_leaf_with_eligibility_partition,
            leaf_tasks,
            workers=worker_count,
            max_in_flight=worker_count * 2,
        ))
        leaf_results = sorted(
            itertools.chain.from_iterable(value["leaves"] for value in leaf_groups),
            key=lambda value: value["path"],
        )
        retagged_samples = sum(
            int(value["retagged_samples"]) for value in leaf_groups
        )
        if retagged_samples != expected_retagged_samples:
            raise RuntimeError(
                "mixed-leaf retag count disagrees with eligibility audit: "
                f"{expected_retagged_samples} != {retagged_samples}"
            )
        observed_samples = sum(int(value["num_samples"]) for value in leaf_results)
        if observed_samples != expected_samples:
            raise RuntimeError(
                "fragment and indexed-leaf sample counts disagree: "
                f"{expected_samples} != {observed_samples}"
            )
        module_path = Path(accelerator_module or __file__).resolve()
        accelerator = {
            "schema_version": ACCELERATOR_SCHEMA_VERSION,
            "module": str(module_path),
            "sha256": _sha256(module_path),
            "shard_schema_version": SHARD_SCHEMA_VERSION,
            "workers": worker_count,
            "num_input_fragments": len(paths),
            "num_reused_shards": sum(bool(value["reused"]) for value in shard_results),
            "eligibility_audit_mode": (
                "single_sequential_pass_per_shard_plus_mixed_leaf_examples"
            ),
            "mixed_leaf_count": len(conflicts),
            "retagged_samples": retagged_samples,
            "memory_pair_eligibility_partition": {
                "policy": "partition_ineligible_rows_only_for_mixed_original_leaf",
                "audited_original_leaf_count": len(eligibility_audits),
                "audited_samples": audited_samples,
                "mixed_leaf_count": len(conflicts),
                "retagged_samples": retagged_samples,
                "conflicts": conflicts,
            },
            "input_fragment_digest": _digest([
                value["fragment_sha256"] for value in shard_results
            ]),
        }
        canonical_sources = sorted(set(itertools.chain.from_iterable(
            value["canonical_raw_sources"] for value in leaf_results
        )))
        leaves = [{key: value[key] for key in (
            "path", "num_samples", "num_episodes", "leaf"
        )} for value in leaf_results]
        manifest = {
            "schema_version": MATERIALIZATION_SCHEMA_VERSION,
            "complete": True,
            "source": SOURCE_NAME,
            "partial": bool(partial),
            "limit": (len(paths) if partial else None),
            "selector": dict(selector or {
                "mode": (
                    "explicit_cached_fragment_prefix" if partial else "all_source_units"
                )
            }),
            "num_samples": observed_samples,
            "num_leaves": len(leaves),
            "num_review_fixtures": 0,
            "canonical_raw_sources": canonical_sources,
            "leaves": leaves,
            "accelerator": accelerator,
        }
        if evaluation_holdout is not None:
            evaluation_holdout_excluded = sum(
                int(value.get("evaluation_holdout_excluded_samples", 0))
                for value in shard_results
            )
            holdout_report = {
                "schema_version": "v5_evaluation_holdout_baseline_reduce_audit_v1",
                "passed": True,
                "holdout": evaluation_holdout.metadata(),
                "policy": "exclude_during_fragment_to_sorted_shard_partition",
                "checked_input_samples": sum(
                    int(value["input_samples"]) for value in shard_results
                ),
                "excluded_input_samples": evaluation_holdout_excluded,
                "published_overlap_samples": 0,
                "published_samples": observed_samples,
            }
            holdout_relative = Path("metadata", "evaluation_holdout_report.json")
            _write_json_atomic(staging / holdout_relative, holdout_report)
            manifest["evaluation_holdout"] = {
                **evaluation_holdout.metadata(),
                "policy": "exclude_during_fragment_to_sorted_shard_partition",
                "checked_input_samples": holdout_report["checked_input_samples"],
                "excluded_input_samples": evaluation_holdout_excluded,
                "published_overlap_samples": 0,
                "report_relative_path": holdout_relative.as_posix(),
                "report_sha256": _sha256(staging / holdout_relative),
            }
        if plan_cache_report is not None:
            report_value = dict(plan_cache_report)
            split_protection = report_value.get("split_protection")
            if isinstance(split_protection, Mapping):
                report_value["split_protection"] = {
                    **dict(split_protection),
                    "excluded_train_samples": sum(
                        int(value["excluded_samples"]) for value in shard_results
                    ),
                }
            _write_json_atomic(staging / "plan_cache_report.json", report_value)
        _write_json_atomic(staging / "manifest.json", manifest)
        _fsync_directory(staging)
        os.replace(staging, output_root)
        _fsync_directory(output_root.parent)
        return {**manifest, "output_root": str(output_root)}
    except BaseException:
        # This prefix is owned exclusively by this accelerator.  Frozen V5
        # staging trees and the shard/pass2 caches are never removed here.
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _write_initial_fragment(
    samples: Sequence[Mapping[str, Any]],
    path: Path,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_key = _digest({"initial_samples": list(samples)})
    metadata_path = path.with_suffix(".meta.json")
    if path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            isinstance(metadata, dict)
            and metadata.get("cache_key") == expected_key
            and metadata.get("fragment_sha256") == _sha256(path)
        ):
            return {**metadata, "reused": True}
        raise RuntimeError(f"owned initial fragment cache has a different key: {path}")
    if path.exists() or metadata_path.exists():
        raise RuntimeError(f"owned initial fragment cache is incomplete: {path}")
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    with temporary.open("wb") as handle:
        for sample in samples:
            handle.write(_json_bytes(sample) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    metadata = {
        "schema_version": "v5_parallel_fragment_v1",
        "cache_key": expected_key,
        "phase": "initial",
        "fragment": str(path),
        "fragment_sha256": _sha256(path),
        "output_count": len(samples),
        "read_physical_lines": len(samples),
        "adapter_statistics": {
            "read_rows": len(samples),
            "emitted_ongoing_rows": 0,
            "excluded_rows": {},
        },
    }
    _write_json_atomic(metadata_path, metadata)
    return metadata


def _prepare_split(
    *,
    split: str,
    work_cache: Path,
    workers: int,
    chunk_bytes: int,
    cache_only: bool = False,
    initial_spool_root: Path | None = None,
) -> tuple[list[dict[str, Any]], set[str], list[dict[str, Any]], dict[str, Any]]:
    if initial_spool_root is not None and (not cache_only or split != "train"):
        raise ValueError(
            "frozen initial spool reuse is supported only for cache-only train"
        )
    adapter = BaselineAdapter()
    paths = _default_paths(adapter, split)
    pass1_paths = tuple(
        path for path in paths
        if not (split == "train" and path.parent.name == "L3L0")
    )
    split_cache = work_cache / split
    plan_cache = split_cache / "plan_cache.json"
    if not plan_cache.is_file():
        raise FileNotFoundError(
            f"frozen baseline plan cache is required; refusing to create it: {plan_cache}"
        )
    cached_plan = json.loads(plan_cache.read_text(encoding="utf-8"))
    if not isinstance(cached_plan, dict):
        raise RuntimeError(f"frozen baseline plan cache is not an object: {split}")
    if cache_only:
        pass1_results = _recover_cached_phase(
            paths=pass1_paths,
            phase="pass1",
            cache_root=split_cache,
            workers=workers,
            chunk_bytes=chunk_bytes,
        )
    else:
        pass1_results = _parallel_phase(
            paths=pass1_paths,
            phase="pass1",
            cache_root=split_cache,
            workers=workers,
            chunk_bytes=chunk_bytes,
            max_rows_per_path=None,
        )

    collector: EpisodeActionPlanCollector | None = None
    episode_keys: set[str]
    initial_samples: list[dict[str, Any]]
    plan_by_episode: dict[str, list[dict[str, Any]]]
    initial_recovery: dict[str, Any] | None = None
    if initial_spool_root is not None:
        plan_by_episode = cached_plan
        initial_samples, hybrid_report = _recover_initial_samples_hybrid(
            initial_spool_root,
            split=split,
            plan_by_episode=plan_by_episode,
            pass1_results=pass1_results,
            workers=workers,
            snapshot_version=SNAPSHOT_VERSION,
            snapshot_content_digest=SNAPSHOT_CONTENT_DIGEST,
        )
        episode_keys = set(plan_by_episode)
        initial_recovery = {
            **hybrid_report,
            "spool_root": str(initial_spool_root.resolve()),
            "plan_cache_sha256": _sha256(plan_cache),
            "num_samples": len(initial_samples),
            "sample_digest": _digest([
                value["sample_id"] for value in initial_samples
            ]),
            "closure": (
                "validated_spool_plus_production_collector_supplement_"
                "plan_ids_snapshot_and_episode_set"
            ),
        }
    else:
        collector = EpisodeActionPlanCollector(
            snapshot_version=SNAPSHOT_VERSION,
            snapshot_content_digest=SNAPSHOT_CONTENT_DIGEST,
        )
        episode_modes: dict[str, str] = {}
        episode_keys = set()
        for record in _iter_fragment_records(pass1_results):
            episode = str(record["canonical_episode_id"])
            episode_keys.add(episode)
            collector.add(record)
            mode = _mode(record)
            old = episode_modes.get(episode)
            episode_modes[episode] = (
                "action_segment"
                if mode == "action_segment" or old == "action_segment"
                else "action"
            )
        initial_samples = []
        plan_by_episode = {}
        for record in collector.iter_records():
            episode = str(record["canonical_episode_id"])
            sample, plan = convert_initial_plan(record, mode=episode_modes[episode])
            initial_samples.append(sample)
            plan_by_episode[episode] = plan
        if cached_plan != plan_by_episode:
            raise RuntimeError(
                f"frozen baseline plan cache does not match pass1: {split}"
            )

    if cache_only:
        pass2_results = _recover_cached_phase(
            paths=paths,
            phase="pass2",
            cache_root=split_cache,
            workers=workers,
            chunk_bytes=chunk_bytes,
            plan_cache=plan_cache,
            # materialize_fragment_shards validates every pass2 digest and
            # output count while producing its shard; avoid reading it twice.
            verify_fragments=False,
        )
    else:
        pass2_results = _parallel_phase(
            paths=paths,
            phase="pass2",
            cache_root=split_cache,
            workers=workers,
            chunk_bytes=chunk_bytes,
            max_rows_per_path=None,
            plan_cache=plan_cache,
        )
    report = {
        "schema_version": "v5_baseline_plan_cache_report_v2",
        "snapshot_version": SNAPSHOT_VERSION,
        "snapshot_content_digest": SNAPSHOT_CONTENT_DIGEST,
        "split": split,
        "source_paths": [str(path) for path in paths],
        "max_rows_per_path": None,
        "pass1_canonical_rows": sum(int(value["output_count"]) for value in pass1_results),
        "pass1_adapter_statistics": _aggregate_statistics(pass1_results),
        "pass2_adapter_statistics": _aggregate_statistics(pass2_results),
        "initial_plan_records": len(initial_samples),
        "collector_conflicts": (
            collector.excluded_conflicts if collector is not None else None
        ),
        "collector_incomplete": (
            collector.excluded_incomplete if collector is not None else None
        ),
        "parallel": {
            "workers": workers,
            "chunk_bytes": chunk_bytes,
            "max_in_flight": workers * 2,
            "nice": 5,
            "pass1_segment_only_source_skipped": split == "train",
            "pass1_chunks": len(pass1_results),
            "pass1_reused_chunks": sum(bool(value["reused"]) for value in pass1_results),
            "pass2_chunks": len(pass2_results),
            "pass2_reused_chunks": sum(bool(value["reused"]) for value in pass2_results),
            "plan_cache_sha256": _sha256(plan_cache),
            "work_cache": str(split_cache),
            "cache_only": cache_only,
            "cache_only_raw_boundary_reads": (
                "one random newline probe per chunk" if cache_only else None
            ),
            "pass2_fragment_verification": (
                "deferred_to_sorted_shard_partition" if cache_only else "phase_worker"
            ),
        },
    }
    if initial_recovery is not None:
        report["initial_sample_recovery"] = initial_recovery
    return initial_samples, episode_keys, pass2_results, report


def _assert_pid_inactive(pid: int) -> None:
    if pid <= 0:
        raise ValueError("--refuse-if-pid-alive must be a positive PID")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    except PermissionError:
        # A permission failure still proves that the process exists.
        pass
    else:
        pass
    if Path(f"/proc/{pid}").exists() or pid == os.getpid():
        raise RuntimeError(
            f"refusing to share the baseline cache with live PID {pid}; "
            "this accelerator cannot hot-switch"
        )
    # Some container runtimes hide /proc entries from the Python mount namespace
    # even though kill(pid, 0) observes the live process.
    raise RuntimeError(
        f"refusing to share the baseline cache with live PID {pid}; "
        "this accelerator cannot hot-switch"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-cache", type=Path, required=True)
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        default=DEFAULT_EVALUATION_MANIFEST,
    )
    parser.add_argument(
        "--evaluation-expected-sha256",
        default=DEFAULT_EVALUATION_SHA256,
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--chunk-mib", type=int, default=256)
    parser.add_argument(
        "--max-pass2-fragments-per-split",
        type=int,
        help=(
            "Publish only this deterministic cached pass2 prefix for each split; "
            "the resulting source is explicitly partial and can be composed early."
        ),
    )
    parser.add_argument(
        "--max-initial-samples-per-split",
        type=int,
        help="Bound initial-plan rows in an explicitly partial early publication.",
    )
    parser.add_argument("--refuse-if-pid-alive", type=int, required=True)
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Recover complete pass1/pass2 caches using random boundary probes.",
    )
    parser.add_argument(
        "--verify-cache-only-no-publish",
        action="store_true",
        help=(
            "Validate the frozen pass1/pass2 cache and initial spool closure, "
            "print the report, and stop before shard/source publication."
        ),
    )
    parser.add_argument(
        "--train-initial-spool-root",
        type=Path,
        help=(
            "Cache-only train initial plans from an explicit frozen .spool; "
            "the complete plan-cache closure is validated before use."
        ),
    )
    args = parser.parse_args(argv)
    evaluation_holdout = EvaluationHoldout.load(
        args.evaluation_manifest,
        expected_sha256=args.evaluation_expected_sha256,
    )
    _assert_pid_inactive(args.refuse_if_pid_alive)
    if args.train_initial_spool_root is not None and not args.cache_only:
        raise ValueError("--train-initial-spool-root requires --cache-only")
    if args.verify_cache_only_no_publish and not args.cache_only:
        raise ValueError("--verify-cache-only-no-publish requires --cache-only")
    workers = resolve_workers(args.workers)
    if args.chunk_mib <= 0:
        raise ValueError("--chunk-mib must be positive")
    for name in (
        "max_pass2_fragments_per_split",
        "max_initial_samples_per_split",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    chunk_bytes = args.chunk_mib * 1024 * 1024
    if chunk_bytes != DEFAULT_CHUNK_BYTES:
        raise ValueError(
            "formal cached baseline build requires the frozen 256 MiB chunk size"
        )

    validation = _prepare_split(
        split="validation",
        work_cache=args.work_cache,
        workers=workers,
        chunk_bytes=chunk_bytes,
        cache_only=args.cache_only,
    )
    train = _prepare_split(
        split="train",
        work_cache=args.work_cache,
        workers=workers,
        chunk_bytes=chunk_bytes,
        cache_only=args.cache_only,
        initial_spool_root=args.train_initial_spool_root,
    )
    validation_initial, validation_episodes, validation_pass2, validation_report = validation
    train_initial, _, train_pass2, train_report = train

    if args.verify_cache_only_no_publish:
        verification = {
            "schema_version": "v5_baseline_frozen_cache_verification_v1",
            "passed": True,
            "published": False,
            "snapshot_version": SNAPSHOT_VERSION,
            "snapshot_content_digest": SNAPSHOT_CONTENT_DIGEST,
            "work_cache": str(args.work_cache.resolve()),
            "train_initial_spool_root": (
                str(args.train_initial_spool_root.resolve())
                if args.train_initial_spool_root is not None
                else None
            ),
            "train_initial_samples": len(train_initial),
            "validation_initial_samples": len(validation_initial),
            "train_pass2_fragments": len(train_pass2),
            "validation_pass2_fragments": len(validation_pass2),
            "train": train_report,
            "validation": validation_report,
            "evaluation_holdout": evaluation_holdout.metadata(),
        }
        print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    partial = (
        args.max_pass2_fragments_per_split is not None
        or args.max_initial_samples_per_split is not None
    )
    original_counts = {
        "train_initial_samples": len(train_initial),
        "validation_initial_samples": len(validation_initial),
        "train_pass2_fragments": len(train_pass2),
        "validation_pass2_fragments": len(validation_pass2),
    }
    if args.max_initial_samples_per_split is not None:
        train_initial = train_initial[: args.max_initial_samples_per_split]
        validation_initial = validation_initial[: args.max_initial_samples_per_split]
    if args.max_pass2_fragments_per_split is not None:
        train_pass2 = train_pass2[: args.max_pass2_fragments_per_split]
        validation_pass2 = validation_pass2[: args.max_pass2_fragments_per_split]
    selected_counts = {
        "train_initial_samples": len(train_initial),
        "validation_initial_samples": len(validation_initial),
        "train_pass2_fragments": len(train_pass2),
        "validation_pass2_fragments": len(validation_pass2),
    }

    initial_root = args.work_cache / ".fragment-reduce-v1" / "initial"
    train_initial_path = initial_root / "train.jsonl"
    validation_initial_path = initial_root / "validation.jsonl"
    _write_initial_fragment(train_initial, train_initial_path)
    _write_initial_fragment(validation_initial, validation_initial_path)

    fragments = [train_initial_path]
    fragments.extend(Path(str(value["fragment"])) for value in train_pass2)
    fragments.append(validation_initial_path)
    fragments.extend(Path(str(value["fragment"])) for value in validation_pass2)
    exclusions: list[Iterable[str]] = [validation_episodes] * (len(train_pass2) + 1)
    exclusions.extend([()] * (len(validation_pass2) + 1))
    report = {
        "schema_version": "v5_baseline_plan_cache_report_v2",
        "snapshot_version": SNAPSHOT_VERSION,
        "snapshot_content_digest": SNAPSHOT_CONTENT_DIGEST,
        "split": "both",
        "source_paths": train_report["source_paths"] + validation_report["source_paths"],
        "max_rows_per_path": None,
        "pass1_canonical_rows": train_report["pass1_canonical_rows"] + validation_report["pass1_canonical_rows"],
        "initial_plan_records": train_report["initial_plan_records"] + validation_report["initial_plan_records"],
        "collector_conflicts": (
            train_report["collector_conflicts"] + validation_report["collector_conflicts"]
            if isinstance(train_report["collector_conflicts"], int)
            and isinstance(validation_report["collector_conflicts"], int)
            else None
        ),
        "collector_incomplete": (
            train_report["collector_incomplete"] + validation_report["collector_incomplete"]
            if isinstance(train_report["collector_incomplete"], int)
            and isinstance(validation_report["collector_incomplete"], int)
            else None
        ),
        "split_protection": {
            "policy": "validation_episode_precedence",
            "protected_validation_episodes": len(validation_episodes),
            "excluded_train_samples": None,
        },
        "splits": {"train": train_report, "validation": validation_report},
        "accelerator_converter_version": CONVERTER_VERSION,
        "partial_prefix_selection": {
            "enabled": partial,
            "original_counts": original_counts,
            "selected_counts": selected_counts,
            "max_pass2_fragments_per_split": args.max_pass2_fragments_per_split,
            "max_initial_samples_per_split": args.max_initial_samples_per_split,
            "can_recompose_after_more_fragments": True,
        },
    }
    manifest = materialize_fragment_shards(
        fragments,
        args.output,
        shard_cache_root=args.work_cache / ".fragment-reduce-v1" / "shards",
        workers=workers,
        excluded_episodes_by_fragment=exclusions,
        plan_cache_report=report,
        evaluation_holdout=evaluation_holdout,
        partial=partial,
        selector={
            "mode": (
                "cached_fragment_prefix_for_early_training"
                if partial
                else "all_source_units"
            ),
            "original_counts": original_counts,
            "selected_counts": selected_counts,
            "can_recompose_after_more_fragments": True,
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACCELERATOR_SCHEMA_VERSION",
    "SHARD_SCHEMA_VERSION",
    "materialize_fragment_shards",
]
