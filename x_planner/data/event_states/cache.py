"""Resumable, Evaluation holdout-fenced V5 fragments for RoboDojo and Takeover-Q.

Final indexed source builders publish atomically, so this module adds an
immutable pre-publication seam: bounded episode batches become JSONL, metadata,
hash, and READY files.  Only complete READY fragments can be sealed into an
indexed partial or full source build.  Media is referenced directly; RoboDojo
uses task instruction indexes and Takeover-Q uses its pinned episode index.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .holdout import (
    EvaluationHoldout,
    DEFAULT_EVALUATION_MANIFEST,
    DEFAULT_EVALUATION_SHA256,
    HoldoutFilter,
)
from .materialize import (
    _convert_robodojo_episode,
    _convert_takeover_group,
    _file_sha256,
    _group_takeover_records,
    _iter_takeover_records_with_report,
    _smoke_all_takeover_failure_types,
    materialize_dataset,
)
from .parallel import bounded_ordered_map, resolve_workers
from .prompt import prompt_renderer_digest, render_user
from .robodojo_adapter import (
    DEFAULT_LABEL_ROOT,
    DEFAULT_MEDIA_ROOT,
    DEFAULT_OFFICIAL_SPLIT,
    ROBODOJO_TASKS,
    load_official_split_assignments,
    scan_robodojo,
)
from .schema import SCHEMA_VERSION, output_profile_id, validate_sample
from .takeover_adapter import (
    ANCHOR_SELECTION_POLICY,
    TAKEOVER_INSTRUCTION_FIELDS,
    TakeoverQAdapter,
    TakeoverQDataError,
    VideoMetadata,
)
from .video_probe_cache import load_video_probe_cache


FRAGMENT_SCHEMA_VERSION = "v10_action_segment_v5_fragment_v1"
GENERATION_SCHEMA_VERSION = "v10_action_segment_v5_fragment_generation_v1"
READY_TEXT = "V5_FRAGMENT_READY_V1\n"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    else:
        text = json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=False
        )
    return (text + "\n").encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _fragment_paths(root: Path, index: int) -> dict[str, Path]:
    stem = f"fragment-{index:06d}"
    return {
        "data": root / f"{stem}.jsonl",
        "review": root / f"{stem}.review.jsonl",
        "meta": root / f"{stem}.meta.json",
        "ready": root / f"{stem}.READY",
    }


def _sample_contract(sample: Mapping[str, Any], source: str) -> dict[str, Any]:
    validated = validate_sample(sample)
    if validated["source"] != source:
        raise ValueError(
            f"fragment source mismatch: {validated['source']!r} != {source!r}"
        )
    spec = validated["output_spec"]
    if validated["output_profile_id"] != output_profile_id(spec):
        raise ValueError("fragment output profile is not canonical")
    if validated["category"] == "takeover":
        target = validated["target"]
        if tuple(target) != ("execution_decision", "decision_detail"):
            raise ValueError("Takeover target contains progress or predictions")
        if target["execution_decision"] != "Takeover":
            raise ValueError("Takeover category has a non-Takeover decision")
        provenance = validated["provenance"]
        if provenance.get("anchor_selection_policy") != ANCHOR_SELECTION_POLICY:
            raise ValueError("Takeover sample does not use the late-Q2 anchor policy")
    render_user(validated)
    return validated


def _ready_fragment(root: Path, index: int) -> dict[str, Any]:
    paths = _fragment_paths(root, index)
    for name in ("data", "review", "meta", "ready"):
        if not paths[name].is_file():
            raise FileNotFoundError(
                f"fragment {index} is missing committed {name}: {paths[name]}"
            )
    if paths["ready"].read_text(encoding="utf-8") != READY_TEXT:
        raise ValueError(f"invalid READY marker: {paths['ready']}")
    meta = _read_object(paths["meta"])
    if meta.get("schema_version") != FRAGMENT_SCHEMA_VERSION:
        raise ValueError(f"invalid fragment metadata schema: {paths['meta']}")
    if meta.get("fragment_index") != index or meta.get("complete") is not True:
        raise ValueError(f"incomplete fragment metadata: {paths['meta']}")
    if _file_sha256(paths["data"]) != meta.get("data_sha256"):
        raise ValueError(f"fragment data digest mismatch: {paths['data']}")
    if _file_sha256(paths["review"]) != meta.get("review_sha256"):
        raise ValueError(f"fragment review digest mismatch: {paths['review']}")
    return meta


def _publish_fragment(
    root: Path,
    *,
    index: int,
    source: str,
    generation_id: str,
    source_unit_ids: Sequence[str],
    samples: Sequence[Mapping[str, Any]],
    review_fixtures: Sequence[Mapping[str, Any]],
    holdout: EvaluationHoldout,
) -> tuple[dict[str, Any], bool]:
    if not source_unit_ids or any(not value for value in source_unit_ids):
        raise ValueError("a fragment must close non-empty source unit ids")
    checker = HoldoutFilter(holdout)
    kept: list[dict[str, Any]] = []
    profiles: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    sample_ids: list[str] = []
    sample_id_set: set[str] = set()
    excluded_ids: list[str] = []
    for raw in samples:
        sample = _sample_contract(raw, source)
        if checker.keep(sample):
            sample_id = str(sample["sample_id"])
            if sample_id in sample_id_set:
                raise ValueError(f"duplicate sample id inside fragment: {sample_id}")
            sample_id_set.add(sample_id)
            sample_ids.append(sample_id)
            profiles[str(sample["output_profile_id"])] += 1
            categories[str(sample["category"])] += 1
            kept.append(sample)
        else:
            excluded_ids.append(str(sample.get("sample_id") or ""))
    if not kept:
        raise ValueError("fragment has no Evaluation holdout-safe V5 samples")

    data_bytes = b"".join(_json_bytes(value) for value in kept)
    review_bytes = b"".join(_json_bytes(dict(value)) for value in review_fixtures)
    paths = _fragment_paths(root, index)
    meta = {
        "schema_version": FRAGMENT_SCHEMA_VERSION,
        "complete": True,
        "source": source,
        "generation_id": generation_id,
        "fragment_index": index,
        "source_unit_count": len(source_unit_ids),
        "source_unit_ids_sha256": _digest_json(list(source_unit_ids)),
        "source_unit_first": source_unit_ids[0],
        "source_unit_last": source_unit_ids[-1],
        "num_samples": len(kept),
        "num_review_fixtures": len(review_fixtures),
        "sample_ids_sha256": _digest_json(sample_ids),
        "data_sha256": _digest_bytes(data_bytes),
        "review_sha256": _digest_bytes(review_bytes),
        "output_profile_counts": dict(sorted(profiles.items())),
        "category_counts": dict(sorted(categories.items())),
        "schema_contract": SCHEMA_VERSION,
        "prompt_renderer_sha256": prompt_renderer_digest(),
        "takeover_anchor_selection_policy": (
            ANCHOR_SELECTION_POLICY if source == "takeover_q" else None
        ),
        "evaluation_holdout": holdout.metadata(),
        "evaluation_holdout_checked_samples": checker.checked_samples,
        "evaluation_holdout_excluded_samples": checker.excluded_samples,
        "evaluation_holdout_excluded_sample_ids": excluded_ids[:100],
        "created_at": _utc_now(),
    }
    present = [path.exists() for path in paths.values()]
    if any(present):
        if not all(present):
            raise RuntimeError(
                f"fragment {index} has a partial prior publication; inspect manually"
            )
        previous = _ready_fragment(root, index)
        stable_fields = (
            "source",
            "generation_id",
            "fragment_index",
            "source_unit_count",
            "source_unit_ids_sha256",
            "num_samples",
            "num_review_fixtures",
            "sample_ids_sha256",
            "data_sha256",
            "review_sha256",
            "schema_contract",
            "prompt_renderer_sha256",
        )
        if any(previous.get(name) != meta.get(name) for name in stable_fields):
            raise RuntimeError(
                f"owned READY fragment differs from deterministic rebuild: {index}"
            )
        return previous, True

    _write_atomic(paths["data"], data_bytes)
    _write_atomic(paths["review"], review_bytes)
    _write_atomic(paths["meta"], _json_bytes(meta, pretty=True))
    _write_atomic(paths["ready"], READY_TEXT.encode("utf-8"))
    return meta, False


def _generation_manifest(
    root: Path,
    *,
    source: str,
    generation_id: str,
    state: str,
    selection: Mapping[str, Any],
    fragments: Sequence[Mapping[str, Any]],
    scan_complete: bool,
    source_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    closure = [
        {
            "fragment_index": value["fragment_index"],
            "source_unit_ids_sha256": value["source_unit_ids_sha256"],
            "num_samples": value["num_samples"],
            "data_sha256": value["data_sha256"],
            "review_sha256": value["review_sha256"],
        }
        for value in fragments
    ]
    result = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "state": state,
        "source": source,
        "generation_id": generation_id,
        "scan_complete": bool(scan_complete),
        "partial": not bool(scan_complete),
        "selection": dict(selection),
        "ready_fragments": len(fragments),
        "source_units": sum(int(value["source_unit_count"]) for value in fragments),
        "num_samples": sum(int(value["num_samples"]) for value in fragments),
        "num_review_fixtures": sum(
            int(value["num_review_fixtures"]) for value in fragments
        ),
        "evaluation_holdout_excluded_samples": sum(
            int(value["evaluation_holdout_excluded_samples"]) for value in fragments
        ),
        "fragment_closure": closure,
        "fragment_closure_sha256": _digest_json(closure),
        "updated_at": _utc_now(),
    }
    if source_report is not None:
        result["source_report"] = dict(source_report)
    _write_atomic(root / "generation.json", _json_bytes(result, pretty=True))
    return result


def _robodojo_units(args: argparse.Namespace) -> tuple[list[Any], dict[str, Any]]:
    split_path = args.robodojo_official_split.resolve(strict=True)
    digest_before = _file_sha256(split_path)
    scan = scan_robodojo(
        media_root=args.robodojo_media_root,
        label_root=args.robodojo_label_root,
        official_split_path=split_path,
        include_splits=("train",),
        allow_holdouts=False,
    )
    if any(value.split != "train" for value in scan.episodes):
        raise RuntimeError("RoboDojo scan returned a non-train episode")
    if _file_sha256(split_path) != digest_before:
        raise RuntimeError("RoboDojo official split changed during direct scan")
    all_episodes = list(scan.episodes)
    episodes = all_episodes
    if args.smoke_one_per_task:
        assignments = load_official_split_assignments(split_path)
        selected: list[Any] = []
        for task in ROBODOJO_TASKS:
            candidates = [
                value for value in episodes
                if value.task_name == task
                and assignments.get(value.canonical_episode_id) == "train"
            ]
            if candidates:
                selected.append(min(
                    candidates,
                    key=lambda value: int(value.trajectory_name.rsplit("_", 1)[1]),
                ))
        episodes = selected
    if args.max_source_units is not None:
        episodes = episodes[: args.max_source_units]
    selection_complete = (
        not args.smoke_one_per_task
        and args.max_source_units is None
        and len(episodes) == len(all_episodes)
    )
    report = {
        "schema_version": "v5_robodojo_incremental_scan_report_v1",
        "scan_complete": True,
        "selection_complete": selection_complete,
        "direct_index_strategy": (
            "task_instruction_json_to_official_trajectory_to_three_video_paths"
        ),
        "recursive_media_walk": False,
        "label_root": str(args.robodojo_label_root.resolve(strict=True)),
        "media_root": str(args.robodojo_media_root.resolve(strict=True)),
        "official_split": str(split_path),
        "official_split_sha256": digest_before,
        "all_train_episodes": len(all_episodes),
        "selected_episodes": len(episodes),
        "scan_summary": scan.summary(),
    }
    return episodes, report


def _takeover_adapter(args: argparse.Namespace) -> tuple[TakeoverQAdapter, dict[str, Any]]:
    probe_cache_path = args.takeover_video_probe_cache.resolve(strict=True)
    entries = load_video_probe_cache(probe_cache_path)
    payload = _read_object(probe_cache_path)

    def cached_probe(path: Path) -> VideoMetadata:
        value = entries.get(Path(path))
        if value is None:
            raise TakeoverQDataError(
                f"video is absent from complete direct-index probe cache: {path}"
            )
        size, mtime_ns, fps, frame_count = value
        observed = path.stat()
        if observed.st_size != size or observed.st_mtime_ns != mtime_ns:
            raise TakeoverQDataError(f"video changed after fast scan: {path}")
        return VideoMetadata(fps=fps, frame_count=frame_count)

    adapter = TakeoverQAdapter(
        snapshot_id=args.takeover_snapshot_id,
        snapshot_root=args.takeover_snapshot_root.resolve(strict=True),
        reviewed_root=args.takeover_reviewed_root.resolve(strict=True),
        video_probe=cached_probe,
        instruction_fields=args.takeover_instruction_fields,
    )
    index_digest = _file_sha256(adapter.index_path.resolve(strict=True))
    if (
        payload.get("snapshot_id") != adapter.snapshot_id
        or payload.get("reviewed_root") != str(adapter.reviewed_root.resolve())
        or payload.get("reviewed_index_sha256") != index_digest
    ):
        raise ValueError("Takeover fast-scan cache does not match pinned index")
    report = {
        "schema_version": "v5_takeover_q_exclusion_report_v1",
        "scan_complete": False,
        "num_exclusions": 0,
        "exclusion_reason_counts": {},
        "exclusions": [],
        "snapshot_id": adapter.snapshot_id,
        "snapshot_root": str(adapter.snapshot_root.resolve()),
        "reviewed_root": str(adapter.reviewed_root.resolve()),
        "reviewed_index": str(adapter.index_path.resolve()),
        "reviewed_index_sha256": index_digest,
        "video_probe_cache_path": str(probe_cache_path),
        "video_probe_cache_sha256": _file_sha256(probe_cache_path),
        "video_probe_cache_entries": len(entries),
        "video_probe_cache_stat_verified_on_use": True,
        "direct_index_strategy": "episodes_jsonl_to_episode_json_to_three_video_paths",
        "recursive_media_walk": False,
        "anchor_selection_policy": ANCHOR_SELECTION_POLICY,
        "task_instruction_fields": list(adapter.instruction_fields),
        "task_instruction_policy": adapter.instruction_policy,
        "task_instruction_source_of_truth": (
            "exact episode_id entry in media-side instruction.json"
        ),
    }
    return adapter, report


def _chunks(values: Iterable[Any], size: int) -> Iterator[tuple[Any, ...]]:
    iterator = iter(values)
    while True:
        chunk = tuple(itertools.islice(iterator, size))
        if not chunk:
            return
        yield chunk


def _memory_available_bytes() -> int:
    with Path("/proc/meminfo").open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo has no MemAvailable value")


def _wait_for_resources(args: argparse.Namespace) -> None:
    affinity = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    load_limit = float(args.max_load_per_cpu) * affinity
    memory_limit = int(float(args.min_available_gib) * 1024**3)
    while True:
        load_1m = os.getloadavg()[0]
        available = _memory_available_bytes()
        if load_1m <= load_limit and available >= memory_limit:
            return
        print(json.dumps({
            "event": "resource_throttle",
            "load_1m": round(load_1m, 3),
            "load_limit": round(load_limit, 3),
            "mem_available_gib": round(available / 1024**3, 3),
            "mem_required_gib": float(args.min_available_gib),
            "poll_seconds": float(args.resource_poll_seconds),
        }, sort_keys=True), flush=True)
        time.sleep(float(args.resource_poll_seconds))


def build_fragments(args: argparse.Namespace) -> dict[str, Any]:
    root = args.fragment_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    workers = resolve_workers(args.workers)
    holdout = EvaluationHoldout.load(
        args.evaluation_manifest,
        expected_sha256=args.evaluation_expected_sha256,
    )
    existing_generation = root / "generation.json"
    if existing_generation.is_file():
        existing = _read_object(existing_generation)
        if (
            existing.get("source") != args.source
            or existing.get("generation_id") != args.generation_id
        ):
            raise RuntimeError("fragment root belongs to another source generation")
    fragments: list[dict[str, Any]] = []
    selection = {
        "fragment_source_units": args.fragment_source_units,
        "max_source_units": args.max_source_units,
        "smoke_one_per_task": bool(args.smoke_one_per_task),
        "smoke_all_failure_types": bool(args.smoke_all_failure_types),
        "workers": workers,
        "max_in_flight": workers * 2,
        "nice": 5,
    }
    _generation_manifest(
        root,
        source=args.source,
        generation_id=args.generation_id,
        state="building",
        selection=selection,
        fragments=fragments,
        scan_complete=False,
    )
    started = time.monotonic()

    if args.source == "robodojo":
        episodes, source_report = _robodojo_units(args)
        scan_complete = bool(source_report["selection_complete"])
        for fragment_index, episode_chunk in enumerate(
            _chunks(episodes, args.fragment_source_units)
        ):
            _wait_for_resources(args)
            samples: list[dict[str, Any]] = []
            fixtures: list[dict[str, Any]] = []
            for converted, review in bounded_ordered_map(
                _convert_robodojo_episode,
                episode_chunk,
                workers=workers,
                max_in_flight=workers * 2,
            ):
                samples.extend(converted)
                fixtures.extend(review)
            meta, reused = _publish_fragment(
                root,
                index=fragment_index,
                source=args.source,
                generation_id=args.generation_id,
                source_unit_ids=[value.canonical_episode_id for value in episode_chunk],
                samples=samples,
                review_fixtures=fixtures,
                holdout=holdout,
            )
            fragments.append(meta)
            _generation_manifest(
                root,
                source=args.source,
                generation_id=args.generation_id,
                state="building",
                selection=selection,
                fragments=fragments,
                scan_complete=False,
                source_report=source_report,
            )
            print(json.dumps({
                "event": "fragment_ready",
                "source": args.source,
                "fragment_index": fragment_index,
                "reused": reused,
                "samples": meta["num_samples"],
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }, sort_keys=True), flush=True)
    else:
        adapter, source_report = _takeover_adapter(args)
        raw_records = _iter_takeover_records_with_report(adapter, source_report)
        if args.smoke_all_failure_types:
            selected_records: Iterable[Mapping[str, Any]] = (
                _smoke_all_takeover_failure_types(raw_records)
            )
        else:
            selected_records = raw_records
        groups: Iterable[tuple[Mapping[str, Any], ...]] = _group_takeover_records(
            selected_records
        )
        if args.max_source_units is not None:
            groups = itertools.islice(groups, args.max_source_units)
        try:
            for fragment_index, group_chunk in enumerate(
                _chunks(groups, args.fragment_source_units)
            ):
                _wait_for_resources(args)
                samples: list[dict[str, Any]] = []
                for converted in bounded_ordered_map(
                    _convert_takeover_group,
                    group_chunk,
                    workers=workers,
                    max_in_flight=workers * 2,
                ):
                    samples.extend(converted)
                unit_ids = []
                for group in group_chunk:
                    provenance = group[0].get("provenance")
                    unit_ids.append(str(
                        provenance.get("episode_key")
                        if isinstance(provenance, Mapping)
                        else ""
                    ))
                meta, reused = _publish_fragment(
                    root,
                    index=fragment_index,
                    source=args.source,
                    generation_id=args.generation_id,
                    source_unit_ids=unit_ids,
                    samples=samples,
                    review_fixtures=(),
                    holdout=holdout,
                )
                fragments.append(meta)
                _generation_manifest(
                    root,
                    source=args.source,
                    generation_id=args.generation_id,
                    state="building",
                    selection=selection,
                    fragments=fragments,
                    scan_complete=False,
                    source_report=source_report,
                )
                print(json.dumps({
                    "event": "fragment_ready",
                    "source": args.source,
                    "fragment_index": fragment_index,
                    "reused": reused,
                    "samples": meta["num_samples"],
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                }, sort_keys=True), flush=True)
        finally:
            close = getattr(raw_records, "close", None)
            if callable(close):
                close()
        scan_complete = bool(source_report.get("scan_complete")) and (
            not args.smoke_all_failure_types and args.max_source_units is None
        )

    if not fragments:
        raise ValueError("source selection produced no READY fragments")
    result = _generation_manifest(
        root,
        source=args.source,
        generation_id=args.generation_id,
        state="ready",
        selection=selection,
        fragments=fragments,
        scan_complete=scan_complete,
        source_report=source_report,
    )
    if args.source == "takeover_q":
        _write_atomic(
            root / "source_report.json", _json_bytes(source_report, pretty=True)
        )
    print(json.dumps({"event": "generation_ready", **result}, sort_keys=True))
    return result


def _iter_fragment_rows(
    root: Path, metas: Sequence[Mapping[str, Any]], name: str
) -> Iterator[dict[str, Any]]:
    for meta in metas:
        path = _fragment_paths(root, int(meta["fragment_index"]))[name]
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"fragment row is not an object: {path}:{line_number}")
                yield value


def seal_fragments(args: argparse.Namespace) -> dict[str, Any]:
    root = args.fragment_root.resolve(strict=True)
    generation = _read_object(root / "generation.json")
    if generation.get("schema_version") != GENERATION_SCHEMA_VERSION:
        raise ValueError("invalid fragment generation schema")
    if generation.get("state") != "ready":
        raise ValueError("only a ready fragment generation can be sealed")
    source = str(generation.get("source") or "")
    indices = [int(value["fragment_index"]) for value in generation["fragment_closure"]]
    if indices != list(range(len(indices))):
        raise ValueError("fragment generation indices are not contiguous from zero")
    metas = [_ready_fragment(root, index) for index in indices]
    closure = [
        {
            "fragment_index": value["fragment_index"],
            "source_unit_ids_sha256": value["source_unit_ids_sha256"],
            "num_samples": value["num_samples"],
            "data_sha256": value["data_sha256"],
            "review_sha256": value["review_sha256"],
        }
        for value in metas
    ]
    if _digest_json(closure) != generation.get("fragment_closure_sha256"):
        raise ValueError("fragment closure differs from generation manifest")

    holdout = EvaluationHoldout.load(
        args.evaluation_manifest,
        expected_sha256=args.evaluation_expected_sha256,
    )
    seen: set[str] = set()

    def samples() -> Iterator[dict[str, Any]]:
        count = 0
        for sample in _iter_fragment_rows(root, metas, "data"):
            sample = _sample_contract(sample, source)
            sample_id = str(sample["sample_id"])
            if sample_id in seen:
                raise ValueError(f"duplicate sample id across fragments: {sample_id}")
            seen.add(sample_id)
            if holdout.match_sample(sample):
                raise ValueError(f"Evaluation holdout row survived into READY fragment: {sample_id}")
            count += 1
            yield sample
        if count != int(generation["num_samples"]):
            raise RuntimeError(
                f"fragment generation count mismatch: {count} != {generation['num_samples']}"
            )

    fixtures = list(_iter_fragment_rows(root, metas, "review"))
    partial = bool(generation.get("partial"))
    if partial and not args.allow_partial:
        raise PermissionError(
            "generation is partial; pass --allow-partial to seal it explicitly"
        )
    plan_report: dict[str, Any] | None = None
    split_path: Path | None = None
    split_digest: str | None = None
    if source == "takeover_q":
        plan_report = _read_object(root / "source_report.json")
        if not partial and plan_report.get("scan_complete") is not True:
            raise ValueError("full Takeover generation lacks a scan-complete report")
    elif source == "robodojo":
        split_path = args.robodojo_official_split.resolve(strict=True)
        split_digest = _file_sha256(split_path)
    else:
        raise ValueError(f"unsupported incremental source: {source!r}")
    selector = {
        "mode": "immutable_ready_fragment_generation",
        "generation_id": generation["generation_id"],
        "generation_partial": partial,
        "fragment_count": len(metas),
        "fragment_closure_sha256": generation["fragment_closure_sha256"],
        "fragment_root": str(root),
        "source_units": generation["source_units"],
        "allows_early_downstream": True,
    }
    manifest = materialize_dataset(
        samples(),
        args.output,
        source=source,
        partial=partial,
        limit=(int(generation["source_units"]) if partial else None),
        review_fixtures=fixtures,
        selector=selector,
        robodojo_official_split_path=split_path,
        expected_robodojo_official_split_sha256=split_digest,
        plan_cache_report=plan_report,
        leaf_workers=resolve_workers(args.workers),
        evaluation_holdout=holdout,
    )
    print(json.dumps({"event": "generation_sealed", **manifest}, sort_keys=True))
    return manifest


def status(args: argparse.Namespace) -> dict[str, Any]:
    root = args.fragment_root.resolve(strict=True)
    generation = _read_object(root / "generation.json")
    ready_on_disk = sorted(root.glob("fragment-*.READY"))
    result = {
        "fragment_root": str(root),
        "state": generation.get("state"),
        "source": generation.get("source"),
        "generation_id": generation.get("generation_id"),
        "scan_complete": generation.get("scan_complete"),
        "partial": generation.get("partial"),
        "ready_fragments_manifest": generation.get("ready_fragments"),
        "ready_fragments_on_disk": len(ready_on_disk),
        "source_units": generation.get("source_units"),
        "num_samples": generation.get("num_samples"),
        "evaluation_holdout_excluded_samples": generation.get(
            "evaluation_holdout_excluded_samples"
        ),
        "fragment_closure_sha256": generation.get("fragment_closure_sha256"),
        "updated_at": generation.get("updated_at"),
    }
    if args.verify:
        for index in range(int(generation.get("ready_fragments") or 0)):
            _ready_fragment(root, index)
        result["verified"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return result


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="publish resumable source fragments")
    build.add_argument("--source", required=True, choices=("robodojo", "takeover_q"))
    build.add_argument("--fragment-root", type=Path, required=True)
    build.add_argument("--generation-id", required=True)
    build.add_argument("--workers", type=int, default=20)
    build.add_argument("--max-load-per-cpu", type=float, default=0.90)
    build.add_argument("--min-available-gib", type=float, default=64.0)
    build.add_argument("--resource-poll-seconds", type=float, default=5.0)
    build.add_argument("--fragment-source-units", type=_positive, default=25)
    build.add_argument("--max-source-units", type=_positive)
    build.add_argument("--smoke-one-per-task", action="store_true")
    build.add_argument("--smoke-all-failure-types", action="store_true")
    build.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_EVALUATION_MANIFEST)
    build.add_argument("--evaluation-expected-sha256", default=DEFAULT_EVALUATION_SHA256)
    build.add_argument("--robodojo-media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    build.add_argument("--robodojo-label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    build.add_argument("--robodojo-official-split", type=Path, default=DEFAULT_OFFICIAL_SPLIT)
    build.add_argument("--takeover-snapshot-id")
    build.add_argument("--takeover-snapshot-root", type=Path)
    build.add_argument("--takeover-reviewed-root", type=Path)
    build.add_argument("--takeover-video-probe-cache", type=Path)
    build.add_argument(
        "--takeover-instruction-fields",
        nargs="+",
        choices=("detailed_instruction", "instruction"),
        default=TAKEOVER_INSTRUCTION_FIELDS,
        help=(
            "ordered fields read from the exact episode entry in the corresponding "
            "media instruction.json"
        ),
    )

    seal = commands.add_parser("seal", help="seal READY fragments into indexed source")
    seal.add_argument("--fragment-root", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--workers", type=int, default=20)
    seal.add_argument("--allow-partial", action="store_true")
    seal.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_EVALUATION_MANIFEST)
    seal.add_argument("--evaluation-expected-sha256", default=DEFAULT_EVALUATION_SHA256)
    seal.add_argument("--robodojo-official-split", type=Path, default=DEFAULT_OFFICIAL_SPLIT)

    report = commands.add_parser("status", help="inspect a fragment generation")
    report.add_argument("--fragment-root", type=Path, required=True)
    report.add_argument("--verify", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        if args.source == "robodojo":
            if args.smoke_all_failure_types:
                raise ValueError("--smoke-all-failure-types requires Takeover-Q")
        else:
            if args.smoke_one_per_task:
                raise ValueError("--smoke-one-per-task requires RoboDojo")
            required = (
                args.takeover_snapshot_id,
                args.takeover_snapshot_root,
                args.takeover_reviewed_root,
                args.takeover_video_probe_cache,
            )
            if not all(value is not None for value in required):
                raise ValueError(
                    "Takeover build requires snapshot id/root, reviewed root and probe cache"
                )
        build_fragments(args)
    elif args.command == "seal":
        seal_fragments(args)
    else:
        status(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "FRAGMENT_SCHEMA_VERSION",
    "GENERATION_SCHEMA_VERSION",
    "build_fragments",
    "seal_fragments",
    "status",
]
