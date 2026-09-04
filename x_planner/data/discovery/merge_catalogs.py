"""Disk-bucket merge/dedup for completed V2 shards and optional V1 catalogs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from ..pipeline.constants import UNIT_LEVEL
from .artifact_ledger import register_artifacts
from .common.atomic import BatchedJsonlWriter, iter_jsonl, read_json, write_json
from .common.hashing import config_hash, sampling_config_hash, stable_hash
from .sampling import sample_key
from .shard_state import completed_attempts, load_current_inventory, version_root


def _legacy_roots() -> tuple[tuple[str, Path], ...]:
    """Read optional ``{source_id: [root, ...]}`` legacy-root aliases."""

    raw = os.environ.get("XPLANNER_LEGACY_ROOTS", "").strip()
    if not raw:
        return ()
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise ValueError("XPLANNER_LEGACY_ROOTS must be a JSON object")
    roots: list[tuple[str, Path]] = []
    for source_id, source_roots in value.items():
        if not isinstance(source_id, str) or not isinstance(source_roots, list):
            raise ValueError("legacy roots must map source names to path lists")
        roots.extend((source_id, Path(root)) for root in source_roots)
    return tuple(roots)


def _sha256(path: Path) -> str:
    if path.is_dir():
        path = path / "manifest.json"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_data_paths(root: Path) -> tuple[Path, ...]:
    manifest = read_json(str(root / "manifest.json"), {}) or {}
    if manifest.get("schema_version") != "v10_training_snapshot_v1" or not manifest.get("complete"):
        return ()
    paths = tuple(root / split / "data.jsonl" for split in ("train", "validation"))
    return paths if all(path.is_file() for path in paths) else ()


def _current_catalogs(run_root: Path) -> list[Path]:
    inventory = load_current_inventory(run_root)
    result = []
    for plan in inventory.shards:
        attempts = completed_attempts(version_root(run_root, plan))
        if attempts and (attempts[-1] / "catalog.jsonl").is_file():
            result.append(attempts[-1] / "catalog.jsonl")
    return result


def _completion_state(run_root: Path) -> dict[str, Any]:
    inventory = load_current_inventory(run_root)
    incomplete: list[int] = []
    completion_hashes: set[str] = set()
    for plan in inventory.shards:
        root = version_root(run_root, plan)
        attempts = sorted(path for path in root.glob("attempt-*") if path.is_dir()) if root.is_dir() else []
        # A newer in-progress retry invalidates the older completed attempt for
        # merge purposes. Otherwise a merge racing a rescan can silently emit
        # stale semantics.
        if not attempts or not (attempts[-1] / ".done").is_file():
            incomplete.append(plan.shard_id)
            continue
        marker = read_json(str(attempts[-1] / ".done"), {}) or {}
        if int(marker.get("episode_count", -1)) != plan.episode_count:
            incomplete.append(plan.shard_id)
            continue
        if int(marker.get("retryable_remaining", 0)) > 0:
            incomplete.append(plan.shard_id)
            continue
        completion_hash = str(marker.get("completion_hash") or "")
        if not completion_hash:
            incomplete.append(plan.shard_id)
            continue
        completion_hashes.add(completion_hash)
    return {
        "run_root": str(run_root),
        "inventory_hash": inventory.inventory_hash,
        "total_shards": len(inventory.shards),
        "completed_shards": len(inventory.shards) - len(incomplete),
        "incomplete_shards": incomplete,
        "completion_hashes": sorted(completion_hashes),
    }


@lru_cache(maxsize=8)
def _resolved_legacy_roots(
    signature: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, Path], ...]:
    return tuple((source_id, Path(root).resolve()) for source_id, root in signature)


@lru_cache(maxsize=65_536)
def _legacy_identity_from_paths(
    paths: tuple[str, ...], root_signature: tuple[tuple[str, str], ...]
) -> tuple[str, str, str, str] | None:
    for raw in paths:
        path = Path(str(raw)).resolve()
        for source_id, root in _resolved_legacy_roots(root_signature):
            try:
                relative = path.relative_to(root)
            except ValueError:
                continue
            episode_dir = path.parent if path.is_file() else path
            try:
                episode_key = episode_dir.relative_to(root).as_posix()
            except ValueError:
                episode_key = relative.parent.as_posix()
            return source_id, episode_key, episode_key.split("/", 1)[0], os.path.realpath(episode_dir)
    return None


def _legacy_identity(episode: Mapping[str, Any]) -> tuple[str, str, str, str] | None:
    paths = tuple(str(path) for path in (
        list((episode.get("videos") or {}).values())
        + list(episode.get("annotation_sources") or ())
    ))
    signature = tuple((source_id, str(root)) for source_id, root in _legacy_roots())
    return _legacy_identity_from_paths(paths, signature)


def iter_legacy_samples(
    catalog: Path, *, sampling_hash: str, issues: list[dict[str, Any]] | None = None
) -> Iterator[dict[str, Any]]:
    snapshot_paths = _snapshot_data_paths(catalog) if catalog.is_dir() else ()
    if snapshot_paths:
        yield from _iter_legacy_snapshot_samples(
            catalog, snapshot_paths=snapshot_paths, sampling_hash=sampling_hash, issues=issues
        )
        return
    for accepted in iter_jsonl(str(catalog)):
        shard_path = Path(str(accepted.get("shard_path") or ""))
        if not shard_path.is_file():
            if issues is not None:
                issues.append({"catalog": str(catalog), "kind": "missing_shard", "path": str(shard_path)})
            continue
        try:
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if issues is not None:
                issues.append({"catalog": str(catalog), "kind": "invalid_shard", "path": str(shard_path)})
            continue
        episode = shard.get("episode") or {}
        identity = _legacy_identity(episode)
        if identity is None:
            if issues is not None:
                issues.append({"catalog": str(catalog), "kind": "unknown_media_root", "path": str(shard_path)})
            continue
        source_id, episode_key, dataset_name, media_realpath = identity
        global_key = f"{source_id}:{episode_key}"
        profile = str(episode.get("profile") or "")
        unit_level = UNIT_LEVEL.get(profile, str(episode.get("unit_type") or ""))
        views = tuple((episode.get("videos") or {}).keys())
        input_paths = list((episode.get("annotation_sources") or ())) + list((episode.get("videos") or {}).values())
        for raw in shard.get("samples") or ():
            row = dict(raw)
            current_frame = int(row["current_frame"])
            key = sample_key(
                source_id=source_id,
                episode_key=episode_key,
                current_frame=current_frame,
                unit_level=unit_level,
                profile=profile,
                views=views,
                sampling_hash=sampling_hash,
            )
            row.update({
                "sample_key": key,
                "global_episode_key": global_key,
                "source_id": source_id,
                "episode_key": episode_key,
                "dataset_name": dataset_name,
                "media_realpath": media_realpath,
                "anchor_index": current_frame,
                "unit_level": unit_level,
                "views": list(views),
                "input_paths": input_paths,
                "label": row.get("target"),
                "metadata": {"sampling_config_hash": sampling_hash, "origin": "v1"},
                "origin": "v1",
            })
            yield row


def _iter_legacy_snapshot_samples(
    root: Path,
    *,
    snapshot_paths: tuple[Path, ...],
    sampling_hash: str,
    issues: list[dict[str, Any]] | None,
) -> Iterator[dict[str, Any]]:
    """Read the immutable V1 training snapshot sequentially instead of 365k shards."""
    for data_path in snapshot_paths:
        for wrapped in iter_jsonl(str(data_path)):
            raw = wrapped.get("v10_sample") if isinstance(wrapped, Mapping) else None
            if not isinstance(raw, Mapping):
                if issues is not None:
                    issues.append({
                        "catalog": str(root), "kind": "invalid_snapshot_row",
                        "path": str(data_path),
                    })
                continue
            row = dict(raw)
            images = list(row.get("images") or ())
            video_paths = list(dict.fromkeys(
                str(image.get("video")) for image in images
                if isinstance(image, Mapping) and image.get("video")
            ))
            identity = _legacy_identity({"videos": {str(index): path for index, path in enumerate(video_paths)}})
            if identity is None:
                if issues is not None:
                    issues.append({
                        "catalog": str(root), "kind": "unknown_media_root",
                        "data_id": wrapped.get("data_id"),
                    })
                continue
            source_id, episode_key, dataset_name, media_realpath = identity
            profile = str(row.get("profile") or wrapped.get("profile") or "")
            unit_level = UNIT_LEVEL.get(profile, str(row.get("unit_type") or ""))
            views = tuple(dict.fromkeys(
                str(image.get("view")) for image in images
                if isinstance(image, Mapping) and image.get("view")
            ))
            current_frame = int(row["current_frame"])
            global_key = f"{source_id}:{episode_key}"
            row.update({
                "sample_key": sample_key(
                    source_id=source_id,
                    episode_key=episode_key,
                    current_frame=current_frame,
                    unit_level=unit_level,
                    profile=profile,
                    views=views,
                    sampling_hash=sampling_hash,
                ),
                "global_episode_key": global_key,
                "source_id": source_id,
                "episode_key": episode_key,
                "dataset_name": dataset_name,
                "media_realpath": media_realpath,
                "anchor_index": current_frame,
                "unit_level": unit_level,
                "views": list(views),
                "input_paths": video_paths,
                "label": row.get("target"),
                "metadata": {"sampling_config_hash": sampling_hash, "origin": "v1"},
                "origin": "v1",
            })
            yield row


def resolve_legacy_catalogs(inputs: Iterable[Path]) -> list[Path]:
    resolved: list[Path] = []
    for raw in inputs:
        path = raw.resolve()
        if path.is_file():
            resolved.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"legacy catalog input is missing: {path}")
        if _snapshot_data_paths(path):
            resolved.append(path)
            continue
        direct = [path / "accepted.jsonl", path / "catalog.jsonl"]
        match = next((candidate for candidate in direct if candidate.is_file()), None)
        if match is None:
            candidates = list((path / "merged" / "catalog_snapshots").glob("*/accepted.jsonl"))
            candidates.extend((path / "catalog_snapshots").glob("*/accepted.jsonl"))
            candidates = [candidate for candidate in candidates if (candidate.parent / "manifest.json").is_file()]
            if candidates:
                match = max(candidates, key=lambda candidate: candidate.stat().st_mtime_ns)
        if match is None:
            raise FileNotFoundError(f"no complete accepted.jsonl/catalog.jsonl under {path}")
        resolved.append(match.resolve())
    return list(dict.fromkeys(resolved))


def _merge_identity(row: Mapping[str, Any]) -> str:
    media = str(row.get("media_realpath") or "")
    if not media:
        media_path = next((
            Path(str(path)).parent for path in row.get("input_paths") or ()
            if str(path).lower().endswith(".mp4")
        ), None)
        media = os.path.realpath(media_path) if media_path is not None else str(
            row.get("global_episode_key") or ""
        )
    payload = {
        "media_realpath": os.path.realpath(media) if media.startswith("/") else media,
        "anchor_index": row.get("anchor_index", row.get("current_frame")),
        "unit_level": row.get("unit_level"),
        "profile": row.get("profile"),
        "views": sorted(row.get("views") or ()),
    }
    return config_hash(payload)


def _content_fingerprint(row: Mapping[str, Any]) -> str:
    return config_hash({
        "current_frame": row.get("current_frame"),
        "profile": row.get("profile"),
        "unit_type": row.get("unit_type"),
        "unit_index": row.get("unit_index"),
        "task_caption": row.get("task_caption"),
        "long_memory": row.get("long_memory"),
        "images": row.get("images"),
        "target": row.get("target"),
    })


def _reduce_bucket(
    bucket_path: Path, catalog_part: Path, conflict_part: Path
) -> dict[str, Any]:
    """Deduplicate one independent hash bucket into deterministic part files."""
    unique: dict[str, dict[str, Any]] = {}
    conflict_rows: list[dict[str, Any]] = []
    duplicate_samples = 0
    conflicts = 0
    for row in iter_jsonl(str(bucket_path)):
        key = _merge_identity(row)
        previous = unique.get(key)
        if previous is not None:
            duplicate_samples += 1
            if _content_fingerprint(previous) != _content_fingerprint(row):
                conflicts += 1
                winner = row if row.get("origin") == "v2" else previous
                conflict_rows.append({
                    "merge_identity": key,
                    "winner_origin": winner.get("origin"),
                    "winner_task_caption": winner.get("task_caption"),
                    "previous_origin": previous.get("origin"),
                    "incoming_origin": row.get("origin"),
                    "previous_sample_key": previous.get("sample_key"),
                    "incoming_sample_key": row.get("sample_key"),
                    "previous_task_caption": previous.get("task_caption"),
                    "incoming_task_caption": row.get("task_caption"),
                    "previous_content_fingerprint": _content_fingerprint(previous),
                    "incoming_content_fingerprint": _content_fingerprint(row),
                    "media_realpath": row.get("media_realpath") or previous.get("media_realpath"),
                })
        if previous is None or row.get("origin") == "v2":
            unique[key] = row

    sources = Counter()
    datasets = Counter()
    with BatchedJsonlWriter(str(catalog_part)) as output, BatchedJsonlWriter(
        str(conflict_part)
    ) as conflict_output:
        for conflict in conflict_rows:
            conflict_output.write(conflict)
        for key in sorted(unique):
            row = unique[key]
            output.write(row)
            sources[str(row.get("source_id") or "unknown")] += 1
            datasets[str(row.get("dataset_name") or "unknown")] += 1
    return {
        "bucket": bucket_path.name,
        "duplicate_samples": duplicate_samples,
        "conflicts": conflicts,
        "sample_count": len(unique),
        "sources": dict(sources),
        "datasets": dict(datasets),
    }


def _concatenate_parts(parts: Iterable[Path], destination: Path) -> None:
    with destination.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())


def merge_catalogs(
    run_root: Path,
    *,
    run_id: str,
    sampling: Mapping[str, Any],
    legacy_catalogs: Iterable[Path] = (),
    input_run_roots: Iterable[Path] = (),
    bucket_count: int = 256,
    ledger_path: str | None = None,
    allow_partial: bool = False,
) -> dict[str, Any]:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    roots = list(dict.fromkeys([*(Path(path).resolve() for path in input_run_roots), run_root.resolve()]))
    completion = [_completion_state(input_root) for input_root in roots]
    unsafe = [
        state for state in completion
        if state["incomplete_shards"] or len(state["completion_hashes"]) != 1
    ]
    if unsafe and not allow_partial:
        summaries = [
            {
                "run_root": state["run_root"],
                "completed_shards": state["completed_shards"],
                "total_shards": state["total_shards"],
                "first_incomplete_shards": state["incomplete_shards"][:20],
                "completion_hashes": state["completion_hashes"],
            }
            for state in unsafe
        ]
        raise RuntimeError(
            "refusing incomplete or mixed-semantics merge; rerun after scan completion "
            f"or pass allow_partial=True: {summaries}"
        )
    v2_catalogs = list(dict.fromkeys(
        path for input_root in roots for path in _current_catalogs(input_root)
    ))
    legacy = resolve_legacy_catalogs(legacy_catalogs)
    inventories = [load_current_inventory(input_root).inventory_hash for input_root in roots]
    semantic = {
        "inventories": inventories,
        "v2": [(str(path), _sha256(path)) for path in v2_catalogs],
        "legacy": [(str(path), _sha256(path)) for path in legacy],
        "sampling": sampling_config_hash(sampling),
        "completion": completion,
        "allow_partial": allow_partial,
    }
    merge_id = config_hash(semantic)[:24]
    final = run_root / "merged" / merge_id
    if final.is_dir():
        return json.loads((final / "statistics.json").read_text(encoding="utf-8"))
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{merge_id}-", dir=final.parent))
    buckets = temporary / "buckets"
    buckets.mkdir()
    handles: dict[int, Any] = {}
    episode_inputs: dict[str, set[str]] = defaultdict(set)
    issues: list[dict[str, Any]] = []
    before = 0
    try:
        def add(row: dict[str, Any], input_id: str) -> None:
            nonlocal before
            if not row.get("media_realpath"):
                video = next((
                    Path(str(path)).parent for path in row.get("input_paths") or ()
                    if str(path).lower().endswith(".mp4")
                ), None)
                if video is not None:
                    row["media_realpath"] = os.path.realpath(video)
            key = _merge_identity(row)
            bucket = stable_hash(key) % bucket_count
            handle = handles.get(bucket)
            if handle is None:
                handle = (buckets / f"{bucket:04d}.jsonl").open("a", encoding="utf-8")
                handles[bucket] = handle
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            episode_inputs[str(row.get("media_realpath") or row.get("global_episode_key") or "")].add(input_id)
            before += 1

        sampling_hash = sampling_config_hash(sampling)
        for path in legacy:
            for row in iter_legacy_samples(path, sampling_hash=sampling_hash, issues=issues):
                add(row, str(path))
        for path in v2_catalogs:
            for row in iter_jsonl(str(path)):
                row["origin"] = "v2"
                add(row, str(path))
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        handles.clear()

        duplicate_samples = 0
        conflicts = 0
        sample_count = 0
        sources = Counter()
        datasets = Counter()
        bucket_paths = sorted(buckets.glob("*.jsonl"))
        parts = temporary / "parts"
        parts.mkdir()
        results: dict[str, dict[str, Any]] = {}
        workers = min(16, len(bucket_paths), os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _reduce_bucket,
                    bucket_path,
                    parts / f"{bucket_path.stem}.catalog.jsonl",
                    parts / f"{bucket_path.stem}.conflicts.jsonl",
                ): bucket_path.name
                for bucket_path in bucket_paths
            }
            for future in as_completed(futures):
                result = future.result()
                results[str(result["bucket"])] = result
        for bucket_path in bucket_paths:
            result = results[bucket_path.name]
            duplicate_samples += int(result["duplicate_samples"])
            conflicts += int(result["conflicts"])
            sample_count += int(result["sample_count"])
            sources.update(result["sources"])
            datasets.update(result["datasets"])
        _concatenate_parts(
            (parts / f"{path.stem}.catalog.jsonl" for path in bucket_paths),
            temporary / "catalog.jsonl",
        )
        _concatenate_parts(
            (parts / f"{path.stem}.conflicts.jsonl" for path in bucket_paths),
            temporary / "conflicts.jsonl",
        )
        shutil.rmtree(parts)
        statistics = {
            "run_id": run_id,
            "merge_id": merge_id,
            "catalog": str(final / "catalog.jsonl"),
            "input_catalogs": [str(path) for path in legacy + v2_catalogs],
            "samples_before_dedup": before,
            "samples_after_dedup": sample_count,
            "duplicate_sample_count": duplicate_samples,
            "conflict_count": conflicts,
            "duplicate_episode_count": sum(1 for inputs in episode_inputs.values() if len(inputs) > 1),
            "by_source": dict(sources),
            "by_dataset": dict(datasets),
            "legacy_issue_count": len(issues),
            "dedupe_precedence": "v2",
            "partial_inputs": bool(unsafe),
            "completion": completion,
        }
        with BatchedJsonlWriter(str(temporary / "legacy_issues.jsonl")) as writer:
            for issue in issues:
                writer.write(issue)
        write_json(str(temporary / "statistics.json"), statistics)
        write_json(str(temporary / "inputs.json"), semantic)
        shutil.rmtree(buckets)
        os.replace(temporary, final)
        write_json(str(run_root / "current_merge.json"), {**statistics, "root": str(final)})
        register_artifacts(
            ledger_path,
            [final / "catalog.jsonl", final / "conflicts.jsonl", final / "legacy_issues.jsonl",
             final / "statistics.json", final / "inputs.json", run_root / "current_merge.json"],
            purpose="V2 globally deduplicated merged catalog",
            source_id=",".join(sorted(sources)),
            run_id=run_id,
        )
        return {**statistics, "root": str(final)}
    except BaseException:
        for handle in handles.values():
            handle.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
