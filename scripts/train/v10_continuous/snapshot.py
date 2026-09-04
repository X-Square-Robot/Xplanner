"""Atomic episode shards, incremental catalogs and immutable training snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import CanonicalEpisode, V10Sample
from .prompt import sample_to_indexed_jsonl
from .state import ScanState


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_episode_shard(
    run_root: Path,
    episode: CanonicalEpisode,
    samples: tuple[V10Sample, ...],
) -> Path:
    digest = hashlib.sha256(episode.episode_key.encode()).hexdigest()
    path = run_root / "episode_shards" / digest[:2] / f"{digest}.json"
    payload = {
        "episode": episode.to_dict(),
        "samples": [sample.to_dict() for sample in samples],
    }
    atomic_write_json(path, payload)
    return path


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "episode_key", "source", "topic", "status", "reason", "detail",
            "shard_path", "profile", "split", "num_samples", "views", "updated_at",
        )
    }


def publish_catalog_snapshot(state: ScanState, run_root: Path) -> Path:
    catalog_root = run_root / "catalog_snapshots"
    catalog_root.mkdir(parents=True, exist_ok=True)
    existing = [int(path.name) for path in catalog_root.iterdir() if path.is_dir() and path.name.isdigit()]
    sequence = max(existing, default=0) + 1
    final = catalog_root / f"{sequence:06d}"
    temporary = Path(tempfile.mkdtemp(prefix=f".{sequence:06d}-", dir=catalog_root))
    try:
        accepted = [_public_row(row) for row in state.iter_rows("accepted")]
        rejected = [_public_row(row) for row in state.iter_rows("rejected")]
        atomic_write_jsonl(temporary / "accepted.jsonl", accepted)
        atomic_write_jsonl(temporary / "rejected.jsonl", rejected)
        manifest = {
            "sequence": sequence,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "stats": state.stats(),
            "accepted_sha256": sha256_file(temporary / "accepted.jsonl"),
            "rejected_sha256": sha256_file(temporary / "rejected.jsonl"),
        }
        atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, final)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final


def _write_split(
    split_root: Path,
    state_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    split_root.mkdir(parents=True)
    data_path = split_root / "data.jsonl"
    index_path = split_root / "data.index"
    episodes_path = split_root / "episodes.jsonl"
    profile_counts: Counter[str] = Counter()
    sample_count = 0
    episode_rows: list[dict[str, Any]] = []
    with data_path.open("wb") as data_output, index_path.open("wb") as index_output:
        for state_row in state_rows:
            shard = json.loads(Path(state_row["shard_path"]).read_text(encoding="utf-8"))
            episode = CanonicalEpisode.from_dict(shard["episode"])
            samples = tuple(V10Sample.from_dict(value) for value in shard["samples"])
            episode_rows.append({
                "episode_key": episode.episode_key,
                "source": episode.source,
                "profile": episode.profile,
                "num_samples": len(samples),
                "views": list(episode.videos),
            })
            profile_counts[episode.profile] += 1
            for sample in samples:
                row = sample_to_indexed_jsonl(sample)
                raw = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
                index_output.write(struct.pack("<Q", data_output.tell()))
                data_output.write(raw)
                sample_count += 1
        data_output.flush()
        os.fsync(data_output.fileno())
        index_output.flush()
        os.fsync(index_output.fileno())
    atomic_write_jsonl(episodes_path, episode_rows)
    result = {
        "episodes": len(episode_rows),
        "samples": sample_count,
        "profiles": dict(sorted(profile_counts.items())),
        "data_sha256": sha256_file(data_path),
        "index_sha256": sha256_file(index_path),
        "episodes_sha256": sha256_file(episodes_path),
    }
    atomic_write_json(split_root / "manifest.json", {
        "schema_version": "v10_indexed_jsonl_v1",
        "jsonl_file": "data.jsonl",
        "index_file": "data.index",
        "num_samples": sample_count,
        "num_episodes": len(episode_rows),
        "profiles": result["profiles"],
        "data_sha256": result["data_sha256"],
        "index_sha256": result["index_sha256"],
    })
    result["manifest_sha256"] = sha256_file(split_root / "manifest.json")
    return result


def publish_training_snapshot(
    state: ScanState,
    run_root: Path,
    *,
    version: str,
    complete: bool,
) -> Path:
    return publish_training_snapshot_from_rows(
        list(state.iter_rows("accepted")),
        [_public_row(row) for row in state.iter_rows("rejected")],
        run_root,
        version=version,
        complete=complete,
        scan_fingerprint=state.get_meta("fingerprint"),
        scan_stats=state.stats(),
    )


def publish_training_snapshot_from_rows(
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    run_root: Path,
    *,
    version: str,
    complete: bool,
    scan_fingerprint: str | None,
    scan_stats: dict[str, Any],
) -> Path:
    """Build an immutable training snapshot from an immutable catalog.

    This path deliberately has no scan lock: catalog snapshots are atomically
    published and content-addressed, so a long JSONL materialization can run in
    parallel with the scanner without observing partially updated scan state.
    """
    if not version or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in version):
        raise ValueError("snapshot version must contain only letters, digits, '-' and '_'")
    snapshots_root = run_root / "training_snapshots"
    snapshots_root.mkdir(parents=True, exist_ok=True)
    final = snapshots_root / version
    if final.exists():
        raise FileExistsError(f"immutable snapshot already exists: {final}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=snapshots_root))
    try:
        split_manifests = {}
        for split in ("train", "validation"):
            split_rows = [row for row in accepted if row["split"] == split]
            split_manifests[split] = _write_split(temporary / split, split_rows)
        train_profiles = {}
        for profile in sorted({row["profile"] for row in accepted if row["split"] == "train"}):
            profile_rows = [
                row for row in accepted
                if row["split"] == "train" and row["profile"] == profile
            ]
            train_profiles[profile] = _write_split(
                temporary / "train_profiles" / profile, profile_rows
            )
        atomic_write_jsonl(temporary / "rejected.jsonl", rejected)
        manifest = {
            "schema_version": "v10_training_snapshot_v1",
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "complete": bool(complete),
            "scan_fingerprint": scan_fingerprint,
            "scan_stats": scan_stats,
            "splits": split_manifests,
            "train_profile_datasets": train_profiles,
            "rejected_sha256": sha256_file(temporary / "rejected.jsonl"),
        }
        manifest["content_digest"] = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
        atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, final)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final
