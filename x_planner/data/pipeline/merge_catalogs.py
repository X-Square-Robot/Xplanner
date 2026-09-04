#!/usr/bin/env python3
"""Merge immutable V10 catalogs and atomically publish a validated snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .snapshot import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    publish_training_snapshot_from_rows,
    sha256_file,
)
from .validate_schema import validate_snapshot


PUBLIC_KEYS = (
    "episode_key",
    "source",
    "topic",
    "status",
    "reason",
    "detail",
    "shard_path",
    "profile",
    "split",
    "num_samples",
    "views",
    "updated_at",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            missing = [key for key in PUBLIC_KEYS if key not in value]
            if missing:
                raise ValueError(f"missing keys at {path}:{line_number}: {missing}")
            rows.append({key: value[key] for key in PUBLIC_KEYS})
    return rows


def _validate_catalog_counts(
    catalog: Path,
    manifest: dict[str, Any],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
) -> None:
    stats = manifest.get("stats")
    if not isinstance(stats, dict):
        raise ValueError(f"catalog stats missing: {catalog}")
    expected_status = stats.get("status") or {}
    expected = {
        "accepted": len(accepted),
        "rejected": len(rejected),
    }
    for status, count in expected.items():
        if int(expected_status.get(status, 0)) != count:
            raise ValueError(
                f"catalog {status} count mismatch in {catalog}: "
                f"manifest={expected_status.get(status, 0)} rows={count}"
            )
    samples = sum(int(row["num_samples"]) for row in accepted)
    if int(stats.get("samples", 0)) != samples:
        raise ValueError(
            f"catalog sample count mismatch in {catalog}: "
            f"manifest={stats.get('samples', 0)} rows={samples}"
        )


def load_catalog(path: Path) -> dict[str, Any]:
    path = path.resolve()
    manifest_path = path / "manifest.json"
    accepted_path = path / "accepted.jsonl"
    rejected_path = path / "rejected.jsonl"
    manifest = _read_json(manifest_path)
    for data_path, digest_key in (
        (accepted_path, "accepted_sha256"),
        (rejected_path, "rejected_sha256"),
    ):
        actual = sha256_file(data_path)
        expected = manifest.get(digest_key)
        if actual != expected:
            raise ValueError(
                f"catalog checksum mismatch for {data_path}: {actual} != {expected}"
            )
    accepted = _read_jsonl(accepted_path)
    rejected = _read_jsonl(rejected_path)
    if any(row["status"] != "accepted" for row in accepted):
        raise ValueError(f"non-accepted row in {accepted_path}")
    if any(row["status"] != "rejected" for row in rejected):
        raise ValueError(f"non-rejected row in {rejected_path}")
    for row in accepted:
        shard = Path(str(row["shard_path"]))
        if not shard.is_file():
            raise FileNotFoundError(f"accepted shard missing: {shard}")
    _validate_catalog_counts(path, manifest, accepted, rejected)
    return {
        "path": path,
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "accepted": accepted,
        "rejected": rejected,
    }


def _row_without_volatile(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"updated_at", "shard_path"}
    }


def _same_accepted(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if _row_without_volatile(left) != _row_without_volatile(right):
        return False
    return sha256_file(Path(str(left["shard_path"]))) == sha256_file(
        Path(str(right["shard_path"]))
    )


def merge_rows(catalogs: Iterable[dict[str, Any]]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, int]
]:
    accepted_by_key: dict[str, dict[str, Any]] = {}
    rejected_by_key: dict[str, dict[str, Any]] = {}
    duplicate_counts: Counter[str] = Counter()
    for catalog in catalogs:
        for row in catalog["accepted"]:
            key = str(row["episode_key"])
            if key in rejected_by_key:
                raise ValueError(f"episode is both accepted and rejected: {key}")
            previous = accepted_by_key.get(key)
            if previous is None:
                accepted_by_key[key] = row
            elif _same_accepted(previous, row):
                duplicate_counts["accepted_exact"] += 1
            else:
                raise ValueError(f"conflicting accepted duplicate: {key}")
        for row in catalog["rejected"]:
            key = str(row["episode_key"])
            if key in accepted_by_key:
                raise ValueError(f"episode is both accepted and rejected: {key}")
            previous = rejected_by_key.get(key)
            if previous is None:
                rejected_by_key[key] = row
            elif _row_without_volatile(previous) == _row_without_volatile(row):
                duplicate_counts["rejected_exact"] += 1
            else:
                raise ValueError(f"conflicting rejected duplicate: {key}")
    accepted = [accepted_by_key[key] for key in sorted(accepted_by_key)]
    rejected = [rejected_by_key[key] for key in sorted(rejected_by_key)]
    return accepted, rejected, dict(sorted(duplicate_counts.items()))


def merged_stats(
    accepted: list[dict[str, Any]], rejected: list[dict[str, Any]]
) -> dict[str, Any]:
    profiles = Counter(str(row["profile"]) for row in accepted)
    splits = Counter(str(row["split"]) for row in accepted)
    return {
        "status": {"accepted": len(accepted), "rejected": len(rejected)},
        "splits": dict(sorted(splits.items())),
        "profiles": dict(sorted(profiles.items())),
        "samples": sum(int(row["num_samples"]) for row in accepted),
        "terminal": len(accepted) + len(rejected),
    }


def _source_record(catalog: dict[str, Any]) -> dict[str, Any]:
    manifest = catalog["manifest"]
    return {
        "catalog": str(catalog["path"]),
        "manifest_sha256": catalog["manifest_sha256"],
        "accepted_sha256": manifest["accepted_sha256"],
        "rejected_sha256": manifest["rejected_sha256"],
        "stats": manifest["stats"],
    }


def publish_merged_catalog(
    output_run_root: Path,
    *,
    version: str,
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    stats: dict[str, Any],
    sources: list[dict[str, Any]],
    duplicates: dict[str, int],
    complete: bool,
) -> tuple[Path, dict[str, Any]]:
    catalog_root = output_run_root / "catalog_snapshots"
    catalog_root.mkdir(parents=True, exist_ok=True)
    final = catalog_root / version
    if final.exists():
        raise FileExistsError(f"immutable merged catalog already exists: {final}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{version}-", dir=catalog_root))
    try:
        atomic_write_jsonl(temporary / "accepted.jsonl", accepted)
        atomic_write_jsonl(temporary / "rejected.jsonl", rejected)
        manifest = {
            "schema_version": "v10_merged_catalog_v1",
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "complete": bool(complete),
            "sources": sources,
            "duplicates_removed": duplicates,
            "stats": stats,
            "accepted_sha256": sha256_file(temporary / "accepted.jsonl"),
            "rejected_sha256": sha256_file(temporary / "rejected.jsonl"),
        }
        manifest["content_digest"] = hashlib.sha256(
            canonical_json(manifest).encode()
        ).hexdigest()
        atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, final)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final, manifest


def merge_catalogs(
    catalog_paths: list[Path],
    *,
    output_run_root: Path,
    version: str,
    complete: bool,
) -> dict[str, Any]:
    if len(catalog_paths) < 2:
        raise ValueError("at least two catalogs are required")
    if not version or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for character in version
    ):
        raise ValueError("version must contain only letters, digits, '-' and '_'")
    catalogs = [load_catalog(path) for path in catalog_paths]
    accepted, rejected, duplicates = merge_rows(catalogs)
    stats = merged_stats(accepted, rejected)
    sources = [_source_record(catalog) for catalog in catalogs]
    merge_fingerprint = hashlib.sha256(canonical_json(sources).encode()).hexdigest()
    output_run_root = output_run_root.resolve()
    merged_catalog, merged_manifest = publish_merged_catalog(
        output_run_root,
        version=version,
        accepted=accepted,
        rejected=rejected,
        stats=stats,
        sources=sources,
        duplicates=duplicates,
        complete=complete,
    )
    snapshot = publish_training_snapshot_from_rows(
        accepted,
        rejected,
        output_run_root,
        version=version,
        complete=complete,
        scan_fingerprint=f"merged:{merge_fingerprint}",
        scan_stats=stats,
    )
    validation = validate_snapshot(snapshot)
    return {
        "catalog": str(merged_catalog),
        "catalog_content_digest": merged_manifest["content_digest"],
        "snapshot": str(snapshot),
        "snapshot_content_digest": _read_json(snapshot / "manifest.json")[
            "content_digest"
        ],
        "stats": stats,
        "duplicates_removed": duplicates,
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, action="append", required=True)
    parser.add_argument("--output-run-root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    result = merge_catalogs(
        args.catalog,
        output_run_root=args.output_run_root,
        version=args.version,
        complete=args.complete,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
