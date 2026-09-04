#!/usr/bin/env python3
"""Publish a compact immutable snapshot from completed train/validation splits.

The regular publisher materializes both the complete train split and a second
physical copy partitioned by Profile.  Full-coverage natural-proportion jobs do
not use those copies.  This module waits for the two primary split manifests,
hard-links their immutable files, retains exact per-Profile metadata, and
atomically publishes a snapshot consumed as one shuffled train dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .snapshot import atomic_write_json, canonical_json, sha256_file


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _link_file(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)


def _split_record(split_root: Path) -> dict[str, Any]:
    manifest_path = split_root / "manifest.json"
    manifest = _read_json(manifest_path)
    return {
        "episodes": int(manifest["num_episodes"]),
        "samples": int(manifest["num_samples"]),
        "profiles": manifest["profiles"],
        "data_sha256": manifest["data_sha256"],
        "index_sha256": manifest["index_sha256"],
        "episodes_sha256": sha256_file(split_root / "episodes.jsonl"),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _wait_for_splits(source: Path, timeout: int, poll: int) -> None:
    deadline = time.monotonic() + timeout
    required = [source / split / "manifest.json" for split in ("train", "validation")]
    while not all(path.is_file() for path in required):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for split manifests under {source}")
        status = ", ".join(f"{path.parent.name}={path.is_file()}" for path in required)
        print(f"[compact-snapshot] waiting: {status}", flush=True)
        time.sleep(poll)


def publish(
    source: Path,
    catalog: Path,
    final: Path,
    *,
    wait_seconds: int,
    poll_seconds: int,
) -> dict[str, Any]:
    source = source.resolve()
    catalog = catalog.resolve()
    final = final.resolve()
    _wait_for_splits(source, wait_seconds, poll_seconds)
    if final.exists():
        return {"snapshot": str(final), "already_exists": True}

    catalog_manifest = _read_json(catalog / "manifest.json")
    profile_episodes: Counter[str] = Counter()
    profile_samples: defaultdict[str, int] = defaultdict(int)
    with (catalog / "accepted.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] != "train":
                continue
            profile = str(row["profile"])
            profile_episodes[profile] += 1
            profile_samples[profile] += int(row["num_samples"])

    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}-compact-", dir=final.parent))
    try:
        split_records: dict[str, Any] = {}
        for split in ("train", "validation"):
            for name in ("data.jsonl", "data.index", "episodes.jsonl", "manifest.json"):
                _link_file(source / split / name, temporary / split / name)
            split_records[split] = _split_record(temporary / split)
        _link_file(catalog / "rejected.jsonl", temporary / "rejected.jsonl")
        profile_records = {
            profile: {
                "episodes": int(profile_episodes[profile]),
                "samples": int(profile_samples[profile]),
                "profiles": {profile: int(profile_episodes[profile])},
                "materialized_in": "train",
            }
            for profile in sorted(profile_episodes)
        }
        if sum(row["samples"] for row in profile_records.values()) != split_records["train"]["samples"]:
            raise ValueError("per-Profile sample counts do not cover the train split")
        manifest = {
            "schema_version": "v10_training_snapshot_v1",
            "version": final.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "complete": bool(catalog_manifest.get("complete", False)),
            "scan_fingerprint": f"merged:{catalog_manifest['content_digest']}",
            "scan_stats": catalog_manifest["stats"],
            "splits": split_records,
            "train_profile_datasets": profile_records,
            "rejected_sha256": sha256_file(temporary / "rejected.jsonl"),
            "materialization": {
                "mode": "compact_full_train",
                "all_profiles_in_train_split": True,
                "source_in_progress_snapshot": str(source),
                "source_catalog": str(catalog),
            },
        }
        manifest["content_digest"] = hashlib.sha256(
            canonical_json(manifest).encode()
        ).hexdigest()
        atomic_write_json(temporary / "manifest.json", manifest)
        try:
            os.replace(temporary, final)
        except OSError:
            if not final.exists():
                raise
            shutil.rmtree(temporary, ignore_errors=True)
        return {
            "snapshot": str(final),
            "train_samples": split_records["train"]["samples"],
            "validation_samples": split_records["validation"]["samples"],
            "profiles": sorted(profile_records),
            "content_digest": manifest["content_digest"],
            "already_exists": False,
        }
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=7200)
    parser.add_argument("--poll-seconds", type=int, default=15)
    args = parser.parse_args()
    result = publish(
        args.source,
        args.catalog,
        args.final,
        wait_seconds=args.wait_seconds,
        poll_seconds=args.poll_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
