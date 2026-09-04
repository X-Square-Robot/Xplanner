#!/usr/bin/env python3
"""Publish an immutable train/validation snapshot from an incremental scan."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from .snapshot import (
    publish_training_snapshot,
    publish_training_snapshot_from_rows,
    sha256_file,
)
from .state import ScanLock, ScanState


def _open_existing(run_root: Path) -> ScanState:
    path = run_root / "scan_state.sqlite3"
    connection = sqlite3.connect(path)
    try:
        values = dict(connection.execute("SELECT key,value FROM meta"))
    finally:
        connection.close()
    return ScanState(
        path,
        fingerprint=values["fingerprint"],
        config_json=values["config_json"],
    )


def _read_scan_meta(run_root: Path) -> dict[str, str]:
    path = (run_root / "scan_state.sqlite3").resolve()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return dict(connection.execute("SELECT key,value FROM meta"))
    finally:
        connection.close()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _publish_from_catalog(
    run_root: Path,
    catalog_snapshot: Path,
    *,
    version: str,
    complete: bool,
) -> Path:
    catalog_snapshot = catalog_snapshot.resolve()
    manifest = json.loads((catalog_snapshot / "manifest.json").read_text(encoding="utf-8"))
    accepted_path = catalog_snapshot / "accepted.jsonl"
    rejected_path = catalog_snapshot / "rejected.jsonl"
    for path, key in (
        (accepted_path, "accepted_sha256"),
        (rejected_path, "rejected_sha256"),
    ):
        actual = sha256_file(path)
        if actual != manifest[key]:
            raise ValueError(f"catalog checksum mismatch for {path}: {actual} != {manifest[key]}")
    meta = _read_scan_meta(run_root)
    return publish_training_snapshot_from_rows(
        _read_jsonl(accepted_path),
        _read_jsonl(rejected_path),
        run_root,
        version=version,
        complete=complete,
        scan_fingerprint=meta.get("fingerprint"),
        scan_stats=manifest["stats"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--complete", action="store_true")
    parser.add_argument(
        "--catalog-snapshot",
        type=Path,
        help="build without the scan lock from an atomically published catalog snapshot",
    )
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if args.catalog_snapshot:
        snapshot = _publish_from_catalog(
            run_root,
            args.catalog_snapshot,
            version=args.version,
            complete=args.complete,
        )
    else:
        with ScanLock(run_root / ".scan.lock"):
            state = _open_existing(run_root)
            try:
                snapshot = publish_training_snapshot(
                    state, run_root, version=args.version, complete=args.complete
                )
            finally:
                state.close()
    print(json.dumps({"snapshot": str(snapshot)}, indent=2))


if __name__ == "__main__":
    main()
