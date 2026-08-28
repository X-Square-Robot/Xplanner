"""Small, dependency-light utilities shared by Memory V3 stages."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

from ..v10_continuous_v2.common.atomic import atomic_write, iter_jsonl, write_json


SUCCESS = "_SUCCESS"


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict) or value.get("version") != "memory_v3":
        raise ValueError(f"not a Memory V3 config: {path}")
    return value


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_bucket(key: str, count: int) -> int:
    if count <= 0:
        raise ValueError("bucket count must be positive")
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) % count


def mark_success(directory: str | Path, payload: Mapping[str, Any]) -> Path:
    root = Path(directory)
    marker = root / SUCCESS
    with atomic_write(str(marker)) as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    return marker


def read_success(directory: str | Path) -> dict[str, Any] | None:
    marker = Path(directory) / SUCCESS
    if not marker.is_file():
        return None
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def write_jsonl_atomic(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with atomic_write(str(path)) as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def iter_numbered_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    for line_number, row in enumerate(iter_jsonl(str(path)), 1):
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSONL row at {path}:{line_number}")
        yield line_number, row


def load_episode_contracts(snapshot: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(snapshot)
    result: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation"):
        path = root / split / "episodes.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for row in iter_jsonl(str(path)):
            key = str(row["episode_key"])
            contract = {
                "global_episode_key": key,
                "source_id": str(row.get("source") or key.split(":", 1)[0]),
                "profile": str(row["profile"]),
                "views": list(row.get("views") or ()),
                "split": split,
            }
            previous = result.get(key)
            if previous is not None:
                if previous != contract:
                    from ..v10_continuous_v2.validate_episode import _split_for

                    canonical_split = _split_for(key, seed=42, validation_ratio=0.05)
                    matching = [
                        value for value in (previous, contract)
                        if value["split"] == canonical_split
                    ]
                    if len(matching) != 1:
                        raise ValueError(
                            f"unresolvable formal Episode contract conflict: {key}; "
                            f"canonical_split={canonical_split} previous={previous} "
                            f"current={contract}"
                        )
                    result[key] = matching[0]
                    continue
                # V2's parallel snapshot concatenates per-part Episode files;
                # one Episode can cross a byte part and appear identically in
                # more than one part.  Contract identity, not line identity,
                # is the split/leakage authority.
                continue
            result[key] = contract
    return result


def validate_formal_snapshot(snapshot: str | Path) -> dict[str, Any]:
    root = Path(snapshot)
    manifest_path = root / "manifest.json"
    metadata_path = root / "snapshot_metadata.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "v10_training_snapshot_v1":
        raise ValueError("input snapshot schema is not v10_training_snapshot_v1")
    if manifest.get("complete") is not True:
        raise ValueError("input V2 snapshot is not complete")
    # The formal 20260807 V2 publisher stored the merged catalog digest in
    # both ``catalog_sha256`` and the legacy-named ``manifest_hash`` field.
    # Preserve that provenance, but never misinterpret it as a file checksum.
    legacy_manifest_hash = str(metadata.get("manifest_hash") or "")
    catalog_sha256 = str(metadata.get("catalog_sha256") or "")
    if legacy_manifest_hash and catalog_sha256 and legacy_manifest_hash != catalog_sha256:
        raise ValueError("input V2 legacy manifest_hash disagrees with catalog_sha256")
    catalogs = [Path(str(value)) for value in metadata.get("input_catalogs") or ()]
    missing = [str(path) for path in catalogs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} input catalogs are missing; first={missing[0]}")
    return {
        "root": str(root.resolve()),
        "content_digest": str(manifest["content_digest"]),
        "episodes": int(manifest["scan_stats"]["accepted"]),
        "samples": int(manifest["scan_stats"]["samples"]),
        "manifest_sha256": file_sha256(manifest_path),
        "metadata_sha256": file_sha256(metadata_path),
        "catalog_sha256": catalog_sha256,
        "legacy_manifest_hash": legacy_manifest_hash,
        "input_catalogs": [str(path) for path in catalogs],
        "catalog_count": len(catalogs),
    }


__all__ = [
    "SUCCESS", "canonical_digest", "file_sha256", "iter_jsonl",
    "iter_numbered_jsonl", "load_config", "load_episode_contracts",
    "mark_success", "read_success", "stable_bucket", "validate_formal_snapshot",
    "write_json", "write_jsonl_atomic",
]
