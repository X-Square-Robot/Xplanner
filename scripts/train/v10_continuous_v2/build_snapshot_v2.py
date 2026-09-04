"""Publish immutable, self-describing V2 training-data snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

from .artifact_ledger import register_artifacts
from .common.atomic import atomic_write, read_json, write_json
from .common.hashing import config_hash, file_signature
from .common.hashing import sampling_config_hash
from .common.constants_v2 import SCAN_RULE_VERSION_V2
from .shard_state import load_current_inventory
from .common.atomic import iter_jsonl
from ..v10_continuous.models import V10Sample
from ..v10_continuous.prompt import sample_to_indexed_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_tree_files(source: Path, target: Path) -> list[Path]:
    copied = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, destination)
        except OSError:
            shutil.copy2(path, destination)
        copied.append(destination)
    return copied


class _IndexedWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.data = (root / "data.jsonl").open("wb")
        self.index = (root / "data.index").open("wb")
        self.episodes = (root / "episodes.jsonl").open("w", encoding="utf-8")
        self.seen_episodes: set[str] = set()
        self.profiles: Counter[str] = Counter()
        self.samples = 0

    def write(self, raw_row: dict) -> None:
        sample = V10Sample.from_dict(raw_row)
        row = sample_to_indexed_jsonl(sample)
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        self.index.write(struct.pack("<Q", self.data.tell()))
        self.data.write(encoded)
        episode_key = str(raw_row.get("global_episode_key") or sample.episode_key)
        if episode_key not in self.seen_episodes:
            self.seen_episodes.add(episode_key)
            self.episodes.write(json.dumps({
                "episode_key": episode_key,
                "source": raw_row.get("source_id"),
                "profile": sample.profile,
                "views": raw_row.get("views") or [],
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.profiles[sample.profile] += 1
        self.samples += 1

    def close(self) -> dict:
        for handle in (self.data, self.index):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        self.episodes.flush()
        os.fsync(self.episodes.fileno())
        self.episodes.close()
        result = {
            "episodes": len(self.seen_episodes),
            "samples": self.samples,
            "profiles": dict(sorted(self.profiles.items())),
            "data_sha256": _sha256(self.root / "data.jsonl"),
            "index_sha256": _sha256(self.root / "data.index"),
            "episodes_sha256": _sha256(self.root / "episodes.jsonl"),
        }
        write_json(str(self.root / "manifest.json"), {
            "schema_version": "v10_indexed_jsonl_v1",
            "jsonl_file": "data.jsonl",
            "index_file": "data.index",
            "num_samples": self.samples,
            "num_episodes": len(self.seen_episodes),
            "profiles": result["profiles"],
            "data_sha256": result["data_sha256"],
            "index_sha256": result["index_sha256"],
        })
        result["manifest_sha256"] = _sha256(self.root / "manifest.json")
        return result


def _write_training_manifest(
    target: Path,
    metadata: dict,
    split_manifests: dict[str, dict[str, Any]],
    profile_manifests: dict[str, dict[str, Any]],
) -> None:
    with atomic_write(str(target / "rejected.jsonl")):
        pass
    manifest = {
        "schema_version": "v10_training_snapshot_v1",
        "version": metadata["snapshot_id"],
        "created_at": metadata["creation_time"],
        "complete": True,
        "scan_fingerprint": metadata["inventory_hash"],
        "scan_stats": {
            "accepted": sum(item["episodes"] for item in split_manifests.values()),
            "samples": sum(item["samples"] for item in split_manifests.values()),
        },
        "splits": split_manifests,
        "train_profile_datasets": profile_manifests,
        "rejected_sha256": _sha256(target / "rejected.jsonl"),
    }
    manifest["content_digest"] = config_hash(manifest)
    write_json(str(target / "manifest.json"), manifest)


def _materialize_v1_training_snapshot_serial(
    catalog: Path, target: Path, metadata: dict
) -> None:
    writers = {
        "train": _IndexedWriter(target / "train"),
        "validation": _IndexedWriter(target / "validation"),
    }
    profile_writers: dict[str, _IndexedWriter] = {}
    try:
        for row in iter_jsonl(str(catalog)):
            split = str(row.get("split") or "train")
            if split not in writers:
                raise ValueError(f"unsupported sample split: {split}")
            writers[split].write(row)
            if split == "train":
                profile = str(row.get("profile") or "")
                if not profile:
                    raise ValueError("sample profile is missing")
                writer = profile_writers.get(profile)
                if writer is None:
                    writer = _IndexedWriter(target / "train_profiles" / profile)
                    profile_writers[profile] = writer
                writer.write(row)
        split_manifests = {name: writer.close() for name, writer in writers.items()}
        profile_manifests = {
            name: writer.close() for name, writer in sorted(profile_writers.items())
        }
    except BaseException:
        for writer in [*writers.values(), *profile_writers.values()]:
            for handle in (writer.data, writer.index, writer.episodes):
                if not handle.closed:
                    handle.close()
        raise
    _write_training_manifest(target, metadata, split_manifests, profile_manifests)


class _PartWriter:
    """Write validated wrapped rows without building global offsets yet."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.data = (root / "data.jsonl").open("wb")
        self.episodes = (root / "episodes.jsonl").open("w", encoding="utf-8")
        self.seen_episodes: set[str] = set()
        self.profiles: Counter[str] = Counter()
        self.samples = 0

    def write(self, raw_row: dict, sample: V10Sample, encoded: bytes) -> None:
        self.data.write(encoded)
        episode_key = str(raw_row.get("global_episode_key") or sample.episode_key)
        if episode_key not in self.seen_episodes:
            self.seen_episodes.add(episode_key)
            self.episodes.write(json.dumps({
                "episode_key": episode_key,
                "source": raw_row.get("source_id"),
                "profile": sample.profile,
                "views": raw_row.get("views") or [],
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.profiles[sample.profile] += 1
        self.samples += 1

    def close(self) -> None:
        self.data.close()
        self.episodes.close()
        write_json(str(self.root / "part_statistics.json"), {
            "samples": self.samples,
            "profiles": dict(self.profiles),
        })

    def abort(self) -> None:
        for handle in (self.data, self.episodes):
            if not handle.closed:
                handle.close()


def _iter_jsonl_range(path: Path, start: int, end: int) -> Iterator[dict[str, Any]]:
    size = path.stat().st_size
    with path.open("rb") as handle:
        if start:
            handle.seek(start - 1)
            if handle.read(1) != b"\n":
                handle.readline()
        while True:
            if handle.tell() >= end and end < size:
                break
            line = handle.readline()
            if not line:
                break
            yield json.loads(line)


def _materialize_snapshot_part(
    catalog: Path, start: int, end: int, part_root: Path
) -> int:
    writers = {
        "train": _PartWriter(part_root / "train"),
        "validation": _PartWriter(part_root / "validation"),
    }
    profile_writers: dict[str, _PartWriter] = {}
    processed = 0
    try:
        for raw_row in _iter_jsonl_range(catalog, start, end):
            sample = V10Sample.from_dict(raw_row)
            wrapped = sample_to_indexed_jsonl(sample)
            encoded = json.dumps(
                wrapped, ensure_ascii=False, separators=(",", ":")
            ).encode() + b"\n"
            split = str(raw_row.get("split") or "train")
            if split not in writers:
                raise ValueError(f"unsupported sample split: {split}")
            writers[split].write(raw_row, sample, encoded)
            if split == "train":
                profile = str(raw_row.get("profile") or "")
                if not profile:
                    raise ValueError("sample profile is missing")
                writer = profile_writers.get(profile)
                if writer is None:
                    writer = _PartWriter(part_root / "train_profiles" / profile)
                    profile_writers[profile] = writer
                writer.write(raw_row, sample, encoded)
            processed += 1
        for writer in [*writers.values(), *profile_writers.values()]:
            writer.close()
        return processed
    except BaseException:
        for writer in [*writers.values(), *profile_writers.values()]:
            writer.abort()
        raise


def _assemble_indexed_parts(part_roots: list[Path], relative: Path, target: Path) -> dict:
    target.mkdir(parents=True, exist_ok=True)
    data_path = target / "data.jsonl"
    index_path = target / "data.index"
    episodes_path = target / "episodes.jsonl"
    seen_episodes: set[str] = set()
    profiles: Counter[str] = Counter()
    samples = 0
    offset = 0
    with data_path.open("wb") as data, index_path.open("wb") as index, episodes_path.open(
        "w", encoding="utf-8"
    ) as episodes:
        for part_root in part_roots:
            source_root = part_root / relative
            if not (source_root / "part_statistics.json").is_file():
                continue
            stats = read_json(str(source_root / "part_statistics.json"), {}) or {}
            samples += int(stats.get("samples", 0))
            profiles.update(stats.get("profiles") or {})
            with (source_root / "data.jsonl").open("rb") as source:
                for line in source:
                    index.write(struct.pack("<Q", offset))
                    data.write(line)
                    offset += len(line)
            with (source_root / "episodes.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    episode_key = str(row["episode_key"])
                    if episode_key not in seen_episodes:
                        seen_episodes.add(episode_key)
                        episodes.write(line if line.endswith("\n") else line + "\n")
        for handle in (data, index, episodes):
            handle.flush()
            os.fsync(handle.fileno())
    result = {
        "episodes": len(seen_episodes),
        "samples": samples,
        "profiles": dict(sorted(profiles.items())),
        "data_sha256": _sha256(data_path),
        "index_sha256": _sha256(index_path),
        "episodes_sha256": _sha256(episodes_path),
    }
    write_json(str(target / "manifest.json"), {
        "schema_version": "v10_indexed_jsonl_v1",
        "jsonl_file": "data.jsonl",
        "index_file": "data.index",
        "num_samples": samples,
        "num_episodes": len(seen_episodes),
        "profiles": result["profiles"],
        "data_sha256": result["data_sha256"],
        "index_sha256": result["index_sha256"],
    })
    result["manifest_sha256"] = _sha256(target / "manifest.json")
    return result


def _materialize_v1_training_snapshot_parallel(
    catalog: Path, target: Path, metadata: dict, workers: int
) -> None:
    parts_root = target / ".snapshot_parts"
    parts_root.mkdir()
    size = catalog.stat().st_size
    ranges = [
        (index * size // workers, (index + 1) * size // workers)
        for index in range(workers)
    ]
    part_roots = [parts_root / f"part-{index:04d}" for index in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_materialize_snapshot_part, catalog, start, end, part_root)
            for (start, end), part_root in zip(ranges, part_roots)
        ]
        for future in as_completed(futures):
            future.result()
    split_manifests = {
        split: _assemble_indexed_parts(part_roots, Path(split), target / split)
        for split in ("train", "validation")
    }
    profiles = sorted({
        path.name
        for part_root in part_roots
        for path in (part_root / "train_profiles").glob("*")
        if path.is_dir()
    })
    profile_manifests = {
        profile: _assemble_indexed_parts(
            part_roots, Path("train_profiles") / profile, target / "train_profiles" / profile
        )
        for profile in profiles
    }
    shutil.rmtree(parts_root)
    _write_training_manifest(target, metadata, split_manifests, profile_manifests)


def _materialize_v1_training_snapshot(catalog: Path, target: Path, metadata: dict) -> None:
    requested = int(os.environ.get("V10_SNAPSHOT_WORKERS", "0") or 0)
    workers = requested or (
        min(16, os.cpu_count() or 1) if catalog.stat().st_size >= 1024 ** 3 else 1
    )
    workers = max(1, min(workers, 64))
    if workers == 1:
        _materialize_v1_training_snapshot_serial(catalog, target, metadata)
    else:
        _materialize_v1_training_snapshot_parallel(catalog, target, metadata, workers)


def build_snapshot(
    run_root: Path,
    *,
    run_id: str,
    config_paths: Iterable[Path],
    ledger_path: str | None = None,
) -> dict:
    merge = read_json(str(run_root / "current_merge.json"))
    manifests = read_json(str(run_root / "current_manifest.json"))
    if not isinstance(merge, dict) or not isinstance(manifests, dict):
        raise FileNotFoundError("merge and manifest stages must complete before snapshot")
    inventory = load_current_inventory(run_root)
    catalog = Path(str(merge["catalog"]))
    merge_inputs = read_json(str(Path(str(merge["root"])) / "inputs.json"), {}) or {}
    inventory_hashes = list(merge_inputs.get("inventories") or [inventory.inventory_hash])
    composite_inventory_hash = config_hash(inventory_hashes)
    manifest_root = Path(str(manifests["root"]))
    configs = [path.resolve() for path in config_paths if path.is_file()]
    semantic = {
        "materialization_version": "v10_v2_indexed_2",
        "run_id": run_id,
        "inventory_hashes": inventory_hashes,
        "merge_id": merge["merge_id"],
        "manifest_id": manifests["manifest_id"],
        "catalog_hash": _sha256(catalog),
        "configs": [file_signature(str(path)) for path in configs],
    }
    snapshot_id = config_hash(semantic)[:24]
    final = run_root / "snapshots" / snapshot_id
    if final.is_dir():
        metadata = read_json(str(final / "snapshot_metadata.json"), {})
        return {**metadata, "root": str(final)}
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}-", dir=final.parent))
    try:
        try:
            os.link(catalog, temporary / "catalog.jsonl")
        except OSError:
            shutil.copy2(catalog, temporary / "catalog.jsonl")
        copied = [temporary / "catalog.jsonl"]
        copied.extend(_copy_tree_files(manifest_root, temporary / "manifests"))
        configs_root = temporary / "configs"
        configs_root.mkdir()
        for path in configs:
            destination = configs_root / path.name
            shutil.copy2(path, destination)
            copied.append(destination)
        stats_root = temporary / "statistics"
        source_stats = run_root / "statistics"
        if source_stats.is_dir():
            copied.extend(_copy_tree_files(source_stats, stats_root))
        source_config = yaml.safe_load(configs[0].read_text(encoding="utf-8")) if configs else {}
        status_summary = read_json(str(run_root / "statistics" / "summary.json"), {}) or {}
        manifest_catalog = manifest_root / "all.jsonl"
        manifest_hash = (
            semantic["catalog_hash"]
            if os.path.samefile(catalog, manifest_catalog)
            else _sha256(manifest_catalog)
        )
        metadata = {
            "snapshot_id": snapshot_id,
            "run_id": run_id,
            "creation_time": datetime.now(timezone.utc).isoformat(),
            "code_version": SCAN_RULE_VERSION_V2,
            "inventory_hash": composite_inventory_hash,
            "inventory_hashes": inventory_hashes,
            "merge_id": merge["merge_id"],
            "manifest_hash": manifest_hash,
            "sample_count": manifests["sample_count"],
            "source_statistics": merge.get("by_source", {}),
            "error_statistics": status_summary.get("by_error_type", {}),
            "input_catalogs": merge.get("input_catalogs", []),
            "config_hash": config_hash([file_signature(str(path)) for path in configs]),
            "source_config_hash": _sha256(configs[0]) if configs else "",
            "view_config_hash": _sha256(configs[1]) if len(configs) > 1 else "",
            "sampling_config_hash": sampling_config_hash((source_config or {}).get("sampling") or {}),
            "catalog_sha256": semantic["catalog_hash"],
        }
        write_json(str(temporary / "snapshot_metadata.json"), metadata)
        _materialize_v1_training_snapshot(temporary / "catalog.jsonl", temporary, metadata)
        copied.append(temporary / "snapshot_metadata.json")
        copied.extend([
            temporary / "manifest.json", temporary / "rejected.jsonl",
            *(path for root_name in ("train", "validation", "train_profiles")
              for path in (temporary / root_name).rglob("*") if path.is_file()),
        ])
        os.replace(temporary, final)
        final_artifacts = [final / path.relative_to(temporary) for path in copied]
        write_json(str(run_root / "current_snapshot.json"), {**metadata, "root": str(final)})
        register_artifacts(
            ledger_path,
            [*final_artifacts, run_root / "current_snapshot.json"],
            purpose="V2 immutable training snapshot",
            source_id=",".join(sorted(merge.get("by_source") or {})),
            run_id=run_id,
        )
        return {**metadata, "root": str(final)}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
