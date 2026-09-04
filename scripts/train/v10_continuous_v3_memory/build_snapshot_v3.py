"""Publish immutable Memory V3 snapshots from merged reference lists."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import struct
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from .common_v3 import (
    canonical_digest,
    file_sha256,
    iter_jsonl,
    load_config,
    mark_success,
    read_success,
    validate_formal_snapshot,
    write_json,
)
from .prompt_v3 import sample_to_indexed_jsonl


PACKAGE_ROOT = Path(__file__).resolve().parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _byte_ranges(path: Path, workers: int) -> list[tuple[int, int]]:
    size = path.stat().st_size
    if size == 0:
        return [(0, 0)]
    count = max(1, min(int(workers), size))
    return [
        (size * index // count, size * (index + 1) // count)
        for index in range(count)
    ]


def _iter_jsonl_range(path: Path, start: int, end: int) -> Iterator[dict[str, Any]]:
    with path.open("rb") as handle:
        handle.seek(start)
        if start:
            handle.seek(start - 1)
            if handle.read(1) != b"\n":
                handle.readline()
        while True:
            offset = handle.tell()
            if offset >= end:
                break
            line = handle.readline()
            if not line:
                break
            yield json.loads(line)


def _iter_ref_rows(
    refs: Iterable[dict[str, Any]],
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    for shard_path, group in itertools.groupby(refs, key=lambda row: str(row["shard_path"])):
        wanted = list(group)
        if not wanted:
            continue
        last_line = 0
        with Path(shard_path).open(encoding="utf-8") as handle:
            for ref in wanted:
                line_number = int(ref["line_number"])
                if line_number <= last_line:
                    raise ValueError(f"non-monotonic refs for {shard_path}")
                line = ""
                for _ in range(last_line, line_number):
                    line = handle.readline()
                    if not line:
                        raise EOFError(f"missing source row {shard_path}:{line_number}")
                last_line = line_number
                sample = json.loads(line)
                if str(sample.get("sample_key")) != str(ref["sample_key"]):
                    raise ValueError(f"sample key mismatch at {shard_path}:{line_number}")
                yield ref, sample


def _iter_ref_samples(path: Path) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    yield from _iter_ref_rows(iter_jsonl(str(path)))


class _IndexedWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.data = (root / "data.jsonl").open("wb")
        self.index = (root / "data.index").open("wb")
        self.episodes = (root / "episodes.jsonl").open("w", encoding="utf-8")
        self.seen_episodes: set[str] = set()
        self.profiles: Counter[str] = Counter()
        self.sources: Counter[str] = Counter()
        self.samples = 0

    def write(self, sample: Mapping[str, Any]) -> None:
        row = sample_to_indexed_jsonl(sample)
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
        self.index.write(struct.pack("<Q", self.data.tell()))
        self.data.write(encoded)
        key = str(sample["global_episode_key"])
        if key not in self.seen_episodes:
            self.seen_episodes.add(key)
            self.episodes.write(json.dumps({
                "episode_key": key,
                "source": sample["source_id"],
                "profile": sample["profile"],
                "views": list(dict.fromkeys(image["view"] for image in sample["images"])),
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.profiles[str(sample["profile"])] += 1
        self.sources[str(sample["source_id"])] += 1
        self.samples += 1

    def close(self, *, hash_outputs: bool = True) -> dict[str, Any]:
        for handle in (self.data, self.index):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        self.episodes.flush()
        os.fsync(self.episodes.fileno())
        self.episodes.close()
        result = {
            "samples": self.samples,
            "episodes": len(self.seen_episodes),
            "profiles": dict(sorted(self.profiles.items())),
            "sources": dict(sorted(self.sources.items())),
        }
        if not hash_outputs:
            return result
        result.update({
            "data_sha256": file_sha256(self.root / "data.jsonl"),
            "index_sha256": file_sha256(self.root / "data.index"),
            "episodes_sha256": file_sha256(self.root / "episodes.jsonl"),
        })
        write_json(str(self.root / "manifest.json"), {
            "schema_version": "v10_memory_v3_indexed_jsonl_v1",
            "jsonl_file": "data.jsonl",
            "index_file": "data.index",
            "num_samples": result["samples"],
            "num_episodes": result["episodes"],
            "profiles": result["profiles"],
            "sources": result["sources"],
            "data_sha256": result["data_sha256"],
            "index_sha256": result["index_sha256"],
        })
        result["manifest_sha256"] = file_sha256(self.root / "manifest.json")
        return result

    def abort(self) -> None:
        for handle in (self.data, self.index, self.episodes):
            if not handle.closed:
                handle.close()


def _split_list_range(payload: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(payload["path"]))
    part_root = Path(str(payload["part_root"]))
    part_root.mkdir(parents=True, exist_ok=False)
    handles = {
        split: (part_root / f"{split}.list").open("wb")
        for split in ("train", "validation")
    }
    counts: Counter[str] = Counter()
    try:
        with path.open("rb") as source:
            start = int(payload["start"])
            end = int(payload["end"])
            source.seek(start)
            if start:
                source.seek(start - 1)
                if source.read(1) != b"\n":
                    source.readline()
            while True:
                offset = source.tell()
                if offset >= end:
                    break
                line = source.readline()
                if not line:
                    break
                row = json.loads(line)
                split = str(row["split"])
                if split not in {"train", "validation"}:
                    raise ValueError(f"unknown split in {path}: {split}")
                handles[split].write(line)
                counts[split] += 1
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        return {
            "task": str(payload["task"]),
            "order": int(payload["order"]),
            "part_root": str(part_root),
            "counts": dict(counts),
        }
    except BaseException:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        raise


def _write_split_lists(merged: Path, target: Path, *, workers: int) -> dict[str, int]:
    target.mkdir(parents=True, exist_ok=True)
    parts_root = target.parent / "split_parts"
    parts_root.mkdir()
    payloads = []
    for task in ("continuous", "initial_plan", "terminal"):
        path = merged / f"{task}.list"
        for order, (start, end) in enumerate(_byte_ranges(path, workers)):
            payloads.append({
                "task": task, "order": order, "path": str(path),
                "start": start, "end": end,
                "part_root": str(parts_root / f"{task}-{order:05d}"),
            })
    results = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_split_list_range, payload) for payload in payloads]
            for future in as_completed(futures):
                results.append(future.result())
        counts: Counter[str] = Counter()
        for task in ("continuous", "initial_plan", "terminal"):
            task_results = sorted(
                (row for row in results if row["task"] == task),
                key=lambda row: int(row["order"]),
            )
            for split, suffix in (("train", "train"), ("validation", "val")):
                output = target / f"{task}_{suffix}.list"
                with output.open("wb") as handle:
                    for result in task_results:
                        part = Path(str(result["part_root"])) / f"{split}.list"
                        with part.open("rb") as source:
                            shutil.copyfileobj(source, handle, length=16 * 1024 * 1024)
                        counts[f"{task}_{split}"] += int(result["counts"].get(split, 0))
                    handle.flush()
                    os.fsync(handle.fileno())
        return dict(counts)
    finally:
        shutil.rmtree(parts_root, ignore_errors=True)


def _materialize_task(
    list_path: Path, root: Path, *, expected_task: str
) -> dict[str, Any]:
    writer = _IndexedWriter(root)
    try:
        for _, sample in _iter_ref_samples(list_path):
            if sample.get("task_type") != expected_task:
                raise ValueError(f"{list_path} contains {sample.get('task_type')}, expected {expected_task}")
            writer.write(sample)
        return writer.close()
    except BaseException:
        writer.abort()
        raise


def _materialize_ref_range(payload: Mapping[str, Any]) -> dict[str, Any]:
    list_path = Path(str(payload["list_path"]))
    part_root = Path(str(payload["part_root"]))
    writer = _IndexedWriter(part_root)
    try:
        refs = _iter_jsonl_range(
            list_path, int(payload["start"]), int(payload["end"])
        )
        for _, sample in _iter_ref_rows(refs):
            if sample.get("task_type") != payload["expected_task"]:
                raise ValueError(
                    f"{list_path} contains {sample.get('task_type')}, "
                    f"expected {payload['expected_task']}"
                )
            writer.write(sample)
        result = writer.close(hash_outputs=False)
        return {
            **result,
            "order": int(payload["order"]),
            "part_root": str(part_root),
        }
    except BaseException:
        writer.abort()
        raise


def _combine_indexed_parts(parts: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    profiles: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    seen_episodes: set[str] = set()
    samples = 0
    data_path = root / "data.jsonl"
    index_path = root / "data.index"
    episodes_path = root / "episodes.jsonl"
    data_digest = hashlib.sha256()
    index_digest = hashlib.sha256()
    episodes_digest = hashlib.sha256()
    with (
        data_path.open("wb") as data,
        index_path.open("wb") as index,
        episodes_path.open("wb") as episodes,
    ):
        for part in sorted(parts, key=lambda row: int(row["order"])):
            part_root = Path(str(part["part_root"]))
            data_base = data.tell()
            with (part_root / "data.jsonl").open("rb") as source:
                while chunk := source.read(16 * 1024 * 1024):
                    data_digest.update(chunk)
                    data.write(chunk)
            offsets = np.fromfile(part_root / "data.index", dtype="<u8")
            if offsets.size:
                offsets += np.uint64(data_base)
                encoded_offsets = offsets.astype("<u8", copy=False).tobytes()
                index_digest.update(encoded_offsets)
                index.write(encoded_offsets)
            with (part_root / "episodes.jsonl").open("rb") as source:
                for line in source:
                    row = json.loads(line)
                    key = str(row["episode_key"])
                    if key in seen_episodes:
                        continue
                    seen_episodes.add(key)
                    episodes_digest.update(line)
                    episodes.write(line)
            samples += int(part["samples"])
            profiles.update({key: int(value) for key, value in part["profiles"].items()})
            sources.update({key: int(value) for key, value in part["sources"].items()})
        for handle in (data, index, episodes):
            handle.flush()
            os.fsync(handle.fileno())
    if index_path.stat().st_size != samples * 8:
        raise ValueError(
            f"combined index size mismatch: {index_path.stat().st_size} != {samples * 8}"
        )
    result = {
        "samples": samples,
        "episodes": len(seen_episodes),
        "profiles": dict(sorted(profiles.items())),
        "sources": dict(sorted(sources.items())),
        "data_sha256": data_digest.hexdigest(),
        "index_sha256": index_digest.hexdigest(),
        "episodes_sha256": episodes_digest.hexdigest(),
    }
    write_json(str(root / "manifest.json"), {
        "schema_version": "v10_memory_v3_indexed_jsonl_v1",
        "jsonl_file": "data.jsonl",
        "index_file": "data.index",
        "num_samples": samples,
        "num_episodes": len(seen_episodes),
        "profiles": result["profiles"],
        "sources": result["sources"],
        "data_sha256": result["data_sha256"],
        "index_sha256": result["index_sha256"],
    })
    result["manifest_sha256"] = file_sha256(root / "manifest.json")
    return result


def _materialize_task_parallel(
    list_path: Path, root: Path, *, expected_task: str, workers: int
) -> dict[str, Any]:
    parts_root = root.parent / f"{root.name}_parts"
    parts_root.mkdir(parents=True, exist_ok=False)
    payloads = [
        {
            "order": order, "list_path": str(list_path),
            "start": start, "end": end, "expected_task": expected_task,
            "part_root": str(parts_root / f"part-{order:05d}"),
        }
        for order, (start, end) in enumerate(_byte_ranges(list_path, workers))
    ]
    results = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_materialize_ref_range, payload) for payload in payloads]
            for future in as_completed(futures):
                results.append(future.result())
        return _combine_indexed_parts(results, root)
    finally:
        shutil.rmtree(parts_root, ignore_errors=True)


def _materialize_terminal_zero_copy(
    terminal_list: Path,
    continuous_root: Path,
    terminal_root: Path,
) -> dict[str, Any]:
    terminal_root.mkdir(parents=True, exist_ok=True)
    wanted = {str(row["sample_key"]): row for row in iter_jsonl(str(terminal_list))}
    original_count = len(wanted)
    data_source = continuous_root / "data.jsonl"
    data_target = terminal_root / "data.jsonl"
    try:
        os.link(data_source, data_target)
        link_mode = "hardlink"
    except OSError:
        os.symlink(data_source, data_target)
        link_mode = "symlink"
    index = (terminal_root / "data.index").open("wb")
    episodes = (terminal_root / "episodes.jsonl").open("w", encoding="utf-8")
    seen: set[str] = set()
    profiles: Counter[str] = Counter()
    try:
        with data_source.open("rb") as data:
            while True:
                offset = data.tell()
                line = data.readline()
                if not line:
                    break
                row = json.loads(line)
                key = str(row.get("sample_key") or row.get("data_id"))
                ref = wanted.pop(key, None)
                if ref is None:
                    continue
                sample = row["v3_sample"]
                if not sample.get("is_terminal_window"):
                    raise ValueError(f"terminal list references non-terminal sample: {key}")
                index.write(struct.pack("<Q", offset))
                episode_key = str(sample["global_episode_key"])
                if episode_key not in seen:
                    seen.add(episode_key)
                    episodes.write(json.dumps({
                        "episode_key": episode_key,
                        "source": sample["source_id"],
                        "profile": sample["profile"],
                        "views": list(dict.fromkeys(image["view"] for image in sample["images"])),
                    }, ensure_ascii=False, separators=(",", ":")) + "\n")
                profiles[str(sample["profile"])] += 1
        if wanted:
            first = next(iter(wanted))
            raise ValueError(f"{len(wanted)} terminal refs missing from continuous data; first={first}")
        for handle in (index, episodes):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
    except BaseException:
        for handle in (index, episodes):
            if not handle.closed:
                handle.close()
        raise
    result = {
        "samples": original_count,
        "episodes": len(seen),
        "profiles": dict(sorted(profiles.items())),
        "data_sha256": file_sha256(data_source),
        "index_sha256": file_sha256(terminal_root / "data.index"),
        "episodes_sha256": file_sha256(terminal_root / "episodes.jsonl"),
        "data_storage": link_mode,
        "shares_data_with": "../../continuous/" + continuous_root.name + "/data.jsonl",
    }
    write_json(str(terminal_root / "manifest.json"), {
        "schema_version": "v10_memory_v3_terminal_index_v1",
        "jsonl_file": "data.jsonl",
        "index_file": "data.index",
        "num_samples": original_count,
        "num_episodes": len(seen),
        "profiles": result["profiles"],
        "data_sha256": result["data_sha256"],
        "index_sha256": result["index_sha256"],
        "data_storage": link_mode,
        "shares_data_with": result["shares_data_with"],
    })
    result["manifest_sha256"] = file_sha256(terminal_root / "manifest.json")
    return result


_DIGEST_MODULUS = 1 << 128


def _key_digest(key: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest(), "little"
    )


def _terminal_list_signature(path: Path) -> tuple[int, int, int]:
    count = xor_value = sum_value = 0
    for row in iter_jsonl(str(path)):
        value = _key_digest(str(row["sample_key"]))
        count += 1
        xor_value ^= value
        sum_value = (sum_value + value) % _DIGEST_MODULUS
    return count, xor_value, sum_value


def _terminal_data_range(payload: Mapping[str, Any]) -> dict[str, Any]:
    data_path = Path(str(payload["data_path"]))
    part_root = Path(str(payload["part_root"]))
    part_root.mkdir(parents=True, exist_ok=False)
    index = (part_root / "data.index").open("wb")
    episodes = (part_root / "episodes.jsonl").open("wb")
    seen: set[str] = set()
    profiles: Counter[str] = Counter()
    count = xor_value = sum_value = 0
    try:
        with data_path.open("rb") as data:
            start = int(payload["start"])
            end = int(payload["end"])
            data.seek(start)
            if start:
                data.seek(start - 1)
                if data.read(1) != b"\n":
                    data.readline()
            while True:
                offset = data.tell()
                if offset >= end:
                    break
                line = data.readline()
                if not line:
                    break
                row = json.loads(line)
                sample = row["v3_sample"]
                if not sample.get("is_terminal_window"):
                    continue
                key = str(row.get("sample_key") or row.get("data_id"))
                value = _key_digest(key)
                count += 1
                xor_value ^= value
                sum_value = (sum_value + value) % _DIGEST_MODULUS
                index.write(struct.pack("<Q", offset))
                episode_key = str(sample["global_episode_key"])
                if episode_key not in seen:
                    seen.add(episode_key)
                    episodes.write(json.dumps({
                        "episode_key": episode_key,
                        "source": sample["source_id"],
                        "profile": sample["profile"],
                        "views": list(dict.fromkeys(
                            image["view"] for image in sample["images"]
                        )),
                    }, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
                profiles[str(sample["profile"])] += 1
        for handle in (index, episodes):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        return {
            "order": int(payload["order"]),
            "part_root": str(part_root),
            "samples": count,
            "xor": xor_value,
            "sum": sum_value,
            "profiles": dict(profiles),
        }
    except BaseException:
        for handle in (index, episodes):
            if not handle.closed:
                handle.close()
        raise


def _materialize_terminal_zero_copy_parallel(
    terminal_list: Path,
    continuous_root: Path,
    terminal_root: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    expected = _terminal_list_signature(terminal_list)
    data_source = continuous_root / "data.jsonl"
    parts_root = terminal_root.parent / f"{terminal_root.name}_parts"
    parts_root.mkdir(parents=True, exist_ok=False)
    payloads = [
        {
            "order": order, "data_path": str(data_source),
            "start": start, "end": end,
            "part_root": str(parts_root / f"part-{order:05d}"),
        }
        for order, (start, end) in enumerate(_byte_ranges(data_source, workers))
    ]
    results = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_terminal_data_range, payload) for payload in payloads]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda row: int(row["order"]))
        observed_count = sum(int(row["samples"]) for row in results)
        observed_xor = 0
        observed_sum = 0
        for row in results:
            observed_xor ^= int(row["xor"])
            observed_sum = (observed_sum + int(row["sum"])) % _DIGEST_MODULUS
        if (observed_count, observed_xor, observed_sum) != expected:
            raise ValueError(
                "terminal list and continuous terminal rows disagree: "
                f"expected={expected} observed={(observed_count, observed_xor, observed_sum)}"
            )
        terminal_root.mkdir(parents=True, exist_ok=False)
        data_target = terminal_root / "data.jsonl"
        try:
            os.link(data_source, data_target)
            link_mode = "hardlink"
        except OSError:
            os.symlink(data_source, data_target)
            link_mode = "symlink"
        seen_episodes: set[str] = set()
        profiles: Counter[str] = Counter()
        with (
            (terminal_root / "data.index").open("wb") as index,
            (terminal_root / "episodes.jsonl").open("wb") as episodes,
        ):
            for result in results:
                part_root = Path(str(result["part_root"]))
                with (part_root / "data.index").open("rb") as source:
                    shutil.copyfileobj(source, index, length=16 * 1024 * 1024)
                with (part_root / "episodes.jsonl").open("rb") as source:
                    for line in source:
                        row = json.loads(line)
                        key = str(row["episode_key"])
                        if key in seen_episodes:
                            continue
                        seen_episodes.add(key)
                        episodes.write(line)
                profiles.update({
                    key: int(value) for key, value in result["profiles"].items()
                })
            for handle in (index, episodes):
                handle.flush()
                os.fsync(handle.fileno())
        if (terminal_root / "data.index").stat().st_size != observed_count * 8:
            raise ValueError("terminal combined index size mismatch")
        result = {
            "samples": observed_count,
            "episodes": len(seen_episodes),
            "profiles": dict(sorted(profiles.items())),
            "data_sha256": json.loads(
                (continuous_root / "manifest.json").read_text(encoding="utf-8")
            )["data_sha256"],
            "index_sha256": file_sha256(terminal_root / "data.index"),
            "episodes_sha256": file_sha256(terminal_root / "episodes.jsonl"),
            "data_storage": link_mode,
            "shares_data_with": "../../continuous/" + continuous_root.name + "/data.jsonl",
            "terminal_key_signature": {
                "algorithm": "blake2b128_count_xor_sum",
                "count": observed_count,
                "xor": f"{observed_xor:032x}",
                "sum": f"{observed_sum:032x}",
            },
        }
        write_json(str(terminal_root / "manifest.json"), {
            "schema_version": "v10_memory_v3_terminal_index_v1",
            "jsonl_file": "data.jsonl",
            "index_file": "data.index",
            "num_samples": observed_count,
            "num_episodes": len(seen_episodes),
            "profiles": result["profiles"],
            "data_sha256": result["data_sha256"],
            "index_sha256": result["index_sha256"],
            "data_storage": link_mode,
            "shares_data_with": result["shares_data_with"],
            "terminal_key_signature": result["terminal_key_signature"],
        })
        result["manifest_sha256"] = file_sha256(terminal_root / "manifest.json")
        return result
    finally:
        shutil.rmtree(parts_root, ignore_errors=True)


def publish(
    config_path: str | Path,
    *,
    merge_root: str | Path | None,
    require_complete: bool,
    workers: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    output_root = Path(str(config["output_root"]))
    if merge_root is None:
        current = json.loads((output_root / "current_merge.json").read_text(encoding="utf-8"))
        merge_root = current["root"]
    merge_root = Path(merge_root).resolve()
    merge_manifest = json.loads((merge_root / "manifest.json").read_text(encoding="utf-8"))
    if require_complete and merge_manifest.get("complete") is not True:
        raise RuntimeError("cannot finalize an incomplete V3 merge")
    publish_workers = max(1, int(
        workers or os.environ.get("MEMORY_V3_PUBLISH_WORKERS", "16")
    ))
    identity = {
        "publisher_schema_revision": 4,
        "merge_content_digest": merge_manifest["content_digest"],
        "config_digest": canonical_digest(config),
        "complete": bool(merge_manifest.get("complete")),
    }
    snapshot_id = canonical_digest(identity)[:24]
    final = output_root / "snapshots" / snapshot_id
    if (final / "_SUCCESS").is_file():
        result = json.loads((final / "manifest.json").read_text(encoding="utf-8")) | {
            "root": str(final)
        }
        write_json(str(output_root / "current_snapshot.json"), result)
        return result
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}-", dir=final.parent))
    writers: list[_IndexedWriter] = []
    try:
        list_counts = _write_split_lists(
            merge_root, temporary / "lists", workers=publish_workers
        )
        shutil.copy2(merge_root / "initial_plan_oversize.list", temporary / "initial_plan_oversize.list")
        datasets: dict[str, dict[str, Any]] = {}
        for split, suffix in (("train", "train"), ("validation", "val")):
            continuous_root = temporary / "datasets" / "continuous" / split
            plan_root = temporary / "datasets" / "initial_plan" / split
            datasets[f"continuous_{split}"] = _materialize_task_parallel(
                temporary / "lists" / f"continuous_{suffix}.list",
                continuous_root,
                expected_task="continuous",
                workers=publish_workers,
            )
            datasets[f"initial_plan_{split}"] = _materialize_task_parallel(
                temporary / "lists" / f"initial_plan_{suffix}.list",
                plan_root,
                expected_task="initial_plan",
                workers=publish_workers,
            )
            datasets[f"terminal_{split}"] = _materialize_terminal_zero_copy_parallel(
                temporary / "lists" / f"terminal_{suffix}.list",
                continuous_root,
                temporary / "datasets" / "terminal" / split,
                workers=publish_workers,
            )
        train_mix = {
            "schema_version": "memory_v3_train_mix_v1",
            "weights": {key: float(value) for key, value in config["train_mix"].items()},
            "sampling": "static_count_without_replacement",
            "integer_allocation": "largest_remainder",
        }
        write_json(str(temporary / "train_mix.json"), train_mix)
        formal = validate_formal_snapshot(config["input_snapshot"])
        statistics = {
            "schema_version": "memory_v3_snapshot_statistics_v1",
            "lists": list_counts,
            "datasets": datasets,
            "initial_plan_oversize": sum(1 for _ in iter_jsonl(str(temporary / "initial_plan_oversize.list"))),
            "builder_counts": merge_manifest.get("builder_counts") or {},
        }
        write_json(str(temporary / "statistics.json"), statistics)
        metadata = {
            "schema_version": "memory_v3_snapshot_metadata_v1",
            "snapshot_id": snapshot_id,
            "created_at": _utc_now(),
            "input_v2_snapshot": formal["root"],
            "input_v2_content_digest": formal["content_digest"],
            "merge_root": str(merge_root),
            "merge_content_digest": merge_manifest["content_digest"],
            "resize_policy_id": config["resize"]["policy_id"],
            "resize_policy": config["resize"],
            "terminal_json_storage": "shared continuous data.jsonl plus terminal-only index",
            "publish_workers": publish_workers,
        }
        write_json(str(temporary / "snapshot_metadata.json"), metadata)
        manifest = {
            "schema_version": "v10_memory_v3_snapshot_v1",
            "version": snapshot_id,
            "created_at": metadata["created_at"],
            "complete": bool(merge_manifest.get("complete")),
            "partial_inputs": bool(merge_manifest.get("partial_inputs")),
            "input_v2_content_digest": formal["content_digest"],
            "merge_content_digest": merge_manifest["content_digest"],
            "lists": {
                path.name: {"sha256": file_sha256(path), "rows": list_counts.get(
                    path.stem.replace("_val", "_validation"), 0
                )}
                for path in sorted((temporary / "lists").glob("*.list"))
            },
            "datasets": datasets,
            "train_mix": train_mix,
            "statistics_sha256": file_sha256(temporary / "statistics.json"),
            "snapshot_metadata_sha256": file_sha256(temporary / "snapshot_metadata.json"),
        }
        manifest["content_digest"] = canonical_digest(manifest)
        write_json(str(temporary / "manifest.json"), manifest)
        mark_success(temporary, {
            "schema_version": "memory_v3_snapshot_success_v1",
            "snapshot_id": snapshot_id,
            "content_digest": manifest["content_digest"],
            "complete": manifest["complete"],
        })
        if final.exists():
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, final)
        result = {**manifest, "root": str(final.resolve())}
        write_json(str(output_root / "current_snapshot.json"), result)
        return result
    except BaseException:
        for writer in writers:
            writer.abort()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "configs" / "v3_memory.yaml"))
    parser.add_argument("--merge-root")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    result = publish(
        args.config, merge_root=args.merge_root,
        require_complete=args.require_complete, workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
