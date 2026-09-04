"""Direct, resumable V3 materialized-snapshot to instruction-conditioned V4 conversion."""

from __future__ import annotations

import argparse
import array
import concurrent.futures
import hashlib
import json
import os
import shutil
import struct
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping

from .prompt import sample_to_indexed_jsonl
from .schema import SCHEMA_VERSION, SNAPSHOT_SCHEMA_VERSION, validate_target


DEFAULT_SOURCE_SNAPSHOT = Path(os.environ.get(
    "XPLANNER_CONTEXT_SNAPSHOT", "/path/to/context_snapshot"
))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get(
    "XPLANNER_ROLLOUT_DATA_ROOT", "work_dirs/rollout_data"
))
EXPECTED_V3_DIGEST = "5843030374ff0b8fe786eb7d4951e9ead022fefdbd2474536b6c14613989cb68"
KEY_CONTRACT = "memory_v4_instruction_conditioned_prompt_v2"
DATASETS = (
    ("continuous", "train"),
    ("continuous", "validation"),
    ("initial_plan", "train"),
    ("initial_plan", "validation"),
)
DEFAULT_PARTS = {
    "continuous_train": 256,
    "continuous_validation": 32,
    "initial_plan_train": 16,
    "initial_plan_validation": 4,
}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: Any, *, pretty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    text = json.dumps(
        value,
        ensure_ascii=False,
        indent=2 if pretty else None,
        sort_keys=pretty,
        separators=None if pretty else (",", ":"),
    ) + "\n"
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    _fsync_dir(path.parent)


def file_sha256(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(snapshot: Path) -> dict[str, Any]:
    value = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if value.get("schema_version") != "v10_memory_snapshot_v1":
        raise ValueError(f"not a Memory V3 snapshot: {snapshot}")
    if value.get("complete") is not True or value.get("partial_inputs") is not False:
        raise ValueError("V4 conversion requires the complete, non-partial V3 snapshot")
    if value.get("content_digest") != EXPECTED_V3_DIGEST:
        raise ValueError(
            f"unexpected V3 content digest: {value.get('content_digest')!r}"
        )
    return value


def _index_count(path: Path) -> int:
    size = path.stat().st_size
    if size % 8:
        raise ValueError(f"invalid uint64 index size: {path}: {size}")
    return size // 8


def preflight(snapshot: Path) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    manifest = _manifest(snapshot)
    datasets: dict[str, Any] = {}
    for task, split in DATASETS:
        name = f"{task}_{split}"
        root = snapshot / "datasets" / task / split
        count = _index_count(root / "data.index")
        expected = int(manifest["datasets"][name]["samples"])
        if count != expected:
            raise ValueError(f"{name}: index={count} manifest={expected}")
        datasets[name] = {
            "samples": count,
            "data_bytes": (root / "data.jsonl").stat().st_size,
            "index_bytes": (root / "data.index").stat().st_size,
        }
    return {
        "schema_version": "memory_v4_preflight_v1",
        "source_snapshot": str(snapshot),
        "source_content_digest": manifest["content_digest"],
        "source_snapshot_manifest_sha256": file_sha256(snapshot / "manifest.json"),
        "datasets": datasets,
        "raw_discovery_or_video_validation_performed": False,
        "passed": True,
    }


def build_id(snapshot: Path, parts: Mapping[str, int]) -> str:
    payload = {
        "contract": KEY_CONTRACT,
        "source": _manifest(snapshot)["content_digest"],
        "parts": dict(parts),
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()[:24]


def v4_sample_key(v3_key: str, source_digest: str = EXPECTED_V3_DIGEST) -> str:
    digest = hashlib.sha256(
        f"{KEY_CONTRACT}\0{source_digest}\0{v3_key}".encode("utf-8")
    ).hexdigest()
    return f"memory-v4-{digest}"


def transform_sample(v3_sample: Mapping[str, Any]) -> dict[str, Any]:
    if v3_sample.get("schema_version") != "memory_v3":
        raise ValueError("row does not contain a canonical memory sample")
    old_key = str(v3_sample["sample_key"])
    instruction = str(v3_sample.get("task_caption") or "").strip()
    old_target = v3_sample.get("target")
    if not instruction or not isinstance(old_target, Mapping):
        raise ValueError(f"V3 sample lacks task instruction or target: {old_key}")
    task = old_target.get("task")
    if not isinstance(task, Mapping) or task.get("caption") != instruction:
        raise ValueError(f"V3 L3 caption mismatch: {old_key}")
    task_type = str(v3_sample["task_type"])
    if task_type == "initial_plan":
        target = {"initial_plan": old_target["initial_plan"]}
    elif task_type == "continuous":
        target = {
            "task_progress_percent": task["progress_percent"],
            "predictions": old_target["predictions"],
        }
    else:
        raise ValueError(f"unsupported materialized task_type: {task_type!r}")
    new_key = v4_sample_key(old_key)
    lineage = {
        "source_version": "memory_v3",
        "source_snapshot_content_digest": EXPECTED_V3_DIGEST,
        "source_sample_key": old_key,
    }
    sample = dict(v3_sample)
    sample.pop("task_caption", None)
    sample["schema_version"] = SCHEMA_VERSION
    sample["sample_key"] = new_key
    sample["sample_id"] = new_key
    sample["task_instruction"] = instruction
    sample["target"] = target
    sample["lineage"] = lineage
    validate_target(
        target,
        str(sample["profile"]),
        task_type,
        instruction=instruction,
        is_terminal_window=bool(sample.get("is_terminal_window", False)),
    )
    return sample


def transform_row(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("v3_sample")
    if not isinstance(raw, Mapping):
        raise ValueError("indexed row is missing v3_sample")
    sample = transform_sample(raw)
    result = sample_to_indexed_jsonl(sample)
    if result["global_episode_key"] != row.get("global_episode_key"):
        raise ValueError("global_episode_key changed during V4 conversion")
    if result["image"] != row.get("image"):
        raise ValueError("image references changed during V4 conversion")
    return result


def _read_offset(index_handle: BinaryIO, row: int) -> int:
    index_handle.seek(row * 8)
    raw = index_handle.read(8)
    if len(raw) != 8:
        raise EOFError(f"missing index offset for row {row}")
    return struct.unpack("<Q", raw)[0]


def _part_ranges(count: int, parts: int) -> list[tuple[int, int]]:
    if parts <= 0:
        raise ValueError("part count must be positive")
    parts = min(parts, max(1, count))
    return [
        (count * index // parts, count * (index + 1) // parts)
        for index in range(parts)
    ]


def _episode_record(row: Mapping[str, Any]) -> dict[str, Any]:
    sample = row["v4_sample"]
    return {
        "episode_key": row["global_episode_key"],
        "source": row["source_id"],
        "profile": row["profile"],
        "views": list(dict.fromkeys(str(image["view"]) for image in sample["images"])),
    }


def _write_part(job: Mapping[str, Any]) -> dict[str, Any]:
    task = str(job["task"])
    split = str(job["split"])
    start = int(job["start"])
    end = int(job["end"])
    source_root = Path(str(job["source_root"]))
    target = Path(str(job["target"]))
    final_data_path = str(job["final_data_path"])
    if (target / "_SUCCESS").is_file():
        stats = json.loads((target / "statistics.json").read_text(encoding="utf-8"))
        if stats["start_row"] != start or stats["end_row"] != end:
            raise ValueError(f"resume part range mismatch: {target}")
        return stats
    temp = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    data_path = temp / "data.jsonl"
    index_path = temp / "data.index"
    terminal_index_path = temp / "terminal.index"
    rows_list_path = temp / "rows.list"
    terminal_list_path = temp / "terminal.list"
    episodes_path = temp / "episodes.jsonl"
    counters = {
        "sources": Counter(),
        "profiles": Counter(),
        "terminal_sources": Counter(),
        "terminal_profiles": Counter(),
    }
    terminal_count = 0
    last_episode = None
    started = time.monotonic()
    with (
        (source_root / "data.index").open("rb") as old_index,
        (source_root / "data.jsonl").open("rb") as old_data,
        data_path.open("wb") as data,
        index_path.open("wb") as index,
        terminal_index_path.open("wb") as terminal_index,
        rows_list_path.open("wb") as rows_list,
        terminal_list_path.open("wb") as terminal_list,
        episodes_path.open("wb") as episodes,
    ):
        old_data.seek(_read_offset(old_index, start))
        for global_row in range(start, end):
            raw_line = old_data.readline()
            if not raw_line:
                raise EOFError(f"unexpected source EOF at row {global_row}")
            old_row = json.loads(raw_line)
            row = transform_row(old_row)
            offset = data.tell()
            index.write(struct.pack("<Q", offset))
            data.write(_json_bytes(row) + b"\n")
            list_row = {
                "sample_key": row["sample_key"],
                "shard_path": final_data_path,
                "line_number": global_row + 1,
                "row_index": global_row,
                "global_episode_key": row["global_episode_key"],
                "split": split,
                "profile": row["profile"],
                "task_type": task,
            }
            rows_list.write(_json_bytes(list_row) + b"\n")
            if bool(row["is_terminal_window"]):
                if task != "continuous":
                    raise ValueError("only continuous data may contain terminal rows")
                terminal_index.write(struct.pack("<Q", offset))
                list_row["task_type"] = "terminal"
                terminal_list.write(_json_bytes(list_row) + b"\n")
                terminal_count += 1
                counters["terminal_sources"][str(row["source_id"])] += 1
                counters["terminal_profiles"][str(row["profile"])] += 1
            episode = str(row["global_episode_key"])
            if episode != last_episode:
                episodes.write(_json_bytes(_episode_record(row)) + b"\n")
                last_episode = episode
            counters["sources"][str(row["source_id"])] += 1
            counters["profiles"][str(row["profile"])] += 1
        handles = (data, index, terminal_index, rows_list, terminal_list, episodes)
        for handle in handles:
            handle.flush()
            os.fsync(handle.fileno())
    stats = {
        "schema_version": "memory_v4_conversion_part_v1",
        "task": task,
        "split": split,
        "start_row": start,
        "end_row": end,
        "rows": end - start,
        "terminal_rows": terminal_count,
        "data_bytes": data_path.stat().st_size,
        "sources": dict(sorted(counters["sources"].items())),
        "profiles": dict(sorted(counters["profiles"].items())),
        "terminal_sources": dict(sorted(counters["terminal_sources"].items())),
        "terminal_profiles": dict(sorted(counters["terminal_profiles"].items())),
        "elapsed_seconds": time.monotonic() - started,
        "raw_discovery_or_video_validation_performed": False,
    }
    atomic_json(temp / "statistics.json", stats)
    with (temp / "_SUCCESS").open("wb") as handle:
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_dir(temp)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp, target)
    _fsync_dir(target.parent)
    return stats


def build(
    snapshot: Path,
    output_root: Path,
    *,
    workers: int,
    parts: Mapping[str, int],
) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    output_root = output_root.resolve()
    check = preflight(snapshot)
    identity = build_id(snapshot, parts)
    root = output_root / "builds" / identity
    root.mkdir(parents=True, exist_ok=True)
    planned_snapshot = output_root / "snapshots" / identity
    manifest_path = root / "build_manifest.json"
    manifest = {
        "schema_version": "memory_v4_conversion_build_v1",
        "build_id": identity,
        "active": True,
        "source_snapshot": str(snapshot),
        "source_content_digest": EXPECTED_V3_DIGEST,
        "planned_snapshot": str(planned_snapshot),
        "workers": workers,
        "parts": dict(parts),
        "preflight": check,
        "raw_discovery_or_video_validation_performed": False,
    }
    atomic_json(manifest_path, manifest)
    jobs: list[dict[str, Any]] = []
    for task, split in DATASETS:
        name = f"{task}_{split}"
        source_root = snapshot / "datasets" / task / split
        count = _index_count(source_root / "data.index")
        final_data_path = planned_snapshot / "datasets" / task / split / "data.jsonl"
        for part_id, (start, end) in enumerate(_part_ranges(count, int(parts[name]))):
            jobs.append({
                "task": task,
                "split": split,
                "start": start,
                "end": end,
                "source_root": str(source_root),
                "target": str(root / "parts" / name / f"part-{part_id:05d}"),
                "final_data_path": str(final_data_path),
            })
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_job = {pool.submit(_write_part, job): job for job in jobs}
        for position, future in enumerate(concurrent.futures.as_completed(future_to_job), 1):
            job = future_to_job[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    json.dumps(
                        {"event": "part_complete", "position": position, "total": len(jobs), **result},
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            except Exception as exc:
                failures.append(f"{job['target']}: {type(exc).__name__}: {exc}")
                print(json.dumps({"event": "part_failed", "job": job, "error": str(exc)}), file=sys.stderr, flush=True)
    manifest.update({
        "active": False,
        "completed_parts": len(results),
        "planned_parts": len(jobs),
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started,
    })
    atomic_json(manifest_path, manifest)
    if failures or len(results) != len(jobs):
        raise RuntimeError(f"V4 build failed: {len(failures)} failures")
    atomic_json(root / "_SUCCESS", {"build_id": identity, "completed_parts": len(results)})
    return manifest


def _copy_with_digest(source: Path, target: BinaryIO, digest: hashlib._Hash) -> int:
    written = 0
    with source.open("rb") as handle:
        while chunk := handle.read(16 << 20):
            target.write(chunk)
            digest.update(chunk)
            written += len(chunk)
    return written


def _append_shifted_index(source: Path, target: BinaryIO, base: int, digest: hashlib._Hash) -> int:
    count = 0
    values = array.array("Q")
    with source.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            if len(chunk) % 8:
                raise ValueError(f"invalid part index: {source}")
            values.frombytes(chunk)
            if sys.byteorder != "little":
                values.byteswap()
            for index in range(len(values)):
                values[index] += base
            if sys.byteorder != "little":
                values.byteswap()
            encoded = values.tobytes()
            target.write(encoded)
            digest.update(encoded)
            count += len(values)
            values = array.array("Q")
    return count


def _parts(build_root: Path, name: str) -> list[Path]:
    result = sorted((build_root / "parts" / name).glob("part-*"))
    if not result or any(not (part / "_SUCCESS").is_file() for part in result):
        raise ValueError(f"incomplete V4 build parts: {name}")
    return result


def _merge_episodes(parts: Iterable[Path], target: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    last_key = None
    with target.open("wb") as output:
        for part in parts:
            with (part / "episodes.jsonl").open("rb") as handle:
                for line in handle:
                    key = json.loads(line)["episode_key"]
                    if key == last_key:
                        continue
                    output.write(line)
                    digest.update(line)
                    count += 1
                    last_key = key
        output.flush()
        os.fsync(output.fileno())
    return count, digest.hexdigest()


def _merge_dataset(
    build_root: Path,
    snapshot_temp: Path,
    task: str,
    split: str,
) -> dict[str, Any]:
    name = f"{task}_{split}"
    parts = _parts(build_root, name)
    root = snapshot_temp / "datasets" / task / split
    root.mkdir(parents=True, exist_ok=True)
    data_digest = hashlib.sha256()
    index_digest = hashlib.sha256()
    list_digest = hashlib.sha256()
    source_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    rows = 0
    byte_base = 0
    list_name = f"{task}_{'val' if split == 'validation' else 'train'}.list"
    list_target = snapshot_temp / "lists" / list_name
    list_target.parent.mkdir(parents=True, exist_ok=True)
    with (
        (root / "data.jsonl").open("wb") as data_out,
        (root / "data.index").open("wb") as index_out,
        list_target.open("wb") as list_out,
    ):
        for part in parts:
            stats = json.loads((part / "statistics.json").read_text(encoding="utf-8"))
            _copy_with_digest(part / "data.jsonl", data_out, data_digest)
            count = _append_shifted_index(part / "data.index", index_out, byte_base, index_digest)
            if count != int(stats["rows"]):
                raise ValueError(f"part index count mismatch: {part}")
            _copy_with_digest(part / "rows.list", list_out, list_digest)
            rows += count
            byte_base += int(stats["data_bytes"])
            source_counts.update(stats["sources"])
            profile_counts.update(stats["profiles"])
        for handle in (data_out, index_out, list_out):
            handle.flush()
            os.fsync(handle.fileno())
    episode_count, episodes_digest = _merge_episodes(parts, root / "episodes.jsonl")
    metadata = {
        "samples": rows,
        "episodes": episode_count,
        "sources": dict(sorted(source_counts.items())),
        "profiles": dict(sorted(profile_counts.items())),
        "data_sha256": data_digest.hexdigest(),
        "index_sha256": index_digest.hexdigest(),
        "episodes_sha256": episodes_digest,
        "list_file": f"lists/{list_name}",
        "list_sha256": list_digest.hexdigest(),
    }
    atomic_json(root / "manifest.json", metadata)
    metadata["manifest_sha256"] = file_sha256(root / "manifest.json")
    return metadata


def _publish_terminal(
    build_root: Path,
    snapshot_temp: Path,
    split: str,
    continuous: Mapping[str, Any],
) -> dict[str, Any]:
    name = f"continuous_{split}"
    parts = _parts(build_root, name)
    root = snapshot_temp / "datasets" / "terminal" / split
    root.mkdir(parents=True, exist_ok=True)
    continuous_root = snapshot_temp / "datasets" / "continuous" / split
    os.link(continuous_root / "data.jsonl", root / "data.jsonl")
    os.link(continuous_root / "episodes.jsonl", root / "episodes.jsonl")
    digest = hashlib.sha256()
    list_digest = hashlib.sha256()
    count = 0
    byte_base = 0
    source_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    list_name = f"terminal_{'val' if split == 'validation' else 'train'}.list"
    list_target = snapshot_temp / "lists" / list_name
    with (root / "data.index").open("wb") as index_out, list_target.open("wb") as list_out:
        for part in parts:
            stats = json.loads((part / "statistics.json").read_text(encoding="utf-8"))
            count += _append_shifted_index(part / "terminal.index", index_out, byte_base, digest)
            _copy_with_digest(part / "terminal.list", list_out, list_digest)
            byte_base += int(stats["data_bytes"])
            source_counts.update(stats["terminal_sources"])
            profile_counts.update(stats["terminal_profiles"])
        for handle in (index_out, list_out):
            handle.flush()
            os.fsync(handle.fileno())
    metadata = {
        "samples": count,
        "episodes": continuous["episodes"],
        "sources": dict(sorted(source_counts.items())),
        "profiles": dict(sorted(profile_counts.items())),
        "data_sha256": continuous["data_sha256"],
        "index_sha256": digest.hexdigest(),
        "episodes_sha256": continuous["episodes_sha256"],
        "data_storage": "hardlink",
        "shares_data_with": f"../../continuous/{split}/data.jsonl",
        "list_file": f"lists/{list_name}",
        "list_sha256": list_digest.hexdigest(),
    }
    atomic_json(root / "manifest.json", metadata)
    metadata["manifest_sha256"] = file_sha256(root / "manifest.json")
    return metadata


def _convert_oversize(source_snapshot: Path, snapshot_temp: Path) -> dict[str, Any]:
    source = source_snapshot / "initial_plan_oversize.list"
    target = snapshot_temp / "initial_plan_oversize.list"
    digest = hashlib.sha256()
    count = 0
    with source.open("r", encoding="utf-8") as inp, target.open("wb") as out:
        for line in inp:
            row = json.loads(line)
            old_key = str(row["sample_key"])
            row["sample_key"] = v4_sample_key(old_key)
            row["lineage"] = {
                "source_version": "memory_v3",
                "source_snapshot_content_digest": EXPECTED_V3_DIGEST,
                "source_sample_key": old_key,
            }
            encoded = _json_bytes(row) + b"\n"
            out.write(encoded)
            digest.update(encoded)
            count += 1
        out.flush()
        os.fsync(out.fileno())
    return {"rows": count, "sha256": digest.hexdigest()}


def publish(snapshot: Path, output_root: Path, parts: Mapping[str, int]) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    output_root = output_root.resolve()
    identity = build_id(snapshot, parts)
    build_root = output_root / "builds" / identity
    if not (build_root / "_SUCCESS").is_file():
        raise ValueError(f"V4 build is incomplete: {build_root}")
    target = output_root / "snapshots" / identity
    if (target / "_SUCCESS").is_file():
        return json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    temp = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    datasets: dict[str, Any] = {}
    for task, split in DATASETS:
        name = f"{task}_{split}"
        datasets[name] = _merge_dataset(build_root, temp, task, split)
    for split in ("train", "validation"):
        datasets[f"terminal_{split}"] = _publish_terminal(
            build_root, temp, split, datasets[f"continuous_{split}"]
        )
    oversize = _convert_oversize(snapshot, temp)
    v3_manifest = _manifest(snapshot)
    for name, metadata in datasets.items():
        expected = v3_manifest["datasets"][name]
        for field in ("samples", "episodes"):
            if int(metadata[field]) != int(expected[field]):
                raise ValueError(
                    f"V4 publish invariant failed: {name}.{field}="
                    f"{metadata[field]} expected={expected[field]}"
                )
        if metadata["profiles"] != expected["profiles"]:
            raise ValueError(f"V4 publish profile counts changed: {name}")
        if "sources" in expected and metadata["sources"] != expected["sources"]:
            raise ValueError(f"V4 publish source counts changed: {name}")
    expected_oversize = int(
        json.loads((snapshot / "statistics.json").read_text(encoding="utf-8"))[
            "initial_plan_oversize"
        ]
    )
    if oversize["rows"] != expected_oversize:
        raise ValueError(
            f"V4 oversize count changed: {oversize['rows']} expected={expected_oversize}"
        )
    train_mix = dict(v3_manifest["train_mix"])
    train_mix["schema_version"] = "memory_v4_train_mix_v1"
    atomic_json(temp / "train_mix.json", train_mix)
    v3_meta = json.loads((snapshot / "snapshot_metadata.json").read_text(encoding="utf-8"))
    snapshot_metadata = {
        "schema_version": "memory_v4_snapshot_metadata_v1",
        "snapshot_id": identity,
        "source_v3_snapshot": str(snapshot),
        "source_v3_content_digest": EXPECTED_V3_DIGEST,
        "conversion_build": str(build_root),
        "conversion_contract": KEY_CONTRACT,
        "instruction_field": "task_instruction",
        "assistant_outputs_l3": False,
        "raw_discovery_or_video_validation_performed": False,
        "resize_policy": v3_meta["resize_policy"],
        "resize_policy_id": v3_meta["resize_policy_id"],
        "terminal_json_storage": "shared V4 continuous data.jsonl plus terminal-only index",
    }
    atomic_json(temp / "snapshot_metadata.json", snapshot_metadata)
    statistics = {
        "schema_version": "memory_v4_snapshot_statistics_v1",
        "datasets": datasets,
        "initial_plan_oversize": oversize["rows"],
        "source_v3_counts_preserved": True,
    }
    atomic_json(temp / "statistics.json", statistics)
    digest_payload = {
        "schema": SNAPSHOT_SCHEMA_VERSION,
        "source": EXPECTED_V3_DIGEST,
        "datasets": {
            name: {
                key: value[key]
                for key in ("samples", "data_sha256", "index_sha256", "episodes_sha256")
            }
            for name, value in datasets.items()
        },
        "oversize": oversize,
        "train_mix": train_mix,
    }
    content_digest = hashlib.sha256(_json_bytes(digest_payload)).hexdigest()
    manifest = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "version": identity,
        "complete": True,
        "partial_inputs": False,
        "content_digest": content_digest,
        "source_v3_content_digest": EXPECTED_V3_DIGEST,
        "source_v3_snapshot": str(snapshot),
        "datasets": datasets,
        "initial_plan_oversize": oversize,
        "train_mix": train_mix,
        "snapshot_metadata_sha256": file_sha256(temp / "snapshot_metadata.json"),
        "statistics_sha256": file_sha256(temp / "statistics.json"),
        "raw_discovery_or_video_validation_performed": False,
    }
    atomic_json(temp / "manifest.json", manifest)
    atomic_json(temp / "_SUCCESS", {"snapshot_id": identity, "content_digest": content_digest})
    _fsync_dir(temp)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp, target)
    _fsync_dir(target.parent)
    atomic_json(output_root / "current_snapshot.json", {
        "snapshot_id": identity,
        "snapshot": str(target),
        "content_digest": content_digest,
    })
    return manifest


def validate_snapshot(snapshot: Path, *, samples_per_dataset: int = 100) -> dict[str, Any]:
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("not a Memory V4 snapshot")
    checked = 0
    failures: list[str] = []
    for task in ("continuous", "initial_plan", "terminal"):
        for split in ("train", "validation"):
            root = snapshot / "datasets" / task / split
            count = _index_count(root / "data.index")
            rows = sorted(set([0, count - 1] + [count * i // samples_per_dataset for i in range(samples_per_dataset)]))
            with (root / "data.index").open("rb") as index, (root / "data.jsonl").open("rb") as data:
                for row_index in rows:
                    try:
                        data.seek(_read_offset(index, row_index))
                        row = json.loads(data.readline())
                        sample = row["v4_sample"]
                        instruction = sample["task_instruction"]
                        assistant = json.loads(row["text"][1]["text"])
                        validate_target(
                            assistant,
                            str(row["profile"]),
                            task,
                            instruction=instruction,
                            is_terminal_window=(task == "terminal" or bool(row["is_terminal_window"])),
                        )
                        encoded_instruction = json.dumps(instruction, ensure_ascii=False)
                        if encoded_instruction not in row["text"][0]["text"]:
                            raise ValueError("user prompt omits exact task instruction")
                        if "\"task\":" in row["text"][1]["text"] or "\"task_caption\":" in row["text"][1]["text"]:
                            raise ValueError("assistant contains forbidden L3 structure")
                        checked += 1
                    except Exception as exc:
                        failures.append(f"{task}/{split} row={row_index}: {type(exc).__name__}: {exc}")
    report = {
        "schema_version": "memory_v4_snapshot_validation_v1",
        "snapshot": str(snapshot),
        "content_digest": manifest["content_digest"],
        "checked_rows": checked,
        "failures": failures,
        "passed": not failures,
    }
    atomic_json(snapshot / "validation_report.json", report)
    if failures:
        raise ValueError(f"V4 validation failed: {failures[:3]}")
    return report


def _parse_parts(value: str) -> dict[str, int]:
    result = dict(DEFAULT_PARTS)
    if value:
        for item in value.split(","):
            name, raw = item.split("=", 1)
            if name not in result:
                raise ValueError(f"unknown dataset part key: {name}")
            result[name] = int(raw)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "build", "publish", "validate", "all"))
    parser.add_argument("--source-snapshot", type=Path, default=DEFAULT_SOURCE_SNAPSHOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--parts", default="")
    parser.add_argument("--samples-per-dataset", type=int, default=100)
    args = parser.parse_args()
    parts = _parse_parts(args.parts)
    if args.command == "preflight":
        result = preflight(args.source_snapshot)
    elif args.command == "build":
        result = build(args.source_snapshot, args.output_root, workers=args.workers, parts=parts)
    elif args.command == "publish":
        result = publish(args.source_snapshot, args.output_root, parts)
    elif args.command == "validate":
        target = args.snapshot or args.output_root / "snapshots" / build_id(args.source_snapshot, parts)
        result = validate_snapshot(target, samples_per_dataset=args.samples_per_dataset)
    else:
        build(args.source_snapshot, args.output_root, workers=args.workers, parts=parts)
        publish(args.source_snapshot, args.output_root, parts)
        target = args.output_root / "snapshots" / build_id(args.source_snapshot, parts)
        result = validate_snapshot(target, samples_per_dataset=args.samples_per_dataset)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
