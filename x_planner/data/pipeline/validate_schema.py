#!/usr/bin/env python3
"""Validate V10 target JSON or every sample in an immutable snapshot."""

from __future__ import annotations

import argparse
import json
import os
import struct
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .captions import normalize_caption
from .models import V10Sample
from .schema import json_schema_document, loads_target, validate_target


def validate_row(row: dict[str, Any]) -> None:
    profile = row.get("profile")
    dialogue = row.get("text")
    if not isinstance(dialogue, list) or len(dialogue) != 2:
        raise ValueError("text must be a two-turn dialogue")
    if dialogue[0].get("role") != "user" or dialogue[1].get("role") != "assistant":
        raise ValueError("dialogue roles must be user then assistant")
    target = loads_target(dialogue[1].get("text", ""), profile)
    sample = V10Sample.from_dict(row["v10_sample"])
    validate_target(sample.target, sample.profile, enforce_key_order=False)
    if target != sample.target:
        raise ValueError("assistant target differs from v10_sample target")
    refs = row.get("image")
    if not isinstance(refs, list) or not 1 <= len(refs) <= 9:
        raise ValueError("image count must be in [1, 9]")
    if dialogue[0].get("text", "").count("<image>") != len(refs):
        raise ValueError("image placeholder count mismatch")
    frames = [ref.get("frame") for ref in refs]
    views = [ref.get("view") for ref in refs]
    if any(isinstance(frame, bool) or not isinstance(frame, int) for frame in frames):
        raise ValueError("invalid frame index")
    if any(frame > sample.current_frame for frame in frames):
        raise ValueError("future visual frame detected")
    if frames[-1] != sample.current_frame:
        raise ValueError("current frame is not last")
    if len(set(frames)) > 3 or len(set(views)) > 3:
        raise ValueError("temporal/view limit exceeded")
    per_frame = {
        frame: tuple(view for current, view in zip(frames, views) if current == frame)
        for frame in sorted(set(frames))
    }
    if len(set(per_frame.values())) != 1:
        raise ValueError("views differ across temporal steps")
    normalized_history = tuple(normalize_caption(item) for item in sample.long_memory)
    if normalized_history != sample.long_memory:
        raise ValueError("memory is not normalized")


def _validate_snapshot_serial(root: Path) -> dict[str, Any]:
    failures: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for split in ("train", "validation"):
        data_path = root / split / "data.jsonl"
        index_path = root / split / "data.index"
        offsets: list[int] = []
        with index_path.open("rb") as handle:
            while raw := handle.read(8):
                if len(raw) != 8:
                    raise ValueError(f"truncated index in {split}")
                offsets.append(struct.unpack("<Q", raw)[0])
        with data_path.open("rb") as handle:
            for row_index, offset in enumerate(offsets):
                handle.seek(offset)
                try:
                    row = json.loads(handle.readline())
                    validate_row(row)
                    counts[f"{split}_samples"] += 1
                    counts[f"profile:{row['profile']}"] += 1
                except Exception as exc:
                    failures[f"{type(exc).__name__}: {str(exc)[:200]}"] += 1
                    if sum(failures.values()) >= 20:
                        raise ValueError(f"snapshot validation failures: {dict(failures)}") from exc
    if failures:
        raise ValueError(f"snapshot validation failures: {dict(failures)}")
    return {"valid": True, "counts": dict(sorted(counts.items()))}


def _validate_index_range(
    root: Path,
    split: str,
    start: int,
    stop: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """Validate one contiguous index range without materialising all offsets."""
    counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    data_path = root / split / "data.jsonl"
    index_path = root / split / "data.index"
    with index_path.open("rb") as index_handle, data_path.open("rb") as data_handle:
        index_handle.seek(start * 8)
        for row_index in range(start, stop):
            raw = index_handle.read(8)
            if len(raw) != 8:
                failures[f"truncated index in {split} at row {row_index}"] += 1
                break
            offset = struct.unpack("<Q", raw)[0]
            data_handle.seek(offset)
            try:
                line = data_handle.readline()
                if not line:
                    raise ValueError(f"empty data row at offset {offset}")
                row = json.loads(line)
                validate_row(row)
                counts[f"{split}_samples"] += 1
                counts["profile:" + str(row.get("profile"))] += 1
            except Exception as exc:
                failures[f"{type(exc).__name__}: {str(exc)[:200]}"] += 1
                if sum(failures.values()) >= 20:
                    break
    return dict(counts), dict(failures)


def _snapshot_split_sizes(root: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for split in ("train", "validation"):
        index_size = (root / split / "data.index").stat().st_size
        if index_size % 8:
            raise ValueError(f"truncated index in {split}: {index_size} bytes")
        sizes[split] = index_size // 8
    return sizes


def _validate_snapshot_parallel(root: Path, num_workers: int) -> dict[str, Any]:
    """Validate a large immutable snapshot with deterministic range workers."""
    sizes = _snapshot_split_sizes(root)
    tasks: list[tuple[str, int, int]] = []
    for split in ("train", "validation"):
        total = sizes[split]
        if not total:
            continue
        task_count = min(num_workers, total)
        chunk = (total + task_count - 1) // task_count
        for start in range(0, total, chunk):
            tasks.append((split, start, min(start + chunk, total)))

    counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(_validate_index_range, root, split, start, stop):
            (split, start, stop)
            for split, start, stop in tasks
        }
        for future in as_completed(futures):
            part_counts, part_failures = future.result()
            counts.update(part_counts)
            failures.update(part_failures)

    if failures:
        raise ValueError(f"snapshot validation failures: {dict(failures)}")
    for split, total in sizes.items():
        actual = counts[f"{split}_samples"]
        if actual != total:
            raise ValueError(
                f"snapshot validation count mismatch in {split}: {actual} != {total}"
            )
    return {"valid": True, "counts": dict(sorted(counts.items()))}


def validate_snapshot(root: Path) -> dict[str, Any]:
    """Validate a snapshot; large snapshots use bounded deterministic parallelism."""
    root = root.resolve()
    sizes = _snapshot_split_sizes(root)
    total = sum(sizes.values())
    configured = os.environ.get("V10_VALIDATE_WORKERS")
    if configured is None:
        num_workers = min(16, os.cpu_count() or 1) if total >= 1_000_000 else 1
    else:
        try:
            num_workers = int(configured)
        except ValueError as exc:
            raise ValueError("V10_VALIDATE_WORKERS must be an integer") from exc
        if not 1 <= num_workers <= 64:
            raise ValueError("V10_VALIDATE_WORKERS must be in [1, 64]")
    if num_workers == 1:
        return _validate_snapshot_serial(root)
    return _validate_snapshot_parallel(root, num_workers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--snapshot", type=Path)
    group.add_argument("--json-file", type=Path)
    group.add_argument("--write-schema", type=Path)
    parser.add_argument("--profile")
    args = parser.parse_args()
    if args.write_schema:
        args.write_schema.parent.mkdir(parents=True, exist_ok=True)
        args.write_schema.write_text(
            json.dumps(json_schema_document(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result = {"schema": str(args.write_schema)}
    elif args.json_file:
        if not args.profile:
            parser.error("--profile is required with --json-file")
        value = json.loads(args.json_file.read_text(encoding="utf-8"))
        validate_target(value, args.profile)
        result = {"valid": True, "profile": args.profile}
    else:
        result = validate_snapshot(args.snapshot.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
