"""Semantic and storage smoke validation for all Memory V3 task datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator

from .common_v3 import file_sha256, iter_jsonl, write_json
from .prompt_v3 import render_user
from .schema_v3 import (
    TERMINAL_CAPTION,
    prediction_one_state,
    validate_continuous_target,
    validate_initial_plan,
    validate_short_memory,
)


def _contains_exact_caption(value: Any, caption: str) -> bool:
    if isinstance(value, dict):
        return value.get("caption") == caption or any(
            _contains_exact_caption(item, caption) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_exact_caption(item, caption) for item in value)
    return False


_FINGERPRINT_MODULUS = 1 << 128


def _add_key_fingerprint(fingerprint: list[int], key: str) -> None:
    """Update the same order-independent terminal-key fingerprint as publisher."""
    value = int.from_bytes(
        hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest(), "little"
    )
    fingerprint[0] += 1
    fingerprint[1] ^= value
    fingerprint[2] = (fingerprint[2] + value) % _FINGERPRINT_MODULUS


def _samples(snapshot: Path, task: str, split: str) -> Iterator[dict[str, Any]]:
    path = snapshot / "datasets" / task / split / "data.jsonl"
    index_path = snapshot / "datasets" / task / split / "data.index"
    count = index_path.stat().st_size // 8
    # Terminal data.jsonl is shared with continuous and only its index selects
    # rows, so use the index offsets rather than walking every physical line.
    if task == "terminal":
        import struct

        with path.open("rb") as data, index_path.open("rb") as index:
            for _ in range(count):
                raw = index.read(8)
                if len(raw) != 8:
                    raise EOFError(index_path)
                data.seek(struct.unpack("<Q", raw)[0])
                wrapped = json.loads(data.readline())
                yield wrapped["v3_sample"]
        return
    for wrapped in iter_jsonl(str(path)):
        yield wrapped["v3_sample"]


def _byte_ranges(path: Path, workers: int) -> list[tuple[int, int]]:
    size = path.stat().st_size
    count = max(1, min(int(workers), size or 1))
    return [
        (size * index // count, size * (index + 1) // count)
        for index in range(count)
    ]


def _samples_in_range(path: Path, start: int, end: int) -> Iterator[dict[str, Any]]:
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
            yield json.loads(line)["v3_sample"]


def _validate_continuous_payload(sample: dict[str, Any]) -> dict[str, Any]:
    if sample["task_type"] != "continuous":
        raise ValueError("continuous dataset has a non-continuous sample")
    validate_continuous_target(
        sample["target"], sample["profile"],
        is_terminal_window=bool(sample["is_terminal_window"]),
    )
    validate_short_memory(sample.get("short_memory"))
    prompt = render_user(sample)
    if "seconds ago" in prompt or "second ago" in prompt:
        raise ValueError(f"seconds leaked into prompt: {sample['sample_key']}")
    has_rate = "Source frame rate: 20 Hz." in prompt
    if has_rate != (sample["source_id"] == "zhengwei"):
        raise ValueError(f"source frame-rate line mismatch: {sample['sample_key']}")
    views = list(dict.fromkeys(image["view"] for image in sample["images"]))
    per_frame: dict[int, list[str]] = defaultdict(list)
    for image in sample["images"]:
        if int(image["frame"]) > int(sample["anchor_frame"]):
            raise ValueError("future visual frame")
        if int(image["relative_frame"]) not in {-20, -10, 0}:
            raise ValueError("unexpected frame offset")
        per_frame[int(image["frame"])].append(str(image["view"]))
    if any(frame_views != views for frame_views in per_frame.values()):
        raise ValueError("multi-view frames do not share a synchronized anchor")
    return prediction_one_state(sample["target"], sample["profile"])


def _continuous_range(payload: dict[str, Any]) -> dict[str, Any]:
    path = Path(payload["path"])
    start = int(payload["start"])
    current_episode: str | None = None
    closed_episodes: set[str] = set()
    episode_order: list[str] = []
    previous_grid_index: int | None = None
    previous_anchor: int | None = None
    previous_state: dict[str, Any] | None = None
    forced_seen = False
    seen_anchors: set[int] = set()
    first: dict[str, Any] | None = None
    last: dict[str, Any] | None = None
    profiles: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    terminal_fingerprint = [0, 0, 0]
    samples = forced = skipped = 0
    for sample in _samples_in_range(path, start, int(payload["end"])):
        state = _validate_continuous_payload(sample)
        episode_key = str(sample["global_episode_key"])
        first_in_range = first is None
        if episode_key != current_episode:
            if current_episode is not None:
                closed_episodes.add(current_episode)
            if episode_key in closed_episodes:
                raise ValueError(f"non-contiguous Episode rows: {episode_key}")
            current_episode = episode_key
            episode_order.append(episode_key)
            previous_grid_index = None
            previous_anchor = None
            previous_state = None
            forced_seen = False
            seen_anchors = set()
        anchor = int(sample["anchor_frame"])
        if anchor in seen_anchors:
            raise ValueError(f"duplicate continuous anchor: {episode_key}:{anchor}")
        seen_anchors.add(anchor)
        is_forced = bool(sample.get("forced_terminal_anchor"))
        grid_index: int | None = None
        if is_forced:
            if forced_seen:
                raise ValueError(f"multiple forced terminal anchors: {episode_key}")
            if not sample["is_terminal_window"]:
                raise ValueError(f"forced anchor is not terminal: {episode_key}")
            if sample.get("anchor_grid_index") is not None:
                raise ValueError(f"forced anchor has a grid index: {episode_key}")
            if previous_anchor is not None and anchor <= previous_anchor:
                raise ValueError(f"forced anchor is not final: {episode_key}")
            forced_seen = True
        else:
            if forced_seen:
                raise ValueError(f"normal anchor follows forced terminal: {episode_key}")
            origin = int(sample["anchor_grid_origin"])
            stride = int(sample["anchor_stride_frames"])
            grid_index = int(sample["anchor_grid_index"])
            if stride != 20 or anchor != origin + grid_index * stride:
                raise ValueError(f"normal anchor is off the 20-frame grid: {episode_key}")
            if previous_grid_index is not None:
                if grid_index <= previous_grid_index:
                    raise ValueError(f"non-increasing anchor grid: {episode_key}")
                skipped += grid_index - previous_grid_index - 1
            previous_grid_index = grid_index
        expected_short = [] if previous_state is None else [previous_state]
        if not (first_in_range and start > 0):
            if sample.get("short_memory") != expected_short:
                raise ValueError(f"short memory is not previous Prediction 1: {episode_key}")
        boundary = {
            "episode": episode_key,
            "anchor": anchor,
            "grid_index": grid_index,
            "forced": is_forced,
            "short_memory": sample.get("short_memory"),
            "state": state,
        }
        if first is None:
            first = boundary
        last = boundary
        previous_anchor = anchor
        previous_state = state
        key = str(sample["sample_key"])
        if sample["is_terminal_window"]:
            _add_key_fingerprint(terminal_fingerprint, key)
        profiles[str(sample["profile"])] += 1
        sources[str(sample["source_id"])] += 1
        samples += 1
        forced += int(is_forced)
    return {
        "order": int(payload["order"]),
        "samples": samples,
        "forced": forced,
        "skipped": skipped,
        "profiles": dict(profiles),
        "sources": dict(sources),
        "terminal_fingerprint": terminal_fingerprint,
        "episode_order": episode_order,
        "first": first,
        "last": last,
    }


def _terminal_index_range(payload: dict[str, Any]) -> dict[str, Any]:
    data_path = Path(payload["data_path"])
    index_path = Path(payload["index_path"])
    start = int(payload["start"])
    end = int(payload["end"])
    fingerprint = [0, 0, 0]
    with data_path.open("rb") as data, index_path.open("rb") as index:
        index.seek(start * 8)
        for _ in range(start, end):
            raw = index.read(8)
            if len(raw) != 8:
                raise EOFError(index_path)
            import struct
            data.seek(struct.unpack("<Q", raw)[0])
            sample = json.loads(data.readline())["v3_sample"]
            key = str(sample["sample_key"])
            if sample["task_type"] != "continuous" or not sample["is_terminal_window"]:
                raise ValueError(f"terminal index mismatch: {key}")
            _add_key_fingerprint(fingerprint, key)
    return {"order": int(payload["order"]), "fingerprint": fingerprint}


def _validate_continuous_split(
    snapshot: Path, split: str, workers: int
) -> dict[str, Any]:
    data_path = snapshot / "datasets" / "continuous" / split / "data.jsonl"
    payloads = [
        {"order": order, "path": str(data_path), "start": start, "end": end}
        for order, (start, end) in enumerate(_byte_ranges(data_path, workers))
    ]
    if len(payloads) == 1:
        parts = [_continuous_range(payloads[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(payloads)) as pool:
            parts = list(pool.map(_continuous_range, payloads))
    parts.sort(key=lambda row: int(row["order"]))

    episodes: set[str] = set()
    closed_episodes: set[str] = set()
    current_episode: str | None = None
    previous_boundary: dict[str, Any] | None = None
    profiles: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    terminal_fingerprint = [0, 0, 0]
    samples = forced = skipped = 0
    for part in parts:
        first = part["first"]
        if first is None:
            continue
        if previous_boundary is None:
            if first["short_memory"] != []:
                raise ValueError(f"first Short Memory is not empty: {first['episode']}")
        elif first["episode"] == previous_boundary["episode"]:
            if first["short_memory"] != [previous_boundary["state"]]:
                raise ValueError(
                    f"cross-range Short Memory mismatch: {first['episode']}"
                )
            if previous_boundary["forced"]:
                raise ValueError(
                    f"sample follows forced terminal anchor: {first['episode']}"
                )
            if int(first["anchor"]) <= int(previous_boundary["anchor"]):
                raise ValueError(f"cross-range non-increasing anchor: {first['episode']}")
            if not first["forced"]:
                previous_index = previous_boundary["grid_index"]
                current_index = first["grid_index"]
                if previous_index is None or current_index is None or current_index <= previous_index:
                    raise ValueError(
                        f"cross-range non-increasing grid: {first['episode']}"
                    )
                skipped += int(current_index) - int(previous_index) - 1
        elif first["short_memory"] != []:
            raise ValueError(f"new Episode Short Memory is not empty: {first['episode']}")

        for episode_key in part["episode_order"]:
            if episode_key == current_episode:
                continue
            if current_episode is not None:
                closed_episodes.add(current_episode)
            if episode_key in closed_episodes:
                raise ValueError(f"non-contiguous Episode rows: {episode_key}")
            current_episode = episode_key
            episodes.add(episode_key)
        previous_boundary = part["last"]
        samples += int(part["samples"])
        forced += int(part["forced"])
        skipped += int(part["skipped"])
        profiles.update({key: int(value) for key, value in part["profiles"].items()})
        sources.update({key: int(value) for key, value in part["sources"].items()})
        fingerprint = part["terminal_fingerprint"]
        terminal_fingerprint[0] += int(fingerprint[0])
        terminal_fingerprint[1] ^= int(fingerprint[1])
        terminal_fingerprint[2] = (
            terminal_fingerprint[2] + int(fingerprint[2])
        ) % _FINGERPRINT_MODULUS

    terminal_data = snapshot / "datasets" / "terminal" / split / "data.jsonl"
    terminal_index = snapshot / "datasets" / "terminal" / split / "data.index"
    terminal_count = terminal_index.stat().st_size // 8
    terminal_workers = max(1, min(workers, terminal_count or 1))
    terminal_payloads = [
        {
            "order": order,
            "data_path": str(terminal_data),
            "index_path": str(terminal_index),
            "start": terminal_count * order // terminal_workers,
            "end": terminal_count * (order + 1) // terminal_workers,
        }
        for order in range(terminal_workers)
    ]
    if len(terminal_payloads) == 1:
        terminal_parts = [_terminal_index_range(terminal_payloads[0])]
    else:
        with ProcessPoolExecutor(max_workers=len(terminal_payloads)) as pool:
            terminal_parts = list(pool.map(_terminal_index_range, terminal_payloads))
    observed = [0, 0, 0]
    for part in terminal_parts:
        fingerprint = part["fingerprint"]
        observed[0] += int(fingerprint[0])
        observed[1] ^= int(fingerprint[1])
        observed[2] = (observed[2] + int(fingerprint[2])) % _FINGERPRINT_MODULUS
    if observed != terminal_fingerprint:
        raise ValueError(f"terminal key set mismatch for {split}")
    return {
        "episodes": episodes,
        "samples": samples,
        "terminal_samples": terminal_count,
        "forced": forced,
        "skipped": skipped,
        "profiles": profiles,
        "sources": sources,
        "terminal_fingerprint": terminal_fingerprint,
    }


def validate(
    snapshot: Path, output: Path | None = None, *, workers: int = 1
) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "v10_memory_v3_snapshot_v1":
        raise ValueError("not a Memory V3 snapshot")
    for name, record in manifest["lists"].items():
        path = snapshot / "lists" / name
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"list checksum mismatch: {name}")

    split_episodes: dict[str, set[str]] = {"train": set(), "validation": set()}
    expected_terminal_fingerprints: dict[str, list[int]] = {}
    counts: Counter[str] = Counter()
    profiles: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    forced = 0
    skipped_unlabelled_grid_anchors = 0
    for split in ("train", "validation"):
        validated = _validate_continuous_split(snapshot, split, max(1, int(workers)))
        split_episodes[split] = validated["episodes"]
        counts[f"continuous_{split}"] = int(validated["samples"])
        counts[f"terminal_{split}"] = int(validated["terminal_samples"])
        forced += int(validated["forced"])
        skipped_unlabelled_grid_anchors += int(validated["skipped"])
        profiles.update(validated["profiles"])
        sources.update(validated["sources"])
        expected_terminal_fingerprints[split] = validated["terminal_fingerprint"]

    leakage = split_episodes["train"].intersection(split_episodes["validation"])
    if leakage:
        raise ValueError(f"Episode split leakage: {len(leakage)} keys; first={next(iter(leakage))}")

    plan_episodes: dict[str, set[str]] = {"train": set(), "validation": set()}
    for split in ("train", "validation"):
        for sample in _samples(snapshot, "initial_plan", split):
            validate_initial_plan(sample["target"], sample["profile"])
            if "long_memory" in sample or "short_memory" in sample:
                raise ValueError("Initial Plan sample contains Memory")
            encoded = json.dumps(sample["target"], ensure_ascii=False)
            if "progress_percent" in encoded or _contains_exact_caption(
                sample["target"], TERMINAL_CAPTION
            ):
                raise ValueError("Initial Plan contains progress or terminal sentinel")
            episode_key = str(sample["global_episode_key"])
            if episode_key in plan_episodes[split]:
                raise ValueError(f"multiple Initial Plans for {episode_key}")
            if episode_key not in split_episodes[split]:
                raise ValueError(f"Initial Plan split disagrees with continuous: {episode_key}")
            plan_episodes[split].add(episode_key)
            counts[f"initial_plan_{split}"] += 1

    oversize = {
        "train": set(), "validation": set(),
    }
    for row in iter_jsonl(str(snapshot / "initial_plan_oversize.list")):
        split = str(row["split"])
        key = str(row["global_episode_key"])
        if key in oversize[split] or key in plan_episodes[split]:
            raise ValueError(f"duplicate plan/oversize Episode: {key}")
        oversize[split].add(key)
    for split in ("train", "validation"):
        if plan_episodes[split] | oversize[split] != split_episodes[split]:
            raise ValueError(f"Initial Plan coverage mismatch for {split}")

    for split in ("train", "validation"):
        continuous_data = snapshot / "datasets" / "continuous" / split / "data.jsonl"
        terminal_data = snapshot / "datasets" / "terminal" / split / "data.jsonl"
        if os.stat(continuous_data).st_ino != os.stat(terminal_data).st_ino:
            raise ValueError(f"terminal JSON is not zero-copy for {split}")

    report = {
        "schema_version": "memory_v3_smoke_report_v1",
        "snapshot": str(snapshot),
        "manifest_digest": manifest["content_digest"],
        "complete": manifest["complete"],
        "partial_inputs": manifest["partial_inputs"],
        "counts": dict(sorted(counts.items())),
        "profiles": dict(sorted(profiles.items())),
        "sources": dict(sorted(sources.items())),
        "forced_terminal_anchors": forced,
        "skipped_unlabelled_grid_anchors": skipped_unlabelled_grid_anchors,
        "train_validation_episode_overlap": 0,
        "initial_plan_oversize": sum(len(value) for value in oversize.values()),
        "terminal_zero_copy": True,
        "terminal_key_fingerprint": {
            split: {
                "algorithm": "blake2b128_count_xor_sum",
                "count": value[0],
                "xor": f"{value[1]:032x}",
                "sum": f"{value[2]:032x}",
            }
            for split, value in sorted(expected_terminal_fingerprints.items())
        },
        "json_profile_semantic_prompt_smoke": "passed",
    }
    if output is not None:
        write_json(str(output), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--workers", type=int,
        default=int(os.environ.get("MEMORY_V3_SMOKE_WORKERS", "16")),
    )
    args = parser.parse_args()
    result = validate(args.snapshot, args.output, workers=args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
