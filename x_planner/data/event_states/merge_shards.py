"""Merge contiguous V5.3 index-materialization shards without re-encoding data."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, BinaryIO, Callable
import uuid

from .bucket_io import atomic_json, file_sha256
from .holdout import (
    EvaluationHoldout,
    DEFAULT_EVALUATION_MANIFEST,
    DEFAULT_EVALUATION_SHA256,
)
from .prompt import prompt_renderer_digest_v53


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _counter_add(target: Counter[str], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        target[str(key)] += int(value)


def _atomic_binary(
    path: Path,
    writer: Callable[[BinaryIO], int],
) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.fchmod(descriptor, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            count = writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return count, file_sha256(path)


def _concat(paths: Sequence[Path], destination: Path) -> tuple[int, str]:
    def write(handle: BinaryIO) -> int:
        lines = 0
        for path in paths:
            with path.open("rb") as source:
                while chunk := source.read(8 * 1024 * 1024):
                    handle.write(chunk)
                    lines += chunk.count(b"\n")
        return lines

    return _atomic_binary(destination, write)


def _merge_contracts(paths: Sequence[Path], destination: Path) -> tuple[int, str]:
    def write(handle: BinaryIO) -> int:
        seen: set[tuple[str, str, str, str, str]] = set()
        count = 0
        for path in paths:
            with path.open("rb") as source:
                for raw in source:
                    if not raw.strip():
                        continue
                    row = json.loads(raw)
                    key = (
                        str(row.get("training_bucket") or ""),
                        str(row.get("split") or ""),
                        str(row.get("category") or ""),
                        str(row.get("context_variant") or ""),
                        str(row.get("output_profile_id") or ""),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    handle.write(raw if raw.endswith(b"\n") else raw + b"\n")
                    count += 1
        return count

    return _atomic_binary(destination, write)


def _merge_frame_cache(paths: Sequence[Path], destination: Path) -> tuple[int, str]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row["path"])
                previous = rows.get(key)
                if previous is not None and previous != row:
                    raise ValueError(f"conflicting frame-cache rows for {key}")
                rows[key] = row

    def write(handle: BinaryIO) -> int:
        for key in sorted(rows):
            handle.write(
                (json.dumps(rows[key], separators=(",", ":")) + "\n").encode("utf-8")
            )
        return len(rows)

    return _atomic_binary(destination, write)


def merge_shards(
    *,
    shard_roots: Sequence[Path],
    output_root: Path,
    instruction_index: Path,
    expected_index_rows: int,
    holdout: EvaluationHoldout,
) -> dict[str, Any]:
    if len(shard_roots) < 2:
        raise ValueError("at least two shard roots are required")
    if expected_index_rows <= 0:
        raise ValueError("expected_index_rows must be positive")
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")

    shard_rows: list[tuple[int, int, Path, dict[str, Any]]] = []
    for root in shard_roots:
        root = root.resolve()
        manifest = _load(root / "manifest.json")
        provenance = manifest.get("provenance") or {}
        start = int(provenance.get("line_start") or 0)
        end = int(provenance.get("line_end") or 0)
        if manifest.get("complete") is not True:
            raise ValueError(f"incomplete shard: {root}")
        if provenance.get("partial") is not True:
            raise ValueError(f"shard is not marked partial: {root}")
        if provenance.get("include_robodojo") is not False:
            raise ValueError(f"RoboDojo-enabled shard is forbidden: {root}")
        if start <= 0 or end < start:
            raise ValueError(f"invalid shard range {start}..{end}: {root}")
        shard_rows.append((start, end, root, manifest))
    shard_rows.sort(key=lambda value: value[0])
    cursor = 1
    for start, end, root, _manifest in shard_rows:
        if start != cursor:
            raise ValueError(f"non-contiguous shard coverage at {root}: expected {cursor}, got {start}")
        cursor = end + 1
    if cursor - 1 != expected_index_rows:
        raise ValueError(
            f"shards cover {cursor - 1} rows, expected {expected_index_rows}"
        )

    full_index_sha = file_sha256(instruction_index)
    expected_benchmark = holdout.metadata()
    for _start, _end, root, manifest in shard_rows:
        provenance = manifest.get("provenance") or {}
        if provenance.get("instruction_index_sha256") != full_index_sha:
            raise ValueError(f"instruction-index SHA mismatch: {root}")
        benchmark = provenance.get("evaluation_holdout") or {}
        if benchmark.get("manifest_sha256") != expected_benchmark.get("manifest_sha256"):
            raise ValueError(f"Evaluation holdout SHA mismatch: {root}")

    staging = output_root.with_name(
        f".{output_root.name}.merge-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir(parents=True)
    source_counts: Counter[str] = Counter()
    quarantine_reasons: Counter[str] = Counter()
    read_rows = accepted_episodes = emitted_samples = 0
    benchmark_excluded = missing_context_variants = missing_task_instruction = 0
    require_profiles: list[str] | None = None
    try:
        observed_buckets = sorted({
            bucket
            for _start, _end, _root, manifest in shard_rows
            for bucket in (manifest.get("buckets") or {})
        })
        bucket_summaries: dict[str, dict[str, Any]] = {}
        for bucket in observed_buckets:
            train_paths: list[Path] = []
            contract_paths: list[Path] = []
            split_counts: Counter[str] = Counter()
            category_counts: Counter[str] = Counter()
            context_counts: Counter[str] = Counter()
            profile_counts: Counter[str] = Counter()
            for _start, _end, root, manifest in shard_rows:
                if bucket not in (manifest.get("buckets") or {}):
                    continue
                bucket_root = root / "buckets" / bucket
                bucket_manifest = _load(bucket_root / "manifest.json")
                train = bucket_root / "train.jsonl"
                contracts = bucket_root / "contracts.jsonl"
                if train.is_file():
                    train_paths.append(train)
                if contracts.is_file():
                    contract_paths.append(contracts)
                _counter_add(split_counts, bucket_manifest.get("split_counts") or {})
                _counter_add(category_counts, bucket_manifest.get("category_counts") or {})
                _counter_add(context_counts, bucket_manifest.get("context_variant_counts") or {})
                _counter_add(profile_counts, bucket_manifest.get("output_profile_counts") or {})
            destination = staging / "buckets" / bucket
            train_records, train_sha = _concat(train_paths, destination / "train.jsonl")
            contract_records, contract_sha = _merge_contracts(
                contract_paths, destination / "contracts.jsonl"
            )
            expected_records = int(split_counts.get("train", 0))
            if train_records != expected_records:
                raise ValueError(
                    f"{bucket} merged lines={train_records}, manifests={expected_records}"
                )
            bucket_manifest = {
                "schema_version": "v10_action_segment_v5_3_bucket_v1",
                "complete": True,
                "training_bucket": bucket,
                "records": train_records,
                "split_counts": dict(sorted(split_counts.items())),
                "category_counts": dict(sorted(category_counts.items())),
                "context_variant_counts": dict(sorted(context_counts.items())),
                "output_profile_counts": dict(sorted(profile_counts.items())),
                "prompt_renderer_sha256": prompt_renderer_digest_v53(),
                "files": {
                    "train.jsonl": {"records": train_records, "sha256": train_sha},
                    "contracts.jsonl": {
                        "records": contract_records,
                        "sha256": contract_sha,
                    },
                },
            }
            bucket_summaries[bucket] = bucket_manifest

        quarantine_paths = [root / "quarantine.jsonl" for _, _, root, _ in shard_rows]
        quarantine_records, quarantine_sha = _concat(
            quarantine_paths, staging / "quarantine.jsonl"
        )
        cache_paths = [root / "media_frame_cache.jsonl" for _, _, root, _ in shard_rows]
        cache_records, cache_sha = _merge_frame_cache(
            cache_paths, staging / "media_frame_cache.jsonl"
        )

        frame_workers: list[int] = []
        batch_sizes: list[int] = []
        for _start, _end, _root, manifest in shard_rows:
            provenance = manifest.get("provenance") or {}
            read_rows += int(provenance.get("read_rows") or 0)
            accepted_episodes += int(provenance.get("accepted_episodes") or 0)
            emitted_samples += int(provenance.get("emitted_samples") or 0)
            benchmark_excluded += int(provenance.get("evaluation_holdout_excluded") or 0)
            missing_context_variants += int(provenance.get("missing_context_variants") or 0)
            missing_task_instruction += int(provenance.get("missing_task_instruction") or 0)
            _counter_add(source_counts, provenance.get("source_counts") or {})
            _counter_add(quarantine_reasons, provenance.get("quarantine_reasons") or {})
            frame_workers.append(int(provenance.get("frame_probe_workers") or 0))
            batch_sizes.append(int(provenance.get("materialize_batch_size") or 0))
            profiles = [str(value) for value in (provenance.get("require_profiles") or [])]
            if require_profiles is None:
                require_profiles = profiles
            elif require_profiles != profiles:
                raise ValueError("require_profiles differ across shards")
        if read_rows != expected_index_rows:
            raise ValueError(f"merged read_rows={read_rows}, expected {expected_index_rows}")
        if emitted_samples != sum(value["records"] for value in bucket_summaries.values()):
            raise ValueError("emitted_samples differ from merged bucket records")

        provenance = {
            "source": "v2v3umi_instruction_index",
            "instruction_index": str(instruction_index.resolve()),
            "instruction_index_sha256": full_index_sha,
            "partial": False,
            "max_episodes": None,
            "line_start": 1,
            "line_end": expected_index_rows,
            "include_robodojo": False,
            "require_profiles": require_profiles or [],
            "read_rows": read_rows,
            "accepted_episodes": accepted_episodes,
            "emitted_samples": emitted_samples,
            "source_counts": dict(sorted(source_counts.items())),
            "evaluation_holdout_excluded": benchmark_excluded,
            "missing_context_variants": missing_context_variants,
            "missing_task_instruction": missing_task_instruction,
            "quarantine_reasons": dict(sorted(quarantine_reasons.items())),
            "evaluation_holdout": expected_benchmark,
            "media_scan_policy": "index_paths_only_no_recursive_media_walk",
            "end_policy": "exact_min_video_sample_count_minus_one",
            "frame_probe_workers": max(frame_workers),
            "frame_probe_workers_per_shard": frame_workers,
            "materialize_batch_size": max(batch_sizes),
            "materialize_shard_count": len(shard_rows),
            "materialize_shard_ranges": [
                {"line_start": start, "line_end": end, "root": str(root)}
                for start, end, root, _manifest in shard_rows
            ],
            "quarantine_records": quarantine_records,
            "media_frame_cache_records": cache_records,
            "quarantine_sha256": quarantine_sha,
            "media_frame_cache_sha256": cache_sha,
        }
        for bucket, manifest in bucket_summaries.items():
            manifest["provenance"] = provenance
            atomic_json(staging / "buckets" / bucket / "manifest.json", manifest)
        root_manifest = {
            "schema_version": "v10_action_segment_v5_3_artifact_v1",
            "complete": True,
            "total_records": emitted_samples,
            "buckets": {
                bucket: {
                    "records": manifest["records"],
                    "manifest": f"buckets/{bucket}/manifest.json",
                }
                for bucket, manifest in sorted(bucket_summaries.items())
            },
            "prompt_renderer_sha256": prompt_renderer_digest_v53(),
            "provenance": provenance,
        }
        atomic_json(staging / "manifest.json", root_manifest)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_root)
        return root_manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--instruction-index", type=Path, required=True)
    parser.add_argument("--expected-index-rows", type=int, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_EVALUATION_MANIFEST)
    parser.add_argument("--evaluation-sha256", default=DEFAULT_EVALUATION_SHA256)
    args = parser.parse_args(argv)
    holdout = EvaluationHoldout.load(
        args.evaluation_holdout, expected_sha256=args.evaluation_holdout_sha256
    )
    report = merge_shards(
        shard_roots=args.shard_root,
        output_root=args.output_root,
        instruction_index=args.instruction_index,
        expected_index_rows=args.expected_index_rows,
        holdout=holdout,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["merge_shards"]
