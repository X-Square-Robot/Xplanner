"""Compose physical V5.3 bucket files into an indexed training snapshot."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from typing import Any
import uuid

from .bucket_io_v53 import atomic_json, file_sha256
from .holdout_v5 import Benchmark3Holdout, DEFAULT_BENCHMARK3_MANIFEST, DEFAULT_BENCHMARK3_SHA256
from .mix_v53 import (
    DEFAULT_WEIGHTS,
    ROBODOJO_FRACTION,
    allocate_exposures,
    allocate_weighted_exposures,
    discover_buckets,
)
from .prompt_v5 import prompt_renderer_digest_v53, render_user
from .schema_v5 import SNAPSHOT_SCHEMA_VERSION_V53, validate_sample


LEAF_SCHEMA_VERSION_V53 = "v10_action_segment_v5_3_bucket_leaf_v1"
BASELINE_ONLY_BUCKETS = ("initial_plan", "ongoing", "end")
BASELINE_ONLY_WEIGHTS: Mapping[str, float] = {
    "initial_plan": 0.25,
    "ongoing": 0.50,
    "end": 0.25,
}
SNAPSHOT_PROFILES = ("seven_bucket", "baseline_only")
BASELINE_INITIAL_CONTEXTS = frozenset({"no_memory_no_initial"})
BASELINE_EXECUTION_REQUIRED_CONTEXTS = frozenset({
    "no_memory_no_initial",
    "with_memory_with_initial",
    "with_memory_with_initial_noisy",
})


def validate_baseline_context_contract(
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require Baseline stage buckets to retain the requested context modes."""

    by_bucket = {str(item.get("training_bucket") or ""): item for item in manifests}
    errors: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    for bucket in BASELINE_ONLY_BUCKETS:
        raw = (by_bucket.get(bucket) or {}).get("context_variant_counts") or {}
        bucket_counts = {str(name): int(value) for name, value in raw.items()}
        counts[bucket] = dict(sorted(bucket_counts.items()))
        observed = {name for name, value in bucket_counts.items() if value > 0}
        required = (
            BASELINE_INITIAL_CONTEXTS
            if bucket == "initial_plan"
            else BASELINE_EXECUTION_REQUIRED_CONTEXTS
        )
        missing = sorted(required - observed)
        if missing:
            errors.append(f"{bucket} missing context variants: {missing}")
        if bucket == "initial_plan" and observed != BASELINE_INITIAL_CONTEXTS:
            errors.append(
                f"initial_plan has forbidden memory context variants: "
                f"{sorted(observed - BASELINE_INITIAL_CONTEXTS)}"
            )
        if bucket != "initial_plan":
            clean = bucket_counts.get("with_memory_with_initial", 0)
            noisy = bucket_counts.get("with_memory_with_initial_noisy", 0)
            if clean != noisy:
                errors.append(
                    f"{bucket} clean/noisy with-initial counts differ: "
                    f"clean={clean}, noisy={noisy}"
                )
    if errors:
        raise ValueError("Baseline context contract failed: " + "; ".join(errors))
    return {
        "schema_version": "v5_3_baseline_context_contract_v1",
        "passed": True,
        "initial_plan_required": sorted(BASELINE_INITIAL_CONTEXTS),
        "ongoing_end_required": sorted(BASELINE_EXECUTION_REQUIRED_CONTEXTS),
        "context_variant_counts": counts,
        "errors": [],
    }


def _link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copyfile(source, destination)
        return "copy"


def _index_bucket(
    *,
    source: Path,
    leaf: Path,
    bucket: str,
    holdout: Benchmark3Holdout,
) -> tuple[dict[str, Any], dict[str, Any]]:
    leaf.mkdir(parents=True)
    index_path = leaf / "data.index"
    episodes_path = leaf / "episodes.jsonl"
    episodes: OrderedDict[str, dict[str, Any]] = OrderedDict()
    categories: dict[str, int] = {}
    contexts: dict[str, int] = {}
    profiles: dict[str, int] = {}
    samples = 0
    benchmark_matches: list[dict[str, Any]] = []
    with source.open("rb") as data, index_path.open("wb") as index:
        while True:
            offset = data.tell()
            line = data.readline()
            if not line:
                break
            if not line.strip():
                continue
            outer = json.loads(line)
            sample = validate_sample(outer.get("v5_sample"))
            if sample["training_bucket"] != bucket or sample["split"] != "train":
                raise ValueError(f"bucket/split mismatch in {source} at sample {samples}")
            # Render every row while composing; prompt drift cannot enter the
            # indexed snapshot unnoticed.
            render_user(sample)
            matches = holdout.match_sample(sample)
            if matches:
                benchmark_matches.append({
                    "sample_id": sample["sample_id"],
                    "episode_key": sample["provenance"].get("episode_key"),
                    "matches": matches,
                })
                if len(benchmark_matches) >= 20:
                    break
            index.write(struct.pack("<Q", offset))
            category = str(sample["category"])
            context = str(sample["context_variant"])
            profile = str(sample["output_profile_id"])
            categories[category] = categories.get(category, 0) + 1
            contexts[context] = contexts.get(context, 0) + 1
            profiles[profile] = profiles.get(profile, 0) + 1
            episode_key = str(sample["provenance"].get("episode_key") or "")
            if not episode_key:
                raise ValueError(f"sample has no episode_key: {sample['sample_id']}")
            episode = episodes.setdefault(episode_key, {
                "episode_key": episode_key,
                "source": bucket,
                "task": bucket,
                "split": "train",
                "first_sample_index": samples,
                "last_sample_index": samples,
                "num_samples": 0,
            })
            episode["last_sample_index"] = samples
            episode["num_samples"] += 1
            samples += 1
        index.flush()
        os.fsync(index.fileno())
    if benchmark_matches:
        raise ValueError(f"Benchmark3 overlap in {bucket}: {benchmark_matches[:3]}")
    if samples <= 0:
        raise ValueError(f"bucket has no train samples: {bucket}")
    with episodes_path.open("w", encoding="utf-8") as handle:
        for episode in episodes.values():
            handle.write(json.dumps(episode, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    mode = _link_or_copy(source, leaf / "data.jsonl")
    manifest = {
        "schema_version": LEAF_SCHEMA_VERSION_V53,
        "complete": True,
        "source": bucket,
        "training_bucket": bucket,
        "category": "mixed" if len(categories) > 1 else next(iter(categories)),
        "context_variant": "mixed" if len(contexts) > 1 else next(iter(contexts)),
        "output_profile_id": "mixed" if len(profiles) > 1 else next(iter(profiles)),
        "task_name": bucket,
        "split": "train",
        "jsonl_file": "data.jsonl",
        "index_file": "data.index",
        "episodes_file": "episodes.jsonl",
        "num_samples": samples,
        "num_episodes": len(episodes),
        "category_counts": dict(sorted(categories.items())),
        "context_variant_counts": dict(sorted(contexts.items())),
        "output_profile_counts": dict(sorted(profiles.items())),
        "source_bucket_path": str(source.resolve()),
        "source_bucket_sha256": file_sha256(source),
        "publish_mode": mode,
        "data_sha256": file_sha256(leaf / "data.jsonl"),
        "index_sha256": file_sha256(index_path),
        "episodes_sha256": file_sha256(episodes_path),
    }
    atomic_json(leaf / "manifest.json", manifest)
    return manifest, {
        "checked_samples": samples,
        "overlap_samples": 0,
        "passed": True,
    }


def compose(
    *,
    artifact_roots: Sequence[Path],
    output_root: Path,
    holdout: Benchmark3Holdout,
    total_exposures: int,
    profile: str = "seven_bucket",
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"snapshot output already exists: {output_root}")
    buckets = discover_buckets(artifact_roots)
    if profile not in SNAPSHOT_PROFILES:
        raise ValueError(f"unknown V5.3 snapshot profile: {profile}")
    observed = set(buckets)
    if profile == "seven_bucket":
        expected = set(DEFAULT_WEIGHTS)
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing or extra:
            raise ValueError(
                f"V5.3 seven-bucket snapshot mismatch: missing={missing}, extra={extra}"
            )
        exposures = allocate_exposures(
            {name: value["records"] for name, value in buckets.items()},
            total=total_exposures,
        )
    else:
        expected = set(BASELINE_ONLY_BUCKETS)
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing or extra:
            raise ValueError(
                f"V5.3 baseline-only snapshot mismatch: missing={missing}, extra={extra}"
            )
        exposures = allocate_weighted_exposures(
            {name: value["records"] for name, value in buckets.items()},
            total=total_exposures,
            weights=BASELINE_ONLY_WEIGHTS,
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.with_name(f".{output_root.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir()
    manifests: list[dict[str, Any]] = []
    holdout_checked = 0
    baseline_context_contract: dict[str, Any] | None = None
    try:
        for bucket in sorted(buckets):
            leaf = staging / "data" / bucket / "train"
            manifest, audit = _index_bucket(
                source=Path(buckets[bucket]["path"]),
                leaf=leaf,
                bucket=bucket,
                holdout=holdout,
            )
            manifests.append({
                **manifest,
                "relative_path": leaf.relative_to(staging).as_posix(),
                "requested_exposures": exposures[bucket],
            })
            holdout_checked += audit["checked_samples"]
        if profile == "baseline_only":
            baseline_context_contract = validate_baseline_context_contract(manifests)
        metadata = staging / "metadata"
        metadata.mkdir()
        holdout_report = {
            "schema_version": "v5_benchmark3_holdout_guard_v1",
            "complete": True,
            "passed": True,
            "checked_samples": holdout_checked,
            "overlap_samples": 0,
            "manifest": holdout.metadata(),
        }
        holdout_path = metadata / "benchmark3_holdout_report.json"
        atomic_json(holdout_path, holdout_report)
        exposure_plan: dict[str, Any] = {
            "schema_version": "v10_action_segment_v5_3_exposure_plan_v1",
            "snapshot_profile": profile,
            "total_exposures": total_exposures,
            "counts_per_bucket": exposures,
            "physical_counts": {name: value["records"] for name, value in buckets.items()},
        }
        if profile == "seven_bucket":
            exposure_plan.update({
                "robodojo_fraction": exposures["robodojo"] / total_exposures,
                "robodojo_fraction_exact_policy": ROBODOJO_FRACTION,
            })
        else:
            exposure_plan["baseline_only_weights"] = dict(BASELINE_ONLY_WEIGHTS)
        atomic_json(metadata / "exposure_plan.json", exposure_plan)
        digest_payload = "\n".join(
            f"{item['training_bucket']}\t{item['data_sha256']}\t{item['num_samples']}"
            for item in sorted(manifests, key=lambda value: value["training_bucket"])
        )
        content_digest = hashlib.sha256(digest_payload.encode()).hexdigest()
        root_manifest = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION_V53,
            "snapshot_id": output_root.name,
            "snapshot_profile": profile,
            "complete": True,
            "partial": any(
                bool((json.loads((Path(value["manifest"])).read_text())
                      .get("provenance") or {}).get("partial"))
                for value in buckets.values()
            ),
            "content_digest": content_digest,
            "num_samples": sum(item["num_samples"] for item in manifests),
            "num_leaves": len(manifests),
            "leaves": manifests,
            "training_buckets": sorted(buckets),
            "prompt_renderer_sha256": prompt_renderer_digest_v53(),
            "exposure_plan": "metadata/exposure_plan.json",
            "benchmark3_holdout": {
                **holdout.metadata(),
                "policy": "compose_requires_zero_source_overlap",
                "overlap_samples": 0,
                "report_relative_path": "metadata/benchmark3_holdout_report.json",
                "report_sha256": file_sha256(holdout_path),
            },
            "artifact_roots": [str(path.resolve()) for path in artifact_roots],
        }
        if baseline_context_contract is not None:
            root_manifest["baseline_context_contract"] = baseline_context_contract
        atomic_json(staging / "manifest.json", root_manifest)
        os.replace(staging, output_root)
        return root_manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--total-exposures", type=int, default=100_000)
    parser.add_argument("--profile", choices=SNAPSHOT_PROFILES, default="seven_bucket")
    parser.add_argument("--benchmark3", type=Path, default=DEFAULT_BENCHMARK3_MANIFEST)
    parser.add_argument("--benchmark3-sha256", default=DEFAULT_BENCHMARK3_SHA256)
    args = parser.parse_args(argv)
    holdout = Benchmark3Holdout.load(args.benchmark3, expected_sha256=args.benchmark3_sha256)
    report = compose(
        artifact_roots=args.artifact_root,
        output_root=args.output_root,
        holdout=holdout,
        total_exposures=args.total_exposures,
        profile=args.profile,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASELINE_ONLY_BUCKETS",
    "BASELINE_ONLY_WEIGHTS",
    "BASELINE_EXECUTION_REQUIRED_CONTEXTS",
    "BASELINE_INITIAL_CONTEXTS",
    "LEAF_SCHEMA_VERSION_V53",
    "SNAPSHOT_PROFILES",
    "compose",
    "validate_baseline_context_contract",
]
