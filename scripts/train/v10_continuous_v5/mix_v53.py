"""Create an adjustable V5.3 exposure plan with an exact 20% RoboDojo quota."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

from .bucket_io_v53 import atomic_json, file_sha256
from .schema_v5 import TRAINING_BUCKETS


DEFAULT_WEIGHTS: Mapping[str, float] = {
    "initial_plan": 0.10,
    "ongoing": 0.20,
    "end": 0.10,
    "robodojo": 0.20,
    "takeover": 0.20,
    "replan_self": 0.10,
    "replan_open": 0.10,
}
ROBODOJO_FRACTION = 0.20


def allocate_weighted_exposures(
    available: Mapping[str, int],
    *,
    total: int,
    weights: Mapping[str, float],
) -> dict[str, int]:
    """Allocate an exact exposure budget across an explicit bucket subset.

    This helper is used by the Baseline-only training profile.  The seven-bucket
    profile continues to use :func:`allocate_exposures`, which separately
    enforces the fixed 20% RoboDojo contract.
    """

    if total <= 0:
        raise ValueError("total exposures must be positive")
    present = {name: int(count) for name, count in available.items() if int(count) > 0}
    if not present:
        raise ValueError("at least one non-empty bucket is required")
    unknown = sorted(set(present) - set(TRAINING_BUCKETS))
    if unknown:
        raise ValueError(f"unknown buckets: {unknown}")
    missing_weights = sorted(name for name in present if float(weights.get(name, 0)) <= 0)
    if missing_weights:
        raise ValueError(f"buckets lack positive exposure weights: {missing_weights}")
    denominator = sum(float(weights[name]) for name in present)
    raw = {name: total * float(weights[name]) / denominator for name in present}
    result = {name: math.floor(value) for name, value in raw.items()}
    residue = total - sum(result.values())
    order = sorted(
        present,
        key=lambda name: (raw[name] - math.floor(raw[name]), name),
        reverse=True,
    )
    for name in order[:residue]:
        result[name] += 1
    if sum(result.values()) != total or any(value <= 0 for value in result.values()):
        raise AssertionError("weighted exposure allocation lost budget or a bucket")
    return {name: result[name] for name in TRAINING_BUCKETS if name in result}


def allocate_exposures(
    available: Mapping[str, int],
    *,
    total: int,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
) -> dict[str, int]:
    if total <= 0:
        raise ValueError("total exposures must be positive")
    present = {name: int(count) for name, count in available.items() if int(count) > 0}
    if "robodojo" not in present:
        raise ValueError("RoboDojo is required for the fixed 20% quota")
    unknown = sorted(set(present) - set(TRAINING_BUCKETS))
    if unknown:
        raise ValueError(f"unknown buckets: {unknown}")
    robodojo = round(total * ROBODOJO_FRACTION)
    remaining = total - robodojo
    others = [name for name in TRAINING_BUCKETS if name != "robodojo" and name in present]
    denominator = sum(float(weights.get(name, 0)) for name in others)
    if denominator <= 0:
        raise ValueError("non-RoboDojo bucket weights must have positive mass")
    raw = {name: remaining * float(weights.get(name, 0)) / denominator for name in others}
    result = {name: math.floor(value) for name, value in raw.items()}
    result["robodojo"] = robodojo
    residue = total - sum(result.values())
    order = sorted(
        others,
        key=lambda name: (raw[name] - math.floor(raw[name]), name),
        reverse=True,
    )
    for name in order[:residue]:
        result[name] += 1
    if sum(result.values()) != total or result["robodojo"] != robodojo:
        raise AssertionError("exposure allocation lost exact quota")
    return {name: result[name] for name in TRAINING_BUCKETS if name in result}


def discover_buckets(artifact_roots: Sequence[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for artifact in artifact_roots:
        buckets = artifact / "buckets"
        if not buckets.is_dir():
            continue
        for directory in sorted(buckets.iterdir()):
            manifest_path = directory / "manifest.json"
            train_path = directory / "train.jsonl"
            if not manifest_path.is_file() or not train_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            bucket = str(manifest.get("training_bucket") or directory.name)
            if bucket in result:
                raise ValueError(f"duplicate physical training bucket: {bucket}")
            count = int(
                (manifest.get("split_counts") or {}).get("train")
                or manifest.get("train_records")
                or 0
            )
            if count <= 0:
                continue
            result[bucket] = {
                "path": str(train_path.resolve()),
                "records": count,
                "sha256": file_sha256(train_path),
                "manifest": str(manifest_path.resolve()),
            }
    return result


def create_plan(
    *,
    artifact_roots: Sequence[Path],
    output_root: Path,
    total_exposures: int,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
) -> dict[str, Any]:
    buckets = discover_buckets(artifact_roots)
    exposures = allocate_exposures(
        {name: value["records"] for name, value in buckets.items()},
        total=total_exposures,
        weights=weights,
    )
    plan = {
        "schema_version": "v10_action_segment_v5_3_exposure_plan_v1",
        "complete": True,
        "total_exposures": total_exposures,
        "robodojo_fraction": exposures["robodojo"] / total_exposures,
        "robodojo_fraction_exact_policy": ROBODOJO_FRACTION,
        "buckets": {
            name: {
                **buckets[name],
                "requested_weight": float(weights.get(name, 0)),
                "exposures": exposures[name],
                "observed_fraction": exposures[name] / total_exposures,
                "repeat_factor": exposures[name] / buckets[name]["records"],
            }
            for name in exposures
        },
        "test_files_excluded": True,
        "artifact_roots": [str(path.resolve()) for path in artifact_roots],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "exposure_plan.json", plan)
    # Deliberately simple YAML, readable by standard loaders without aliases.
    lines = ["version: v10_action_segment_v5_3", "datasets:"]
    for name, value in plan["buckets"].items():
        lines.extend([
            f"  {name}:",
            f"    path: {json.dumps(value['path'])}",
            f"    records: {value['records']}",
            f"    exposures: {value['exposures']}",
            f"    weight: {value['observed_fraction']:.12f}",
        ])
    (output_root / "data.yml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plan


def _weight(value: str) -> tuple[str, float]:
    name, separator, raw = value.partition("=")
    if not separator or name not in TRAINING_BUCKETS:
        raise argparse.ArgumentTypeError("weight must be BUCKET=FLOAT")
    try:
        result = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weight must be BUCKET=FLOAT") from exc
    if result < 0:
        raise argparse.ArgumentTypeError("weight cannot be negative")
    return name, result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--total-exposures", type=int, default=100_000)
    parser.add_argument("--weight", action="append", type=_weight, default=[])
    args = parser.parse_args(argv)
    weights = dict(DEFAULT_WEIGHTS)
    weights.update(dict(args.weight))
    report = create_plan(
        artifact_roots=args.artifact_root,
        output_root=args.output_root,
        total_exposures=args.total_exposures,
        weights=weights,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_WEIGHTS", "ROBODOJO_FRACTION", "allocate_exposures", "create_plan", "discover_buckets"]
