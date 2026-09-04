"""Project audited Takeover V5.2 rows into the frozen V5.3 train/test split."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any

from .bucket_io import AtomicJsonlWriter, BucketWriter, file_sha256
from .holdout import EvaluationHoldout, DEFAULT_EVALUATION_MANIFEST, DEFAULT_EVALUATION_SHA256
from .schema import SCHEMA_VERSION_V53, validate_sample


DEFAULT_SNAPSHOT = Path(os.environ.get(
    "XPLANNER_TAKEOVER_SNAPSHOT", "/path/to/takeover_snapshot"
))
DEFAULT_SPLIT = Path(os.environ.get(
    "XPLANNER_TAKEOVER_SPLIT", "/path/to/takeover_split.jsonl"
))


def load_split(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("episode_key") or "")
            split = str(row.get("split") or "")
            if not key or split not in {"train", "test"}:
                raise ValueError(f"invalid split row {line_number}")
            if key in result:
                raise ValueError(f"duplicate split episode: {key}")
            result[key] = split
    if not result:
        raise ValueError("Takeover split is empty")
    return result


def iter_takeover_rows(snapshot: Path) -> Iterator[tuple[str, int, Mapping[str, Any]]]:
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    for leaf in manifest.get("leaves") or ():
        if leaf.get("source") != "takeover_q" or leaf.get("category") != "takeover":
            continue
        path = snapshot / str(leaf["relative_path"]) / "data.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    yield str(path), line_number, json.loads(line)


def project_sample(sample: Mapping[str, Any], *, split: str) -> dict[str, Any]:
    provenance = sample.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("cached Takeover sample has no provenance")
    value = {
        "schema_version": SCHEMA_VERSION_V53,
        "sample_id": f"{sample['sample_id']}_v53",
        "base_sample_id": str(sample["base_sample_id"]),
        "source": "takeover_q",
        "training_bucket": "takeover",
        "category": "takeover",
        "context_variant": "no_memory_no_initial",
        "split": split,
        "output_spec": dict(sample["output_spec"]),
        "output_profile_id": str(sample["output_profile_id"]),
        "task_instruction": str(sample["task_instruction"]),
        "images": list(sample["images"]),
        "prompt_context": {},
        "target": dict(sample["target"]),
        "supervision": {"loss_mask_paths": list(
            (sample.get("supervision") or {}).get("loss_mask_paths") or ()
        )},
        "provenance": {
            **dict(provenance),
            "split": split,
            "projected_from_schema": sample.get("schema_version"),
            "projected_from_sample_id": sample.get("sample_id"),
            "takeover_split_version": "v5_3_takeover_episode_split_v1",
            "legacy_gold_considered": False,
            "memory_pair_eligible": False,
        },
    }
    return validate_sample(value)


def materialize(
    *,
    snapshot: Path,
    split_path: Path,
    output_root: Path,
    holdout: EvaluationHoldout,
    max_samples_per_split: int | None = None,
) -> dict[str, Any]:
    split_map = load_split(split_path)
    writer = BucketWriter(output_root, contract_examples_per_key=1)
    quarantine = AtomicJsonlWriter(output_root / "buckets" / "takeover" / "quarantine.jsonl")
    counts: Counter[str] = Counter()
    failure_counts: Counter[tuple[str, str]] = Counter()
    reasons: Counter[str] = Counter()
    source_rows = 0
    benchmark_excluded = 0
    try:
        for source_file, line_number, row in iter_takeover_rows(snapshot):
            source_rows += 1
            sample = row.get("v5_sample") if isinstance(row, Mapping) else None
            provenance = sample.get("provenance") if isinstance(sample, Mapping) else None
            episode_key = str(provenance.get("episode_key") or "") if isinstance(provenance, Mapping) else ""
            split = split_map.get(episode_key)
            if split is None:
                reasons["episode_not_in_new_completed_split"] += 1
                quarantine.write({
                    "source_file": source_file,
                    "source_line": line_number,
                    "episode_key": episode_key,
                    "reason": "episode_not_in_new_completed_split",
                })
                continue
            if max_samples_per_split is not None and counts[split] >= max_samples_per_split:
                if counts["train"] >= max_samples_per_split and counts["test"] >= max_samples_per_split:
                    break
                continue
            try:
                projected = project_sample(sample, split=split)
                matches = holdout.match_sample(projected)
                if matches:
                    benchmark_excluded += 1
                    reasons["evaluation_holdout_overlap"] += 1
                    quarantine.write({
                        "source_file": source_file,
                        "source_line": line_number,
                        "episode_key": episode_key,
                        "reason": "evaluation_holdout_overlap",
                        "matches": matches,
                    })
                    continue
                writer.write(projected, source_record={
                    "source_snapshot_sample_id": sample.get("sample_id"),
                    "episode_key": episode_key,
                    "case_id": provenance.get("case_id"),
                })
                counts[split] += 1
                failure = projected["target"]["decision_detail"]["failure_analysis"]["failure_type"]
                failure_counts[(split, failure)] += 1
            except Exception as exc:
                reason = type(exc).__name__
                reasons[reason] += 1
                quarantine.write({
                    "source_file": source_file,
                    "source_line": line_number,
                    "episode_key": episode_key,
                    "reason": reason,
                    "detail": str(exc),
                })
        provenance = {
            "source_snapshot": str(snapshot.resolve()),
            "source_snapshot_manifest_sha256": file_sha256(snapshot / "manifest.json"),
            "episode_split": str(split_path.resolve()),
            "episode_split_sha256": file_sha256(split_path),
            "episode_split_size": len(split_map),
            "partial": max_samples_per_split is not None,
            "max_samples_per_split": max_samples_per_split,
            "source_rows_seen": source_rows,
            "split_counts": dict(sorted(counts.items())),
            "failure_type_sample_counts": {
                split: {
                    failure: count
                    for (side, failure), count in sorted(failure_counts.items())
                    if side == split
                }
                for split in ("train", "test")
            },
            "quarantine_reasons": dict(sorted(reasons.items())),
            "evaluation_holdout_excluded": benchmark_excluded,
            "evaluation_holdout": holdout.metadata(),
            "legacy_gold_considered": False,
            "split_unit": "episode",
        }
        report = writer.close(provenance=provenance)
        quarantine.close()
        return {**report, "takeover_projection": provenance, "quarantine_records": quarantine.count}
    except BaseException:
        writer.abort()
        quarantine.close(publish=False)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--episode-split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_EVALUATION_MANIFEST)
    parser.add_argument("--evaluation-sha256", default=DEFAULT_EVALUATION_SHA256)
    args = parser.parse_args(argv)
    holdout = EvaluationHoldout.load(args.evaluation_holdout, expected_sha256=args.evaluation_holdout_sha256)
    report = materialize(
        snapshot=args.snapshot,
        split_path=args.episode_split,
        output_root=args.output_root,
        holdout=holdout,
        max_samples_per_split=args.max_samples_per_split,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["iter_takeover_rows", "load_split", "materialize", "project_sample"]
