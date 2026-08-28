#!/usr/bin/env python3
"""Configure a prepared V10 data YAML for full-coverage shuffled epochs.

The normal V10 training preparation enables power-law task balancing, which
intentionally downsamples common Profiles.  This utility removes that balance
for jobs whose contract is "visit every training sample once per epoch" while
retaining X2RobotSampler's deterministic global shuffle and distributed shard.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def configure(
    data_config: Path,
    snapshot: Path,
    audit_output: Path,
    *,
    seed: int,
    world_size: int,
    per_device_batch_size: int,
    epochs: int,
    single_train_dataset: bool = False,
) -> dict[str, Any]:
    data_config = data_config.resolve()
    snapshot = snapshot.resolve()
    value = yaml.safe_load(data_config.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("dataset"), dict):
        raise ValueError(f"invalid dataset config: {data_config}")

    dataset = value["dataset"]
    sampler = dataset.get("sampler")
    if not isinstance(sampler, dict):
        raise ValueError("dataset.sampler must be a mapping")
    sampler["type"] = "default"
    sampler["seed"] = int(seed)
    sampler["batch_size"] = 1
    removed_balance = sampler.pop("task_balance", None)
    removed_balance_report = sampler.pop("task_balance_report_path", None)

    sources = dataset.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("dataset.sources must be a non-empty list")
    snapshot_manifest = _read_json(snapshot / "manifest.json")
    expected_train_samples = int(snapshot_manifest["splits"]["train"]["samples"])
    profile_names = sorted(snapshot_manifest.get("train_profile_datasets", {}))
    profile_paths: list[str] = []
    observed_train_samples = 0
    if single_train_dataset:
        train_path = (snapshot / "train").resolve()
        train_manifest = _read_json(train_path / "manifest.json")
        observed_train_samples = int(train_manifest["num_samples"])
        profile_paths.append(str(train_path))
        # One physical dataset contains every Profile.  This is the natural
        # representation when task balancing is disabled and avoids storing a
        # second copy of the entire train split under train_profiles/.
        first_source = sources[0]
        if not isinstance(first_source, dict):
            raise ValueError("dataset source must be a mapping")
        first_source["paths"] = [{
            "path": str(train_path),
            "episode_type": "x2_multimodal",
            "task_name": "v10_all_profiles",
        }]
        dataset["sources"] = [first_source]
    else:
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("dataset source must be a mapping")
            paths = source.get("paths")
            if not isinstance(paths, list) or not paths:
                raise ValueError("dataset source paths must be non-empty")
            for item in paths:
                profile_path = Path(str(item["path"])).resolve()
                profile_manifest_path = profile_path / "manifest.json"
                profile_manifest = _read_json(profile_manifest_path)
                observed_train_samples += int(profile_manifest["num_samples"])
                profile_paths.append(str(profile_path))
    if observed_train_samples != expected_train_samples:
        raise ValueError(
            "profile sample total does not cover the complete train split: "
            f"profiles={observed_train_samples} snapshot={expected_train_samples}"
        )
    if world_size <= 0 or per_device_batch_size <= 0 or epochs <= 0:
        raise ValueError("world-size, per-device-batch-size and epochs must be positive")
    padded_samples = math.ceil(expected_train_samples / world_size) * world_size
    samples_per_rank = padded_samples // world_size
    optimizer_steps_per_epoch = math.ceil(samples_per_rank / per_device_batch_size)

    rendered = yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
    _atomic_write_text(data_config, rendered)
    audit = {
        "schema_version": "v10_full_coverage_shuffle_v1",
        "data_config": str(data_config),
        "snapshot": str(snapshot),
        "snapshot_content_digest": snapshot_manifest["content_digest"],
        "expected_train_samples_per_epoch": expected_train_samples,
        "observed_profile_samples": observed_train_samples,
        "profiles": profile_paths,
        "profile_names": profile_names,
        "single_train_dataset": bool(single_train_dataset),
        "seed": int(seed),
        "world_size": int(world_size),
        "per_device_batch_size": int(per_device_batch_size),
        "epochs": int(epochs),
        "distributed_padding_samples_per_epoch": padded_samples - expected_train_samples,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "expected_total_optimizer_steps": optimizer_steps_per_epoch * epochs,
        "sampler_type": "X2RobotSampler/default",
        "task_balance_removed": removed_balance,
        "task_balance_report_path_removed": removed_balance_report,
        "sampling_contract": (
            "all samples once per epoch before DistributedSampler-style padding; "
            "global shuffle uses seed+epoch, then rank sharding and per-rank shuffle"
        ),
    }
    _atomic_write_text(
        audit_output.resolve(),
        json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--single-train-dataset", action="store_true")
    args = parser.parse_args()
    result = configure(
        args.data_config,
        args.snapshot,
        args.audit_output,
        seed=args.seed,
        world_size=args.world_size,
        per_device_batch_size=args.per_device_batch_size,
        epochs=args.epochs,
        single_train_dataset=args.single_train_dataset,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
