#!/usr/bin/env python3
"""Scan collection root partitions while preserving canonical episode keys."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from . import scanner as scanner_module
from .adapters import EpisodeJob, _discover_collection
from .constants import (
    DEFAULT_MAX_CAMERA_VIEWS,
    DEFAULT_MIN_INTERVAL_FRAMES,
    DEFAULT_MIN_L0_COUNT,
    DEFAULT_MIN_L1_COUNT,
    SCAN_RULE_VERSION,
)


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "sources_collection_reverse_accel.yml"
DEFAULT_OUTPUT = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/luhao/checkpoint/v10_continuous/collection_accel_scans"
)


def discover_partitioned_jobs(
    config: Mapping[str, Any], max_episodes: int = 0
) -> Iterator[EpisodeJob]:
    """Discover disjoint roots but key them relative to one canonical root."""

    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources config must contain a non-empty sources list")
    seen_dirs: set[str] = set()
    emitted = 0
    for source in sources:
        if not isinstance(source, Mapping) or source.get("kind") != "collection_partition":
            raise ValueError("accelerator only accepts collection_partition sources")
        source_name = str(source["name"])
        key_root = Path(str(source["episode_key_root"])).resolve()
        if not key_root.is_dir():
            raise FileNotFoundError(key_root)
        for root_value in source.get("roots") or ():
            root = Path(str(root_value)).resolve()
            try:
                root.relative_to(key_root)
            except ValueError as exc:
                raise ValueError(f"partition root {root} is outside key root {key_root}") from exc
            partition_source = {"name": source_name, "roots": [str(root)]}
            for job in _discover_collection(partition_source):
                physical = os.path.realpath(job.episode_dir)
                if physical in seen_dirs:
                    continue
                seen_dirs.add(physical)
                relative = Path(physical).relative_to(key_root).as_posix()
                yield replace(job, episode_key=f"{source_name}/{relative}")
                emitted += 1
                if max_episodes and emitted >= max_episodes:
                    return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--episode-timeout", type=float, default=900.0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--publish-every", type=int, default=1000)
    parser.add_argument("--publish-seconds", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--min-interval-frames", type=int, default=DEFAULT_MIN_INTERVAL_FRAMES)
    parser.add_argument("--min-l1-count", type=int, default=DEFAULT_MIN_L1_COUNT)
    parser.add_argument("--min-l0-count", type=int, default=DEFAULT_MIN_L0_COUNT)
    parser.add_argument("--max-camera-views", type=int, default=DEFAULT_MAX_CAMERA_VIEWS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-final-snapshot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.episode_timeout <= 0:
        raise ValueError("workers and episode-timeout must be positive")
    if not 0 < args.validation_ratio < 1:
        raise ValueError("validation-ratio must be in (0, 1)")
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    settings = {
        "seed": args.seed,
        "validation_ratio": args.validation_ratio,
        "min_interval_frames": args.min_interval_frames,
        "min_l1_count": args.min_l1_count,
        "min_l0_count": args.min_l0_count,
        "max_camera_views": args.max_camera_views,
        "rule_version": SCAN_RULE_VERSION,
    }
    original_discover = scanner_module.discover_jobs
    scanner_module.discover_jobs = discover_partitioned_jobs
    try:
        run_root = scanner_module.run_scan(
            config=config,
            output_root=args.output_root,
            settings=settings,
            workers=args.workers,
            episode_timeout=args.episode_timeout,
            max_episodes=args.max_episodes,
            publish_every=args.publish_every,
            publish_seconds=args.publish_seconds,
            auto_training_snapshot=False,
            final_on_complete=not args.no_final_snapshot,
        )
    finally:
        scanner_module.discover_jobs = original_discover
    print(json.dumps({"status": "complete", "run_root": str(run_root)}, indent=2))


if __name__ == "__main__":
    main()
