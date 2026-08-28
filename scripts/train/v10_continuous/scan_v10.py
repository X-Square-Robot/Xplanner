#!/usr/bin/env python3
"""Incremental V10 scanner. Re-running the same command resumes automatically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .constants import (
    DEFAULT_MAX_CAMERA_VIEWS,
    DEFAULT_MIN_INTERVAL_FRAMES,
    DEFAULT_MIN_L0_COUNT,
    DEFAULT_MIN_L1_COUNT,
    SCAN_RULE_VERSION,
)
from .scanner import run_scan


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "sources.yml"
DEFAULT_OUTPUT = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/luhao/checkpoint/v10_continuous/scans"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--episode-timeout", type=float, default=900.0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--publish-every", type=int, default=100)
    parser.add_argument("--publish-seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--min-interval-frames", type=int, default=DEFAULT_MIN_INTERVAL_FRAMES)
    parser.add_argument("--min-l1-count", type=int, default=DEFAULT_MIN_L1_COUNT)
    parser.add_argument("--min-l0-count", type=int, default=DEFAULT_MIN_L0_COUNT)
    parser.add_argument("--max-camera-views", type=int, default=DEFAULT_MAX_CAMERA_VIEWS)
    parser.add_argument("--resume", action="store_true", help="Explicit documentation flag; resume is automatic.")
    parser.add_argument("--no-auto-training-snapshot", action="store_true")
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
    run_root = run_scan(
        config=config,
        output_root=args.output_root,
        settings=settings,
        workers=args.workers,
        episode_timeout=args.episode_timeout,
        max_episodes=args.max_episodes,
        publish_every=args.publish_every,
        publish_seconds=args.publish_seconds,
        auto_training_snapshot=not args.no_auto_training_snapshot,
        final_on_complete=not args.no_final_snapshot,
    )
    print(json.dumps({"status": "complete", "run_root": str(run_root)}, indent=2))


if __name__ == "__main__":
    main()

