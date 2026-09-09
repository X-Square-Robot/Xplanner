#!/usr/bin/env python3
"""Prepare the deterministic evaluation progress-parity episode bundle."""

from pathlib import Path
import os
import sys

_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root))
_dataset_repo = Path(os.environ.get("XPLANNER_DATASET_REPO", _repo_root / "third_party" / "x2robot_dataset_v2"))
if not (_dataset_repo / "x2robot_dataset_v2").is_dir() and (_repo_root.parent / "x2robot_dataset_v2").is_dir():
    _dataset_repo = _repo_root.parent / "x2robot_dataset_v2"
if (_dataset_repo / "x2robot_dataset_v2").is_dir():
    sys.path.insert(0, str(_dataset_repo))

from x_planner.evaluation.progress.prepare_specs import main


if __name__ == "__main__":
    raise SystemExit(main())
