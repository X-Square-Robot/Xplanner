#!/usr/bin/env python3
"""Prepare the deterministic Benchmark3 progress-parity episode bundle."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from x_planner.evaluation.benchmark3.prepare_specs import main


if __name__ == "__main__":
    raise SystemExit(main())
