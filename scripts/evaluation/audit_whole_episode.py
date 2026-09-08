#!/usr/bin/env python3
"""Run the whole-episode context, progress, and terminal audit."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from x_planner.evaluation.whole_episode.audit import main


if __name__ == "__main__":
    raise SystemExit(main())
