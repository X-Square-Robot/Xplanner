#!/usr/bin/env python3
"""Combine teacher-forced/rollout metrics and compute consistency_gap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .snapshot import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    teacher = json.loads(args.teacher.read_text(encoding="utf-8"))
    rollout = json.loads(args.rollout.read_text(encoding="utf-8"))
    result = {
        "teacher_forced_score": float(teacher["teacher_forced_score"]),
        "rollout_score": float(rollout["rollout_score"]),
    }
    result["consistency_gap"] = (
        result["teacher_forced_score"] - result["rollout_score"]
    )
    atomic_write_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
