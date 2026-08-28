#!/usr/bin/env python3
"""Print incremental V10 scan state without taking the scan lock."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    database = args.run_root.resolve() / "scan_state.sqlite3"
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    try:
        counts = {
            row["status"]: row["count"]
            for row in connection.execute(
                "SELECT status, COUNT(*) AS count FROM episodes GROUP BY status"
            )
        }
        profiles = {
            row["profile"]: row["count"]
            for row in connection.execute(
                "SELECT profile, COUNT(*) AS count FROM episodes "
                "WHERE status='accepted' GROUP BY profile"
            )
        }
        splits = {
            row["split"]: row["count"]
            for row in connection.execute(
                "SELECT split, COUNT(*) AS count FROM episodes "
                "WHERE status='accepted' GROUP BY split"
            )
        }
        failures = {
            row["reason"]: row["count"]
            for row in connection.execute(
                "SELECT reason, COUNT(*) AS count FROM episodes "
                "WHERE status='rejected' GROUP BY reason ORDER BY count DESC LIMIT 20"
            )
        }
        result = {
            "run_root": str(args.run_root.resolve()),
            "terminal": sum(counts.values()),
            "status": counts,
            "profiles": profiles,
            "splits": splits,
            "failure_reasons": failures,
        }
    finally:
        connection.close()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
