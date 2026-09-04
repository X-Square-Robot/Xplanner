"""Deterministically audit what the existing V4 terminal index actually labels."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

from x2robot_dataset_v2.readers.multimodal_jsonl_reader import (
    get_indexed_jsonl_sample_count,
    load_indexed_jsonl_item,
)

from ..v10_continuous_v3_memory.schema_v3 import TERMINAL_CAPTION


UNIT_FIELD = {"subtask": "subtask", "action": "action", "segment": "l0"}


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _indices(total: int, requested: int) -> list[int]:
    count = min(total, requested)
    if count == total:
        return list(range(total))
    return sorted({index * (total - 1) // (count - 1) for index in range(count)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    total = get_indexed_jsonl_sample_count(str(dataset))
    selected = _indices(total, args.samples)
    active_progress: list[int] = []
    task_progress: list[int] = []
    sources: Counter[str] = Counter()
    profiles: Counter[str] = Counter()
    sentinel_errors: list[int] = []
    flag_errors: list[int] = []
    for index in selected:
        row = load_indexed_jsonl_item(str(dataset), index)
        sample = row["v4_sample"]
        field = UNIT_FIELD[str(sample["unit_type"])]
        target = sample["target"]
        active_progress.append(int(target["predictions"][0][field]["progress_percent"]))
        task_progress.append(int(target["task_progress_percent"]))
        sources[str(sample["source_id"])] += 1
        profiles[str(sample["profile"])] += 1
        if not bool(sample.get("is_terminal_window")):
            flag_errors.append(index)
        second = target["predictions"][1]
        if any(
            unit.get("caption") != TERMINAL_CAPTION
            for unit in second.values()
            if isinstance(unit, dict) and "caption" in unit
        ):
            sentinel_errors.append(index)
    report = {
        "schema_version": "memory_v4_terminal_semantics_audit_v1",
        "dataset": str(dataset),
        "dataset_count": total,
        "sampling": "evenly_spaced_without_replacement",
        "sample_count": len(selected),
        "sample_first_index": selected[0],
        "sample_last_index": selected[-1],
        "sources": dict(sorted(sources.items())),
        "profiles": dict(sorted(profiles.items())),
        "is_terminal_window_false_count": len(flag_errors),
        "prediction_2_sentinel_error_count": len(sentinel_errors),
        "prediction_1_active_unit_progress": {
            "minimum": min(active_progress),
            "median": median(active_progress),
            "maximum": max(active_progress),
            "below_20_count": sum(value < 20 for value in active_progress),
            "below_20_rate": sum(value < 20 for value in active_progress) / len(active_progress),
            "below_80_count": sum(value < 80 for value in active_progress),
            "below_80_rate": sum(value < 80 for value in active_progress) / len(active_progress),
        },
        "task_progress": {
            "minimum": min(task_progress),
            "median": median(task_progress),
            "maximum": max(task_progress),
        },
        "interpretation": (
            "The index selects windows anywhere inside the final active same-scale unit. "
            "It does not establish that physical execution has completed at the observation."
        ),
        "physical_completion_label_available": False,
    }
    _atomic(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
