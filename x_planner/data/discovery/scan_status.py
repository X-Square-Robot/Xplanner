"""Aggregate current V2 inventory, shard progress, failures, and paths."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common.atomic import iter_jsonl, read_json, write_json
from .shard_state import completed_attempts, load_current_inventory, version_root


MAX_STATUS_PATHS = 10_000


def _failure_summary(row: dict[str, Any], shard_id: int) -> dict[str, Any]:
    error = row.get("error") or {}
    paths = list(error.get("input_paths") or row.get("input_paths") or ())
    return {
        "source_id": row.get("source_id"),
        "episode_key": row.get("episode_key"),
        "global_episode_key": row.get("global_episode_key"),
        "stage": error.get("stage"),
        "view_name": error.get("view_name"),
        "error_type": error.get("error_type"),
        "error_message": error.get("error_message"),
        "retryable": row.get("retryable"),
        "attempts": row.get("attempts"),
        "input_paths": paths,
        "shard_id": shard_id,
    }


def _success_summary(row: dict[str, Any], shard_id: int) -> dict[str, Any]:
    return {
        "source_id": row.get("source_id"),
        "episode_key": row.get("episode_key"),
        "global_episode_key": row.get("global_episode_key"),
        "input_paths": row.get("input_paths") or [],
        "views": row.get("views") or [],
        "shard_id": shard_id,
    }


def _inventory_build_status(run_root: Path) -> dict[str, Any] | None:
    state = read_json(str(run_root / "inventory_build.json"))
    if not isinstance(state, dict):
        return None
    processed = int(state.get("processed", 0))
    updated = float(state.get("updated_at", 0.0))
    by_source = {
        str(source): {"discovered": int(count), "pending": 0}
        for source, count in (state.get("by_source") or {}).items()
    }
    return {
        "phase": str(state.get("status") or "building_inventory"),
        "run_root": str(run_root.resolve()),
        "inventory_hash": None,
        "discovered": processed,
        "discovered_by_source": dict(state.get("by_source") or {}),
        "completed": 0,
        "failed": 0,
        "terminal": 0,
        "pending": 0,
        "samples": 0,
        "timing": {
            "started_at": None,
            "updated_at": datetime.fromtimestamp(updated, timezone.utc).isoformat() if updated else None,
            "elapsed_seconds": 0.0,
            "episodes_per_second": 0.0,
            "samples_per_second": 0.0,
            "eta_seconds": None,
        },
        "shards": {
            "total": int((state.get("request") or {}).get("num_shards", 0)),
            "completed": 0,
            "failed": 0,
            "with_episode_failures": 0,
            "pending": int((state.get("request") or {}).get("num_shards", 0)),
        },
        "by_source": by_source,
        "by_error_type": {},
        "by_path": {},
        "by_view": {},
        "failed_episodes": [],
        "success_episodes": [],
        "_attempts": [],
        "inventory_build": state,
    }


def collect_status(run_root: Path, *, include_episode_rows: bool = False) -> dict[str, Any]:
    building = _inventory_build_status(run_root)
    if building is not None and building["phase"] in {"building_inventory", "finalizing_inventory"}:
        return building
    try:
        inventory = load_current_inventory(run_root)
    except FileNotFoundError:
        if building is not None:
            return building
        raise
    status = Counter()
    by_source: dict[str, Counter] = defaultdict(Counter)
    by_error = Counter()
    by_path = Counter()
    by_view = Counter()
    failed: list[dict[str, Any]] = []
    success: list[dict[str, Any]] = []
    completed_shards = shards_with_episode_failures = samples = 0
    terminal = 0
    path_overflow = 0
    updated_times: list[float] = []
    frozen_attempts: list[dict[str, Any]] = []
    for plan in inventory.shards:
        attempts = completed_attempts(version_root(run_root, plan))
        if not attempts:
            continue
        attempt = attempts[-1]
        frozen_attempts.append({"attempt": str(attempt), "shard_id": plan.shard_id})
        updated_times.append((attempt / ".done").stat().st_mtime)
        stats = read_json(str(attempt / "statistics.json"), {}) or {}
        completed_shards += 1
        samples += int(stats.get("samples", 0))
        local_status = Counter(stats.get("status") or {})
        status.update(local_status)
        terminal += sum(local_status.values())
        if local_status.get("failed"):
            shards_with_episode_failures += 1
        for source, counts in (stats.get("by_source") or {}).items():
            by_source[str(source)].update(counts)
        for error_type, count in (stats.get("by_error_type") or {}).items():
            by_error[str(error_type)] += int(count)
        if include_episode_rows:
            for row in iter_jsonl(str(attempt / "episodes_failed.jsonl")):
                summary = _failure_summary(row, plan.shard_id)
                paths = summary["input_paths"]
                for path in paths:
                    normalized = str(path)
                    if normalized in by_path or len(by_path) < MAX_STATUS_PATHS:
                        by_path[normalized] += 1
                    else:
                        path_overflow += 1
                failed.append(summary)
            for row in iter_jsonl(str(attempt / "episodes_success.jsonl")):
                for view in row.get("views") or ():
                    by_view[str(view)] += 1
                success.append(_success_summary(row, plan.shard_id))
    discovered = inventory.total_episodes
    inventory_time = Path(inventory.path).stat().st_mtime
    updated_time = max(updated_times, default=inventory_time)
    elapsed_seconds = max(0.0, updated_time - inventory_time)
    episodes_per_second = terminal / elapsed_seconds if elapsed_seconds else 0.0
    samples_per_second = samples / elapsed_seconds if elapsed_seconds else 0.0
    pending = max(0, discovered - terminal)
    eta_seconds = pending / episodes_per_second if episodes_per_second else None
    for source, count in inventory.by_source.items():
        by_source[source]["discovered"] = count
        terminal_for_source = by_source[source].get("success", 0) + by_source[source].get("failed", 0)
        by_source[source]["pending"] = max(0, count - terminal_for_source)
    if path_overflow:
        by_path["__other_paths__"] += path_overflow
    return {
        "phase": "complete" if pending == 0 else "scanning",
        "run_root": str(run_root.resolve()),
        "inventory_hash": inventory.inventory_hash,
        "discovered": discovered,
        "discovered_by_source": inventory.by_source,
        "completed": int(status.get("success", 0)),
        "failed": int(status.get("failed", 0)),
        "pending": pending,
        "samples": samples,
        "terminal": terminal,
        "timing": {
            "started_at": datetime.fromtimestamp(inventory_time, timezone.utc).isoformat(),
            "updated_at": datetime.fromtimestamp(updated_time, timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "episodes_per_second": episodes_per_second,
            "samples_per_second": samples_per_second,
            "eta_seconds": eta_seconds,
        },
        "shards": {
            "total": len(inventory.shards),
            "completed": completed_shards,
            "failed": 0,
            "with_episode_failures": shards_with_episode_failures,
            "pending": max(0, len(inventory.shards) - completed_shards),
        },
        "by_source": {key: dict(value) for key, value in sorted(by_source.items())},
        "by_error_type": dict(by_error.most_common()),
        "by_path": dict(by_path.most_common()),
        "by_path_truncated": bool(path_overflow),
        "by_view": dict(by_view.most_common()),
        "failed_episodes": failed,
        "success_episodes": success,
        "_attempts": frozen_attempts,
    }


def collect_status_many(
    run_roots: list[Path], *, include_episode_rows: bool = False
) -> dict[str, Any]:
    reports = [
        collect_status(root, include_episode_rows=include_episode_rows)
        for root in dict.fromkeys(path.resolve() for path in run_roots)
    ]
    if len(reports) == 1:
        return reports[0]
    by_source: dict[str, Counter] = defaultdict(Counter)
    by_error = Counter()
    by_path = Counter()
    by_view = Counter()
    shards = Counter()
    for report in reports:
        for source, counts in report["by_source"].items():
            by_source[source].update(counts)
        by_error.update(report["by_error_type"])
        by_path.update(report["by_path"])
        by_view.update(report["by_view"])
        shards.update(report["shards"])
    started = min(report["timing"]["started_at"] for report in reports)
    updated = max(report["timing"]["updated_at"] for report in reports)
    elapsed = sum(float(report["timing"]["elapsed_seconds"]) for report in reports)
    completed = sum(int(report["completed"]) for report in reports)
    failed = sum(int(report["failed"]) for report in reports)
    terminal = completed + failed
    samples = sum(int(report["samples"]) for report in reports)
    pending = sum(int(report["pending"]) for report in reports)
    episode_rate = (completed + failed) / elapsed if elapsed else 0.0
    return {
        "phase": "complete" if pending == 0 else "scanning",
        "run_root": [report["run_root"] for report in reports],
        "inventory_hash": [report["inventory_hash"] for report in reports],
        "discovered": sum(int(report["discovered"]) for report in reports),
        "discovered_by_source": {
            source: counts["discovered"] for source, counts in sorted(by_source.items())
        },
        "completed": completed,
        "failed": failed,
        "pending": pending,
        "samples": samples,
        "terminal": terminal,
        "shards": dict(shards),
        "by_source": {source: dict(counts) for source, counts in sorted(by_source.items())},
        "by_error_type": dict(by_error.most_common()),
        "by_path": dict(by_path.most_common()),
        "by_path_truncated": any(bool(report.get("by_path_truncated")) for report in reports),
        "by_view": dict(by_view.most_common()),
        "failed_episodes": [
            row for report in reports for row in report["failed_episodes"]
        ],
        "success_episodes": [
            row for report in reports for row in report["success_episodes"]
        ],
        "_attempts": [
            item for report in reports for item in report.get("_attempts", [])
        ],
        "timing": {
            "started_at": started, "updated_at": updated,
            "elapsed_seconds": elapsed, "episodes_per_second": episode_rate,
            "samples_per_second": samples / elapsed if elapsed else 0.0,
            "eta_seconds": pending / episode_rate if episode_rate else None,
        },
        "runs": [{
            "run_root": report["run_root"],
            "inventory_hash": report["inventory_hash"],
            "discovered": report["discovered"],
            "completed": report["completed"],
            "failed": report["failed"],
            "pending": report["pending"],
            "samples": report["samples"],
        } for report in reports],
    }


def publish_statistics(run_root: Path, report: dict[str, Any]) -> list[Path]:
    root = run_root / "statistics"
    from .common.atomic import BatchedJsonlWriter
    by_path = Counter()
    by_view = Counter()
    path_overflow = 0
    failed_path = root / "failed_paths.jsonl"
    with BatchedJsonlWriter(str(failed_path)) as writer:
        for row in _iter_terminal_summaries(report, success=False):
            writer.write(row)
            for raw in row.get("input_paths") or ():
                path = str(raw)
                if path in by_path or len(by_path) < MAX_STATUS_PATHS:
                    by_path[path] += 1
                else:
                    path_overflow += 1
    success_path = root / "success_paths.jsonl"
    with BatchedJsonlWriter(str(success_path)) as writer:
        for row in _iter_terminal_summaries(report, success=True):
            writer.write(row)
            for view in row.get("views") or ():
                by_view[str(view)] += 1
    if path_overflow:
        by_path["__other_paths__"] += path_overflow
    report["by_path"] = dict(by_path.most_common())
    report["by_path_truncated"] = bool(path_overflow)
    report["by_view"] = dict(by_view.most_common())
    paths = {
        "summary.json": {
            key: value for key, value in report.items()
            if key not in {"failed_episodes", "success_episodes", "by_path"}
            and not key.startswith("_")
        },
        "by_source.json": report["by_source"],
        "by_error_type.json": report["by_error_type"],
        "by_path.json": report["by_path"],
        "by_view.json": report["by_view"],
    }
    output = []
    for name, value in paths.items():
        path = root / name
        write_json(str(path), value)
        output.append(path)
    output.append(failed_path)
    output.append(success_path)
    return output


def _iter_terminal_summaries(
    report: dict[str, Any], *, success: bool
):
    cached = report.get("success_episodes" if success else "failed_episodes") or ()
    if cached:
        yield from cached
        return
    frozen = report.get("_attempts")
    if frozen is not None:
        name = "episodes_success.jsonl" if success else "episodes_failed.jsonl"
        for reference in frozen:
            attempt = Path(str(reference["attempt"]))
            shard_id = int(reference["shard_id"])
            for row in iter_jsonl(str(attempt / name)):
                yield (
                    _success_summary(row, shard_id)
                    if success else _failure_summary(row, shard_id)
                )
        return
    roots = report.get("run_root")
    if isinstance(roots, str):
        roots = [roots]
    for root_value in roots or ():
        source_root = Path(str(root_value))
        inventory = load_current_inventory(source_root)
        for plan in inventory.shards:
            attempts = completed_attempts(version_root(source_root, plan))
            if not attempts:
                continue
            name = "episodes_success.jsonl" if success else "episodes_failed.jsonl"
            for row in iter_jsonl(str(attempts[-1] / name)):
                yield (
                    _success_summary(row, plan.shard_id)
                    if success else _failure_summary(row, plan.shard_id)
                )


def compact_status(report: dict[str, Any], *, top_paths: int = 100) -> dict[str, Any]:
    """Keep CLI/status JSON bounded while full rows stay in statistics files."""
    compact = {
        key: value for key, value in report.items()
        if key not in {"failed_episodes", "success_episodes", "by_path"}
        and not key.startswith("_")
    }
    paths = report.get("by_path") or {}
    compact["by_path_count"] = len(paths)
    compact["top_paths"] = dict(list(paths.items())[:top_paths])
    compact["failed_episode_count"] = int(report.get("failed", 0))
    compact["success_episode_count"] = int(report.get("completed", 0))
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--write-statistics", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    report = collect_status(args.run_root)
    if args.write_statistics:
        publish_statistics(args.run_root, report)
    print(json.dumps(report if args.verbose else compact_status(report), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
