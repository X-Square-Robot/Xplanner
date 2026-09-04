#!/usr/bin/env python3
"""Reproducible inventory and successful-episode sampling audit for V10 V2."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import yaml

from .artifact_ledger import register_artifacts
from .common.atomic import BatchedJsonlWriter, iter_jsonl, write_json
from .common.constants_v2 import SCAN_RULE_VERSION_V2
from .common.hashing import sampling_config_hash
from .merge_catalogs_v2 import _completion_state
from .sampling_v2 import validate_sampling_config
from .shard_state import (
    Inventory,
    completed_attempts,
    iter_plan,
    load_current_inventory,
    version_root,
)
from .source_discovery import DiscoveredEpisode
from .validate_episode import process_episode


PACKAGE_ROOT = Path(__file__).resolve().parent


def _yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def _reservoir(values: Iterable[Any], count: int, rng: random.Random) -> list[Any]:
    selected: list[Any] = []
    for position, value in enumerate(values, 1):
        if len(selected) < count:
            selected.append(value)
            continue
        replacement = rng.randrange(position)
        if replacement < count:
            selected[replacement] = value
    return selected


def _inventory_rows(inventory: Inventory) -> Iterable[DiscoveredEpisode]:
    for plan in inventory.shards:
        yield from iter_plan(plan)


def _successful_rows(run_root: Path, inventory: Inventory) -> Iterable[dict[str, Any]]:
    for plan in inventory.shards:
        attempts = completed_attempts(version_root(run_root, plan))
        if not attempts:
            continue
        for row in iter_jsonl(str(attempts[-1] / "episodes_success.jsonl")):
            yield {**row, "_shard_id": plan.shard_id, "_attempt": str(attempts[-1])}


def _lookup_inventory_items(
    inventory: Inventory, selected: list[dict[str, Any]]
) -> dict[str, DiscoveredEpisode]:
    by_shard: dict[int, set[str]] = defaultdict(set)
    for row in selected:
        by_shard[int(row["_shard_id"])].add(str(row["global_episode_key"]))
    result: dict[str, DiscoveredEpisode] = {}
    for plan in inventory.shards:
        wanted = by_shard.get(plan.shard_id)
        if not wanted:
            continue
        for item in iter_plan(plan):
            if item.global_episode_key in wanted:
                result[item.global_episode_key] = item
                if len(result) == len(selected):
                    return result
    return result


def _old_sample_keys(selected: list[dict[str, Any]]) -> dict[str, set[str]]:
    by_attempt: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        by_attempt[str(row["_attempt"])].add(str(row["global_episode_key"]))
    result: dict[str, set[str]] = defaultdict(set)
    for attempt, wanted in by_attempt.items():
        for sample in iter_jsonl(str(Path(attempt) / "catalog.jsonl")):
            key = str(sample.get("global_episode_key") or "")
            if key in wanted:
                result[key].add(str(sample["sample_key"]))
    return result


def _result_summary(result: dict[str, Any]) -> tuple[dict[str, Any], set[str]]:
    value = dict(result)
    samples = list(value.pop("samples", ()))
    keys = {str(sample["sample_key"]) for sample in samples}
    value.pop("canonical_episode", None)
    if samples:
        first = samples[0]
        value["first_sample"] = {
            "sample_key": first.get("sample_key"),
            "anchor_index": first.get("anchor_index"),
            "profile": first.get("profile"),
            "task_caption": first.get("task_caption"),
            "target": first.get("target"),
            "views": first.get("views"),
        }
    return value, keys


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status = Counter(str(row.get("status")) for row in rows)
    errors = Counter(
        str((row.get("error") or {}).get("error_type") or "")
        for row in rows if row.get("status") == "failed"
    )
    profiles = Counter(str(row.get("profile") or "") for row in rows if row.get("profile"))
    return {
        "selected": len(rows),
        "status": dict(status),
        "by_error_type": dict(errors),
        "by_profile": dict(profiles),
        "samples": sum(int(row.get("sample_count", 0)) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "sources.yml")
    parser.add_argument("--views-config", type=Path, default=PACKAGE_ROOT / "configs" / "views.yml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--inventory-samples", type=int, default=100)
    parser.add_argument("--successful-samples", type=int, default=100)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--cc-ledger", type=Path)
    args = parser.parse_args()
    if args.inventory_samples < 0 or args.successful_samples < 0:
        raise ValueError("sample counts must be non-negative")

    run_root = args.run_root.resolve()
    inventory = load_current_inventory(run_root)
    completion = _completion_state(run_root)
    if completion["incomplete_shards"] and not args.allow_incomplete:
        raise RuntimeError(
            f"run is incomplete: {completion['completed_shards']}/{completion['total_shards']} shards; "
            "pass --allow-incomplete only for a progress audit"
        )
    config = _yaml(args.config)
    views = _yaml(args.views_config)
    sampling = dict(config.get("sampling") or {})
    validate_sampling_config(sampling)
    settings = {
        **dict(config.get("runtime") or {}),
        "sampling": sampling,
        "max_camera_views": int(sampling.get("max_camera_views", 3)),
        "rule_version": SCAN_RULE_VERSION_V2,
    }
    sampling_hash = sampling_config_hash(sampling)
    output = (args.output_dir or run_root / "audit" / f"seed-{args.seed}").resolve()
    output.mkdir(parents=True, exist_ok=True)

    uniform_items = _reservoir(
        _inventory_rows(inventory), args.inventory_samples, random.Random(args.seed)
    )
    selected_success = _reservoir(
        _successful_rows(run_root, inventory),
        args.successful_samples,
        random.Random(args.seed + 1),
    )
    success_items = _lookup_inventory_items(inventory, selected_success)
    old_keys = _old_sample_keys(selected_success)

    uniform_rows: list[dict[str, Any]] = []
    for item in uniform_items:
        result = process_episode(
            item, run_id=run_root.name, shard_id=-1, worker_id="audit",
            settings=settings, view_config=views, sampling_hash=sampling_hash,
        )
        summary, _keys = _result_summary(result)
        uniform_rows.append(summary)

    successful_rows: list[dict[str, Any]] = []
    for previous in selected_success:
        key = str(previous["global_episode_key"])
        item = success_items.get(key)
        if item is None:
            successful_rows.append({
                "status": "failed", "global_episode_key": key,
                "error": {"error_type": "inventory_lookup_failed"},
            })
            continue
        result = process_episode(
            item, run_id=run_root.name, shard_id=int(previous["_shard_id"]),
            worker_id="audit", settings=settings, view_config=views,
            sampling_hash=sampling_hash,
        )
        summary, new_keys = _result_summary(result)
        previous_keys = old_keys.get(key, set())
        summary.update({
            "previous_status": previous.get("status"),
            "previous_profile": previous.get("profile"),
            "previous_sample_count": int(previous.get("sample_count", len(previous_keys))),
            "previous_sample_keys": len(previous_keys),
            "sample_keys_match": new_keys == previous_keys,
            "new_only_sample_keys": len(new_keys - previous_keys),
            "old_only_sample_keys": len(previous_keys - new_keys),
        })
        successful_rows.append(summary)

    uniform_path = output / "uniform.jsonl"
    successful_path = output / "successful.jsonl"
    with BatchedJsonlWriter(str(uniform_path)) as writer:
        for row in uniform_rows:
            writer.write(row)
    with BatchedJsonlWriter(str(successful_path)) as writer:
        for row in successful_rows:
            writer.write(row)
    summary = {
        "run_root": str(run_root),
        "inventory_hash": inventory.inventory_hash,
        "inventory_total": inventory.total_episodes,
        "seed": args.seed,
        "scan_rule_version": SCAN_RULE_VERSION_V2,
        "sampling_config_hash": sampling_hash,
        "completion": completion,
        "uniform": _summarize(uniform_rows),
        "successful": {
            **_summarize(successful_rows),
            "sample_keys_match": sum(bool(row.get("sample_keys_match")) for row in successful_rows),
        },
        "uniform_rows": str(uniform_path),
        "successful_rows": str(successful_path),
    }
    summary_path = output / "summary.json"
    write_json(str(summary_path), summary)
    register_artifacts(
        str(args.cc_ledger) if args.cc_ledger else None,
        [summary_path, uniform_path, successful_path],
        purpose="V2 fixed-seed inventory and successful-episode audit",
        source_id=",".join(sorted(inventory.by_source)),
        run_id=run_root.name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
