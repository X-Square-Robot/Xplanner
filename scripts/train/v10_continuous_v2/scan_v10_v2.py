#!/usr/bin/env python3
"""Unified command line for the V10 continuous V2 data pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from .artifact_ledger import register_artifacts
from .build_dataset_manifests import build_manifests
from .build_snapshot_v2 import build_snapshot
from .common.atomic import read_json, write_json
from .common.constants_v2 import DEFAULT_OUTPUT_ROOT, SCAN_RULE_VERSION_V2
from .common.hashing import config_hash
from .merge_catalogs_v2 import merge_catalogs
from .sampling_v2 import validate_sampling_config
from .scan_status_v2 import compact_status, collect_status, collect_status_many, publish_statistics
from .scanner_v2 import run_inventory
from .shard_state import (
    build_inventory,
    discovery_config_hash,
    inventory_matches,
    load_current_inventory,
    resolve_discovery_inputs,
)


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "sources.yml"
DEFAULT_VIEWS = PACKAGE_ROOT / "configs" / "views.yml"
DEFAULT_MIXTURE = PACKAGE_ROOT / "configs" / "mixture.yml"
DEFAULT_LEDGER = Path("/mnt/cpfs/zbl-cpfs-new/USERS/luhao/APlan/0806/scan_check_cc.md")


def _yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "validate", "scan", "resume", "status", "merge", "manifest", "snapshot", "all"
    ))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--views-config", type=Path, default=DEFAULT_VIEWS)
    parser.add_argument("--mixture-config", type=Path, default=DEFAULT_MIXTURE)
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id")
    parser.add_argument(
        "--inventory-from-run-id",
        help="For a new scan/validate run, reuse another run's immutable inventory by reference.",
    )
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument(
        "--discovery-mode", choices=("auto", "fast", "root"),
        help=(
            "auto (default) uses immutable path inventories when every selected "
            "source is covered, otherwise root discovery; fast requires those "
            "inventories; root always re-enumerates source roots"
        ),
    )
    parser.add_argument(
        "--target-inventory", action="append", default=[], metavar="SOURCE=PATH",
        help="Override a fast-discovery immutable inventory for one source (repeatable).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Documentation alias; use the resume command.")
    parser.add_argument("--legacy-catalog", type=Path, action="append", default=[])
    parser.add_argument(
        "--input-run-id", action="append", default=[],
        help="Additional completed V2 run to include in merge (repeatable).",
    )
    parser.add_argument("--bucket-count", type=int, default=256)
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Allow merge from incomplete shards or mixed completion hashes.",
    )
    parser.add_argument(
        "--publish-statistics", action="store_true",
        help="For status, stream exact failed/success path files and path/view aggregates.",
    )
    parser.add_argument("--cc-ledger", type=Path, default=DEFAULT_LEDGER)
    return parser.parse_args()


def _settings(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    runtime = dict(config.get("runtime") or {})
    runtime.update({
        "sampling": dict(config.get("sampling") or {}),
        "max_camera_views": int((config.get("sampling") or {}).get("max_camera_views", 3)),
        "rule_version": SCAN_RULE_VERSION_V2,
        "video_validation": str(runtime.get("video_validation", "metadata")),
    })
    if args.num_workers is not None:
        runtime["num_workers"] = args.num_workers
    if args.num_shards is not None:
        runtime["num_shards"] = args.num_shards
    validate_sampling_config(runtime["sampling"])
    return runtime


def _inventory_overrides(args: argparse.Namespace) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for value in getattr(args, "target_inventory", ()) or ():
        source_id, separator, raw_path = str(value).partition("=")
        if not separator or not source_id or not raw_path:
            raise ValueError("--target-inventory must use SOURCE=PATH")
        if source_id in overrides:
            raise ValueError(f"duplicate --target-inventory source: {source_id}")
        overrides[source_id] = Path(raw_path)
    return overrides


def _discovery_request(config: dict[str, Any], args: argparse.Namespace):
    return resolve_discovery_inputs(
        config,
        source_filter=set(args.source) or None,
        requested_mode=getattr(args, "discovery_mode", None),
        inventory_overrides=_inventory_overrides(args),
    )


def _default_run_id(config: dict[str, Any], args: argparse.Namespace) -> str:
    runtime = dict(config.get("runtime") or {})
    num_shards = args.num_shards if args.num_shards is not None else int(runtime.get("num_shards", 256))
    discovery_mode, input_inventories = _discovery_request(config, args)
    semantic = {
        # Keep merge destinations, worker counts, and checkpoint cadence out of
        # the scan identity. They cannot change either the inventory or rows.
        "discovery_config_hash": discovery_config_hash(config),
        "sources": sorted(args.source),
        "max_episodes": args.max_episodes,
        "num_shards": num_shards,
        "discovery_mode": discovery_mode,
        "input_inventory_hashes": {
            source_id: inventory.inventory_hash
            for source_id, inventory in input_inventories
        },
        "sampling": config.get("sampling") or {},
        "validation": {
            key: runtime.get(key)
            for key in (
                "video_validation",
                "validation_ratio",
                "seed",
                "min_frames",
                "min_width",
                "min_height",
            )
        },
        "rule_version": SCAN_RULE_VERSION_V2,
    }
    return f"scan-{config_hash(semantic)[:12]}"


def _resolve_run(args: argparse.Namespace, config: dict[str, Any]) -> tuple[str, Path]:
    output = args.output_root.resolve()
    if args.run_id:
        run_id = args.run_id
    elif args.command in {"status", "resume", "merge", "manifest", "snapshot"}:
        current = read_json(str(output / "current_run.json"))
        if not isinstance(current, dict) or not current.get("run_id"):
            raise FileNotFoundError(f"no current V2 run under {output}")
        run_id = str(current["run_id"])
    else:
        run_id = _default_run_id(config, args)
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in run_id):
        raise ValueError("run-id may contain only letters, digits, '-' and '_'")
    return run_id, output / "runs" / run_id


def _inventory_for_command(
    command: str,
    *,
    config: dict[str, Any],
    run_root: Path,
    run_id: str,
    args: argparse.Namespace,
    settings: dict[str, Any],
):
    if command in {"validate", "scan", "all"}:
        requested_shards = int(settings.get("num_shards", 256))
        requested_sources = set(args.source) or None
        discovery_mode, input_inventories = _discovery_request(config, args)
        input_inventory_hashes = {
            source_id: inventory.inventory_hash
            for source_id, inventory in input_inventories
        }
        try:
            existing = load_current_inventory(run_root)
        except FileNotFoundError:
            existing = None
        if args.inventory_from_run_id:
            source_run_id = str(args.inventory_from_run_id)
            if not source_run_id or any(
                char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for char in source_run_id
            ):
                raise ValueError(f"invalid inventory-from-run-id: {source_run_id}")
            source_root = run_root.parent / source_run_id
            if source_root.resolve() == run_root.resolve():
                raise ValueError("inventory-from-run-id must name a different run")
            borrowed = load_current_inventory(source_root)
            if not inventory_matches(
                borrowed,
                config,
                num_shards=requested_shards,
                source_filter=requested_sources,
                max_episodes=args.max_episodes,
                discovery_mode=discovery_mode,
                input_inventory_hashes=input_inventory_hashes,
            ):
                raise ValueError(
                    f"inventory from {source_run_id} does not match requested "
                    "discovery/source/shard/max-episodes semantics"
                )
            if existing is not None and existing.inventory_hash != borrowed.inventory_hash:
                raise ValueError(
                    f"target run already references inventory {existing.inventory_hash}, "
                    f"not borrowed inventory {borrowed.inventory_hash}"
                )
            if existing is None:
                write_json(str(run_root / "current_inventory.json"), borrowed.to_dict())
                write_json(str(run_root / "inventory_reference.json"), {
                    "inventory_hash": borrowed.inventory_hash,
                    "inventory_manifest": borrowed.path,
                    "source_run_id": source_run_id,
                    "source_run_root": str(source_root.resolve()),
                })
            existing = borrowed
        if existing is not None and inventory_matches(
            existing,
            config,
            num_shards=requested_shards,
            source_filter=requested_sources,
            max_episodes=args.max_episodes,
            discovery_mode=discovery_mode,
            input_inventory_hashes=input_inventory_hashes,
        ):
            return existing
        return build_inventory(
            config,
            run_root,
            num_shards=requested_shards,
            source_filter=requested_sources,
            max_episodes=args.max_episodes,
            run_id=run_id,
            ledger_path=str(args.cc_ledger) if args.cc_ledger else None,
            discovery_mode=discovery_mode,
            input_inventories=input_inventories,
        )
    inventory = load_current_inventory(run_root)
    requested_sources = set(args.source)
    if requested_sources and requested_sources != set(inventory.by_source):
        raise ValueError(
            f"requested sources {sorted(requested_sources)} do not match frozen inventory "
            f"sources {sorted(inventory.by_source)}"
        )
    return inventory


def _input_run_roots(args: argparse.Namespace, run_root: Path) -> list[Path]:
    roots = [run_root]
    for run_id in args.input_run_id:
        if not run_id or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for char in run_id
        ):
            raise ValueError(f"invalid input-run-id: {run_id}")
        roots.append(args.output_root.resolve() / "runs" / run_id)
    return list(dict.fromkeys(path.resolve() for path in roots))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    config = _yaml(args.config)
    views = _yaml(args.views_config)
    settings = _settings(config, args)
    num_workers = int(settings.get("num_workers", 16))
    if num_workers <= 0 or int(settings.get("num_shards", 256)) <= 0:
        raise ValueError("num-workers and num-shards must be positive")
    run_id, run_root = _resolve_run(args, config)
    if args.dry_run:
        discovery_mode, input_inventories = _discovery_request(config, args)
        print(json.dumps({
            "status": "dry-run",
            "run_id": run_id,
            "run_root": str(run_root),
            "sources": sorted(args.source),
            "settings": settings,
            "inventory_from_run_id": args.inventory_from_run_id,
            "discovery_mode": discovery_mode,
            "input_inventories": {
                source_id: {
                    "inventory_hash": inventory.inventory_hash,
                    "path": inventory.path,
                    "episodes": inventory.by_source[source_id],
                }
                for source_id, inventory in input_inventories
            },
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(str(args.output_root.resolve() / "current_run.json"), {
        "run_id": run_id, "run_root": str(run_root), "rule_version": SCAN_RULE_VERSION_V2
    })
    register_artifacts(
        str(args.cc_ledger) if args.cc_ledger else None,
        [args.output_root.resolve() / "current_run.json"],
        purpose="V2 current run pointer",
        source_id=",".join(sorted(args.source)) or "all",
        run_id=run_id,
    )
    inventory = _inventory_for_command(
        args.command,
        config=config,
        run_root=run_root,
        run_id=run_id,
        args=args,
        settings=settings,
    )
    common = {
        "run_id": run_id,
        "settings": settings,
        "view_config": views,
        "num_workers": num_workers,
        "shard_id": args.shard_id,
        "ledger_path": str(args.cc_ledger) if args.cc_ledger else None,
        "fail_fast": args.fail_fast,
    }
    result: Any
    if args.command == "validate":
        result = run_inventory(inventory, run_root, stage="validate", mode="scan", **common)
    elif args.command == "scan":
        result = run_inventory(inventory, run_root, stage="scan", mode="scan", **common)
    elif args.command == "resume":
        result = run_inventory(inventory, run_root, stage="scan", mode="resume", **common)
    elif args.command == "status":
        full_status = collect_status_many(_input_run_roots(args, run_root))
        if args.publish_statistics:
            paths = publish_statistics(run_root, full_status)
            register_artifacts(
                common["ledger_path"], paths, purpose="V2 status statistics",
                source_id=",".join(sorted(inventory.by_source)), run_id=run_id,
            )
        result = compact_status(full_status)
    elif args.command == "merge":
        merge_config = config.get("merge") or {}
        legacy_inputs = args.legacy_catalog or [
            Path(value) for value in merge_config.get("prior_catalogs") or ()
        ]
        result = merge_catalogs(
            run_root,
            run_id=run_id,
            sampling=settings["sampling"],
            legacy_catalogs=legacy_inputs,
            input_run_roots=_input_run_roots(args, run_root),
            bucket_count=args.bucket_count,
            ledger_path=common["ledger_path"],
            allow_partial=args.allow_partial,
        )
    elif args.command == "manifest":
        result = build_manifests(run_root, run_id=run_id, ledger_path=common["ledger_path"])
    elif args.command == "snapshot":
        result = build_snapshot(
            run_root,
            run_id=run_id,
            config_paths=(args.config, args.views_config, args.mixture_config),
            ledger_path=common["ledger_path"],
        )
    else:
        stages: dict[str, Any] = {}
        stages["validate"] = run_inventory(inventory, run_root, stage="validate", mode="scan", **common)
        stages["scan"] = run_inventory(inventory, run_root, stage="scan", mode="scan", **common)
        status = collect_status(run_root)
        paths = publish_statistics(run_root, status)
        register_artifacts(
            common["ledger_path"], paths, purpose="V2 status statistics",
            source_id=",".join(sorted(inventory.by_source)), run_id=run_id,
        )
        stages["status"] = compact_status(status)
        stages["merge"] = merge_catalogs(
            run_root, run_id=run_id, sampling=settings["sampling"],
            legacy_catalogs=args.legacy_catalog, input_run_roots=_input_run_roots(args, run_root),
            bucket_count=args.bucket_count,
            ledger_path=common["ledger_path"],
            allow_partial=args.allow_partial,
        )
        stages["manifest"] = build_manifests(run_root, run_id=run_id, ledger_path=common["ledger_path"])
        stages["snapshot"] = build_snapshot(
            run_root, run_id=run_id,
            config_paths=(args.config, args.views_config, args.mixture_config),
            ledger_path=common["ledger_path"],
        )
        result = stages
    print(json.dumps({"run_id": run_id, "run_root": str(run_root), "result": result}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
