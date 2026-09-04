"""Incrementally merge only completed Memory V3 shards into reference lists."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .common_v3 import (
    canonical_digest,
    file_sha256,
    iter_jsonl,
    iter_numbered_jsonl,
    load_config,
    mark_success,
    read_success,
    write_json,
)


PACKAGE_ROOT = Path(__file__).resolve().parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ref(row: Mapping[str, Any], path: Path, line_number: int, task_type: str) -> dict[str, Any]:
    return {
        "sample_key": str(row["sample_key"]),
        "shard_path": str(path.resolve()),
        "line_number": int(line_number),
        "global_episode_key": str(row["global_episode_key"]),
        "split": str(row["split"]),
        "profile": str(row["profile"]),
        "task_type": task_type,
    }


def _merge_shard_part(payload: Mapping[str, Any]) -> dict[str, Any]:
    shard = Path(str(payload["shard"]))
    part_root = Path(str(payload["part_root"]))
    part_root.mkdir(parents=True, exist_ok=False)
    counts: Counter[str] = Counter()
    seen: dict[str, set[str]] = {
        name: set() for name in ("continuous", "initial_plan", "terminal")
    }
    handles = {
        name: (part_root / f"{name}.list").open("w", encoding="utf-8")
        for name in ("continuous", "initial_plan", "terminal", "initial_plan_oversize")
    }
    try:
        for task_type, filename in (
            ("continuous", "continuous.jsonl"),
            ("initial_plan", "initial_plan.jsonl"),
        ):
            path = shard / filename
            for line_number, row in iter_numbered_jsonl(path):
                key = str(row["sample_key"])
                if key in seen[task_type]:
                    raise ValueError(f"duplicate {task_type} sample_key in {shard}: {key}")
                seen[task_type].add(key)
                handles[task_type].write(json.dumps(
                    _ref(row, path, line_number, task_type),
                    ensure_ascii=False, separators=(",", ":"),
                ) + "\n")
                counts[task_type] += 1
        for _, row in iter_numbered_jsonl(shard / "terminal_refs.jsonl"):
            key = str(row["sample_key"])
            if key in seen["terminal"]:
                raise ValueError(f"duplicate terminal sample_key in {shard}: {key}")
            if key not in seen["continuous"]:
                raise ValueError(f"terminal ref does not point at a continuous sample: {key}")
            seen["terminal"].add(key)
            handles["terminal"].write(json.dumps(
                row, ensure_ascii=False, separators=(",", ":")
            ) + "\n")
            counts["terminal"] += 1
        for _, row in iter_numbered_jsonl(shard / "initial_plan_oversize.jsonl"):
            handles["initial_plan_oversize"].write(json.dumps(
                row, ensure_ascii=False, separators=(",", ":")
            ) + "\n")
            counts["initial_plan_oversize"] += 1
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
        return {
            "order": int(payload["order"]),
            "shard": str(shard),
            "part_root": str(part_root),
            "counts": dict(counts),
        }
    finally:
        for handle in handles.values():
            handle.close()


def _validate_unique_episode_partition(
    build_root: Path, completed: list[tuple[Path, Mapping[str, Any]]]
) -> int:
    seen: set[str] = set()
    for shard, _ in completed:
        index_path = build_root / "episode_index" / f"{shard.name}.jsonl"
        for row in iter_jsonl(str(index_path)):
            key = str(row["global_episode_key"])
            if key in seen:
                raise ValueError(f"Episode key occurs in multiple completed shards: {key}")
            seen.add(key)
    return len(seen)


def merge(
    config_path: str | Path,
    *,
    build_root: str | Path | None,
    allow_partial: bool,
    workers: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    output_root = Path(str(config["output_root"]))
    if build_root is None:
        current = json.loads((output_root / "current_build.json").read_text(encoding="utf-8"))
        build_root = current["build_root"]
    build_root = Path(build_root).resolve()
    manifest_path = build_root / "build_manifest.json"
    if manifest_path.is_file():
        build_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        if not allow_partial:
            raise RuntimeError(
                f"active build has no final build_manifest; pass --allow-partial: {build_root}"
            )
        index_manifest = read_success(build_root / "episode_index")
        if index_manifest is None:
            raise RuntimeError(f"active build has no completed Episode index: {build_root}")
        build_manifest = {
            "schema_version": "memory_v3_active_build_v1",
            "build_id": build_root.name,
            "build_root": str(build_root),
            "expected_shards": int(index_manifest["nonempty_shards"]),
            "episode_index": index_manifest,
            "counts": {},
            "active": True,
        }
    completed = []
    marker_hashes = []
    for shard in sorted((build_root / "shards").glob("shard-*")):
        marker = read_success(shard)
        if marker is None:
            continue
        completed.append((shard, marker))
        marker_hashes.append(file_sha256(shard / "_SUCCESS"))
    expected = int(build_manifest["expected_shards"])
    failure_count = sum(int(marker.get("failure_count", 0)) for _, marker in completed)
    index_manifest = build_manifest.get("episode_index") or {}
    full_coverage = (
        index_manifest.get("requested_max_episodes") is None
        and not index_manifest.get("source_filter")
        and int(index_manifest.get("selected_episode_count", -1))
        == int(index_manifest.get("formal_episode_count", -2))
    )
    complete = len(completed) == expected and failure_count == 0 and full_coverage
    if not allow_partial and not complete:
        raise RuntimeError(
            f"build is not complete: completed={len(completed)}/{expected} failures={failure_count}"
        )
    if not completed:
        raise RuntimeError("no completed zero-failure shards are available to merge")
    episode_partition_count = _validate_unique_episode_partition(build_root, completed)
    merge_workers = max(1, min(
        int(workers or os.environ.get("MEMORY_V3_MERGE_WORKERS", "16")),
        len(completed),
    ))
    identity = {
        "merge_schema_revision": 4,
        "build_id": build_manifest["build_id"],
        "completed_marker_hashes": marker_hashes,
        "allow_partial": bool(allow_partial),
        "config_digest": canonical_digest(config),
    }
    merge_id = canonical_digest(identity)[:24]
    final = output_root / "merged" / merge_id
    marker = read_success(final)
    if marker is not None:
        result = {**marker, "root": str(final.resolve())}
        write_json(str(output_root / "current_merge.json"), result)
        return result
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{merge_id}-", dir=final.parent))
    handles: dict[str, Any] = {}
    counts: Counter[str] = Counter()
    try:
        parts_root = temporary / "parts"
        parts_root.mkdir()
        results = []
        with ProcessPoolExecutor(max_workers=merge_workers) as pool:
            futures = [
                pool.submit(_merge_shard_part, {
                    "order": order,
                    "shard": str(shard),
                    "part_root": str(parts_root / f"part-{order:05d}"),
                })
                for order, (shard, _) in enumerate(completed)
            ]
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda row: int(row["order"]))
        for result in results:
            counts.update({name: int(value) for name, value in result["counts"].items()})
        output_digests = {
            name: hashlib.sha256()
            for name in ("continuous", "initial_plan", "terminal", "initial_plan_oversize")
        }
        for name in ("continuous", "initial_plan", "terminal", "initial_plan_oversize"):
            handles[name] = (temporary / f"{name}.list").open("wb")
            for result in results:
                with (Path(str(result["part_root"])) / f"{name}.list").open("rb") as source:
                    while chunk := source.read(16 * 1024 * 1024):
                        output_digests[name].update(chunk)
                        handles[name].write(chunk)
            handles[name].flush()
            os.fsync(handles[name].fileno())
            handles[name].close()
        handles.clear()
        shutil.rmtree(parts_root)
        files = {
            f"{name}.list": {
                "sha256": output_digests[name].hexdigest(),
                "bytes": (temporary / f"{name}.list").stat().st_size,
                "rows": counts[name],
            }
            for name in ("continuous", "initial_plan", "terminal", "initial_plan_oversize")
        }
        manifest = {
            "schema_version": "memory_v3_merged_lists_v1",
            "merge_id": merge_id,
            "created_at": _utc_now(),
            "build_root": str(build_root),
            "build_id": build_manifest["build_id"],
            "complete": complete,
            "partial_inputs": not complete,
            "expected_shards": expected,
            "completed_shards": len(completed),
            "failure_count": failure_count,
            "full_episode_coverage": full_coverage,
            "episode_partition_count": episode_partition_count,
            "merge_workers": merge_workers,
            "builder_counts": build_manifest.get("counts") or {},
            "files": files,
            "identity": identity,
        }
        manifest["content_digest"] = canonical_digest(manifest)
        write_json(str(temporary / "manifest.json"), manifest)
        mark_success(temporary, manifest)
        if final.exists():
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, final)
        result = {**manifest, "root": str(final.resolve())}
        write_json(str(output_root / "current_merge.json"), result)
        return result
    except BaseException:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "configs" / "v3_memory.yaml"))
    parser.add_argument("--build-root")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    result = merge(
        args.config, build_root=args.build_root,
        allow_partial=args.allow_partial, workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
