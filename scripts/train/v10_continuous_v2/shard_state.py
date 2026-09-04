"""Immutable inventories and cumulative, crash-safe V2 shard attempts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from .artifact_ledger import register_artifacts
from .common.atomic import atomic_write, iter_jsonl, read_json, write_json
from .common.hashing import config_hash, shard_for
from .source_discovery import DiscoveredEpisode, discover_episodes


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ShardPlan:
    shard_id: int
    plan_hash: str
    path: str
    episode_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "plan_hash": self.plan_hash,
            "path": self.path,
            "episode_count": self.episode_count,
        }


@dataclass(frozen=True, slots=True)
class Inventory:
    inventory_hash: str
    path: str
    total_episodes: int
    by_source: dict[str, int]
    shards: tuple[ShardPlan, ...]
    effective_config_hash: str = ""
    source_filter: tuple[str, ...] = ()
    num_shards: int = 0
    max_episodes: int = 0
    discovery_mode: str = "root"
    input_inventory_hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inventory_hash": self.inventory_hash,
            "path": self.path,
            "total_episodes": self.total_episodes,
            "by_source": self.by_source,
            "shards": [item.to_dict() for item in self.shards],
            "effective_config_hash": self.effective_config_hash,
            "source_filter": list(self.source_filter),
            "num_shards": self.num_shards or len(self.shards),
            "max_episodes": self.max_episodes,
            "discovery_mode": self.discovery_mode,
            "input_inventory_hashes": self.input_inventory_hashes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Inventory":
        return cls(
            inventory_hash=str(value["inventory_hash"]),
            path=str(value["path"]),
            total_episodes=int(value["total_episodes"]),
            by_source={str(key): int(count) for key, count in value.get("by_source", {}).items()},
            shards=tuple(ShardPlan(**item) for item in value.get("shards", ())),
            effective_config_hash=str(value.get("effective_config_hash") or ""),
            source_filter=tuple(value.get("source_filter") or ()),
            num_shards=int(value.get("num_shards") or len(value.get("shards", ()))),
            max_episodes=int(value.get("max_episodes") or 0),
            discovery_mode=str(value.get("discovery_mode") or "root"),
            input_inventory_hashes={
                str(key): str(inventory_hash)
                for key, inventory_hash in (value.get("input_inventory_hashes") or {}).items()
            },
        )


def load_inventory_manifest(path: Path) -> Inventory:
    """Load an immutable inventory pointer or manifest without changing it."""
    value = read_json(str(path))
    if not isinstance(value, Mapping):
        raise FileNotFoundError(f"inventory manifest is missing or invalid: {path}")
    return Inventory.from_dict(value)


def _configured_source_ids(config: Mapping[str, Any]) -> set[str]:
    return {
        str(source.get("source_id") or "")
        for source in config.get("sources") or ()
        if isinstance(source, Mapping) and source.get("source_id")
    }


def resolve_discovery_inputs(
    config: Mapping[str, Any],
    *,
    source_filter: set[str] | None,
    requested_mode: str | None,
    inventory_overrides: Mapping[str, Path] | None = None,
) -> tuple[str, tuple[tuple[str, Inventory], ...]]:
    """Resolve ``auto`` to root discovery or immutable-list fast discovery.

    Fast mode is selected only when every requested source has a configured
    inventory. This avoids silently mixing cached and live discovery, whose
    cross-source realpath de-duplication order could otherwise change source
    identity.
    """
    discovery = config.get("discovery") or {}
    if not isinstance(discovery, Mapping):
        raise ValueError("discovery config must be a mapping")
    mode = str(requested_mode or discovery.get("default_mode") or "auto")
    if mode not in {"auto", "fast", "root"}:
        raise ValueError(f"unsupported discovery mode: {mode}")
    if mode == "root":
        return "root", ()

    configured = discovery.get("fast_inventories") or {}
    if not isinstance(configured, Mapping):
        raise ValueError("discovery.fast_inventories must be a mapping")
    paths = {str(key): Path(str(value)) for key, value in configured.items()}
    paths.update({str(key): Path(value) for key, value in (inventory_overrides or {}).items()})
    requested_sources = set(source_filter or _configured_source_ids(config))
    missing = sorted(requested_sources - set(paths))
    if missing:
        if mode == "fast":
            raise ValueError(
                "fast discovery has no immutable inventory for sources: "
                + ", ".join(missing)
            )
        return "root", ()

    resolved: list[tuple[str, Inventory]] = []
    try:
        for source_id in sorted(requested_sources):
            inventory = load_inventory_manifest(paths[source_id].resolve())
            if source_id not in inventory.by_source:
                raise ValueError(
                    f"fast inventory {paths[source_id]} does not contain source {source_id}"
                )
            resolved.append((source_id, inventory))
    except (FileNotFoundError, ValueError) as exc:
        if mode == "fast":
            raise
        LOGGER.warning("fast discovery unavailable; falling back to root: %s", exc)
        return "root", ()
    return "fast", tuple(resolved)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sort_plan(path: Path) -> None:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows.sort(key=lambda row: str(row["global_episode_key"]))
    with atomic_write(str(path)) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def discovery_config_hash(config: Mapping[str, Any]) -> str:
    """Hash only fields that can change the immutable directory inventory."""
    return config_hash({
        key: config.get(key)
        for key in (
            "version", "annotation_root", "l3_roots", "topic_max_depth",
            "inventory_shard_key", "sources",
        )
    })


def _inventory_shard_key(item: DiscoveredEpisode, strategy: str) -> str:
    if strategy == "episode":
        return item.global_episode_key
    if strategy == "topic":
        # Keep all records backed by the same topic-level instruction together.
        # The worker's JSON cache can then load each multi-MB annotation once.
        return f"{item.source_id}\0{os.path.abspath(item.job.topic)}"
    raise ValueError(f"unsupported inventory_shard_key: {strategy}")


class _InventoryShardBuffers:
    """Bound file descriptors while retaining append-only crash recovery.

    Inventory rows hash to effectively random shards. An LRU of open files
    would therefore reopen a file for almost every row. Small per-shard
    buffers give us zero long-lived descriptors and amortise those opens.
    """

    def __init__(self, root: Path, *, batch_size: int = 64) -> None:
        self.root = root
        self.batch_size = max(1, batch_size)
        self.buffers: dict[int, list[str]] = defaultdict(list)

    def path(self, shard_id: int) -> Path:
        return self.root / f"shard-{shard_id:05d}.jsonl"

    def write(self, shard_id: int, value: Mapping[str, Any]) -> None:
        rows = self.buffers[shard_id]
        rows.append(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        if len(rows) >= self.batch_size:
            self.flush(shard_id)

    def flush(self, shard_id: int) -> None:
        rows = self.buffers.get(shard_id)
        if not rows:
            return
        with self.path(shard_id).open("a", encoding="utf-8") as handle:
            handle.writelines(rows)
        rows.clear()

    def close(self) -> None:
        for shard_id in list(self.buffers):
            self.flush(shard_id)


def _load_partial_inventory(
    root: Path,
) -> tuple[Counter[int], Counter[str], set[str], int]:
    """Recover durable rows and truncate a possible torn final JSONL line."""
    counts: Counter[int] = Counter()
    by_source: Counter[str] = Counter()
    seen_realpaths: set[str] = set()
    total = 0
    for path in sorted(root.glob("shard-*.jsonl")):
        try:
            shard_id = int(path.stem.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        with path.open("rb+") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                try:
                    row = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    handle.truncate(offset)
                    break
                counts[shard_id] += 1
                by_source[str(row["source_id"])] += 1
                seen_realpaths.add(str(row["media_realpath"]))
                total += 1
    return counts, by_source, seen_realpaths, total


def inventory_matches(
    inventory: Inventory,
    config: Mapping[str, Any],
    *,
    num_shards: int,
    source_filter: set[str] | None,
    max_episodes: int,
    discovery_mode: str = "root",
    input_inventory_hashes: Mapping[str, str] | None = None,
) -> bool:
    """Return whether an immutable inventory exactly satisfies a request."""
    requested_sources = tuple(sorted(source_filter or ()))
    requested_hash = discovery_config_hash(config)
    compatible_hash = inventory.effective_config_hash in {
        requested_hash,
        config_hash(config),  # pre-v3 inventories stored the full config hash
    }
    if not compatible_hash:
        frozen = read_json(str(Path(inventory.path).parent / "effective_config.json"))
        compatible_hash = isinstance(frozen, Mapping) and discovery_config_hash(frozen) == requested_hash
    return (
        compatible_hash
        and inventory.source_filter == requested_sources
        and (inventory.num_shards or len(inventory.shards)) == num_shards
        and inventory.max_episodes == max_episodes
        and inventory.discovery_mode == discovery_mode
        and inventory.input_inventory_hashes == dict(sorted((input_inventory_hashes or {}).items()))
    )


def _iter_inventory_source(
    source_id: str, inventory: Inventory
) -> Iterator[DiscoveredEpisode]:
    count = 0
    for plan in sorted(inventory.shards, key=lambda item: item.shard_id):
        for item in iter_plan(plan):
            if item.source_id != source_id:
                continue
            count += 1
            yield item
    expected = inventory.by_source[source_id]
    if count != expected:
        raise ValueError(
            f"fast inventory source count mismatch for {source_id}: "
            f"manifest={expected}, rows={count}"
        )


def _discover_from_inventories(
    inventories: tuple[tuple[str, Inventory], ...], *, max_episodes: int
) -> Iterator[DiscoveredEpisode]:
    """Round-robin frozen path lists and de-duplicate exactly like root discovery."""
    active = [iter(_iter_inventory_source(source_id, inventory)) for source_id, inventory in inventories]
    seen_realpaths: set[str] = set()
    emitted = 0
    try:
        while active:
            next_active: list[Iterator[DiscoveredEpisode]] = []
            for iterator in active:
                try:
                    item = next(iterator)
                except StopIteration:
                    continue
                next_active.append(iterator)
                if item.media_realpath in seen_realpaths:
                    continue
                seen_realpaths.add(item.media_realpath)
                yield item
                emitted += 1
                if max_episodes and emitted >= max_episodes:
                    return
            active = next_active
    finally:
        for iterator in active:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()


def build_inventory(
    config: Mapping[str, Any],
    run_root: Path,
    *,
    num_shards: int,
    source_filter: set[str] | None,
    max_episodes: int,
    run_id: str,
    ledger_path: str | None,
    discovery_mode: str = "root",
    input_inventories: tuple[tuple[str, Inventory], ...] = (),
) -> Inventory:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    run_root.mkdir(parents=True, exist_ok=True)
    input_inventory_hashes = dict(sorted(
        (source_id, inventory.inventory_hash)
        for source_id, inventory in input_inventories
    ))
    if discovery_mode not in {"fast", "root"}:
        raise ValueError(f"unsupported resolved discovery mode: {discovery_mode}")
    if discovery_mode == "fast" and not input_inventories:
        raise ValueError("fast discovery requires at least one immutable inventory")
    if discovery_mode == "root" and input_inventories:
        raise ValueError("root discovery cannot accept input inventories")
    request = {
        "config_hash": discovery_config_hash(config),
        "num_shards": num_shards,
        "sources": sorted(source_filter or ()),
        "max_episodes": max_episodes,
        "discovery_mode": discovery_mode,
        "input_inventory_hashes": input_inventory_hashes,
    }
    build_id = config_hash(request)[:24]
    temporary_root = run_root / f".inventory-build-{build_id}"
    temporary_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "inventory_build.json"
    resume = temporary_root.is_dir() and any(temporary_root.glob("shard-*.jsonl"))
    if resume:
        counts, by_source, seen_realpaths, total = _load_partial_inventory(temporary_root)
    else:
        counts, by_source, seen_realpaths, total = Counter(), Counter(), set(), 0
    writer = _InventoryShardBuffers(
        temporary_root,
        batch_size=int((config.get("runtime") or {}).get("inventory_flush_records", 64)),
    )
    shard_key_strategy = str(config.get("inventory_shard_key") or "episode")

    def publish_progress(status: str) -> None:
        write_json(str(state_path), {
            "build_id": build_id,
            "pid": os.getpid(),
            "run_id": run_id,
            "status": status,
            "processed": total,
            "by_source": dict(sorted(by_source.items())),
            "temporary_root": str(temporary_root),
            "request": request,
            "resumed": resume,
            "updated_at": time.time(),
        })

    publish_progress("building_inventory")
    try:
        discovered = (
            _discover_from_inventories(input_inventories, max_episodes=max_episodes)
            if discovery_mode == "fast"
            else discover_episodes(
                config, source_filter=source_filter, max_episodes=max_episodes
            )
        )
        for item in discovered:
            if resume and item.media_realpath in seen_realpaths:
                continue
            shard_id = shard_for(
                _inventory_shard_key(item, shard_key_strategy), num_shards
            )
            writer.write(shard_id, item.to_dict())
            counts[shard_id] += 1
            by_source[item.source_id] += 1
            if resume:
                seen_realpaths.add(item.media_realpath)
            total += 1
            if total % 10_000 == 0:
                publish_progress("building_inventory")
        writer.close()
        publish_progress("finalizing_inventory")
        plans: list[ShardPlan] = []
        plans_root = run_root / "plans"
        for shard_id in sorted(counts):
            temporary = temporary_root / f"shard-{shard_id:05d}.jsonl"
            _sort_plan(temporary)
            plan_hash = _hash_file(temporary)
            final = plans_root / f"shard-{shard_id:05d}" / f"{plan_hash}.jsonl"
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                temporary.unlink()
            else:
                os.replace(temporary, final)
            plans.append(ShardPlan(shard_id, plan_hash, str(final), counts[shard_id]))
        semantic = {
            "config_hash": discovery_config_hash(config),
            "num_shards": num_shards,
            "sources": sorted(source_filter or ()),
            "max_episodes": max_episodes,
            "discovery_mode": discovery_mode,
            "input_inventory_hashes": input_inventory_hashes,
            "plans": [(plan.shard_id, plan.plan_hash, plan.episode_count) for plan in plans],
        }
        inventory_hash = config_hash(semantic)
        inventory_root = run_root / "inventories" / inventory_hash
        manifest_path = inventory_root / "manifest.json"
        inventory = Inventory(
            inventory_hash=inventory_hash,
            path=str(manifest_path),
            total_episodes=total,
            by_source=dict(sorted(by_source.items())),
            shards=tuple(plans),
            effective_config_hash=discovery_config_hash(config),
            source_filter=tuple(sorted(source_filter or ())),
            num_shards=num_shards,
            max_episodes=max_episodes,
            discovery_mode=discovery_mode,
            input_inventory_hashes=input_inventory_hashes,
        )
        if not manifest_path.exists():
            write_json(str(manifest_path), inventory.to_dict())
            write_json(str(inventory_root / "effective_config.json"), config)
        write_json(str(run_root / "current_inventory.json"), inventory.to_dict())
        register_artifacts(
            ledger_path,
            [manifest_path, run_root / "current_inventory.json", *(plan.path for plan in plans)],
            purpose="V2 deterministic Episode inventory and shard plans",
            source_id=",".join(sorted(by_source)) or "none",
            run_id=run_id,
        )
        try:
            state_path.unlink()
        except FileNotFoundError:
            pass
        return inventory
    except BaseException:
        writer.close()
        publish_progress("interrupted")
        raise
    finally:
        try:
            temporary_root.rmdir()
        except OSError:
            pass


def load_current_inventory(run_root: Path) -> Inventory:
    value = read_json(str(run_root / "current_inventory.json"))
    if not isinstance(value, Mapping):
        raise FileNotFoundError(f"current inventory missing under {run_root}")
    return Inventory.from_dict(value)


def iter_plan(plan: ShardPlan) -> Iterator[DiscoveredEpisode]:
    for value in iter_jsonl(plan.path):
        yield DiscoveredEpisode.from_dict(value)


def version_root(run_root: Path, plan: ShardPlan, *, stage: str = "scan") -> Path:
    base = "validation_shards" if stage == "validate" else "shards"
    return run_root / base / f"shard-{plan.shard_id:05d}" / plan.plan_hash


def completed_attempts(root: Path) -> list[Path]:
    return sorted(
        path for path in root.glob("attempt-*")
        if path.is_dir() and (path / ".done").is_file()
    ) if root.is_dir() else []


def next_attempt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in root.glob("attempt-*"):
        try:
            numbers.append(int(path.name.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return root / f"attempt-{max(numbers, default=0) + 1:04d}"


def load_attempt_records(attempt: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    episodes: dict[str, dict[str, Any]] = {}
    samples: dict[str, dict[str, Any]] = {}
    if not attempt:
        return episodes, samples
    for name in ("episodes_success.jsonl", "episodes_failed.jsonl"):
        for row in iter_jsonl(str(attempt / name)):
            episodes[str(row["global_episode_key"])] = row
    for row in iter_jsonl(str(attempt / "catalog.jsonl")):
        samples[str(row["sample_key"])] = row
    return episodes, samples


def artifact_paths(attempt: Path, *, quick: bool) -> list[Path]:
    names = [
        "episodes_success.jsonl", "episodes_failed.jsonl", "errors.jsonl",
        "statistics.json", ".done",
    ]
    if not quick:
        names.insert(0, "catalog.jsonl")
    return [attempt / name for name in names]
