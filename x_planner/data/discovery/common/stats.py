"""Five-level statistics: episode / video / view / sample / shard (spec section 9)."""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

MAX_REPRESENTATIVES = 3
MAX_PATHS_PER_ERROR = 200


@dataclass
class PathStat:
    path: str
    error_count: int = 0
    episodes: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    error_types: Counter = field(default_factory=Counter)
    representatives: list[str] = field(default_factory=list)

    def add(self, *, episode_key: str, source_id: str, error_type: str) -> None:
        self.error_count += 1
        self.episodes.add(episode_key)
        self.sources.add(source_id)
        self.error_types[error_type] += 1
        if len(self.representatives) < MAX_REPRESENTATIVES and episode_key not in self.representatives:
            self.representatives.append(episode_key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "error_count": self.error_count,
            "affected_episodes": len(self.episodes),
            "affected_sources": sorted(self.sources),
            "error_types": dict(self.error_types.most_common()),
            "representative_episodes": list(self.representatives),
        }


@dataclass
class ScanStats:
    """Mergeable counter bundle. Workers produce one, the coordinator sums them."""

    episodes_discovered: int = 0
    episodes_validated: int = 0
    episodes_validation_failed: int = 0
    episodes_scanned: int = 0
    episodes_scan_failed: int = 0

    videos_discovered: int = 0
    videos_valid: int = 0
    videos_invalid: int = 0

    samples_generated: int = 0

    shards_completed: int = 0
    shards_failed: int = 0

    by_source: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    by_error_type: Counter = field(default_factory=Counter)
    by_profile: Counter = field(default_factory=Counter)
    by_unit_level: Counter = field(default_factory=Counter)
    by_view_combination: Counter = field(default_factory=Counter)
    by_view: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    by_stage: Counter = field(default_factory=Counter)
    by_dataset: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    error_paths: dict[str, PathStat] = field(default_factory=dict)
    error_type_paths: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    error_type_episodes: dict[str, set] = field(default_factory=lambda: defaultdict(set))
    error_type_sources: dict[str, set] = field(default_factory=lambda: defaultdict(set))
    error_type_representatives: dict[str, list] = field(default_factory=lambda: defaultdict(list))

    unknown_view_stems: Counter = field(default_factory=Counter)

    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()

    def record_discovered(self, source_id: str, dataset_name: str = "") -> None:
        self.episodes_discovered += 1
        self.by_source[source_id]["discovered"] += 1
        if dataset_name:
            self.by_dataset[dataset_name]["discovered"] += 1
        self.touch()

    def record_validated(self, source_id: str, dataset_name: str = "", *, videos: int = 0) -> None:
        self.episodes_validated += 1
        self.by_source[source_id]["validated"] += 1
        if dataset_name:
            self.by_dataset[dataset_name]["validated"] += 1
        self.videos_discovered += videos
        self.videos_valid += videos
        self.touch()

    def record_success(
        self,
        source_id: str,
        *,
        dataset_name: str = "",
        profile: str = "",
        unit_level: str = "",
        views: tuple[str, ...] = (),
        sample_count: int = 0,
    ) -> None:
        self.episodes_scanned += 1
        self.samples_generated += sample_count
        counter = self.by_source[source_id]
        counter["scanned"] += 1
        counter["samples"] += sample_count
        if dataset_name:
            self.by_dataset[dataset_name]["scanned"] += 1
            self.by_dataset[dataset_name]["samples"] += sample_count
        if profile:
            self.by_profile[profile] += sample_count
        if unit_level:
            self.by_unit_level[unit_level] += sample_count
        if views:
            self.by_view_combination["+".join(views)] += sample_count
            for view in views:
                self.by_view[view]["samples"] += sample_count
                self.by_view[view]["episodes"] += 1
        self.touch()

    def record_failure(
        self,
        source_id: str,
        *,
        dataset_name: str = "",
        episode_key: str = "",
        error_type: str = "unknown_error",
        stage: str = "",
        view_name: str = "",
        input_paths: tuple[str, ...] = (),
        validation_stage: bool = False,
    ) -> None:
        if validation_stage:
            self.episodes_validation_failed += 1
            self.by_source[source_id]["validation_failed"] += 1
            if dataset_name:
                self.by_dataset[dataset_name]["validation_failed"] += 1
        else:
            self.episodes_scan_failed += 1
            self.by_source[source_id]["scan_failed"] += 1
            if dataset_name:
                self.by_dataset[dataset_name]["scan_failed"] += 1

        self.by_error_type[error_type] += 1
        self.by_source[source_id][f"error:{error_type}"] += 1
        if dataset_name:
            self.by_dataset[dataset_name][f"error:{error_type}"] += 1
        if stage:
            self.by_stage[stage] += 1
        if view_name:
            self.by_view[view_name]["errors"] += 1
            self.by_view[view_name][f"error:{error_type}"] += 1

        self.error_type_episodes[error_type].add(episode_key)
        self.error_type_sources[error_type].add(source_id)
        reps = self.error_type_representatives[error_type]
        if len(reps) < MAX_REPRESENTATIVES and episode_key not in reps:
            reps.append(episode_key)

        for path in input_paths:
            self.error_type_paths[error_type][path] += 1
            stat = self.error_paths.get(path)
            if stat is None:
                if len(self.error_paths) >= MAX_PATHS_PER_ERROR * len(self.by_error_type or {1: 1}):
                    continue
                stat = PathStat(path=path)
                self.error_paths[path] = stat
            stat.add(episode_key=episode_key, source_id=source_id, error_type=error_type)
        self.touch()

    def record_invalid_video(self, count: int = 1) -> None:
        self.videos_discovered += count
        self.videos_invalid += count
        self.touch()

    def record_unknown_view(self, stem: str) -> None:
        self.unknown_view_stems[stem] += 1

    # -- aggregation ----------------------------------------------------

    def merge(self, other: "ScanStats") -> None:
        for name in (
            "episodes_discovered",
            "episodes_validated",
            "episodes_validation_failed",
            "episodes_scanned",
            "episodes_scan_failed",
            "videos_discovered",
            "videos_valid",
            "videos_invalid",
            "samples_generated",
            "shards_completed",
            "shards_failed",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))

        for source, counter in other.by_source.items():
            self.by_source[source].update(counter)
        for dataset, counter in other.by_dataset.items():
            self.by_dataset[dataset].update(counter)
        for view, counter in other.by_view.items():
            self.by_view[view].update(counter)
        self.by_error_type.update(other.by_error_type)
        self.by_profile.update(other.by_profile)
        self.by_unit_level.update(other.by_unit_level)
        self.by_view_combination.update(other.by_view_combination)
        self.by_stage.update(other.by_stage)
        self.unknown_view_stems.update(other.unknown_view_stems)

        for error_type, counter in other.error_type_paths.items():
            self.error_type_paths[error_type].update(counter)
        for error_type, episodes in other.error_type_episodes.items():
            self.error_type_episodes[error_type].update(episodes)
        for error_type, sources in other.error_type_sources.items():
            self.error_type_sources[error_type].update(sources)
        for error_type, reps in other.error_type_representatives.items():
            target = self.error_type_representatives[error_type]
            for rep in reps:
                if len(target) < MAX_REPRESENTATIVES and rep not in target:
                    target.append(rep)

        for path, stat in other.error_paths.items():
            existing = self.error_paths.get(path)
            if existing is None:
                self.error_paths[path] = stat
                continue
            existing.error_count += stat.error_count
            existing.episodes.update(stat.episodes)
            existing.sources.update(stat.sources)
            existing.error_types.update(stat.error_types)
            for rep in stat.representatives:
                if len(existing.representatives) < MAX_REPRESENTATIVES and rep not in existing.representatives:
                    existing.representatives.append(rep)

        self.started_at = min(self.started_at, other.started_at)
        self.updated_at = max(self.updated_at, other.updated_at)

    # -- serialization ---------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Portable form used both for IPC and for statistics/*.json."""
        return {
            "episodes_discovered": self.episodes_discovered,
            "episodes_validated": self.episodes_validated,
            "episodes_validation_failed": self.episodes_validation_failed,
            "episodes_scanned": self.episodes_scanned,
            "episodes_scan_failed": self.episodes_scan_failed,
            "videos_discovered": self.videos_discovered,
            "videos_valid": self.videos_valid,
            "videos_invalid": self.videos_invalid,
            "samples_generated": self.samples_generated,
            "shards_completed": self.shards_completed,
            "shards_failed": self.shards_failed,
            "by_source": {key: dict(value) for key, value in self.by_source.items()},
            "by_dataset": {key: dict(value) for key, value in self.by_dataset.items()},
            "by_error_type": dict(self.by_error_type),
            "by_profile": dict(self.by_profile),
            "by_unit_level": dict(self.by_unit_level),
            "by_view_combination": dict(self.by_view_combination),
            "by_view": {key: dict(value) for key, value in self.by_view.items()},
            "by_stage": dict(self.by_stage),
            "unknown_view_stems": dict(self.unknown_view_stems),
            "error_paths": {key: value.to_dict() for key, value in self.error_paths.items()},
            "error_type_paths": {
                key: dict(value) for key, value in self.error_type_paths.items()
            },
            "error_type_episodes": {
                key: len(value) for key, value in self.error_type_episodes.items()
            },
            "error_type_sources": {
                key: sorted(value) for key, value in self.error_type_sources.items()
            },
            "error_type_representatives": {
                key: list(value) for key, value in self.error_type_representatives.items()
            },
            "started_at": self.started_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ScanStats":
        stats = cls()
        for name in (
            "episodes_discovered",
            "episodes_validated",
            "episodes_validation_failed",
            "episodes_scanned",
            "episodes_scan_failed",
            "videos_discovered",
            "videos_valid",
            "videos_invalid",
            "samples_generated",
            "shards_completed",
            "shards_failed",
        ):
            setattr(stats, name, int(value.get(name, 0)))
        for source, counter in (value.get("by_source") or {}).items():
            stats.by_source[source] = Counter(counter)
        for dataset, counter in (value.get("by_dataset") or {}).items():
            stats.by_dataset[dataset] = Counter(counter)
        for view, counter in (value.get("by_view") or {}).items():
            stats.by_view[view] = Counter(counter)
        stats.by_error_type = Counter(value.get("by_error_type") or {})
        stats.by_profile = Counter(value.get("by_profile") or {})
        stats.by_unit_level = Counter(value.get("by_unit_level") or {})
        stats.by_view_combination = Counter(value.get("by_view_combination") or {})
        stats.by_stage = Counter(value.get("by_stage") or {})
        stats.unknown_view_stems = Counter(value.get("unknown_view_stems") or {})
        for error_type, counter in (value.get("error_type_paths") or {}).items():
            stats.error_type_paths[error_type] = Counter(counter)
        for error_type, sources in (value.get("error_type_sources") or {}).items():
            stats.error_type_sources[error_type] = set(sources)
        for error_type, reps in (value.get("error_type_representatives") or {}).items():
            stats.error_type_representatives[error_type] = list(reps)
        for path, payload in (value.get("error_paths") or {}).items():
            stat = PathStat(path=path, error_count=int(payload.get("error_count", 0)))
            stat.sources = set(payload.get("affected_sources") or ())
            stat.error_types = Counter(payload.get("error_types") or {})
            stat.representatives = list(payload.get("representative_episodes") or ())
            # ``affected_episodes`` is stored as a count; re-seed with the
            # representatives so merges stay monotonic without holding every key.
            stat.episodes = set(stat.representatives)
            stats.error_paths[path] = stat
        stats.started_at = float(value.get("started_at", time.time()))
        stats.updated_at = float(value.get("updated_at", time.time()))
        return stats

    def throughput(self) -> dict[str, float]:
        elapsed = max(1e-6, self.updated_at - self.started_at)
        processed = self.episodes_scanned + self.episodes_scan_failed
        return {
            "elapsed_seconds": elapsed,
            "episodes_per_second": processed / elapsed,
            "samples_per_second": self.samples_generated / elapsed,
        }

    def summary(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["throughput"] = self.throughput()
        return payload
