"""Fail-closed evaluation-holdout protection for every training stage.

The same physical episode is visible on another deployment through several mount spellings.
This module therefore compares logical episode IDs, normalized paths, resolved
paths, and (when the referenced file exists) device/inode identities.  It is
deliberately independent of a particular source adapter so Takeover, RoboDojo,
Baseline, composed snapshots, prepared datasets, and launchers share one gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "v5_evaluation_holdout_guard_v1"
AUDIT_SCHEMA_VERSION = "v5_evaluation_holdout_audit_v1"
DEFAULT_EVALUATION_MANIFEST = Path(os.environ.get(
    "XPLANNER_EVALUATION_MANIFEST",
    "benchmarks/evaluation_holdout/manifest.jsonl",
))
DEFAULT_EVALUATION_SHA256 = os.environ.get("XPLANNER_EVALUATION_SHA256") or None


def _load_path_aliases() -> tuple[tuple[str, str], ...]:
    """Load deployment-specific physical-to-logical path aliases from JSON.

    Example::

        export XPLANNER_PATH_ALIASES='{\"/data/open_action\":\"/open_data/video\"}'
    """

    raw = os.environ.get("XPLANNER_PATH_ALIASES", "").strip()
    if not raw:
        return ()
    value = json.loads(raw)
    if not isinstance(value, Mapping):
        raise ValueError("XPLANNER_PATH_ALIASES must be a JSON object")
    aliases: list[tuple[str, str]] = []
    for physical, logical in value.items():
        if not isinstance(physical, str) or not isinstance(logical, str):
            raise ValueError("XPLANNER_PATH_ALIASES keys and values must be strings")
        aliases.append((physical.rstrip("/"), logical.rstrip("/")))
    return tuple(sorted(aliases, key=lambda item: len(item[0]), reverse=True))


_PATH_ALIASES = _load_path_aliases()

_OPEN_DATASET_ALIASES = {
    "AgiBotWorld-Alpha": "AgiBotWorld-Alpha-v2",
    "AgiBotWorld-Beta": "AgiBotWorld-Beta-v2",
    "RH20T_transfer": "RH20T_transfer_v2",
    "driod": "droid_success_only",
    "droid_success_only_v2": "droid_success_only",
}


class HoldoutViolationError(RuntimeError):
    """Raised when a training artifact contains an evaluation-holdout identity."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_path(value: Any) -> str:
    """Return a stable logical spelling without requiring the path to exist."""

    if not isinstance(value, (str, os.PathLike)):
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    # PurePosixPath removes duplicate slashes and harmless ``.`` components.
    normalized = PurePosixPath(raw).as_posix()
    if raw.startswith("/") and not normalized.startswith("/"):
        normalized = "/" + normalized
    for physical, logical in _PATH_ALIASES:
        if normalized == physical:
            normalized = logical
            break
        if normalized.startswith(physical + "/"):
            normalized = logical + normalized[len(physical):]
            break
    prefix = "/open_data/video/"
    if normalized.startswith(prefix):
        tail = normalized[len(prefix):]
        dataset, separator, remainder = tail.partition("/")
        dataset = _OPEN_DATASET_ALIASES.get(dataset, dataset)
        normalized = prefix + dataset + (separator + remainder if separator else "")
    return normalized.rstrip("/") or "/"


def _inode_identity(path_text: str) -> tuple[int, int] | None:
    try:
        value = Path(path_text).stat()
    except (OSError, ValueError):
        return None
    return value.st_dev, value.st_ino


def _path_values(row: Mapping[str, Any]) -> Iterator[str]:
    for name in (
        "logical_episode_path",
        "resolved_episode_path",
        "existing_episode_path",
    ):
        value = row.get(name)
        if isinstance(value, str) and value:
            yield value
    candidates = row.get("path_candidates")
    if isinstance(candidates, list):
        for value in candidates:
            if isinstance(value, str) and value:
                yield value


def _video_values(row: Mapping[str, Any]) -> Iterator[str]:
    videos = row.get("camera_videos")
    if isinstance(videos, list):
        for item in videos:
            if isinstance(item, Mapping):
                value = item.get("mp4_path")
                if isinstance(value, str) and value:
                    yield value


def _sample_payload(value: Mapping[str, Any]) -> Mapping[str, Any]:
    for field_name in ("event_sample", "v5_sample"):
        nested = value.get(field_name)
        if isinstance(nested, Mapping):
            return nested
    return value


def _sample_episode_ids(sample: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    provenance = sample.get("provenance")
    if isinstance(provenance, Mapping):
        for name in (
            "episode_key",
            "episode_id",
            "canonical_episode_id",
            "uid",
        ):
            value = provenance.get(name)
            if isinstance(value, str) and value.strip():
                result.add(value.strip().strip("/"))
    return result


def _sample_videos(sample: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    for field_name in ("images", "image"):
        images = sample.get(field_name)
        if not isinstance(images, list):
            continue
        for item in images:
            if isinstance(item, str) and item:
                result.add(item)
            elif isinstance(item, Mapping):
                for name in ("video", "path", "mp4_path"):
                    value = item.get(name)
                    if isinstance(value, str) and value:
                        result.add(value)
    provenance = sample.get("provenance")
    if isinstance(provenance, Mapping):
        for name in (
            "episode_path",
            "logical_episode_path",
            "resolved_episode_path",
            "existing_episode_path",
        ):
            value = provenance.get(name)
            if isinstance(value, str) and value:
                # Episode directories participate through the same normalized
                # parent check as video paths.
                result.add(value + "/__evaluation_holdout_episode_sentinel__")
        raw_paths = provenance.get("raw_video_paths")
        if isinstance(raw_paths, Mapping):
            result.update(
                value for value in raw_paths.values()
                if isinstance(value, str) and value
            )
    return result


@dataclass(slots=True)
class EvaluationHoldout:
    manifest_path: Path
    manifest_sha256: str
    expected_sha256: str | None
    row_count: int
    episode_ids: frozenset[str]
    episode_paths: frozenset[str]
    video_paths: frozenset[str]
    inode_identities: frozenset[tuple[int, int]]
    _inode_cache: dict[str, tuple[int, int] | None] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        manifest_path: Path | str = DEFAULT_EVALUATION_MANIFEST,
        *,
        expected_sha256: str | None = DEFAULT_EVALUATION_SHA256,
    ) -> "EvaluationHoldout":
        path = Path(manifest_path).resolve(strict=True)
        observed = file_sha256(path)
        if expected_sha256 is not None and observed != expected_sha256:
            raise ValueError(
                "Evaluation holdout manifest SHA-256 differs from the pinned holdout: "
                f"{observed} != {expected_sha256}"
            )
        episode_ids: set[str] = set()
        episode_paths: set[str] = set()
        video_paths: set[str] = set()
        inodes: set[tuple[int, int]] = set()
        row_count = 0
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"Evaluation holdout line {line_number} is not an object"
                    )
                uid = value.get("uid")
                episode_key = value.get("episode_key")
                if not isinstance(uid, str) or not uid.strip():
                    raise ValueError(f"Evaluation holdout line {line_number} has no uid")
                uid = uid.strip().strip("/")
                if uid in episode_ids:
                    raise ValueError(f"duplicate Evaluation holdout uid: {uid}")
                episode_ids.add(uid)
                if isinstance(episode_key, str) and episode_key.strip():
                    episode_ids.add(episode_key.strip().strip("/"))
                for raw in _path_values(value):
                    normalized = normalize_path(raw)
                    if normalized:
                        episode_paths.add(normalized)
                    inode = _inode_identity(raw)
                    if inode is not None:
                        inodes.add(inode)
                for raw in _video_values(value):
                    normalized = normalize_path(raw)
                    if normalized:
                        video_paths.add(normalized)
                        episode_paths.add(str(PurePosixPath(normalized).parent))
                    inode = _inode_identity(raw)
                    if inode is not None:
                        inodes.add(inode)
                row_count += 1
        if row_count <= 0:
            raise ValueError("Evaluation holdout manifest is empty")
        return cls(
            manifest_path=path,
            manifest_sha256=observed,
            expected_sha256=expected_sha256,
            row_count=row_count,
            episode_ids=frozenset(episode_ids),
            episode_paths=frozenset(episode_paths),
            video_paths=frozenset(video_paths),
            inode_identities=frozenset(inodes),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "expected_sha256": self.expected_sha256,
            "row_count": self.row_count,
            "episode_identity_count": len(self.episode_ids),
            "episode_path_count": len(self.episode_paths),
            "video_path_count": len(self.video_paths),
            "existing_inode_count": len(self.inode_identities),
            "match_modes": [
                "episode_identity",
                "normalized_episode_path",
                "normalized_video_path",
                "device_inode",
            ],
        }

    def match_sample(self, value: Mapping[str, Any]) -> list[dict[str, str]]:
        sample = _sample_payload(value)
        matches: list[dict[str, str]] = []
        for episode_id in sorted(_sample_episode_ids(sample)):
            if episode_id in self.episode_ids:
                matches.append({"mode": "episode_identity", "value": episode_id})
        for raw_video in sorted(_sample_videos(sample)):
            normalized = normalize_path(raw_video)
            parent = str(PurePosixPath(normalized).parent) if normalized else ""
            if normalized in self.video_paths:
                matches.append({"mode": "normalized_video_path", "value": normalized})
            if parent in self.episode_paths:
                matches.append({"mode": "normalized_episode_path", "value": parent})
            inode = self._inode_cache.get(raw_video)
            if raw_video not in self._inode_cache:
                inode = _inode_identity(raw_video)
                self._inode_cache[raw_video] = inode
            if inode is not None and inode in self.inode_identities:
                matches.append({
                    "mode": "device_inode",
                    "value": f"{inode[0]}:{inode[1]}",
                })
        # A row may match both a logical path and inode.  Keep each evidence
        # kind once so reports stay bounded and deterministic.
        unique: dict[tuple[str, str], dict[str, str]] = {}
        for match in matches:
            unique[(match["mode"], match["value"])] = match
        return [unique[key] for key in sorted(unique)]


@dataclass(slots=True)
class HoldoutFilter:
    holdout: EvaluationHoldout
    checked_samples: int = 0
    excluded_samples: int = 0
    excluded_base_sample_ids: set[str] = field(default_factory=set)
    match_mode_counts: Counter[str] = field(default_factory=Counter)
    examples: list[dict[str, Any]] = field(default_factory=list)
    max_examples: int = 100

    def keep(self, value: Mapping[str, Any]) -> bool:
        self.checked_samples += 1
        sample = _sample_payload(value)
        matches = self.holdout.match_sample(sample)
        if not matches:
            return True
        self.excluded_samples += 1
        base = str(sample.get("base_sample_id") or sample.get("sample_id") or "")
        if base:
            self.excluded_base_sample_ids.add(base)
        self.match_mode_counts.update(match["mode"] for match in matches)
        if len(self.examples) < self.max_examples:
            provenance = sample.get("provenance")
            self.examples.append({
                "sample_id": str(sample.get("sample_id") or ""),
                "base_sample_id": base,
                "source": str(sample.get("source") or ""),
                "episode_key": str(
                    provenance.get("episode_key")
                    if isinstance(provenance, Mapping)
                    else ""
                ),
                "matches": matches,
            })
        return False

    def filter(self, samples: Iterable[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
        for sample in samples:
            if self.keep(sample):
                yield sample

    def report(self) -> dict[str, Any]:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "passed": self.excluded_samples == 0,
            "holdout": self.holdout.metadata(),
            "checked_samples": self.checked_samples,
            "excluded_samples": self.excluded_samples,
            "excluded_base_samples": len(self.excluded_base_sample_ids),
            "match_mode_counts": dict(sorted(self.match_mode_counts.items())),
            "examples": self.examples,
        }


def audit_jsonl_files(
    paths: Sequence[Path],
    holdout: EvaluationHoldout,
    *,
    fail_on_match: bool = False,
) -> dict[str, Any]:
    checker = HoldoutFilter(holdout)
    file_count = 0
    for path in paths:
        file_count += 1
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"{path}:{line_number} is not an object")
                checker.keep(value)
    report = checker.report()
    report["file_count"] = file_count
    report["paths"] = [str(path) for path in paths]
    if fail_on_match and checker.excluded_samples:
        raise HoldoutViolationError(
            f"Evaluation holdout leaked into artifact: {checker.excluded_samples} rows"
        )
    return report


def audit_artifact(
    root: Path | str,
    holdout: EvaluationHoldout,
    *,
    fail_on_match: bool = False,
) -> dict[str, Any]:
    resolved = Path(root).resolve(strict=True)
    paths = sorted(resolved.rglob("data.jsonl"))
    if not paths:
        raise ValueError(f"artifact contains no data.jsonl files: {resolved}")
    report = audit_jsonl_files(paths, holdout, fail_on_match=fail_on_match)
    report["artifact_root"] = str(resolved)
    report.pop("paths", None)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_EVALUATION_MANIFEST)
    parser.add_argument("--expected-sha256", default=DEFAULT_EVALUATION_SHA256)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-match", action="store_true")
    args = parser.parse_args(argv)
    holdout = EvaluationHoldout.load(
        args.manifest,
        expected_sha256=args.expected_sha256,
    )
    report = audit_artifact(
        args.artifact_root,
        holdout,
        # The CLI always persists the evidence report before returning a
        # failure status.  Library callers can still request the exception
        # based fail-closed behavior directly from ``audit_artifact``.
        fail_on_match=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.fail_on_match and report["excluded_samples"]:
        return 3
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "EvaluationHoldout",
    "DEFAULT_EVALUATION_MANIFEST",
    "DEFAULT_EVALUATION_SHA256",
    "HoldoutFilter",
    "HoldoutViolationError",
    "SCHEMA_VERSION",
    "audit_artifact",
    "audit_jsonl_files",
    "file_sha256",
    "normalize_path",
]
