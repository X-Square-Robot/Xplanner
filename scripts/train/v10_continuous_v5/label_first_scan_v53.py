"""Fast label-first V5.3 scan driven by v2v3umi ``instruction.json`` files."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from .holdout_v5 import Benchmark3Holdout, DEFAULT_BENCHMARK3_MANIFEST, DEFAULT_BENCHMARK3_SHA256
from .task_instruction_v53 import resolve_episode_task_instruction


DEFAULT_LABEL_ROOT = Path(
    "/mnt/cpfs/zbl-cpfs-new/open_data/video_caption/"
    "video_caption_for_x2_check_v2v3umi"
)
DEFAULT_ZHENGWEI_ROOT = Path("/mnt/jfs/x2robot-prod/x2robot_data/zhengwei")
DEFAULT_COLLECTION_ROOT = Path("/mnt/jfs/x2robot-prod/x2robot_data/collection")
DEFAULT_OPEN_ACTION_ROOT = Path(
    "/mnt/jfs/x2robot-prod/open-data/Open_Action_datasets_as_mp4"
)
DEFAULT_ROBODOJO_MEDIA_ROOT = Path(
    "/mnt/cpfs/zbl-cpfs-new/share/qudelin/DATA/robotwin30_x2/arx_x5"
)
SCANNER_VERSION = "v5_3_label_first_v4_task_instruction_fix"
_CJK = re.compile(r"[\u3400-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")
_INTERVAL = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_lines(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    # mkstemp intentionally creates 0600 files.  Cluster workers run as
    # ``nobody`` while the controller audits artifacts as the workspace user,
    # so publish read-only evidence for all users before the atomic rename.
    os.fchmod(fd, 0o644)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return count


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_lines(path, (value,))


def discover_instruction_files(
    label_root: Path, limit: int | None = None
) -> list[Path]:
    """Walk only the compact label tree; media roots are never traversed."""

    root = label_root.resolve(strict=True)
    pending = [root]
    result: list[Path] = []
    while pending:
        current = pending.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda entry: entry.name)
        except OSError:
            continue
        directories: list[Path] = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir(follow_symlinks=False):
                directories.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False) and entry.name == "instruction.json":
                result.append(Path(entry.path))
                if limit is not None and len(result) >= limit:
                    return result
        # Reverse push makes the lexicographically smallest directory the next
        # one popped.  A smoke prefix therefore does not enumerate the entire
        # compact label tree before stopping.
        pending.extend(reversed(directories))
    return result


def build_collection_device_index(root: Path) -> dict[str, str | tuple[str, ...]]:
    """Map XRRD IDs to every shallow candidate root.

    Collection reuses a physical device ID in multiple batches.  Retaining all
    candidates is required because the label-relative path, not an arbitrary
    first match, determines the episode directory.
    """

    candidates: dict[str, list[str]] = {}
    if not root.is_dir():
        return {}
    for batch in sorted(root.iterdir()):
        if not batch.is_dir():
            continue
        for rig in sorted(batch.iterdir()):
            if not rig.is_dir():
                continue
            for device in sorted(rig.iterdir()):
                if not device.is_dir() or not device.name.startswith("XRRD"):
                    continue
                candidates.setdefault(device.name, []).append(str(device.resolve()))
    result: dict[str, str | tuple[str, ...]] = {}
    for device, values in sorted(candidates.items()):
        unique = tuple(sorted(set(values)))
        result[device] = unique[0] if len(unique) == 1 else unique
    return result


def _source_group(relative: Path) -> str:
    first = relative.parts[0]
    if first == "robotwin30_x2":
        return "robodojo"
    if first.isdigit():
        return "zhengwei"
    if first.startswith("XRRD"):
        return "collection"
    return "open_action"


def episode_candidates(
    *,
    instruction_relative: Path,
    episode_key: str,
    zhengwei_root: Path,
    collection_root: Path,
    collection_devices: Mapping[str, str | Sequence[str]],
    open_action_root: Path,
    robodojo_media_root: Path,
) -> tuple[str, list[Path]]:
    group = _source_group(instruction_relative)
    parent = instruction_relative.parent
    if group == "zhengwei":
        return group, [zhengwei_root / parent / episode_key]
    if group == "collection":
        device = parent.parts[0]
        raw_prefixes = collection_devices.get(device)
        if raw_prefixes is None:
            return group, []
        prefixes = (
            [raw_prefixes]
            if isinstance(raw_prefixes, str)
            else list(raw_prefixes)
        )
        return group, [
            Path(prefix).joinpath(*parent.parts[1:], episode_key)
            for prefix in prefixes
        ]
    if group == "robodojo":
        relative = parent.parts[2:] if parent.parts[:2] == ("robotwin30_x2", "arx_x5") else parent.parts
        return group, [robodojo_media_root.joinpath(*relative, episode_key)]
    dataset = parent.parts[0]
    remainder = parent.parts[1:]
    candidates = [
        open_action_root.joinpath(dataset, *remainder, episode_key),
        open_action_root.joinpath(dataset, episode_key),
        open_action_root.joinpath(*parent.parts, episode_key),
    ]
    aliases = {
        "AgiBotWorld-Alpha-v2": "AgiBotWorld-Alpha",
        "AgiBotWorld-Beta-v2": "AgiBotWorld-Beta",
        "RH20T_transfer_v2": "RH20T_transfer",
    }
    if dataset in aliases:
        candidates.extend([
            open_action_root.joinpath(aliases[dataset], *remainder, episode_key),
            open_action_root.joinpath(aliases[dataset], episode_key),
        ])
    unique: dict[str, Path] = {}
    for candidate in candidates:
        unique[str(candidate)] = candidate
    return group, list(unique.values())


def _english_intervals(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    result: list[dict[str, Any]] = []
    for raw_interval, raw_caption in value.items():
        match = _INTERVAL.match(str(raw_interval))
        caption = str(raw_caption or "").strip()
        if not match or not caption or not _LATIN.search(caption) or _CJK.search(caption):
            continue
        start, end = map(int, match.groups())
        if end <= start:
            continue
        result.append({"start_frame": start, "end_frame": end, "caption": caption})
    return sorted(result, key=lambda item: (item["start_frame"], item["end_frame"], item["caption"]))


def _joint_alignable(actions: Sequence[Mapping[str, Any]], segments: Sequence[Mapping[str, Any]]) -> bool:
    if not actions or not segments:
        return False
    def overlaps(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        return min(int(left["end_frame"]), int(right["end_frame"])) > max(
            int(left["start_frame"]), int(right["start_frame"])
        )
    return all(any(overlaps(item, other) for other in segments) for item in actions) and all(
        any(overlaps(item, other) for other in actions) for item in segments
    )


def _camera_videos(directory: Path) -> list[str]:
    if not directory.is_dir():
        return []
    preferred = [directory / name for name in ("faceImg.mp4", "leftImg.mp4", "rightImg.mp4")]
    found = [str(path.resolve()) for path in preferred if path.is_file()]
    if found:
        return found
    try:
        return [str(path.resolve()) for path in sorted(directory.glob("*.mp4"))[:3]]
    except OSError:
        return []


def scan_instruction(
    path: Path,
    *,
    label_root: Path,
    zhengwei_root: Path,
    collection_root: Path,
    collection_devices: Mapping[str, str | Sequence[str]],
    open_action_root: Path,
    robodojo_media_root: Path,
    holdout: Benchmark3Holdout,
) -> list[dict[str, Any]]:
    relative = path.resolve().relative_to(label_root.resolve())
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain an episode map")
    stat = path.stat()
    rows: list[dict[str, Any]] = []
    instruction_cache: dict[str, Mapping[str, Any] | None] = {}
    for episode_key, annotation in sorted(value.items(), key=lambda item: str(item[0])):
        if not isinstance(annotation, Mapping):
            continue
        actions = _english_intervals(annotation.get("action_caption"))
        segments = _english_intervals(annotation.get("human_segment_caption"))
        profiles: list[str] = []
        excluded_profiles: dict[str, str] = {}
        if len(actions) > 3:
            profiles.append("action_only")
        else:
            excluded_profiles["action_only"] = "subtask_count_le_3"
        if len(segments) > 3:
            profiles.append("segment_only")
        else:
            excluded_profiles["segment_only"] = "subtask_count_le_3"
        if len(actions) > 3 and len(segments) > 3 and _joint_alignable(actions, segments):
            profiles.append("action_segment_joint")
        else:
            excluded_profiles["action_segment_joint"] = (
                "subtask_count_le_3" if len(actions) <= 3 or len(segments) <= 3 else "unaligned_intervals"
            )
        group, candidates = episode_candidates(
            instruction_relative=relative,
            episode_key=str(episode_key),
            zhengwei_root=zhengwei_root,
            collection_root=collection_root,
            collection_devices=collection_devices,
            open_action_root=open_action_root,
            robodojo_media_root=robodojo_media_root,
        )
        resolved: Path | None = None
        videos: list[str] = []
        for candidate in candidates:
            candidate_videos = _camera_videos(candidate)
            if candidate_videos:
                resolved = candidate.resolve()
                videos = candidate_videos
                break
        instruction = resolve_episode_task_instruction(
            annotation=annotation,
            annotation_path=path,
            episode_key=str(episode_key),
            resolved_episode_path=resolved,
            json_cache=instruction_cache,
        )
        if instruction.status != "resolved":
            for profile in profiles:
                excluded_profiles[profile] = "missing_task_instruction"
            profiles = []
        pseudo_sample = {
            "images": videos,
            "provenance": {
                "episode_key": str(episode_key),
                "episode_path": str(resolved or (candidates[0] if candidates else "")),
                "raw_video_paths": {str(index): video for index, video in enumerate(videos)},
            },
        }
        matches = holdout.match_sample(pseudo_sample)
        if matches:
            profiles = []
        rows.append({
            "schema_version": "v10_action_segment_v5_3_instruction_index_v2",
            "scanner_version": SCANNER_VERSION,
            "instruction_relative": relative.as_posix(),
            "instruction_size": stat.st_size,
            "instruction_mtime_ns": stat.st_mtime_ns,
            "episode_key": str(episode_key),
            "source_group": group,
            "task_hint": relative.parent.name,
            "task_instruction": instruction.text,
            "task_instruction_status": instruction.status,
            "task_instruction_source": instruction.source,
            "task_instruction_source_path": instruction.source_path,
            "task_instruction_source_field": instruction.source_field,
            "task_instruction_checked_paths": list(instruction.checked_paths),
            "task_instruction_rejected_candidates": list(instruction.rejected_candidates),
            "action_count": len(actions),
            "segment_count": len(segments),
            "eligible_profiles": profiles,
            "excluded_profiles": excluded_profiles,
            "actions": actions,
            "segments": segments,
            "path_candidates": [str(candidate) for candidate in candidates],
            "resolved_episode_path": str(resolved) if resolved else None,
            "camera_videos": videos,
            "media_status": "available" if videos else "missing",
            "benchmark3_excluded": bool(matches),
            "benchmark3_matches": matches,
        })
    return rows


def scan(
    *,
    label_root: Path,
    output_root: Path,
    zhengwei_root: Path = DEFAULT_ZHENGWEI_ROOT,
    collection_root: Path = DEFAULT_COLLECTION_ROOT,
    open_action_root: Path = DEFAULT_OPEN_ACTION_ROOT,
    robodojo_media_root: Path = DEFAULT_ROBODOJO_MEDIA_ROOT,
    workers: int = 16,
    max_instructions: int | None = None,
    reuse: bool = False,
    holdout: Benchmark3Holdout,
) -> dict[str, Any]:
    index_path = output_root / "instruction_index.jsonl"
    manifest_path = output_root / "instruction_index_manifest.json"
    if reuse and index_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("complete") is True
            and manifest.get("scanner_version") == SCANNER_VERSION
            and manifest.get("label_root") == str(label_root.resolve())
            and manifest.get("index_sha256") == _sha256(index_path)
        ):
            return {**manifest, "cache_reused": True}
    started = time.monotonic()
    paths = discover_instruction_files(label_root, max_instructions)
    resolved_label_root = label_root.resolve()
    needs_collection = any(
        path.resolve().relative_to(resolved_label_root).parts[0].startswith("XRRD")
        for path in paths
    )
    collection_cache_path = output_root / "collection_device_index.json"
    collection_devices: dict[str, str | tuple[str, ...]] = {}
    collection_cache_reused = False
    if needs_collection and collection_cache_path.is_file():
        cached = json.loads(collection_cache_path.read_text(encoding="utf-8"))
        if cached.get("collection_root") == str(collection_root.resolve()):
            collection_devices = {
                str(device): (
                    tuple(value) if isinstance(value, list) else str(value)
                )
                for device, value in (cached.get("devices") or {}).items()
            }
            collection_cache_reused = True
    if needs_collection and not collection_devices:
        collection_devices = build_collection_device_index(collection_root)
        _atomic_json(collection_cache_path, {
            "schema_version": "v5_3_collection_device_index_v1",
            "collection_root": str(collection_root.resolve()),
            "devices": {
                device: list(value) if isinstance(value, tuple) else value
                for device, value in collection_devices.items()
            },
        })
    kwargs = {
        "label_root": label_root,
        "zhengwei_root": zhengwei_root,
        "collection_root": collection_root,
        "collection_devices": collection_devices,
        "open_action_root": open_action_root,
        "robodojo_media_root": robodojo_media_root,
        "holdout": holdout,
    }
    errors: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="v53-label") as pool:
        futures = [(path, pool.submit(scan_instruction, path, **kwargs)) for path in paths]
        for path, future in futures:
            try:
                rows.extend(future.result())
            except Exception as exc:
                errors.append({"instruction": str(path), "error": type(exc).__name__, "detail": str(exc)})
    rows.sort(key=lambda row: (row["instruction_relative"], row["episode_key"]))
    _atomic_lines(index_path, rows)
    counts = Counter(row["source_group"] for row in rows)
    profiles = Counter(profile for row in rows for profile in row["eligible_profiles"])
    manifest = {
        "schema_version": "v10_action_segment_v5_3_instruction_index_manifest_v2",
        "scanner_version": SCANNER_VERSION,
        "complete": not errors,
        "partial": max_instructions is not None,
        "cache_reused": False,
        "label_root": str(label_root.resolve()),
        "instruction_files": len(paths),
        "episode_rows": len(rows),
        "source_counts": dict(sorted(counts.items())),
        "eligible_profile_counts": dict(sorted(profiles.items())),
        "subtask_le_3_rows": sum(
            not row["eligible_profiles"]
            and row["task_instruction_status"] == "resolved"
            for row in rows
            if not row["benchmark3_excluded"]
        ),
        "missing_media_rows": sum(row["media_status"] != "available" for row in rows),
        "missing_task_instruction_rows": sum(
            row["task_instruction_status"] != "resolved" for row in rows
        ),
        "task_instruction_source_counts": dict(sorted(Counter(
            str(row["task_instruction_source"] or "missing") for row in rows
        ).items())),
        "benchmark3_excluded_rows": sum(row["benchmark3_excluded"] for row in rows),
        "errors": errors,
        "collection_device_count": len(collection_devices),
        "collection_index_required": needs_collection,
        "collection_index_cache_reused": collection_cache_reused,
        "collection_duplicate_device_count": sum(
            isinstance(value, tuple) for value in collection_devices.values()
        ),
        "collection_candidate_root_count": sum(
            len(value) if isinstance(value, tuple) else 1
            for value in collection_devices.values()
        ),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "index_file": "instruction_index.jsonl",
        "index_sha256": _sha256(index_path),
        "benchmark3": holdout.metadata(),
        "roots": {
            "zhengwei": str(zhengwei_root),
            "collection": str(collection_root),
            "open_action": str(open_action_root),
            "robodojo": str(robodojo_media_root),
        },
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--zhengwei-root", type=Path, default=DEFAULT_ZHENGWEI_ROOT)
    parser.add_argument("--collection-root", type=Path, default=DEFAULT_COLLECTION_ROOT)
    parser.add_argument("--open-action-root", type=Path, default=DEFAULT_OPEN_ACTION_ROOT)
    parser.add_argument("--robodojo-media-root", type=Path, default=DEFAULT_ROBODOJO_MEDIA_ROOT)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-instructions", type=int)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--benchmark3", type=Path, default=DEFAULT_BENCHMARK3_MANIFEST)
    parser.add_argument("--benchmark3-sha256", default=DEFAULT_BENCHMARK3_SHA256)
    args = parser.parse_args(argv)
    holdout = Benchmark3Holdout.load(args.benchmark3, expected_sha256=args.benchmark3_sha256)
    report = scan(
        label_root=args.label_root,
        output_root=args.output_root,
        zhengwei_root=args.zhengwei_root,
        collection_root=args.collection_root,
        open_action_root=args.open_action_root,
        robodojo_media_root=args.robodojo_media_root,
        workers=args.workers,
        max_instructions=args.max_instructions,
        reuse=args.reuse,
        holdout=holdout,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_collection_device_index",
    "discover_instruction_files",
    "episode_candidates",
    "scan",
    "scan_instruction",
]
