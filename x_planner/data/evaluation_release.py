#!/usr/bin/env python3
"""Audit and materialize a portable X-Planner evaluation release.

The private selection manifest contains deployment-specific paths.  This module
resolves those paths locally, verifies that every declared video and annotation
exists, and can then copy only the release payload into a path-portable bundle.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

from x_planner.data.analysis_subset import (
    file_sha256,
    iter_jsonl,
    parse_path_maps,
    remap_path,
    safe_component,
)


ITEM_SCHEMA_VERSION = "xplanner_eval_item_v1"
RELEASE_SCHEMA_VERSION = "xplanner_eval_release_v1"
AUDIT_SCHEMA_VERSION = "xplanner_eval_audit_v1"

LABEL_FIELDS = (
    "label_key",
    "current_segment_index",
    "target_subtask_1",
    "target_subtask_2",
)
ANALYSIS_FIELDS = (
    "action_comp",
    "action_evidence",
    "action_inference_status",
    "actual_duration_bucket",
    "camera_count",
    "camera_duration_max_s",
    "camera_duration_min_s",
    "camera_duration_spread_s",
    "complexity_bucket",
    "expertise_tag",
    "inferred_atomic_actions",
    "label_quality",
    "object_tag",
    "scene_tag",
    "subtask_comp",
    "subtask_count",
    "subtask_count_bucket",
    "task_class",
    "time_comp",
)


def portable_id(uid: str) -> str:
    """Return a short collision-resistant component for an episode UID."""

    slug = safe_component(uid)
    digest = hashlib.sha256(uid.encode("utf-8")).hexdigest()[:12]
    return f"{slug[:96].rstrip('-.')}-{digest}"


def _load_json_object(
    path: Path,
    cache: dict[Path, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    if path not in cache:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError(f"expected JSON object: {path}")
        cache[path] = value
    return cache[path]


def _keyed_row(
    path: Path,
    key: str,
    cache: dict[Path, Mapping[str, Any]],
) -> dict[str, Any] | None:
    value = _load_json_object(path, cache)
    if value is None:
        return None
    row = value.get(key)
    return dict(row) if isinstance(row, Mapping) else None


def _episode_path(
    row: Mapping[str, Any],
    mappings: tuple[tuple[str, str], ...],
) -> Path | None:
    for field in ("resolved_episode_path", "existing_episode_path", "logical_episode_path"):
        raw = str(row.get(field) or "")
        if raw:
            candidate = remap_path(raw, mappings)
            if candidate.is_dir():
                return candidate
    return None


def inspect_row(
    row: Mapping[str, Any],
    mappings: tuple[tuple[str, str], ...],
    cache: dict[Path, Mapping[str, Any]],
) -> dict[str, Any]:
    uid = str(row.get("uid") or "").strip()
    if not uid:
        raise ValueError("evaluation row has no uid")
    episode_id = portable_id(uid)
    cameras = row.get("camera_videos")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError(f"{uid} has no camera_videos")

    resolved_media: list[dict[str, Any]] = []
    used_views: set[str] = set()
    for index, camera in enumerate(cameras):
        if not isinstance(camera, Mapping) or not camera.get("mp4_path"):
            raise ValueError(f"{uid} camera {index} has no mp4_path")
        view = safe_component(str(
            camera.get("logical_view") or camera.get("raw_camera") or index
        ))
        if view in used_views:
            view = f"{view}-{index}"
        used_views.add(view)
        source = remap_path(str(camera["mp4_path"]), mappings)
        resolved_media.append({
            "view": view,
            "source": source,
            "exists": source.is_file(),
            "duration_s": camera.get("duration_s"),
            "relative": Path("media") / episode_id / f"{view}{source.suffix.lower() or '.mp4'}",
        })

    label_raw = str(row.get("instruction_path") or "")
    label_path = remap_path(label_raw, mappings) if label_raw else None
    episode_key = str(row.get("episode_key") or uid.rsplit("/", 1)[-1])
    annotation = (
        _keyed_row(label_path, episode_key, cache)
        if label_path is not None
        else None
    )

    episode_path = _episode_path(row, mappings)
    instruction_path = episode_path / "instruction.json" if episode_path else None
    source_instruction = (
        _keyed_row(instruction_path, episode_key, cache)
        if instruction_path is not None
        else None
    )
    missing_views = [media["view"] for media in resolved_media if not media["exists"]]
    ready = not missing_views and annotation is not None
    return {
        "uid": uid,
        "id": episode_id,
        "source_dataset": str(row.get("dataset") or "unknown"),
        "source_group": str(row.get("source_group") or "unknown"),
        "row": dict(row),
        "media": resolved_media,
        "annotation": annotation,
        "source_instruction": source_instruction,
        "annotation_available": annotation is not None,
        "source_instruction_available": source_instruction is not None,
        "episode_directory_available": episode_path is not None,
        "missing_views": missing_views,
        "ready": ready,
    }


def audit_manifest(
    manifest: Path,
    mappings: tuple[tuple[str, str], ...] = (),
    *,
    limit: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cache: dict[Path, Mapping[str, Any]] = {}
    inspected: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    seen_ids: set[str] = set()
    for source in iter_jsonl(manifest):
        if limit is not None and len(inspected) >= limit:
            break
        item = inspect_row(source, mappings, cache)
        if item["uid"] in seen_uids:
            raise ValueError(f"duplicate evaluation uid: {item['uid']}")
        if item["id"] in seen_ids:
            raise ValueError(f"duplicate portable id: {item['id']}")
        seen_uids.add(item["uid"])
        seen_ids.add(item["id"])
        inspected.append(item)

    declared_views = sum(len(item["media"]) for item in inspected)
    available_views = sum(
        int(media["exists"])
        for item in inspected
        for media in item["media"]
    )
    incomplete_by_dataset = Counter(
        item["source_dataset"] for item in inspected if not item["ready"]
    )
    dataset_counts = Counter(item["source_dataset"] for item in inspected)
    dataset_groups: dict[str, set[str]] = {}
    for item in inspected:
        dataset_groups.setdefault(item["source_dataset"], set()).add(item["source_group"])
    view_count_distribution = Counter(len(item["media"]) for item in inspected)
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "source_manifest_sha256": file_sha256(manifest),
        "episode_count": len(inspected),
        "unique_uid_count": len(seen_uids),
        "ready_episode_count": sum(item["ready"] for item in inspected),
        "incomplete_episode_count": sum(not item["ready"] for item in inspected),
        "declared_video_count": declared_views,
        "available_video_count": available_views,
        "missing_video_count": declared_views - available_views,
        "annotation_available_count": sum(
            item["annotation_available"] for item in inspected
        ),
        "source_instruction_available_count": sum(
            item["source_instruction_available"] for item in inspected
        ),
        "episode_directory_available_count": sum(
            item["episode_directory_available"] for item in inspected
        ),
        "declared_view_count_distribution": {
            str(key): value for key, value in sorted(view_count_distribution.items())
        },
        "incomplete_by_dataset": dict(incomplete_by_dataset.most_common()),
        "source_ledger": [
            {
                "source_dataset": dataset,
                "episode_count": dataset_counts[dataset],
                "source_groups": sorted(dataset_groups[dataset]),
                "approval": "pending",
            }
            for dataset in sorted(dataset_counts)
        ],
        "release_ready": bool(inspected) and all(item["ready"] for item in inspected),
        "incomplete_episodes": [
            {
                "id": item["id"],
                "source_uid": item["uid"],
                "source_dataset": item["source_dataset"],
                "missing_views": item["missing_views"],
                "annotation_available": item["annotation_available"],
                "episode_directory_available": item["episode_directory_available"],
            }
            for item in inspected
            if not item["ready"]
        ],
    }
    return report, inspected


def portable_item(
    inspected: Mapping[str, Any],
    media: list[dict[str, Any]],
) -> dict[str, Any]:
    row = inspected["row"]
    source_instruction = inspected.get("source_instruction") or {}
    instruction = {
        key: value
        for key, value in source_instruction.items()
        if key in {"instruction", "instruction_zh", "detailed_instruction", "detailed_instruction_zh", "task"}
    }
    if "instruction" not in instruction:
        instruction["instruction"] = str(row.get("instruction") or "").strip()
    if "task" not in instruction:
        instruction["task"] = str(row.get("task") or "").strip()
    labels = {
        key: row[key] for key in LABEL_FIELDS if key in row
    }
    labels["temporal_annotations"] = inspected["annotation"]
    result = {
        "schema_version": ITEM_SCHEMA_VERSION,
        "id": inspected["id"],
        "source_uid": inspected["uid"],
        "split": "evaluation",
        "source_dataset": inspected["source_dataset"],
        "source_group": inspected["source_group"],
        "instruction": instruction,
        "task": str(row.get("task") or "").strip(),
        "anchor_frame": int(row.get("anchor_frame") or 0),
        "media": media,
        "labels": labels,
        "attributes": {
            key: row[key] for key in ANALYSIS_FIELDS if key in row
        },
    }
    if not result["instruction"]["instruction"] or not result["task"]:
        raise ValueError(f"{inspected['uid']} has no portable instruction or task")
    return result


def _copy_and_hash(source: Path, destination: Path) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as source_handle, destination.open("xb") as output_handle:
        for chunk in iter(lambda: source_handle.read(8 * 1024 * 1024), b""):
            output_handle.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _release_readme(episode_count: int, video_count: int) -> str:
    return f"""# X-Planner Evaluation Set

This immutable evaluation artifact contains {episode_count} episodes and {video_count} declared
camera videos. `items.jsonl` contains portable task, annotation, and media records. All media paths
are relative to this directory and every copied file is covered by `checksums.sha256`.

The artifact is evaluation-only. It is not an event-grounded training-data release. Source-specific
licenses and attribution in the accompanying dataset card continue to apply.
"""


def export_release(
    manifest: Path,
    output: Path,
    report: Mapping[str, Any],
    inspected: list[dict[str, Any]],
) -> dict[str, Any]:
    if not report.get("release_ready"):
        raise ValueError(
            "evaluation release is incomplete; repair every missing video/annotation before export"
        )
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    items_path = temporary / "items.jsonl"
    video_count = 0
    media_bytes = 0
    try:
        with items_path.open("x", encoding="utf-8") as items_handle:
            for item in inspected:
                media_rows: list[dict[str, Any]] = []
                for source_media in item["media"]:
                    destination = temporary / source_media["relative"]
                    digest, size = _copy_and_hash(source_media["source"], destination)
                    media_row = {
                        "view": source_media["view"],
                        "path": source_media["relative"].as_posix(),
                        "sha256": digest,
                        "bytes": size,
                    }
                    if source_media["duration_s"] is not None:
                        media_row["duration_s"] = source_media["duration_s"]
                    media_rows.append(media_row)
                    video_count += 1
                    media_bytes += size
                items_handle.write(json.dumps(
                    portable_item(item, media_rows),
                    ensure_ascii=False,
                    sort_keys=True,
                ) + "\n")

        release = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "split": "evaluation",
            "episode_count": len(inspected),
            "video_count": video_count,
            "media_bytes": media_bytes,
            "source_manifest_sha256": file_sha256(manifest),
            "items_sha256": file_sha256(items_path),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(release, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "README.md").write_text(
            _release_readme(len(inspected), video_count),
            encoding="utf-8",
        )
        checksum_paths = sorted(
            path for path in temporary.rglob("*")
            if path.is_file() and path.name != "checksums.sha256"
        )
        with (temporary / "checksums.sha256").open("x", encoding="utf-8") as handle:
            for path in checksum_paths:
                handle.write(
                    f"{file_sha256(path)}  {path.relative_to(temporary).as_posix()}\n"
                )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return release


def write_audit_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--release-output", type=Path)
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help="repeatable mapping from private manifest paths to local mounts",
    )
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = args.manifest.resolve(strict=True)
    mappings = parse_path_maps(args.path_map)
    report, inspected = audit_manifest(manifest, mappings, limit=args.limit)
    write_audit_report(args.audit_report.resolve(), report)
    result: dict[str, Any] = {
        "audit_report": str(args.audit_report.resolve()),
        **{key: report[key] for key in (
            "episode_count",
            "ready_episode_count",
            "incomplete_episode_count",
            "declared_video_count",
            "available_video_count",
            "missing_video_count",
            "annotation_available_count",
            "release_ready",
        )},
    }
    if args.release_output is not None:
        release = export_release(
            manifest,
            args.release_output.resolve(),
            report,
            inspected,
        )
        result["release_output"] = str(args.release_output.resolve())
        result["release"] = release
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
