#!/usr/bin/env python3
"""Build an atomic V10 datalist for captioned Open Action topics not yet listed."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ROOT = Path(
    "/mnt/cpfs/zbl-cpfs-new/open_data/Open_Action_datasets_as_mp4"
)
DEFAULT_CAPTION_ROOT = Path(
    "/mnt/cpfs/zbl-cpfs-new/open_data/video_caption/"
    "video_caption_for_x2_check_v2v3umi"
)
DEFAULT_EXISTING = (
    Path(
        "/mnt/cpfs/zbl-cpfs-new/USERS/luhao/wall-x_2604/workspace/example/"
        "vga2_textlm/config/data_list/public_datalist_v10json.yml"
    ),
    Path(
        "/mnt/cpfs/zbl-cpfs-new/USERS/luhao/wall-x_2604/workspace/example/"
        "vga2_textlm/config/data_list/v2v3umi_datalist_v10json.yml"
    ),
)
DEFAULT_OUTPUT = Path(__file__).with_name("configs") / "open_action_delta_datalist.yml"


VIEW_CANDIDATES = {
    "face_view": (
        "faceImg",
        "head_rgb",
        "camera_head",
        "camera_top",
        "camera_front",
        "third_view_rgb",
        "chest_rgb",
    ),
    "left_wrist_view": (
        "leftImg",
        "wrist_left_rgb",
        "camera_left_wrist",
        "camera_wrist_left",
        "camera_left",
    ),
    "right_wrist_view": (
        "rightImg",
        "wrist_right_rgb",
        "camera_right_wrist",
        "camera_wrist_right",
        "camera_right",
    ),
    "side_view": (
        "sideImg",
        "camera_side",
        "head_right_rgb",
        "back_right_fisheye_rgb",
        "high_center_fisheye_rgb",
    ),
}


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML object: {path}")
    return value


def _listed_topics(paths: list[Path]) -> set[str]:
    topics: set[str] = set()
    for path in paths:
        for item in _load_yaml(path).get("dataset_path") or ():
            if isinstance(item, Mapping) and item.get("path"):
                topics.add(os.path.realpath(str(item["path"])))
    return topics


def _annotation_keys(path: Path) -> list[str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    if not isinstance(value, Mapping):
        return []
    return sorted(str(key) for key, item in value.items() if isinstance(item, Mapping))


def _episode_dir(topic: Path, name: str) -> Path | None:
    for candidate in (topic / name, topic / f"episode_{name}"):
        if candidate.is_dir():
            return candidate
    return None


def _camera_mapping(topic: Path, names: list[str]) -> dict[str, str]:
    stems: set[str] = set()
    for name in names[:32]:
        episode = _episode_dir(topic, name)
        if episode is None:
            continue
        for video in episode.glob("*.mp4"):
            if not video.name.startswith("."):
                stems.add(video.stem)
    mapping: dict[str, str] = {}
    for view, candidates in VIEW_CANDIDATES.items():
        selected = next((stem for stem in candidates if stem in stems), None)
        if selected is not None:
            mapping[selected] = view
    return mapping


def _task_name(root: Path, topic: Path) -> str:
    relative = topic.relative_to(root).as_posix().lower()
    return re.sub(r"[^a-z0-9]+", "_", relative).strip("_")


def build(
    root: Path,
    caption_root: Path,
    existing: list[Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    caption_root = caption_root.resolve()
    listed = _listed_topics(existing)
    listed_datasets: set[str] = set()
    for raw_topic in listed:
        try:
            relative = Path(raw_topic).relative_to(root)
        except ValueError:
            continue
        if relative.parts:
            listed_datasets.add(relative.parts[0])
    rows: list[dict[str, Any]] = []
    annotation_files = 0
    empty_annotations = 0
    delta_datasets = sorted(
        path for path in root.iterdir()
        if path.is_dir() and path.name not in listed_datasets
    )
    missing_caption_roots: list[str] = []
    for dataset in delta_datasets:
        caption_dataset = caption_root / dataset.name
        if not caption_dataset.is_dir():
            missing_caption_roots.append(dataset.name)
            continue
        for instruction in sorted(caption_dataset.rglob("instruction.json")):
            annotation_files += 1
            relative_topic = instruction.parent.relative_to(caption_dataset)
            topic = (dataset / relative_topic).resolve()
            if not topic.is_dir():
                continue
            if os.path.realpath(topic) in listed:
                continue
            names = _annotation_keys(instruction)
            if not names:
                empty_annotations += 1
                continue
            rows.append(
                {
                    "task_name": _task_name(root, topic),
                    "cam_mapping": _camera_mapping(topic, names),
                    "path": str(topic),
                    # Keep the external English caption first, but retain the
                    # topic-local annotation as a per-episode fallback.  Some
                    # external files contain an empty object for an otherwise
                    # valid local episode.
                    "instruction_templates": [
                        str(instruction.resolve()),
                        "{topic_path}/instruction.json",
                    ],
                    "hq_caption": True,
                    "dataset_type": "general_video",
                }
            )
    document = {"dataset_path": rows}
    summary = {
        "root": str(root),
        "caption_root": str(caption_root),
        "existing_datalists": [str(path.resolve()) for path in existing],
        "existing_topics": len(listed),
        "existing_dataset_roots": len(listed_datasets),
        "delta_dataset_roots": [path.name for path in delta_datasets],
        "delta_datasets_without_caption_root": missing_caption_roots,
        "instruction_files_seen": annotation_files,
        "empty_or_invalid_annotations_skipped": empty_annotations,
        "delta_topics": len(rows),
        "delta_topics_with_mapped_camera": sum(bool(row["cam_mapping"]) for row in rows),
        "delta_topics_without_mapped_camera": sum(not row["cam_mapping"] for row in rows),
    }
    return document, summary


def _atomic_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(value, handle, allow_unicode=True, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--caption-root", type=Path, default=DEFAULT_CAPTION_ROOT)
    parser.add_argument("--existing", type=Path, action="append")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    existing = args.existing or list(DEFAULT_EXISTING)
    document, summary = build(args.root, args.caption_root, existing)
    _atomic_yaml(args.output.resolve(), document)
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.summary.with_name(f".{args.summary.name}.tmp")
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, args.summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
