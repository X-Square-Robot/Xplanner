#!/usr/bin/env python3
"""Materialize the portable image/JSON artifact used for data analysis.

The source manifest may contain deployment-specific absolute paths. Exported
records retain only stable metadata and paths relative to the artifact root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "xplanner_analysis_item_v1"
LABEL_FIELDS = (
    "target_subtask_1",
    "target_subtask_2",
    "action_comp",
    "subtask_comp",
    "time_comp",
    "body",
)
ANALYSIS_FIELDS = (
    "source_group",
    "scene_tag",
    "object_tag",
    "expertise_tag",
    "complexity_bucket",
    "actual_duration_bucket",
    "episode_duration_s",
    "subtask_count",
    "subtask_count_bucket",
    "inferred_atomic_actions",
    "label_quality",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_component(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    if not result:
        raise ValueError(f"value cannot form a safe path component: {value!r}")
    return result


def parse_path_maps(values: Iterable[str]) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for value in values:
        source, separator, target = value.partition("=")
        if not separator or not source or not target:
            raise ValueError(f"invalid --path-map {value!r}; expected SOURCE=TARGET")
        result.append((source.rstrip("/"), target.rstrip("/")))
    return tuple(sorted(result, key=lambda item: len(item[0]), reverse=True))


def remap_path(value: str, mappings: tuple[tuple[str, str], ...]) -> Path:
    for source, target in mappings:
        if value == source or value.startswith(source + "/"):
            return Path(target + value[len(source):])
    return Path(value)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            yield value


def extract_frame(
    ffmpeg: str,
    video: Path,
    frame_index: int,
    output: Path,
    timeout: int,
) -> None:
    if not video.is_file():
        raise FileNotFoundError(video)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(video),
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-y",
        str(output),
    ]
    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    if result.returncode or not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(
            f"failed to extract frame {frame_index} from {video}: "
            f"{result.stderr.strip()[-500:]}"
        )


def portable_row(
    source: dict[str, Any],
    *,
    sample_id: str,
    media: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": sample_id,
        "split": "analysis",
        "instruction": str(source.get("instruction") or "").strip(),
        "task": str(source.get("task") or "").strip(),
        "task_classes": sorted(set(source.get("task_class") or [])),
        "source_dataset": str(source.get("dataset") or "unknown"),
        "anchor_frame": int(source.get("anchor_frame") or 0),
        "media": media,
        "analysis": {
            name: source[name]
            for name in ANALYSIS_FIELDS
            if name in source
        },
        "labels": {
            name: source[name]
            for name in LABEL_FIELDS
            if name in source
        },
    }
    if not result["instruction"] or not result["task"]:
        raise ValueError(f"{sample_id} has no instruction or task")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--path-map",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help="repeatable mapping from manifest paths to locally mounted paths",
    )
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    source_manifest = args.manifest.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    mappings = parse_path_maps(args.path_map)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    rows_path = temporary / "items.jsonl"
    rows_path.parent.mkdir(parents=True)
    seen: set[str] = set()
    exported = 0
    media_count = 0
    try:
        with rows_path.open("w", encoding="utf-8") as rows_out:
            for source in iter_jsonl(source_manifest):
                if args.limit is not None and exported >= args.limit:
                    break
                sample_id = safe_component(str(source.get("uid") or ""))
                if sample_id in seen:
                    raise ValueError(f"duplicate portable sample id: {sample_id}")
                seen.add(sample_id)
                anchor_frame = int(source.get("anchor_frame") or 0)
                cameras = source.get("camera_videos")
                if not isinstance(cameras, list) or not cameras:
                    raise ValueError(f"{sample_id} has no camera_videos")
                media: list[dict[str, Any]] = []
                used_views: set[str] = set()
                for camera_index, camera in enumerate(cameras):
                    if not isinstance(camera, dict) or not camera.get("mp4_path"):
                        raise ValueError(f"{sample_id} camera {camera_index} has no mp4_path")
                    view = safe_component(str(
                        camera.get("logical_view") or camera.get("raw_camera") or camera_index
                    ))
                    if view in used_views:
                        view = f"{view}-{camera_index}"
                    used_views.add(view)
                    relative = Path("media") / sample_id / f"{view}.jpg"
                    destination = temporary / relative
                    source_video = remap_path(str(camera["mp4_path"]), mappings)
                    extract_frame(
                        args.ffmpeg,
                        source_video,
                        anchor_frame,
                        destination,
                        args.timeout,
                    )
                    media.append({
                        "view": view,
                        "path": relative.as_posix(),
                        "sha256": file_sha256(destination),
                    })
                    media_count += 1
                row = portable_row(
                    source,
                    sample_id=sample_id,
                    media=media,
                )
                rows_out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                exported += 1
        release_manifest = {
            "schema_version": "xplanner_analysis_release_v1",
            "split": "analysis",
            "labels_included": True,
            "source_manifest_sha256": file_sha256(source_manifest),
            "sample_count": exported,
            "media_count": media_count,
            "items_sha256": file_sha256(rows_path),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(release_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksum_paths = sorted(
            path for path in temporary.rglob("*")
            if path.is_file() and path.name != "checksums.sha256"
        )
        with (temporary / "checksums.sha256").open("w", encoding="utf-8") as handle:
            for path in checksum_paths:
                relative_path = path.relative_to(temporary).as_posix()
                handle.write(f"{file_sha256(path)}  {relative_path}\n")
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({"output": str(output), **release_manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
