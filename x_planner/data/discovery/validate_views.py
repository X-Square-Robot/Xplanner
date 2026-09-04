"""Cheap and full V2 view validation with explicit view-name diagnostics."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import av

from ..pipeline.video import VideoValidationError, validate_views as full_validate_views
from .common.errors import ScanError


def _canonical_aliases(view_config: Mapping[str, Any]) -> dict[str, str]:
    views = view_config.get("views") if isinstance(view_config.get("views"), Mapping) else view_config
    aliases = dict((views or {}).get("aliases") or {})
    if not bool((views or {}).get("case_sensitive", False)):
        aliases.update({str(key).lower(): str(value) for key, value in list(aliases.items())})
    return {str(key): str(value) for key, value in aliases.items()}


def validate_view_names(
    episode_dir: str,
    candidates: Mapping[str, str],
    view_config: Mapping[str, Any],
) -> None:
    resolve_view_candidates(episode_dir, candidates, view_config)


def resolve_view_candidates(
    episode_dir: str,
    candidates: Mapping[str, str],
    view_config: Mapping[str, Any],
) -> tuple[dict[str, str], list[str]]:
    """Resolve actual stems to canonical views using configured priority."""
    views = view_config.get("views") if isinstance(view_config.get("views"), Mapping) else view_config
    views = views or {}
    allowed = set(views.get("order") or ())
    required = set(views.get("required") or ())
    aliases = _canonical_aliases(view_config)
    unknown: list[str] = []
    canonical_paths: dict[str, list[Path]] = {}
    for path in sorted(Path(episode_dir).glob("*.mp4")):
        stem = path.stem
        key = stem if bool(views.get("case_sensitive", False)) else stem.lower()
        canonical = aliases.get(key)
        if canonical is None:
            unknown.append(stem)
            continue
        canonical_paths.setdefault(canonical, []).append(path)
    priorities = {str(stem): index for index, stem in enumerate(views.get("stem_priority") or ())}
    selected: dict[str, str] = {}
    warnings: list[str] = []
    for canonical, paths in canonical_paths.items():
        paths.sort(key=lambda path: (priorities.get(path.stem, len(priorities)), path.stem, str(path)))
        if len(paths) > 1 and bool(views.get("duplicate_is_fatal", True)):
            raise ScanError(
                "duplicate_view",
                f"{', '.join(path.stem for path in paths)} map to {canonical}",
                stage="validate",
                view_name=canonical,
                input_paths=tuple(str(path) for path in paths),
            )
        if len(paths) > 1:
            warnings.append(
                f"duplicate_view:{canonical}:{','.join(path.stem for path in paths)}"
            )
        selected[canonical] = str(paths[0].resolve())
    if bool(views.get("reject_unknown_views", False)) and unknown:
        raise ScanError(
            "unknown_view",
            ",".join(unknown[:20]),
            stage="validate",
            input_paths=(episode_dir,),
        )
    if unknown:
        warnings.append("unknown_view:" + ",".join(unknown[:20]))
    if not selected:
        selected = {str(key): os.path.abspath(str(value)) for key, value in candidates.items()}
    invalid = set(selected) - allowed if allowed else set()
    if invalid:
        raise ScanError("unknown_view", ",".join(sorted(invalid)), stage="validate")
    missing = required - set(selected)
    if missing:
        raise ScanError("missing_view", ",".join(sorted(missing)), stage="validate")
    ordered = list(views.get("order") or ())
    return dict(sorted(selected.items(), key=lambda item: (
        ordered.index(item[0]) if item[0] in ordered else len(ordered), item[0]
    ))), warnings


def quick_validate_views(
    candidates: Mapping[str, str], expected_frames: int, *, max_camera_views: int
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, Any]]]:
    valid: dict[str, str] = {}
    failures: dict[str, str] = {}
    probes: dict[str, dict[str, Any]] = {}
    for view, raw_path in candidates.items():
        path = os.path.abspath(raw_path)
        try:
            stat = os.stat(path)
            if stat.st_size <= 0:
                raise OSError("empty video")
            with av.open(path, mode="r") as container:
                if not container.streams.video:
                    raise ValueError("no video stream")
                stream = container.streams.video[0]
                declared_frames = int(stream.frames) if stream.frames else None
                average_rate = float(stream.average_rate) if stream.average_rate is not None else None
                if declared_frames is not None and declared_frames != expected_frames:
                    raise ValueError(
                        f"declared frame mismatch: {declared_frames} != {expected_frames}"
                    )
        except (OSError, ValueError, av.error.FFmpegError) as exc:
            failures[view] = f"{type(exc).__name__}: {exc}"
            continue
        if len(valid) < max_camera_views:
            valid[view] = path
            probes[view] = {
                "path": path,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "declared_frames": declared_frames,
                "average_rate": average_rate,
                "quick": True,
            }
    if not valid:
        detail = "; ".join(f"{key}={value}" for key, value in failures.items())
        raise ScanError("missing_video", detail or "no candidate videos", stage="validate")
    return valid, failures, probes


def validate_episode_views(
    candidates: Mapping[str, str],
    expected_frames: int,
    *,
    max_camera_views: int,
    quick: bool,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, Any]]]:
    if quick:
        return quick_validate_views(
            candidates, expected_frames, max_camera_views=max_camera_views
        )
    try:
        return full_validate_views(
            dict(candidates), expected_frames, max_camera_views=max_camera_views
        )
    except FileNotFoundError as exc:
        raise ScanError(
            "missing_video", str(exc), stage="validate", input_paths=(str(exc.filename or ""),)
        ) from exc
    except VideoValidationError as exc:
        raise ScanError("video_decode_error", str(exc), stage="validate") from exc
