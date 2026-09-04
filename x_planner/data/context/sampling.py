"""Deterministic 20-frame Memory V3 sampling and full-plan construction."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

from ..pipeline.captions import normalize_caption
from ..pipeline.constants import (
    FIELD_TO_LEVEL,
    LEVEL_TO_FIELD,
    PROFILE_FIELDS,
    PROFILE_LEVELS,
    UNIT_LEVEL,
)
from ..pipeline.hierarchy import (
    EpisodeValidationError,
    build_target,
    valid_frame_ranges,
)
from ..pipeline.memory import MemoryCodec
from ..pipeline.models import CanonicalEpisode
from .schema import (
    TERMINAL_CAPTION,
    prediction_one_state,
    validate_continuous_target,
    validate_initial_plan,
)


def _sha256(*values: object) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def frame_indices(anchor: int, offsets: Iterable[int] = (-20, -10, 0)) -> tuple[int, ...]:
    values = tuple(int(anchor + offset) for offset in offsets if anchor + int(offset) >= 0)
    if not values or values[-1] != anchor:
        raise EpisodeValidationError("current_frame_not_last", str(values))
    if any(frame > anchor for frame in values):
        raise EpisodeValidationError("future_visual_frame", str(values))
    return values


def build_image_refs(
    episode: CanonicalEpisode,
    anchor: int,
    *,
    offsets: Iterable[int] = (-20, -10, 0),
) -> tuple[dict[str, Any], ...]:
    indices = frame_indices(anchor, offsets)
    refs = tuple(
        {
            "view": view,
            "video": video,
            "frame": frame,
            "relative_frame": frame - anchor,
        }
        for frame in indices
        for view, video in episode.videos.items()
    )
    if not refs:
        raise EpisodeValidationError("invalid_visual_count", "0")
    return refs


def anchor_specs(
    episode: CanonicalEpisode,
    *,
    stride: int = 20,
) -> tuple[tuple[int, bool], ...]:
    if stride <= 0:
        raise ValueError("anchor stride must be positive")
    units = episode.levels[UNIT_LEVEL[episode.profile]]
    start = units[0].start_frame
    end = units[-1].end_frame
    valid_ranges_by_unit = tuple(valid_frame_ranges(episode, unit) for unit in units)
    if any(not ranges for ranges in valid_ranges_by_unit):
        missing = next(index for index, ranges in enumerate(valid_ranges_by_unit) if not ranges)
        raise EpisodeValidationError("unit_without_complete_profile_labels", str(missing))
    # Some accepted V2 annotations contain genuine gaps both between same-level
    # Units and inside a higher-level Unit's lower-level labels. Keep the single
    # Episode-global grid, but emit only anchors for which build_target has a
    # complete current label at every Profile level.
    grid = list(range(start, end, stride))
    normal = [
        anchor for anchor in grid
        if any(
            range_start <= anchor < range_end
            for ranges in valid_ranges_by_unit
            for range_start, range_end in ranges
        )
    ]
    last_ranges = valid_ranges_by_unit[-1]
    forced_frame: int | None = None
    if not any(
        range_start <= anchor < range_end
        for anchor in normal
        for range_start, range_end in last_ranges
    ):
        forced_frame = last_ranges[0][0]
        normal.append(forced_frame)
    values = sorted(set(normal))
    return tuple((anchor, anchor == forced_frame) for anchor in values)


def _terminal_level(field: str, caption: str) -> dict[str, Any]:
    level = FIELD_TO_LEVEL[field]
    value: dict[str, Any] = {"level": level}
    if level == "L0":
        value["source"] = "segment"
    value["caption"] = caption
    value["progress_percent"] = 0
    return value


def _with_terminal_prediction(
    target: Mapping[str, Any], profile: str, caption: str
) -> dict[str, Any]:
    result = {
        "task": dict(target["task"]),
        "predictions": [dict(target["predictions"][0])],
    }
    terminal: dict[str, Any] = {"index": 2}
    for field in PROFILE_FIELDS[profile]:
        terminal[field] = _terminal_level(field, caption)
    result["predictions"].append(terminal)
    return result


def continuous_samples(
    episode: CanonicalEpisode,
    *,
    source_id: str,
    global_episode_key: str,
    stride: int = 20,
    offsets: Iterable[int] = (-20, -10, 0),
    terminal_caption: str = TERMINAL_CAPTION,
    resize_policy_id: str = "auto_near_640_no_upscale_v1",
    source_frame_rate_hz: float | None = None,
) -> tuple[dict[str, Any], ...]:
    unit_level = UNIT_LEVEL[episode.profile]
    units = episode.levels[unit_level]
    codec = MemoryCodec(short_memory_k=1, visible_long_memory_limit=8)
    completed_archives: list[tuple[str, ...]] = [()]
    archive: list[str] = []
    for unit in units:
        caption = normalize_caption(unit.caption)
        if not archive or not codec.same(archive[-1], caption):
            archive.append(caption)
        completed_archives.append(tuple(archive))
    grid_origin = units[0].start_frame
    previous_state: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    for anchor, forced in anchor_specs(episode, stride=stride):
        if not forced and (anchor - grid_origin) % stride:
            raise EpisodeValidationError("off_grid_anchor", str(anchor))
        target, unit_index = build_target(episode, anchor)
        is_terminal = unit_index == len(units) - 1
        if is_terminal:
            target = _with_terminal_prediction(target, episode.profile, terminal_caption)
        elif len(target["predictions"]) != 2:
            raise EpisodeValidationError(
                "missing_next_prediction", f"anchor={anchor} unit={unit_index}"
            )
        validate_continuous_target(
            target,
            episode.profile,
            is_terminal_window=is_terminal,
            terminal_caption=terminal_caption,
        )
        images = build_image_refs(episode, anchor, offsets=offsets)
        sample_key = "memory-v3-" + _sha256(
            global_episode_key, "continuous", anchor, episode.profile
        )
        row = {
            "schema_version": "memory_v3",
            "task_type": "continuous",
            "sample_key": sample_key,
            "sample_id": sample_key,
            "global_episode_key": global_episode_key,
            "episode_key": episode.episode_key,
            "source_id": source_id,
            "split": episode.split,
            "profile": episode.profile,
            "unit_type": episode.unit_type,
            "unit_index": unit_index,
            "anchor_frame": anchor,
            "anchor_grid_origin": grid_origin,
            "anchor_grid_index": None if forced else (anchor - grid_origin) // stride,
            "anchor_stride_frames": stride,
            "forced_terminal_anchor": forced,
            "is_terminal_window": is_terminal,
            "task_caption": episode.task_caption,
            "long_memory": list(completed_archives[unit_index]),
            "short_memory": [] if previous_state is None else [previous_state],
            "images": list(images),
            "target": target,
            "resize_policy_id": resize_policy_id,
        }
        if source_frame_rate_hz is not None:
            row["source_frame_rate_hz"] = float(source_frame_rate_hz)
        rows.append(row)
        previous_state = prediction_one_state(target, episode.profile)
    if not rows:
        raise EpisodeValidationError("no_valid_current_frame")
    return tuple(rows)


def _contained(parent: Any, children: Iterable[Any]) -> list[Any]:
    return [child for child in children if parent.contains_unit(child)]


def _plan_object(
    episode: CanonicalEpisode,
    unit: Any,
    remaining_levels: tuple[str, ...],
) -> dict[str, Any]:
    value: dict[str, Any] = {"level": unit.level}
    if unit.level == "L0":
        value["source"] = unit.source
    value["caption"] = unit.caption
    if remaining_levels:
        child_level = remaining_levels[0]
        children = _contained(unit, episode.levels[child_level])
        if not children:
            raise EpisodeValidationError(
                "initial_plan_parent_without_child", f"{unit.unit_id}->{child_level}"
            )
        child_field = LEVEL_TO_FIELD[child_level]
        child_key = "actions" if child_level == "L1" else "segments"
        value[child_key] = [
            {
                "index": index,
                child_field: _plan_object(episode, child, remaining_levels[1:]),
            }
            for index, child in enumerate(children, 1)
        ]
    return value


def initial_plan_target(episode: CanonicalEpisode) -> dict[str, Any]:
    levels = PROFILE_LEVELS[episode.profile]
    top_level = levels[0]
    field = LEVEL_TO_FIELD[top_level]
    units = episode.levels[top_level]
    target = {
        "task": {"level": "L3", "caption": episode.task_caption},
        "initial_plan": [
            {
                "index": index,
                field: _plan_object(episode, unit, levels[1:]),
            }
            for index, unit in enumerate(units, 1)
        ],
    }
    validate_initial_plan(target, episode.profile, expected_top_level_units=len(units))
    return target


def initial_plan_sample(
    episode: CanonicalEpisode,
    *,
    source_id: str,
    global_episode_key: str,
    images: Iterable[Mapping[str, Any]],
    resize_policy_id: str = "auto_near_640_no_upscale_v1",
) -> dict[str, Any]:
    image_rows = [dict(image) for image in images]
    if not image_rows:
        raise EpisodeValidationError("invalid_visual_count", "0")
    current = [
        int(image["frame"])
        for image in image_rows
        if int(image.get("relative_frame", 1)) == 0
    ]
    if not current:
        raise EpisodeValidationError("initial_plan_missing_current_frame")
    if len(set(current)) != 1:
        raise EpisodeValidationError("unsynchronized_current_frame", str(current))
    sample_key = "memory-v3-" + _sha256(
        global_episode_key, "initial_plan", episode.profile
    )
    return {
        "schema_version": "memory_v3",
        "task_type": "initial_plan",
        "sample_key": sample_key,
        "sample_id": sample_key,
        "global_episode_key": global_episode_key,
        "episode_key": episode.episode_key,
        "source_id": source_id,
        "split": episode.split,
        "profile": episode.profile,
        "unit_type": episode.unit_type,
        "anchor_frame": current[0],
        "is_terminal_window": False,
        "task_caption": episode.task_caption,
        "images": image_rows,
        "target": initial_plan_target(episode),
        "resize_policy_id": resize_policy_id,
    }
