"""Strict hierarchy validation, target construction and deterministic anchors."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from typing import Any, Iterable

from .captions import is_valid_english_caption, normalize_caption
from .constants import (
    DEFAULT_MAX_CAMERA_VIEWS,
    DEFAULT_MAX_VISUAL_INPUTS,
    DEFAULT_MIN_INTERVAL_FRAMES,
    DEFAULT_MIN_L0_COUNT,
    DEFAULT_MIN_L1_COUNT,
    DEFAULT_STRIDE,
    FIELD_TO_LEVEL,
    LEVEL_ORDER,
    LEVEL_TO_FIELD,
    PROFILE_FIELDS,
    PROFILE_LEVELS,
    PROFILE_PRECEDENCE,
    UNIT_LEVEL,
    UNIT_TYPE,
    VIEW_PRIORITY,
)
from .models import CanonicalEpisode, ImageRef, TemporalUnit, V10Sample
from .schema import validate_target


class EpisodeValidationError(ValueError):
    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


def progress_percent(frame: int, start: int, end: int) -> int:
    progress = (frame - start) / (end - start)
    return round(max(0.0, min(progress, 1.0)) * 100)


def validate_level(
    level: str,
    units: Iterable[TemporalUnit],
    num_frames: int,
    *,
    min_interval_frames: int = DEFAULT_MIN_INTERVAL_FRAMES,
) -> tuple[TemporalUnit, ...]:
    ordered = tuple(sorted(units, key=lambda unit: (unit.start_frame, unit.end_frame)))
    if not ordered:
        raise EpisodeValidationError(f"missing_{level.lower()}")
    seen_ids: set[str] = set()
    previous_end = -1
    for unit in ordered:
        if unit.level != level:
            raise EpisodeValidationError("level_mismatch", f"{unit.unit_id}: {unit.level} != {level}")
        if unit.unit_id in seen_ids:
            raise EpisodeValidationError("duplicate_unit_id", unit.unit_id)
        seen_ids.add(unit.unit_id)
        if not is_valid_english_caption(unit.caption):
            raise EpisodeValidationError("invalid_caption", f"{level}:{unit.unit_id}")
        if (
            isinstance(unit.start_frame, bool)
            or isinstance(unit.end_frame, bool)
            or not isinstance(unit.start_frame, int)
            or not isinstance(unit.end_frame, int)
        ):
            raise EpisodeValidationError("invalid_interval_type", unit.unit_id)
        if not 0 <= unit.start_frame < unit.end_frame <= num_frames:
            raise EpisodeValidationError(
                "interval_out_of_bounds",
                f"{unit.unit_id}=[{unit.start_frame},{unit.end_frame}) num_frames={num_frames}",
            )
        if unit.duration < min_interval_frames:
            raise EpisodeValidationError(
                "interval_too_short", f"{unit.unit_id}:{unit.duration}"
            )
        if unit.start_frame < previous_end:
            raise EpisodeValidationError("same_level_overlap", f"{level}:{unit.unit_id}")
        previous_end = unit.end_frame
        if level == "L0" and unit.source not in {"human_segment", "segment"}:
            raise EpisodeValidationError("invalid_l0_source", unit.unit_id)
    return ordered


def _link_selected_levels(
    valid_levels: dict[str, tuple[TemporalUnit, ...]], profile: str
) -> dict[str, tuple[TemporalUnit, ...]]:
    selected_levels = PROFILE_LEVELS[profile]
    linked: dict[str, tuple[TemporalUnit, ...]] = {
        level: tuple(replace(unit, parent_id=None) for unit in valid_levels[level])
        for level in selected_levels
    }
    for parent_level, child_level in zip(selected_levels, selected_levels[1:]):
        parents = linked[parent_level]
        children: list[TemporalUnit] = []
        child_counts = {parent.unit_id: 0 for parent in parents}
        for child in linked[child_level]:
            matches = [parent for parent in parents if parent.contains_unit(child)]
            if len(matches) != 1:
                raise EpisodeValidationError(
                    "invalid_parent_relation",
                    f"{child.unit_id} has {len(matches)} {parent_level} parents",
                )
            parent = matches[0]
            child_counts[parent.unit_id] += 1
            children.append(replace(child, parent_id=parent.unit_id))
        empty_parents = [unit_id for unit_id, count in child_counts.items() if count == 0]
        if empty_parents:
            raise EpisodeValidationError(
                "parent_without_required_child",
                f"{parent_level}->{child_level}: {empty_parents[:5]}",
            )
        linked[child_level] = tuple(children)
    return linked


def _choose_profile(
    valid_levels: dict[str, tuple[TemporalUnit, ...]],
    *,
    min_l1_count: int,
    min_l0_count: int,
) -> tuple[str, dict[str, tuple[TemporalUnit, ...]]]:
    relation_errors: list[str] = []
    for profile in PROFILE_PRECEDENCE:
        required = PROFILE_LEVELS[profile]
        if any(level not in valid_levels for level in required):
            continue
        if "L1" in required and len(valid_levels["L1"]) < min_l1_count:
            continue
        if "L0" in required and len(valid_levels["L0"]) < min_l0_count:
            continue
        try:
            return profile, _link_selected_levels(valid_levels, profile)
        except EpisodeValidationError as exc:
            relation_errors.append(f"{profile}:{exc}")
    counts = {level: len(valid_levels.get(level, ())) for level in LEVEL_ORDER}
    diagnostics = [
        "valid_counts=" + ",".join(f"{level}:{counts[level]}" for level in LEVEL_ORDER)
    ]
    if not any(counts.values()):
        diagnostics.append("no_valid_temporal_level")
    below_minimum = []
    if 0 < counts["L1"] < min_l1_count:
        below_minimum.append(f"L1:{counts['L1']}<{min_l1_count}")
    if 0 < counts["L0"] < min_l0_count:
        below_minimum.append(f"L0:{counts['L0']}<{min_l0_count}")
    if below_minimum:
        diagnostics.append("below_min=" + ",".join(below_minimum))
    if relation_errors:
        diagnostics.append("relation_errors=" + " | ".join(relation_errors[:5]))
    detail = "; ".join(diagnostics)
    raise EpisodeValidationError("profile_unavailable", detail)


def canonicalize_episode(
    *,
    source: str,
    episode_key: str,
    episode_name: str,
    split: str,
    num_frames: int,
    task_caption: str,
    raw_levels: dict[str, Iterable[TemporalUnit]],
    videos: dict[str, str],
    annotation_sources: Iterable[str] = (),
    metadata: dict[str, Any] | None = None,
    min_interval_frames: int = DEFAULT_MIN_INTERVAL_FRAMES,
    min_l1_count: int = DEFAULT_MIN_L1_COUNT,
    min_l0_count: int = DEFAULT_MIN_L0_COUNT,
) -> CanonicalEpisode:
    if not isinstance(num_frames, int) or isinstance(num_frames, bool) or num_frames <= 0:
        raise EpisodeValidationError("invalid_num_frames", repr(num_frames))
    if not is_valid_english_caption(task_caption):
        raise EpisodeValidationError("invalid_l3")
    if split not in {"train", "validation"}:
        raise EpisodeValidationError("invalid_split", split)
    if not videos:
        raise EpisodeValidationError("no_valid_views")

    valid_levels: dict[str, tuple[TemporalUnit, ...]] = {}
    invalid_level_reasons: dict[str, str] = {}
    for level in LEVEL_ORDER:
        raw = tuple(raw_levels.get(level, ()))
        if not raw:
            continue
        try:
            valid_levels[level] = validate_level(
                level,
                raw,
                num_frames,
                min_interval_frames=min_interval_frames,
            )
        except EpisodeValidationError as exc:
            invalid_level_reasons[level] = str(exc)

    try:
        profile, linked_levels = _choose_profile(
            valid_levels,
            min_l1_count=min_l1_count,
            min_l0_count=min_l0_count,
        )
    except EpisodeValidationError as exc:
        detail = exc.detail or str(exc)
        if invalid_level_reasons:
            detail += f"; invalid_levels={invalid_level_reasons}"
        raise EpisodeValidationError(exc.reason, detail) from exc

    ordered_videos = {
        view: videos[view]
        for view in VIEW_PRIORITY
        if view in videos
    }
    if not ordered_videos:
        raise EpisodeValidationError("no_mapped_valid_views")
    return CanonicalEpisode(
        source=source,
        episode_key=episode_key,
        episode_name=episode_name,
        split=split,
        num_frames=num_frames,
        task_caption=task_caption,
        profile=profile,
        unit_type=UNIT_TYPE[UNIT_LEVEL[profile]],
        levels=linked_levels,
        videos=ordered_videos,
        annotation_sources=tuple(annotation_sources),
        metadata=dict(metadata or {}),
    )


def _containing(units: Iterable[TemporalUnit], frame: int) -> TemporalUnit | None:
    for unit in units:
        if unit.contains_frame(frame):
            return unit
    return None


def _object_for(unit: TemporalUnit, frame: int, future: bool) -> dict[str, Any]:
    value: dict[str, Any] = {"level": unit.level}
    if unit.level == "L0":
        value["source"] = unit.source
    value["caption"] = unit.caption
    value["progress_percent"] = (
        0 if future else progress_percent(frame, unit.start_frame, unit.end_frame)
    )
    return value


def build_target(
    episode: CanonicalEpisode,
    current_frame: int,
) -> tuple[dict[str, Any], int]:
    if not 0 <= current_frame < episode.num_frames:
        raise EpisodeValidationError("current_frame_out_of_bounds", str(current_frame))
    unit_level = UNIT_LEVEL[episode.profile]
    units = episode.levels[unit_level]
    current_index = next(
        (index for index, unit in enumerate(units) if unit.contains_frame(current_frame)),
        None,
    )
    if current_index is None:
        raise EpisodeValidationError("current_unit_not_found", str(current_frame))

    prediction_units = [units[current_index]]
    if current_index + 1 < len(units):
        prediction_units.append(units[current_index + 1])

    predictions: list[dict[str, Any]] = []
    for position, prediction_unit in enumerate(prediction_units, 1):
        future = position == 2
        prediction: dict[str, Any] = {"index": position}
        for field in PROFILE_FIELDS[episode.profile]:
            level = FIELD_TO_LEVEL[field]
            if level == unit_level:
                selected = prediction_unit
            else:
                candidates = [
                    unit
                    for unit in episode.levels[level]
                    if prediction_unit.contains_unit(unit)
                ]
                selected = (
                    min(candidates, key=lambda unit: (unit.start_frame, unit.end_frame))
                    if future
                    else _containing(candidates, current_frame)
                )
            if selected is None:
                raise EpisodeValidationError(
                    "prediction_field_unavailable",
                    f"frame={current_frame} field={field} future={future}",
                )
            prediction[field] = _object_for(selected, current_frame, future)
        predictions.append(prediction)

    target = {
        "task": {
            "level": "L3",
            "caption": episode.task_caption,
            "progress_percent": progress_percent(current_frame, 0, episode.num_frames),
        },
        "predictions": predictions,
    }
    validate_target(target, episode.profile, expected_prediction_count=len(prediction_units))
    return target, current_index


def _intersect_ranges(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    i = j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def valid_frame_ranges(
    episode: CanonicalEpisode, prediction_unit: TemporalUnit
) -> list[tuple[int, int]]:
    ranges = [(prediction_unit.start_frame, prediction_unit.end_frame)]
    unit_level = UNIT_LEVEL[episode.profile]
    lower_levels = PROFILE_LEVELS[episode.profile][
        PROFILE_LEVELS[episode.profile].index(unit_level) + 1:
    ]
    for level in lower_levels:
        child_ranges = [
            (child.start_frame, child.end_frame)
            for child in episode.levels[level]
            if prediction_unit.contains_unit(child)
        ]
        if not child_ranges:
            return []
        ranges = _intersect_ranges(ranges, child_ranges)
        if not ranges:
            return []
    return ranges


def quantile_frames(
    ranges: list[tuple[int, int]], quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)
) -> tuple[int, ...]:
    total = sum(end - start for start, end in ranges)
    if total <= 0:
        return ()
    frames: list[int] = []
    for quantile in quantiles:
        offset = math.floor(quantile * (total - 1))
        for start, end in ranges:
            length = end - start
            if offset < length:
                frames.append(start + offset)
                break
            offset -= length
    return tuple(dict.fromkeys(frames))


def frame_indices(current_frame: int, stride: int = DEFAULT_STRIDE) -> tuple[int, ...]:
    return tuple(
        frame
        for frame in (current_frame - 2 * stride, current_frame - stride, current_frame)
        if frame >= 0
    )


def build_image_refs(
    episode: CanonicalEpisode,
    current_frame: int,
    *,
    stride: int = DEFAULT_STRIDE,
    max_camera_views: int = DEFAULT_MAX_CAMERA_VIEWS,
) -> tuple[ImageRef, ...]:
    views = [view for view in VIEW_PRIORITY if view in episode.videos][:max_camera_views]
    indices = frame_indices(current_frame, stride)
    refs = tuple(
        ImageRef(
            view=view,
            video=episode.videos[view],
            frame=frame,
            relative_frame=frame - current_frame,
        )
        for frame in indices
        for view in views
    )
    if not refs or len(refs) > DEFAULT_MAX_VISUAL_INPUTS:
        raise EpisodeValidationError("invalid_visual_count", str(len(refs)))
    if refs[-1].frame != current_frame:
        raise EpisodeValidationError("current_frame_not_last")
    return refs


def build_samples(episode: CanonicalEpisode) -> tuple[V10Sample, ...]:
    unit_level = UNIT_LEVEL[episode.profile]
    units = episode.levels[unit_level]
    samples: list[V10Sample] = []
    for unit_index, unit in enumerate(units):
        ranges = valid_frame_ranges(episode, unit)
        for current_frame in quantile_frames(ranges):
            try:
                target, found_index = build_target(episode, current_frame)
                if found_index != unit_index:
                    raise EpisodeValidationError("anchor_unit_mismatch")
                images = build_image_refs(episode, current_frame)
            except EpisodeValidationError:
                continue
            history = tuple(normalize_caption(item.caption) for item in units[:unit_index])
            digest = hashlib.sha256(
                f"{episode.episode_key}\0{unit_index}\0{current_frame}".encode()
            ).hexdigest()[:20]
            samples.append(V10Sample(
                sample_id=f"v10-{digest}",
                episode_key=episode.episode_key,
                split=episode.split,
                profile=episode.profile,
                unit_type=episode.unit_type,
                unit_index=unit_index,
                current_frame=current_frame,
                task_caption=episode.task_caption,
                long_memory=history,
                images=images,
                target=target,
            ))
    if not samples:
        raise EpisodeValidationError("no_valid_current_frame")
    return tuple(samples)
