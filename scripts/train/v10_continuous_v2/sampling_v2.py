"""V2 deterministic quantile-plus-stride sampling over frozen V1 semantics."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from ..v10_continuous.captions import normalize_caption
from ..v10_continuous.constants import UNIT_LEVEL
from ..v10_continuous.constants import (
    DEFAULT_MAX_CAMERA_VIEWS,
    DEFAULT_MAX_VISUAL_INPUTS,
    DEFAULT_STRIDE,
)
from ..v10_continuous.hierarchy import (
    EpisodeValidationError,
    build_image_refs,
    build_target,
    quantile_frames,
    valid_frame_ranges,
)
from ..v10_continuous.models import CanonicalEpisode, V10Sample
from .common.hashing import canonical_json, sha256_hex


FROZEN_SAMPLING_VALUES: dict[str, int] = {
    "visual_stride": DEFAULT_STRIDE,
    "visual_timesteps": 3,
    "max_camera_views": DEFAULT_MAX_CAMERA_VIEWS,
    "max_visual_inputs": DEFAULT_MAX_VISUAL_INPUTS,
}


def validate_sampling_config(sampling: Mapping[str, Any]) -> None:
    """Fail fast when a config claims visual semantics V1 does not expose."""
    for key, expected in FROZEN_SAMPLING_VALUES.items():
        actual = int(sampling.get(key, expected))
        if actual != expected:
            raise ValueError(f"sampling.{key} is frozen at {expected}, got {actual}")


def anchors_for_ranges(
    ranges: list[tuple[int, int]], *, quantiles: tuple[float, ...], stride: int
) -> tuple[int, ...]:
    frames = set(quantile_frames(ranges, quantiles))
    for start, end in ranges:
        frames.update(range(start, end, stride))
    return tuple(sorted(frames))


def sample_key(
    *, source_id: str, episode_key: str, current_frame: int, unit_level: str,
    profile: str, views: tuple[str, ...], sampling_hash: str
) -> str:
    payload = {
        "source_id": source_id,
        "episode_key": episode_key,
        "anchor_index": current_frame,
        "unit_level": unit_level,
        "profile": profile,
        "canonical_view_set": list(views),
        "sampling_config_hash": sampling_hash,
    }
    return "v10v2-" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def build_samples_v2(
    episode: CanonicalEpisode,
    *,
    source_id: str,
    global_episode_key: str,
    dataset_name: str,
    sampling: Mapping[str, Any],
    sampling_hash: str,
    input_paths: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    validate_sampling_config(sampling)
    quantiles = tuple(float(item) for item in sampling.get("anchor_quantiles", (0.25, 0.5, 0.75)))
    stride = int(sampling.get("anchor_stride", 10))
    max_anchors = sampling.get("max_anchors_per_episode")
    if stride <= 0 or any(not 0 <= value <= 1 for value in quantiles):
        raise ValueError("invalid anchor sampling configuration")
    unit_level = UNIT_LEVEL[episode.profile]
    units = episode.levels[unit_level]
    rows: list[dict[str, Any]] = []
    views = tuple(episode.videos)
    for unit_index, unit in enumerate(units):
        ranges = valid_frame_ranges(episode, unit)
        for current_frame in anchors_for_ranges(ranges, quantiles=quantiles, stride=stride):
            try:
                target, found_index = build_target(episode, current_frame)
                if found_index != unit_index:
                    raise EpisodeValidationError("anchor_unit_mismatch")
                images = build_image_refs(episode, current_frame)
            except EpisodeValidationError:
                continue
            history = tuple(normalize_caption(item.caption) for item in units[:unit_index])
            digest = sha256_hex(global_episode_key, unit_index, current_frame)[:20]
            sample = V10Sample(
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
            ).to_dict()
            key = sample_key(
                source_id=source_id,
                episode_key=episode.episode_key,
                current_frame=current_frame,
                unit_level=unit_level,
                profile=episode.profile,
                views=views,
                sampling_hash=sampling_hash,
            )
            sample.update({
                "sample_key": key,
                "global_episode_key": global_episode_key,
                "source_id": source_id,
                "dataset_name": dataset_name,
                "anchor_index": current_frame,
                "unit_level": unit_level,
                "views": list(views),
                "input_paths": list(input_paths),
                "label": target,
                "metadata": {"sampling_config_hash": sampling_hash},
            })
            rows.append(sample)
            if max_anchors is not None and len(rows) >= int(max_anchors):
                return tuple(rows)
    if not rows:
        raise EpisodeValidationError("no_valid_current_frame")
    return tuple(rows)
