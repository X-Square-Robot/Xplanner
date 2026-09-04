"""Dynamic visual-first prompt and indexed-JSONL row rendering."""

from __future__ import annotations

from collections.abc import Iterable

from .constants import PROFILE_FIELDS
from .memory import MemoryCodec
from .models import V10Sample
from .schema import dumps_target


def build_prompt(
    *,
    profile: str,
    unit_type: str,
    views: Iterable[str],
) -> str:
    fields = PROFILE_FIELDS[profile]
    view_text = ", ".join(views)
    field_text = ", ".join(fields)
    return (
        "Use the synchronized current and historical robot-camera images as the primary evidence. "
        "Memory is only a prior over already completed units and may be incomplete or noisy.\n"
        "Identify the complete L3 task, locate the current temporal unit, and predict only its "
        "immediately following same-level unit when one exists. Never skip an intermediate unit.\n"
        f"Prediction unit: {unit_type}. Required prediction fields: {field_text}. "
        f"Actual camera views: {view_text}.\n"
        "Return exactly one valid English JSON object. The task field is always L3. "
        "Predictions must contain one or two entries indexed from 1. Future prediction progress "
        "must be 0. Omit unavailable hierarchy levels; do not output null, markdown, explanation, "
        "seconds, or additional fields."
    )


def render_user_text(
    sample: V10Sample,
    *,
    codec: MemoryCodec | None = None,
    long_memory: Iterable[str] | None = None,
    short_memory: Iterable[str] | None = None,
) -> str:
    codec = codec or MemoryCodec()
    long_values = tuple(sample.long_memory if long_memory is None else long_memory)
    short_values = tuple(
        codec.short_from_long(long_values) if short_memory is None else short_memory
    )
    views = tuple(dict.fromkeys(image.view for image in sample.images))
    blocks = [
        build_prompt(profile=sample.profile, unit_type=sample.unit_type, views=views),
        "Long Memory:\n" + codec.render_long(long_values),
        "Short Memory:\n" + codec.render_short(short_values),
    ]
    blocks.extend(
        f"[time={image.relative_frame:+d}f][view={image.view}]\n<image>"
        for image in sample.images
    )
    return "\n\n".join(blocks)


def sample_to_indexed_jsonl(
    sample: V10Sample,
    *,
    codec: MemoryCodec | None = None,
    long_memory: Iterable[str] | None = None,
    short_memory: Iterable[str] | None = None,
) -> dict[str, object]:
    user_text = render_user_text(
        sample,
        codec=codec,
        long_memory=long_memory,
        short_memory=short_memory,
    )
    return {
        "data_id": sample.sample_id,
        "episode_key": sample.episode_key,
        "profile": sample.profile,
        "unit_type": sample.unit_type,
        "unit_index": sample.unit_index,
        "current_frame": sample.current_frame,
        "gt_long_memory": list(sample.long_memory),
        "v10_sample": sample.to_dict(),
        "image": [
            {"video": image.video, "frame": image.frame, "view": image.view}
            for image in sample.images
        ],
        "text": [
            {"role": "user", "text": user_text},
            {"role": "assistant", "text": dumps_target(sample.target, sample.profile)},
        ],
    }
