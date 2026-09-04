"""Frame-only Memory V3 prompts and indexed JSONL rendering."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..pipeline.constants import PROFILE_FIELDS
from .memory import MemoryCodecV3
from .schema import TERMINAL_CAPTION, dumps_assistant


def _source_rate_line(sample: Mapping[str, Any]) -> str:
    value = sample.get("source_frame_rate_hz")
    if value is None:
        return ""
    rate = float(value)
    rendered = str(int(rate)) if rate.is_integer() else f"{rate:g}"
    return f"Source frame rate: {rendered} Hz."


def _image_blocks(images: list[Mapping[str, Any]]) -> list[str]:
    return [
        f"[frame_offset={int(image['relative_frame'])}][view={image['view']}]\n<image>"
        for image in images
    ]


def build_continuous_instruction(sample: Mapping[str, Any]) -> str:
    fields = ", ".join(PROFILE_FIELDS[str(sample["profile"])])
    views = ", ".join(dict.fromkeys(str(image["view"]) for image in sample["images"]))
    blocks = [
        "Use the synchronized robot-camera frames as the primary evidence. "
        "Memory is prior state and may be incomplete or noisy.",
        "Identify the complete L3 task and the active temporal unit. Prediction 1 is the "
        "active unit. Prediction 2 is only the immediately following same-level unit; "
        f"for the final real unit it is the fixed completion caption {TERMINAL_CAPTION!r}.",
        f"Prediction unit: {sample['unit_type']}. Required prediction fields: {fields}. "
        f"Actual camera views: {views}.",
        "Return exactly one valid English JSON object. Keep L3 under task. Return exactly "
        "two indexed predictions. Prediction 2 progress_percent must be 0. Omit unavailable "
        "hierarchy levels and do not output null, markdown, explanation, or additional fields.",
    ]
    rate = _source_rate_line(sample)
    if rate:
        blocks.append(rate)
    return "\n".join(blocks)


def render_continuous_user(sample: Mapping[str, Any]) -> str:
    codec = MemoryCodecV3(
        visible_long_memory_limit=int(sample.get("visible_long_memory_limit", 8))
    )
    blocks = [
        build_continuous_instruction(sample),
        "Long Memory:\n" + codec.render_long(sample.get("long_memory") or ()),
        "Short Memory:\n" + codec.render_short(sample.get("short_memory") or ()),
        *_image_blocks(list(sample["images"])),
    ]
    return "\n\n".join(blocks)


def render_initial_plan_user(sample: Mapping[str, Any]) -> str:
    fields = ", ".join(PROFILE_FIELDS[str(sample["profile"])])
    views = ", ".join(dict.fromkeys(str(image["view"]) for image in sample["images"]))
    blocks = [
        "Use the earliest valid synchronized robot-camera frames to predict the complete "
        "ordered plan for the Episode.",
        f"Profile: {sample['profile']}. Profile fields: {fields}. Actual camera views: {views}.",
        "Return exactly one valid English JSON object with top-level task and initial_plan. "
        "Store L3 once under task. Include every future unit in order, nest lower-level units "
        "under their owning parent, omit progress_percent, and do not truncate, add a completion "
        "sentinel, output markdown, explanation, null, or additional fields.",
    ]
    rate = _source_rate_line(sample)
    if rate:
        blocks.append(rate)
    blocks.extend(_image_blocks(list(sample["images"])))
    return "\n\n".join(blocks)


def render_user(sample: Mapping[str, Any]) -> str:
    task_type = str(sample["task_type"])
    if task_type == "initial_plan":
        return render_initial_plan_user(sample)
    if task_type == "continuous":
        return render_continuous_user(sample)
    raise ValueError(f"unknown V3 task_type: {task_type!r}")


def sample_to_indexed_jsonl(sample: Mapping[str, Any]) -> dict[str, Any]:
    task_type = str(sample["task_type"])
    return {
        "data_id": sample["sample_key"],
        "sample_key": sample["sample_key"],
        "global_episode_key": sample["global_episode_key"],
        "episode_key": sample["episode_key"],
        "source_id": sample["source_id"],
        "split": sample["split"],
        "profile": sample["profile"],
        "task_type": task_type,
        "is_terminal_window": bool(sample.get("is_terminal_window", False)),
        "resize_policy_id": sample.get("resize_policy_id"),
        "v3_sample": dict(sample),
        "image": [
            {"video": image["video"], "frame": image["frame"], "view": image["view"]}
            for image in sample["images"]
        ],
        "text": [
            {"role": "user", "text": render_user(sample)},
            {
                "role": "assistant",
                "text": dumps_assistant(sample["target"], str(sample["profile"]), task_type),
            },
        ],
    }
