"""Memory V4 prompts: the canonical human instruction is always an input."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..v10_continuous.constants import FIELD_TO_LEVEL, PROFILE_FIELDS
from ..v10_continuous_v3_memory.memory_v3 import MemoryCodecV3
from ..v10_continuous_v3_memory.prompt_v3 import _image_blocks, _source_rate_line
from ..v10_continuous_v3_memory.schema_v3 import TERMINAL_CAPTION


_LEVEL_NAMES = {"L3": "Task", "L2": "Subtask", "L1": "Action", "L0": "Segment"}
_UNIT_LEVELS = {"subtask": "L2", "action": "L1", "segment": "L0"}
_HIERARCHY = "Task (L3) > Subtask (L2) > Action (L1) > Segment (L0)"


def _level_label(level: str) -> str:
    return f"{_LEVEL_NAMES[level]} ({level})"


def _profile_labels(profile: str) -> str:
    return ", ".join(
        _level_label(FIELD_TO_LEVEL[field]) for field in PROFILE_FIELDS[profile]
    )


def _tagged_block(name: str, value: str, *, attributes: str = "") -> str:
    return f"[{name}]{attributes}\n{value}\n[/{name}]"


def _instruction_block(sample: Mapping[str, Any]) -> str:
    instruction = str(sample.get("task_instruction") or "").strip()
    if not instruction:
        raise ValueError("Memory V4 sample is missing task_instruction")
    return _tagged_block(
        "task_instruction",
        json.dumps(instruction, ensure_ascii=False),
        attributes="[level=L3]",
    )


def build_continuous_instruction(sample: Mapping[str, Any]) -> str:
    profile = str(sample["profile"])
    unit_type = str(sample["unit_type"])
    if unit_type not in _UNIT_LEVELS:
        raise ValueError(f"unknown V4 prediction unit: {unit_type!r}")
    prediction_scale = _level_label(_UNIT_LEVELS[unit_type])
    views = ", ".join(dict.fromkeys(str(image["view"]) for image in sample["images"]))
    blocks = [
        "Track the given Task (L3) using synchronized robot-camera frames. Memory records "
        "prior state and may be incomplete or noisy.",
        f"Hierarchy (coarse to fine): {_HIERARCHY}.",
        _instruction_block(sample),
        f"Predict two consecutive units at the {prediction_scale} scale: Prediction 1 is "
        "current; Prediction 2 is the next same-scale unit, or "
        f"{TERMINAL_CAPTION!r} after the final real unit.",
        f"Profile: {profile}. Output levels: {_profile_labels(profile)}. Views: {views}.",
        "Return one English JSON object with ordered top-level fields task_progress_percent "
        "and predictions, containing exactly two indexed predictions.",
    ]
    rate = _source_rate_line(sample)
    if rate:
        blocks.append(rate)
    return "\n".join(blocks)


def render_continuous_user(sample: Mapping[str, Any]) -> str:
    codec = MemoryCodecV3(
        visible_long_memory_limit=int(sample.get("visible_long_memory_limit", 8))
    )
    return "\n\n".join(
        [
            build_continuous_instruction(sample),
            _tagged_block(
                "long_memory", codec.render_long(sample.get("long_memory") or ())
            ),
            _tagged_block(
                "short_memory", codec.render_short(sample.get("short_memory") or ())
            ),
            *_image_blocks(list(sample["images"])),
        ]
    )


def render_initial_plan_user(sample: Mapping[str, Any]) -> str:
    profile = str(sample["profile"])
    views = ", ".join(dict.fromkeys(str(image["view"]) for image in sample["images"]))
    blocks = [
        "Build the complete ordered plan for the given Task (L3) from the earliest valid "
        "synchronized robot-camera frames.",
        f"Hierarchy (coarse to fine): {_HIERARCHY}.",
        _instruction_block(sample),
        f"Profile: {profile}. Plan levels: {_profile_labels(profile)}. Views: {views}.",
        "Return one English JSON object with top-level field initial_plan. Include every "
        "future unit in order and nest each lower-scale unit under its parent. Omit "
        "progress_percent; do not truncate or add a completion sentinel.",
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
    raise ValueError(f"unknown V4 task_type: {task_type!r}")


def sample_to_indexed_jsonl(sample: Mapping[str, Any]) -> dict[str, Any]:
    from .schema_v4 import dumps_assistant

    task_type = str(sample["task_type"])
    instruction = str(sample["task_instruction"])
    terminal = bool(sample.get("is_terminal_window", False))
    return {
        "data_id": sample["sample_key"],
        "sample_key": sample["sample_key"],
        "global_episode_key": sample["global_episode_key"],
        "episode_key": sample["episode_key"],
        "source_id": sample["source_id"],
        "split": sample["split"],
        "profile": sample["profile"],
        "task_type": task_type,
        "is_terminal_window": terminal,
        "resize_policy_id": sample.get("resize_policy_id"),
        "lineage": dict(sample.get("lineage") or {}),
        "v4_sample": dict(sample),
        "image": [
            {"video": image["video"], "frame": image["frame"], "view": image["view"]}
            for image in sample["images"]
        ],
        "text": [
            {"role": "user", "text": render_user(sample)},
            {
                "role": "assistant",
                "text": dumps_assistant(
                    sample["target"],
                    str(sample["profile"]),
                    task_type,
                    instruction=instruction,
                    is_terminal_window=terminal,
                ),
            },
        ],
    }
