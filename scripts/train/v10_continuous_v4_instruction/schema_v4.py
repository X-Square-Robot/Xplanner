"""Strict instruction-conditioned Memory V4 wire contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..v10_continuous.captions import is_valid_english_caption
from ..v10_continuous.schema import TargetValidationError
from ..v10_continuous_v3_memory.schema_v3 import (
    TERMINAL_CAPTION,
    validate_continuous_target as validate_v3_continuous_target,
    validate_initial_plan as validate_v3_initial_plan,
)


SCHEMA_VERSION = "memory_v4"
SNAPSHOT_SCHEMA_VERSION = "v10_memory_v4_snapshot_v1"
CONTINUOUS_KEYS = ("task_progress_percent", "predictions")
INITIAL_PLAN_KEYS = ("initial_plan",)


def validate_instruction(value: Any) -> str:
    if not isinstance(value, str) or not is_valid_english_caption(value):
        raise TargetValidationError("task_instruction must be non-empty English")
    return value


def _exact_keys(value: Mapping[str, Any], expected: tuple[str, ...], where: str) -> None:
    actual = tuple(value)
    if actual != expected:
        raise TargetValidationError(f"{where} keys must be {expected}, got {actual}")


def _progress(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise TargetValidationError(
            "task_progress_percent must be an integer in [0, 100]"
        )
    return value


def validate_continuous_target(
    target: Any,
    profile: str,
    *,
    instruction: str,
    is_terminal_window: bool,
) -> dict[str, Any]:
    """Validate V4 without ever making L3 an Assistant output field."""
    instruction = validate_instruction(instruction)
    if not isinstance(target, Mapping):
        raise TargetValidationError("continuous target must be an object")
    _exact_keys(target, CONTINUOUS_KEYS, "continuous target")
    progress = _progress(target["task_progress_percent"])
    reconstructed = {
        "task": {
            "level": "L3",
            "caption": instruction,
            "progress_percent": progress,
        },
        "predictions": target["predictions"],
    }
    validated = validate_v3_continuous_target(
        reconstructed,
        profile,
        is_terminal_window=is_terminal_window,
    )
    return {
        "task_progress_percent": progress,
        "predictions": validated["predictions"],
    }


def validate_initial_plan(
    target: Any,
    profile: str,
    *,
    instruction: str,
    expected_top_level_units: int | None = None,
) -> dict[str, Any]:
    instruction = validate_instruction(instruction)
    if not isinstance(target, Mapping):
        raise TargetValidationError("initial plan target must be an object")
    _exact_keys(target, INITIAL_PLAN_KEYS, "initial plan target")
    reconstructed = {
        "task": {"level": "L3", "caption": instruction},
        "initial_plan": target["initial_plan"],
    }
    validated = validate_v3_initial_plan(
        reconstructed,
        profile,
        expected_top_level_units=expected_top_level_units,
    )
    return {"initial_plan": validated["initial_plan"]}


def validate_target(
    target: Any,
    profile: str,
    task_type: str,
    *,
    instruction: str,
    is_terminal_window: bool = False,
) -> dict[str, Any]:
    if task_type == "initial_plan":
        if is_terminal_window:
            raise TargetValidationError("initial_plan cannot be terminal")
        return validate_initial_plan(target, profile, instruction=instruction)
    if task_type in {"continuous", "terminal"}:
        return validate_continuous_target(
            target,
            profile,
            instruction=instruction,
            is_terminal_window=is_terminal_window,
        )
    raise TargetValidationError(f"unknown V4 task_type: {task_type!r}")


def dumps_assistant(
    target: Mapping[str, Any],
    profile: str,
    task_type: str,
    *,
    instruction: str,
    is_terminal_window: bool = False,
) -> str:
    validated = validate_target(
        target,
        profile,
        task_type,
        instruction=instruction,
        is_terminal_window=is_terminal_window,
    )
    return json.dumps(validated, ensure_ascii=False, separators=(",", ":"))


def loads_assistant(
    text: str,
    profile: str,
    task_type: str,
    *,
    instruction: str,
    is_terminal_window: bool = False,
) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TargetValidationError(f"invalid JSON: {exc}") from exc
    return validate_target(
        value,
        profile,
        task_type,
        instruction=instruction,
        is_terminal_window=is_terminal_window,
    )

