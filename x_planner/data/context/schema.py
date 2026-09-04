"""Strict Memory V3 continuous and Initial Plan wire contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ..pipeline.captions import is_valid_english_caption
from ..pipeline.constants import (
    FIELD_TO_LEVEL,
    LEVEL_TO_FIELD,
    PROFILE_FIELDS,
    PROFILE_LEVELS,
    UNIT_LEVEL,
)
from ..pipeline.schema import TargetValidationError, validate_target


TERMINAL_CAPTION = "the task is complete"
_CHILD_LIST = {"L1": "actions", "L0": "segments"}


def _exact_keys(value: Mapping[str, Any], expected: tuple[str, ...], where: str) -> None:
    actual = tuple(value)
    if actual != expected:
        raise TargetValidationError(f"{where} keys must be {expected}, got {actual}")


def _caption(value: Any, where: str) -> str:
    if not isinstance(value, str) or not is_valid_english_caption(value):
        raise TargetValidationError(f"{where} must be non-empty English")
    return value


def active_field(profile: str) -> str:
    if profile not in PROFILE_FIELDS:
        raise TargetValidationError(f"unknown profile: {profile!r}")
    return LEVEL_TO_FIELD[UNIT_LEVEL[profile]]


def prediction_one_state(target: Mapping[str, Any], profile: str) -> dict[str, Any]:
    validate_target(target, profile)
    value = target["predictions"][0][active_field(profile)]
    return {
        "caption": str(value["caption"]),
        "progress_percent": int(value["progress_percent"]),
    }


def is_terminal_prediction(target: Mapping[str, Any], profile: str) -> bool:
    predictions = target.get("predictions") if isinstance(target, Mapping) else None
    if not isinstance(predictions, list) or len(predictions) < 2:
        return False
    field = active_field(profile)
    value = predictions[1].get(field) if isinstance(predictions[1], Mapping) else None
    return bool(
        isinstance(value, Mapping)
        and value.get("caption") == TERMINAL_CAPTION
    )


def validate_continuous_target(
    target: Any,
    profile: str,
    *,
    is_terminal_window: bool,
    terminal_caption: str = TERMINAL_CAPTION,
) -> dict[str, Any]:
    validated = validate_target(target, profile, expected_prediction_count=2)
    terminal = is_terminal_prediction(validated, profile)
    if is_terminal_window != terminal:
        raise TargetValidationError(
            "terminal metadata and active Prediction 2 caption disagree"
        )
    if is_terminal_window:
        second = validated["predictions"][1]
        for field in PROFILE_FIELDS[profile]:
            value = second[field]
            if value["caption"] != terminal_caption:
                raise TargetValidationError(
                    f"terminal predictions[1].{field}.caption must be {terminal_caption!r}"
                )
            if value["progress_percent"] != 0:
                raise TargetValidationError(
                    f"terminal predictions[1].{field}.progress_percent must be 0"
                )
    else:
        for prediction in validated["predictions"]:
            for field in PROFILE_FIELDS[profile]:
                if prediction[field]["caption"] == terminal_caption:
                    raise TargetValidationError("non-terminal target contains terminal caption")
    return validated


def _plan_level_object(
    value: Any,
    *,
    level: str,
    remaining_levels: tuple[str, ...],
    where: str,
) -> None:
    if not isinstance(value, Mapping):
        raise TargetValidationError(f"{where} must be an object")
    keys: list[str] = ["level"]
    if level == "L0":
        keys.append("source")
    keys.append("caption")
    if remaining_levels:
        keys.append(_CHILD_LIST[remaining_levels[0]])
    _exact_keys(value, tuple(keys), where)
    if value["level"] != level:
        raise TargetValidationError(f"{where}.level must be {level!r}")
    if level == "L0" and value["source"] not in {"human_segment", "segment"}:
        raise TargetValidationError(f"{where}.source is invalid")
    _caption(value["caption"], f"{where}.caption")
    if not remaining_levels:
        return
    child_level = remaining_levels[0]
    child_field = LEVEL_TO_FIELD[child_level]
    child_key = _CHILD_LIST[child_level]
    children = value[child_key]
    if not isinstance(children, list) or not children:
        raise TargetValidationError(f"{where}.{child_key} must be a non-empty list")
    for position, child in enumerate(children, 1):
        child_where = f"{where}.{child_key}[{position - 1}]"
        if not isinstance(child, Mapping):
            raise TargetValidationError(f"{child_where} must be an object")
        _exact_keys(child, ("index", child_field), child_where)
        if child["index"] != position:
            raise TargetValidationError(f"{child_where}.index must be {position}")
        _plan_level_object(
            child[child_field],
            level=child_level,
            remaining_levels=remaining_levels[1:],
            where=f"{child_where}.{child_field}",
        )


def validate_initial_plan(
    target: Any,
    profile: str,
    *,
    expected_top_level_units: int | None = None,
) -> dict[str, Any]:
    if profile not in PROFILE_FIELDS:
        raise TargetValidationError(f"unknown profile: {profile!r}")
    if not isinstance(target, Mapping):
        raise TargetValidationError("initial plan target must be an object")
    _exact_keys(target, ("task", "initial_plan"), "initial plan target")
    task = target["task"]
    if not isinstance(task, Mapping):
        raise TargetValidationError("task must be an object")
    _exact_keys(task, ("level", "caption"), "task")
    if task["level"] != "L3":
        raise TargetValidationError("task.level must be 'L3'")
    _caption(task["caption"], "task.caption")

    plan = target["initial_plan"]
    if not isinstance(plan, list) or not plan:
        raise TargetValidationError("initial_plan must be a non-empty list")
    if expected_top_level_units is not None and len(plan) != expected_top_level_units:
        raise TargetValidationError(
            f"expected {expected_top_level_units} top-level plan units, got {len(plan)}"
        )
    levels = PROFILE_LEVELS[profile]
    top_level = levels[0]
    top_field = LEVEL_TO_FIELD[top_level]
    for position, entry in enumerate(plan, 1):
        where = f"initial_plan[{position - 1}]"
        if not isinstance(entry, Mapping):
            raise TargetValidationError(f"{where} must be an object")
        _exact_keys(entry, ("index", top_field), where)
        if entry["index"] != position:
            raise TargetValidationError(f"{where}.index must be {position}")
        _plan_level_object(
            entry[top_field],
            level=top_level,
            remaining_levels=levels[1:],
            where=f"{where}.{top_field}",
        )
    return dict(target)


def dumps_assistant(target: Mapping[str, Any], profile: str, task_type: str) -> str:
    if task_type == "initial_plan":
        validated = validate_initial_plan(target, profile)
    elif task_type in {"continuous", "terminal"}:
        validated = validate_target(target, profile, expected_prediction_count=2)
    else:
        raise TargetValidationError(f"unknown V3 task_type: {task_type!r}")
    return json.dumps(validated, ensure_ascii=False, separators=(",", ":"))


def loads_assistant(text: str, profile: str, task_type: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TargetValidationError(f"invalid JSON: {exc}") from exc
    if task_type == "initial_plan":
        return validate_initial_plan(value, profile)
    if task_type in {"continuous", "terminal"}:
        return validate_target(value, profile, expected_prediction_count=2)
    raise TargetValidationError(f"unknown V3 task_type: {task_type!r}")


def validate_short_memory(value: Any) -> tuple[dict[str, Any], ...]:
    if value in (None, (), []):
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TargetValidationError("short_memory must be a sequence")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            item = {"caption": item, "progress_percent": 100}
        if not isinstance(item, Mapping):
            raise TargetValidationError(f"short_memory[{index}] must be an object or string")
        _exact_keys(item, ("caption", "progress_percent"), f"short_memory[{index}]")
        _caption(item["caption"], f"short_memory[{index}].caption")
        progress = item["progress_percent"]
        if isinstance(progress, bool) or not isinstance(progress, int) or not 0 <= progress <= 100:
            raise TargetValidationError(
                f"short_memory[{index}].progress_percent must be an integer in [0, 100]"
            )
        result.append({"caption": item["caption"], "progress_percent": progress})
    return tuple(result)

