"""Strict target JSON contract without a third-party runtime dependency."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .captions import is_valid_english_caption
from .constants import FIELD_TO_LEVEL, PROFILE_FIELDS


class TargetValidationError(ValueError):
    pass


def _require_exact_keys(
    value: Mapping[str, Any],
    keys: tuple[str, ...],
    where: str,
    *,
    enforce_order: bool,
) -> None:
    actual = tuple(value.keys())
    valid = actual == keys if enforce_order else set(actual) == set(keys) and len(actual) == len(keys)
    if not valid:
        raise TargetValidationError(f"{where} keys must be {keys}, got {actual}")


def _progress(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise TargetValidationError(f"{where} must be an integer in [0, 100]")
    return value


def _validate_level_object(
    field: str,
    value: Any,
    future: bool,
    where: str,
    *,
    enforce_key_order: bool,
) -> None:
    if not isinstance(value, Mapping):
        raise TargetValidationError(f"{where} must be an object")
    if field == "l0":
        keys = ("level", "source", "caption", "progress_percent")
    else:
        keys = ("level", "caption", "progress_percent")
    _require_exact_keys(value, keys, where, enforce_order=enforce_key_order)
    if value["level"] != FIELD_TO_LEVEL[field]:
        raise TargetValidationError(
            f"{where}.level must be {FIELD_TO_LEVEL[field]!r}"
        )
    if field == "l0" and value["source"] not in {"human_segment", "segment"}:
        raise TargetValidationError(f"{where}.source is invalid")
    if not is_valid_english_caption(value["caption"]):
        raise TargetValidationError(f"{where}.caption must be non-empty English")
    progress = _progress(value["progress_percent"], f"{where}.progress_percent")
    if future and progress != 0:
        raise TargetValidationError(f"{where} future progress must be 0")


def validate_target(
    target: Any,
    profile: str,
    *,
    expected_prediction_count: int | None = None,
    enforce_key_order: bool = True,
) -> dict[str, Any]:
    if profile not in PROFILE_FIELDS:
        raise TargetValidationError(f"unknown profile: {profile!r}")
    if not isinstance(target, Mapping):
        raise TargetValidationError("target must be an object")
    _require_exact_keys(
        target, ("task", "predictions"), "target", enforce_order=enforce_key_order
    )

    task = target["task"]
    if not isinstance(task, Mapping):
        raise TargetValidationError("task must be an object")
    _require_exact_keys(
        task,
        ("level", "caption", "progress_percent"),
        "task",
        enforce_order=enforce_key_order,
    )
    if task["level"] != "L3":
        raise TargetValidationError("task.level must be 'L3'")
    if not is_valid_english_caption(task["caption"]):
        raise TargetValidationError("task.caption must be non-empty English")
    _progress(task["progress_percent"], "task.progress_percent")

    predictions = target["predictions"]
    if not isinstance(predictions, list) or len(predictions) not in {1, 2}:
        raise TargetValidationError("predictions must contain exactly 1 or 2 objects")
    if expected_prediction_count is not None and len(predictions) != expected_prediction_count:
        raise TargetValidationError(
            f"expected {expected_prediction_count} predictions, got {len(predictions)}"
        )
    fields = PROFILE_FIELDS[profile]
    expected_keys = ("index",) + fields
    for position, prediction in enumerate(predictions, 1):
        where = f"predictions[{position - 1}]"
        if not isinstance(prediction, Mapping):
            raise TargetValidationError(f"{where} must be an object")
        _require_exact_keys(
            prediction, expected_keys, where, enforce_order=enforce_key_order
        )
        if prediction["index"] != position:
            raise TargetValidationError(f"{where}.index must be {position}")
        for field in fields:
            _validate_level_object(
                field,
                prediction[field],
                future=position == 2,
                where=f"{where}.{field}",
                enforce_key_order=enforce_key_order,
            )
    return dict(target)


def canonicalize_target(target: Mapping[str, Any], profile: str) -> dict[str, Any]:
    """Validate stored content and restore the exact Assistant wire-key order."""

    validate_target(target, profile, enforce_key_order=False)
    task = target["task"]
    ordered_predictions = []
    for prediction in target["predictions"]:
        ordered_prediction: dict[str, Any] = {"index": prediction["index"]}
        for field in PROFILE_FIELDS[profile]:
            value = prediction[field]
            ordered_value: dict[str, Any] = {"level": value["level"]}
            if field == "l0":
                ordered_value["source"] = value["source"]
            ordered_value["caption"] = value["caption"]
            ordered_value["progress_percent"] = value["progress_percent"]
            ordered_prediction[field] = ordered_value
        ordered_predictions.append(ordered_prediction)
    return {
        "task": {
            "level": task["level"],
            "caption": task["caption"],
            "progress_percent": task["progress_percent"],
        },
        "predictions": ordered_predictions,
    }


def dumps_target(target: Mapping[str, Any], profile: str) -> str:
    ordered = canonicalize_target(target, profile)
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def loads_target(text: str, profile: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TargetValidationError(f"invalid JSON: {exc}") from exc
    return validate_target(value, profile)


def json_schema_document() -> dict[str, Any]:
    """Publishable JSON Schema; semantic/order checks remain in validate_target."""

    level_defs: dict[str, Any] = {}
    for field, level in FIELD_TO_LEVEL.items():
        properties: dict[str, Any] = {
            "level": {"const": level},
            "caption": {"type": "string", "minLength": 1},
            "progress_percent": {"type": "integer", "minimum": 0, "maximum": 100},
        }
        required = ["level", "caption", "progress_percent"]
        if field == "l0":
            properties = {
                "level": {"const": "L0"},
                "source": {"enum": ["human_segment", "segment"]},
                "caption": properties["caption"],
                "progress_percent": properties["progress_percent"],
            }
            required = ["level", "source", "caption", "progress_percent"]
        level_defs[field] = {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": properties,
        }

    profile_variants = []
    for profile, fields in PROFILE_FIELDS.items():
        properties = {"index": {"type": "integer", "enum": [1, 2]}}
        properties.update({field: {"$ref": f"#/$defs/{field}"} for field in fields})
        profile_variants.append({
            "title": profile,
            "type": "object",
            "additionalProperties": False,
            "required": ["index", *fields],
            "properties": properties,
        })
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "v10-continuous-target.schema.json",
        "type": "object",
        "additionalProperties": False,
        "required": ["task", "predictions"],
        "properties": {
            "task": {
                "type": "object",
                "additionalProperties": False,
                "required": ["level", "caption", "progress_percent"],
                "properties": {
                    "level": {"const": "L3"},
                    "caption": {"type": "string", "minLength": 1},
                    "progress_percent": {"type": "integer", "minimum": 0, "maximum": 100},
                },
            },
            "predictions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": {"oneOf": profile_variants},
            },
        },
        "$defs": level_defs,
    }
