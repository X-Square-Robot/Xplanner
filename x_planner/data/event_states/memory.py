"""Causal V5 initial-plan, long-memory, and short-memory contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .schema import (
    CONTEXT_VARIANTS,
    OUTPUT_UNITS,
    V5ValidationError,
    validate_model_visible_text,
    validate_output_spec,
)


LONG_MEMORY_LIMIT = 8


def _progress(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise V5ValidationError(f"{where} must be an integer in [0, 100]")
    return value


def validate_long_memory(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise V5ValidationError("long_memory must be an array")
    if len(value) > LONG_MEMORY_LIMIT:
        raise V5ValidationError(f"long_memory cannot exceed {LONG_MEMORY_LIMIT} items")
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(value, start=1):
        if not isinstance(raw, Mapping) or not raw or tuple(raw)[0] != "index":
            raise V5ValidationError("long_memory items must start with index")
        unit_names = tuple(raw)[1:]
        if (
            not unit_names
            or len(unit_names) != len(set(unit_names))
            or unit_names != tuple(unit for unit in OUTPUT_UNITS if unit in unit_names)
        ):
            raise V5ValidationError(
                "long_memory items must contain available Action and/or Segment units"
            )
        if raw["index"] != index:
            raise V5ValidationError("long_memory indices must be contiguous")
        units = {
            name: validate_model_visible_text(raw[name], f"long_memory.{name}")
            for name in unit_names
        }
        identity = tuple(units.get(name, "").casefold() for name in OUTPUT_UNITS)
        if identity in seen:
            raise V5ValidationError("long_memory must be ordered and deduplicated")
        seen.add(identity)
        result.append({"index": index, **units})
    return result


def _short_unit(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or tuple(value) != (
        "available", "caption", "progress_percent"
    ):
        raise V5ValidationError(
            f"{where} must use available/caption/progress_percent"
        )
    available = value["available"]
    if not isinstance(available, bool):
        raise V5ValidationError(f"{where}.available must be boolean")
    progress = _progress(value["progress_percent"], f"{where}.progress_percent")
    if not available:
        raise V5ValidationError(
            f"{where} is present, so available must be true; omit unavailable units"
        )
    return {
        "available": True,
        "caption": validate_model_visible_text(value["caption"], f"{where}.caption"),
        "progress_percent": progress,
    }


def validate_short_memory(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or tuple(value) != (
        "task_progress_percent", "prediction1"
    ):
        raise V5ValidationError("short_memory must use task_progress_percent/prediction1")
    prediction = value["prediction1"]
    if not isinstance(prediction, Mapping) or not prediction:
        raise V5ValidationError("short_memory.prediction1 must be a non-empty object")
    unit_names = tuple(prediction)
    if (
        len(unit_names) != len(set(unit_names))
        or unit_names != tuple(unit for unit in OUTPUT_UNITS if unit in unit_names)
    ):
        raise V5ValidationError(
            "short_memory.prediction1 must use available Action and/or Segment units"
        )
    return {
        "task_progress_percent": _progress(
            value["task_progress_percent"], "short_memory.task_progress_percent"
        ),
        "prediction1": {
            name: _short_unit(
                prediction[name], f"short_memory.prediction1.{name}"
            )
            for name in unit_names
        },
    }


def validate_prompt_context(
    value: Any,
    *,
    memory_variant: str,
    category: str,
    output_spec: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise V5ValidationError("prompt_context must be an object")
    if memory_variant == "no_memory":
        if value:
            raise V5ValidationError("no_memory prompt_context must be empty")
        return {}
    if memory_variant != "with_memory" or category not in {"ongoing", "end"}:
        raise V5ValidationError("with_memory is valid only for ongoing/end")
    if tuple(value) != ("initial_plan_memory", "long_memory", "short_memory"):
        raise V5ValidationError(
            "with_memory context must use initial_plan_memory/long_memory/short_memory"
        )
    spec = validate_output_spec(output_spec, category=category)
    initial_plan = value["initial_plan_memory"]
    if not isinstance(initial_plan, list) or not initial_plan:
        raise V5ValidationError("initial_plan_memory must be a non-empty validated plan")
    # Plan nodes are validated with the same public target contract by schema;
    # this module deliberately avoids importing that private recursive helper.
    from .schema import validate_target

    plan_spec = {
        "prediction1_units": [],
        "prediction2_units": [],
        "plan_units": spec["plan_units"],
    }
    validated_plan = validate_target(
        {"initial_plan": initial_plan}, "initial_plan", plan_spec
    )
    return {
        "initial_plan_memory": validated_plan["initial_plan"],
        "long_memory": validate_long_memory(value["long_memory"]),
        "short_memory": validate_short_memory(value["short_memory"]),
    }


def validate_prompt_context_variants(
    value: Any,
    *,
    context_variant: str,
    category: str,
    output_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the five explicit V5.3 inference-context contracts.

    Noise is deliberately validated from offline provenance by ``schema``;
    model-visible context keeps the same wire shape as its clean counterpart.
    """

    if not isinstance(value, Mapping):
        raise V5ValidationError("prompt_context must be an object")
    if context_variant not in CONTEXT_VARIANTS:
        raise V5ValidationError(f"unknown context_variant: {context_variant!r}")
    if context_variant == "no_memory_no_initial":
        if value:
            raise V5ValidationError("no_memory_no_initial prompt_context must be empty")
        return {}
    if category == "initial_plan":
        raise V5ValidationError("initial_plan cannot contain execution memory")

    includes_initial = context_variant in {
        "with_memory_with_initial",
        "with_memory_with_initial_noisy",
    }
    expected = (
        ("initial_plan_memory", "long_memory", "short_memory")
        if includes_initial
        else ("long_memory", "short_memory")
    )
    if tuple(value) != expected:
        raise V5ValidationError(
            f"{context_variant} context must use {'/'.join(expected)}"
        )
    result: dict[str, Any] = {}
    if includes_initial:
        initial_plan = value["initial_plan_memory"]
        if not isinstance(initial_plan, list) or not initial_plan:
            raise V5ValidationError("initial_plan_memory must be a non-empty plan")
        from .schema import validate_target

        spec = validate_output_spec(output_spec, category=category)
        plan_spec = {
            "prediction1_units": [],
            "prediction2_units": [],
            "plan_units": spec["plan_units"],
        }
        result["initial_plan_memory"] = validate_target(
            {"initial_plan": initial_plan}, "initial_plan", plan_spec
        )["initial_plan"]
    result["long_memory"] = validate_long_memory(value["long_memory"])
    result["short_memory"] = validate_short_memory(value["short_memory"])
    return result


__all__ = [
    "LONG_MEMORY_LIMIT",
    "validate_long_memory",
    "validate_prompt_context",
    "validate_prompt_context_variants",
    "validate_short_memory",
]
