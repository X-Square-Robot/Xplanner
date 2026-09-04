"""Single-source V5 prompt rendering with optional causal memory context."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .memory_v5 import validate_prompt_context
from .schema_v5 import (
    PROMPT_VERSION,
    PROMPT_VERSION_V53,
    SCHEMA_VERSION_V53,
    V5ValidationError,
    validate_model_visible_text,
    validate_output_spec,
)


INITIAL_PLAN_INSTRUCTION = (
    "Use the synchronized robot observations and task instruction to produce one "
    "complete ordered plan as JSON. {unit_requirement} Return only the JSON object."
)

EXECUTION_INSTRUCTION = (
    "Use the synchronized robot observations, task instruction, and any supplied "
    "causal memory to assess the current execution. Requested output profile: "
    "{profile_requirement}. Choose execution_decision from Continue, Replan, Takeover, "
    "or End. If the decision is Takeover, return exactly execution_decision and "
    "decision_detail; decision_detail must contain evidence-grounded failure_analysis "
    "and a machine-executable recovery_plan, and no progress or predictions. Otherwise, "
    "return task_progress_percent, exactly two predictions named by index and role, "
    "execution_decision, and decision_detail. Prediction 1 is current and Prediction 2 "
    "is next. Omit every Action or Segment key that is not requested. Return only one "
    "JSON object."
)

EXECUTION_INSTRUCTION_V53 = (
    "Use the synchronized robot observations, task instruction, and any supplied "
    "causal memory to assess the current execution. Requested output profile: "
    "{profile_requirement}. Choose execution_decision from Continue, Replan, Takeover, "
    "or End. For Replan, return exactly execution_decision and decision_detail with an "
    "evidence-grounded reason and machine-executable updated_plan. For Takeover, return "
    "exactly execution_decision and decision_detail with evidence-grounded "
    "failure_analysis and a machine-executable recovery_plan. For Continue or End, "
    "return task_progress_percent, exactly two predictions named by index and role, "
    "execution_decision, and decision_detail. Prediction 1 is current and Prediction 2 "
    "is next. Omit every Action or Segment key that is not requested. Return only one "
    "JSON object."
)


def _unit_phrase(units: list[str]) -> str:
    names = [name.capitalize() for name in units]
    if not names:
        return "index and role only"
    if len(names) == 1:
        return f"{names[0]} only"
    return " and ".join(names)


def initial_plan_instruction(output_spec: dict[str, Any]) -> str:
    spec = validate_output_spec(output_spec, category="initial_plan")
    requirement = (
        "The plan must contain ordered Actions only; do not output Segment keys."
        if spec["plan_units"] == ["action"]
        else "The plan must contain ordered Actions and ordered Segments for each Action."
    )
    return INITIAL_PLAN_INSTRUCTION.format(unit_requirement=requirement)


def execution_instruction(output_spec: dict[str, Any]) -> str:
    spec = validate_output_spec(output_spec)
    profile = (
        f"Prediction 1 {_unit_phrase(spec['prediction1_units'])}; "
        f"Prediction 2 {_unit_phrase(spec['prediction2_units'])}; "
        f"plan {_unit_phrase(spec['plan_units'])}"
    )
    return EXECUTION_INSTRUCTION.format(profile_requirement=profile)


def execution_instruction_v53(output_spec: dict[str, Any]) -> str:
    spec = validate_output_spec(output_spec)
    profile = (
        f"Prediction 1 {_unit_phrase(spec['prediction1_units'])}; "
        f"Prediction 2 {_unit_phrase(spec['prediction2_units'])}; "
        f"decision plan {_unit_phrase(spec['plan_units'])}"
    )
    return EXECUTION_INSTRUCTION_V53.format(profile_requirement=profile)


def prompt_renderer_digest() -> str:
    payload = f"{PROMPT_VERSION}\n{INITIAL_PLAN_INSTRUCTION}\n{EXECUTION_INSTRUCTION}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prompt_renderer_digest_v53() -> str:
    payload = (
        f"{PROMPT_VERSION_V53}\n{INITIAL_PLAN_INSTRUCTION}\n"
        f"{EXECUTION_INSTRUCTION_V53}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def render_user(sample: dict[str, Any]) -> str:
    category = str(sample.get("category", ""))
    is_v53 = sample.get("schema_version") == SCHEMA_VERSION_V53
    memory_variant = str(sample.get("memory_variant", ""))
    context_variant = str(sample.get("context_variant", ""))
    images = sample.get("images")
    if not isinstance(images, list) or not images:
        raise V5ValidationError("render_user requires at least one image")
    instruction = validate_model_visible_text(
        sample.get("task_instruction"), "task_instruction"
    )
    if is_v53:
        from .memory_v5 import validate_prompt_context_v53

        context = validate_prompt_context_v53(
            sample.get("prompt_context"),
            context_variant=context_variant,
            category=category,
            output_spec=sample.get("output_spec"),
        )
    else:
        context = validate_prompt_context(
            sample.get("prompt_context"),
            memory_variant=memory_variant,
            category=category,
            output_spec=sample.get("output_spec"),
        )
    parts = ["<image>" * len(images), f"Task instruction: {instruction}"]
    if category == "initial_plan":
        if (is_v53 and context_variant != "no_memory_no_initial") or (
            not is_v53 and memory_variant != "no_memory"
        ):
            raise V5ValidationError("initial_plan prompt must not contain memory")
        parts.append(initial_plan_instruction(sample["output_spec"]))
        return "\n".join(parts)
    if category not in {"ongoing", "end", "takeover", "replan"}:
        raise V5ValidationError(f"unknown execution category: {category!r}")
    if context:
        if "initial_plan_memory" in context:
            parts.append(
                "Initial plan memory: " + _compact(context["initial_plan_memory"])
            )
        parts.extend([
            "Long memory: " + _compact(context["long_memory"]),
            "Short memory: " + _compact(context["short_memory"]),
        ])
    parts.append(
        execution_instruction_v53(sample["output_spec"])
        if is_v53
        else execution_instruction(sample["output_spec"])
    )
    prompt = "\n".join(parts)
    # The category and target decision never participate in rendering.  This is
    # intentionally one renderer for normal execution, failure, and completion.
    return prompt


def normalized_execution_instruction(prompt: str) -> str:
    """Return the profile-specific canonical suffix used by equality audits."""
    marker = prompt.rsplit("\n", 1)[-1]
    if not marker.startswith(
        "Use the synchronized robot observations, task instruction, and any supplied "
    ) or "Requested output profile:" not in marker:
        raise V5ValidationError("execution prompt does not use a canonical V5 suffix")
    return marker


__all__ = [
    "EXECUTION_INSTRUCTION",
    "EXECUTION_INSTRUCTION_V53",
    "INITIAL_PLAN_INSTRUCTION",
    "execution_instruction",
    "execution_instruction_v53",
    "initial_plan_instruction",
    "normalized_execution_instruction",
    "prompt_renderer_digest",
    "prompt_renderer_digest_v53",
    "render_user",
]
