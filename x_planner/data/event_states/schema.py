"""Strict V5 model-visible and offline-supervision contracts.

The model sees only Task, Action, and Segment semantics.  Data availability is
kept in offline ``supervision.loss_mask_paths`` metadata and is enforced by the
V5 epilogue rather than being learned from placeholder values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import re
from typing import Any


SCHEMA_VERSION = "v10_action_segment_v5_2"
SCHEMA_VERSION_V53 = "v10_action_segment_v5_3"
SNAPSHOT_SCHEMA_VERSION = "v10_action_segment_v5_snapshot_v3"
SNAPSHOT_SCHEMA_VERSION_V53 = "v10_action_segment_v5_snapshot_v4"
PROMPT_VERSION = "v10_action_segment_v5_prompt_v2"
PROMPT_VERSION_V53 = "v10_action_segment_v5_prompt"

CATEGORIES = ("initial_plan", "ongoing", "end", "takeover", "replan")
LEGACY_CATEGORIES = ("initial_plan", "ongoing", "end", "takeover")
MEMORY_VARIANTS = ("no_memory", "with_memory")
CONTEXT_VARIANTS = (
    "no_memory_no_initial",
    "with_memory_no_initial",
    "with_memory_with_initial",
    "with_memory_no_initial_noisy",
    "with_memory_with_initial_noisy",
)
NOISY_CONTEXT_VARIANTS = (
    "with_memory_no_initial_noisy",
    "with_memory_with_initial_noisy",
)
TRAINING_BUCKETS = (
    "initial_plan",
    "ongoing",
    "end",
    "robodojo",
    "takeover",
    "replan_self",
    "replan_open",
)
EXECUTION_DECISIONS = ("Continue", "Replan", "Takeover", "End")
OUTPUT_UNITS = ("action", "segment")

FAILURE_TYPE_BY_SOURCE_CODE = {
    "1.1": "Arm remains stationary instead of approaching the object",
    "1.2": "Arm remains stationary above the object without grasping",
    "1.3": "Arm remains stationary after grasping the object",
    "1.4": "Arm moves back and forth without manipulating the object",
    "1.5": "Arm does not retract after completing the task",
    "2.1": "Wrong object is grasped",
    "2.2": "An already placed object is grasped",
    "3.1": "Operations are executed in the wrong order",
    "4.1": "The object is not grasped",
    "4.2": "The object is released unexpectedly after grasping",
    "4.3": "Repeated grasp attempts fail",
    "5.1": "The object is placed at the wrong location",
    "6.1": "The object is dropped during transport",
    "7.1": "The gripper cannot release the object",
    "8.1": "The arm collides with another object during motion",
}
FAILURE_TYPES = tuple(FAILURE_TYPE_BY_SOURCE_CODE.values())

_FORBIDDEN_MODEL_TERMS = re.compile(
    r"(?i)(?:\bsubtask\b|\bL[0-3]\b|\bprofile\b)"
)
_RAW_FAILURE_CODE = re.compile(r"(?<!\d)(?:[1-8]\.[1-5])(?!\d)")
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_MEASUREMENT_AFTER_DECIMAL = re.compile(
    r"(?ix)^\s*-?\s*(?:(?:"
    r"milliseconds?|seconds?|minutes?|hours?|ms|"
    r"milligrams?|grams?|kilograms?|mg|g|kg|"
    r"millimeters?|centimeters?|meters?|millimetres?|centimetres?|metres?|mm|cm|m|"
    r"milliliters?|liters?|millilitres?|litres?|ml|l|"
    r"degrees?|radians?|rad|turns?|circles?|rotations?|times?|x|"
    r"percent|fps|hz|newtons?|n"
    r")\b|%)"
)

_TASK_INSTRUCTION_SOURCES = {
    "baseline_v2v3umi": {
        "label_annotation.task_caption",
        "label_annotation.instruction",
        "media_task_instruction_json.episode.task_caption",
        "media_task_instruction_json.episode.instruction",
        "media_task_instruction_json.root.task_caption",
        "media_task_instruction_json.root.instruction",
        "media_episode_instruction_json.episode.task_caption",
        "media_episode_instruction_json.episode.instruction",
        "media_episode_instruction_json.root.task_caption",
        "media_episode_instruction_json.root.instruction",
    },
    "robodojo": {
        "media_instruction_json.episode.task_caption",
        "media_instruction_json.episode.instruction",
    },
    "takeover_q": {
        "media_task_instruction_json.episode.detailed_instruction",
        "media_task_instruction_json.episode.instruction",
        "media_episode_instruction_json.episode.detailed_instruction",
        "media_episode_instruction_json.episode.instruction",
    },
    "replan_self": {
        "samples_json.task_caption",
        "samples_json.instruction",
    },
    "replan_open": {
        "samples_json.task_caption",
        "samples_json.instruction",
    },
}
_TASK_INSTRUCTION_PATH_REQUIRED = {
    "baseline_v2v3umi",
    "robodojo",
    "takeover_q",
    "replan_self",
    "replan_open",
}


class V5ValidationError(ValueError):
    """Raised when a V5 sample violates a wire or supervision invariant."""


def _exact_keys(value: Mapping[str, Any], expected: tuple[str, ...], where: str) -> None:
    actual = tuple(value)
    if actual != expected:
        raise V5ValidationError(f"{where} keys must be {expected}, got {actual}")


def _english(value: Any, where: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise V5ValidationError(f"{where} must be a string")
    if not value.strip():
        if allow_empty:
            return ""
        raise V5ValidationError(f"{where} must be non-empty")
    if not re.search(r"[A-Za-z]", value):
        raise V5ValidationError(f"{where} must contain English text")
    if re.search(r"[\u3400-\u9fff]", value):
        raise V5ValidationError(f"{where} must not contain CJK text")
    if _FORBIDDEN_MODEL_TERMS.search(value):
        raise V5ValidationError(f"{where} contains a forbidden legacy term")
    return value.strip()


def validate_model_visible_text(value: str, where: str = "model-visible text") -> str:
    value = _english(value, where)
    for match in _RAW_FAILURE_CODE.finditer(value):
        # The failure taxonomy uses decimal-looking source identifiers, but
        # robot captions also legitimately contain quantities such as
        # ``1.5 kg``, ``1.5 circles`` and ``4.2-second``.  Only a matched token
        # without an immediately following physical/count unit is treated as a
        # raw taxonomy code.  This preserves the user's no-code contract
        # without deleting valid motion supervision.
        if not _MEASUREMENT_AFTER_DECIMAL.match(value[match.end():]):
            raise V5ValidationError(f"{where} contains a raw failure code")
    return value


def _progress(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 100:
        raise V5ValidationError(f"{where} must be an integer in [0, 100]")
    return value


def _validate_unit(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise V5ValidationError(f"{where} must be an object")
    _exact_keys(value, ("available", "caption", "progress_percent"), where)
    available = value["available"]
    if not isinstance(available, bool):
        raise V5ValidationError(f"{where}.available must be boolean")
    caption = value["caption"]
    progress = _progress(value["progress_percent"], f"{where}.progress_percent")
    if not available:
        raise V5ValidationError(
            f"{where} is present, so available must be true; omit unavailable units"
        )
    caption = validate_model_visible_text(caption, f"{where}.caption")
    return {
        "available": available,
        "caption": caption,
        "progress_percent": progress,
    }


def _validate_units(value: Any, where: str, *, allow_empty: bool) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise V5ValidationError(f"{where} must be an array")
    result = [str(unit) for unit in value]
    if not allow_empty and not result:
        raise V5ValidationError(f"{where} must not be empty")
    if len(result) != len(set(result)) or any(unit not in OUTPUT_UNITS for unit in result):
        raise V5ValidationError(f"{where} contains an invalid or duplicate unit")
    canonical = [unit for unit in OUTPUT_UNITS if unit in result]
    if result != canonical:
        raise V5ValidationError(f"{where} must use canonical action/segment order")
    return result


def validate_output_spec(value: Any, *, category: str | None = None) -> dict[str, list[str]]:
    if not isinstance(value, Mapping):
        raise V5ValidationError("output_spec must be an object")
    _exact_keys(
        value,
        ("prediction1_units", "prediction2_units", "plan_units"),
        "output_spec",
    )
    result = {
        "prediction1_units": _validate_units(
            value["prediction1_units"], "output_spec.prediction1_units", allow_empty=True
        ),
        "prediction2_units": _validate_units(
            value["prediction2_units"], "output_spec.prediction2_units", allow_empty=True
        ),
        "plan_units": _validate_units(
            value["plan_units"], "output_spec.plan_units", allow_empty=True
        ),
    }
    if category == "initial_plan":
        if result["prediction1_units"] or result["prediction2_units"]:
            raise V5ValidationError("initial_plan output_spec cannot request predictions")
        if "action" not in result["plan_units"]:
            raise V5ValidationError("initial_plan output_spec requires Action")
    elif category in {"ongoing", "end", "takeover", "replan"}:
        if not result["prediction1_units"]:
            raise V5ValidationError(f"{category} output_spec requires Prediction 1 units")
        if result["plan_units"] not in (["action"], ["action", "segment"]):
            raise V5ValidationError(f"{category} output_spec requires an Action plan profile")
    return result


def output_profile_id(output_spec: Mapping[str, Any]) -> str:
    value = validate_output_spec(output_spec)

    def units(name: str) -> str:
        selected = value[name]
        return "-".join(selected) if selected else "none"

    return (
        f"p1-{units('prediction1_units')}__"
        f"p2-{units('prediction2_units')}__"
        f"plan-{units('plan_units')}"
    )


def _validate_predictions(
    value: Any,
    output_spec: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise V5ValidationError("predictions must contain exactly two items")
    result: list[dict[str, Any]] = []
    for offset, raw in enumerate(value, start=1):
        where = f"predictions[{offset - 1}]"
        if not isinstance(raw, Mapping):
            raise V5ValidationError(f"{where} must be an object")
        requested = tuple(output_spec[f"prediction{offset}_units"])
        _exact_keys(raw, ("index", "role", *requested), where)
        role = "current" if offset == 1 else "next"
        if raw["index"] != offset or raw["role"] != role:
            raise V5ValidationError(f"{where} must use index={offset}, role={role!r}")
        units = {
            name: _validate_unit(raw[name], f"{where}.{name}")
            for name in requested
        }
        if offset == 2:
            for name, unit in units.items():
                if unit["progress_percent"] != 0:
                    raise V5ValidationError(f"Prediction 2 {name} progress must be 0")
        result.append({"index": offset, "role": role, **units})
    return result


def _validate_plan(
    value: Any,
    plan_units: Sequence[str],
    where: str = "initial_plan",
) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise V5ValidationError(f"{where} must be a non-empty array")
    result: list[dict[str, Any]] = []
    for action_index, raw in enumerate(value, start=1):
        item_where = f"{where}[{action_index - 1}]"
        if not isinstance(raw, Mapping):
            raise V5ValidationError(f"{item_where} must be an object")
        _exact_keys(raw, ("index", "action"), item_where)
        if raw["index"] != action_index:
            raise V5ValidationError(f"{item_where}.index must be {action_index}")
        action = raw["action"]
        if not isinstance(action, Mapping):
            raise V5ValidationError(f"{item_where}.action must be an object")
        expected_action_keys = (
            ("caption", "segments") if "segment" in plan_units else ("caption",)
        )
        _exact_keys(action, expected_action_keys, f"{item_where}.action")
        caption = validate_model_visible_text(
            action["caption"], f"{item_where}.action.caption"
        )
        if "segment" not in plan_units:
            result.append({"index": action_index, "action": {"caption": caption}})
            continue
        segments_raw = action["segments"]
        if isinstance(segments_raw, (str, bytes)) or not isinstance(segments_raw, Sequence):
            raise V5ValidationError(f"{item_where}.action.segments must be an array")
        segments: list[dict[str, Any]] = []
        for segment_index, segment_raw in enumerate(segments_raw, start=1):
            segment_where = f"{item_where}.action.segments[{segment_index - 1}]"
            if not isinstance(segment_raw, Mapping):
                raise V5ValidationError(f"{segment_where} must be an object")
            _exact_keys(segment_raw, ("index", "segment"), segment_where)
            if segment_raw["index"] != segment_index:
                raise V5ValidationError(f"{segment_where}.index must be {segment_index}")
            segment = segment_raw["segment"]
            if not isinstance(segment, Mapping):
                raise V5ValidationError(f"{segment_where}.segment must be an object")
            _exact_keys(segment, ("caption",), f"{segment_where}.segment")
            segments.append({
                "index": segment_index,
                "segment": {
                    "caption": validate_model_visible_text(
                        segment["caption"], f"{segment_where}.segment.caption"
                    )
                },
            })
        result.append({"index": action_index, "action": {"caption": caption, "segments": segments}})
    return result


def _validate_failure_analysis(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise V5ValidationError("failure_analysis must be an object")
    expected = (
        "failed_action_context",
        "expected_action",
        "observed_failure",
        "failure_type",
    )
    _exact_keys(value, expected, "failure_analysis")
    result = {
        key: validate_model_visible_text(value[key], f"failure_analysis.{key}")
        for key in expected
    }
    if result["failure_type"] not in FAILURE_TYPES:
        raise V5ValidationError("failure_analysis.failure_type is not in the 15-class taxonomy")
    return result


def _validate_decision_detail(
    value: Any,
    decision: str,
    plan_units: Sequence[str],
) -> Any:
    if decision == "Continue":
        if value is not None:
            raise V5ValidationError("Continue decision_detail must be null")
        return None
    if not isinstance(value, Mapping):
        raise V5ValidationError(f"{decision} decision_detail must be an object")
    if decision == "Replan":
        _exact_keys(value, ("reason", "updated_plan"), "Replan decision_detail")
        return {
            "reason": validate_model_visible_text(value["reason"], "Replan reason"),
            "updated_plan": _validate_plan(value["updated_plan"], plan_units, "updated_plan"),
        }
    if decision == "Takeover":
        _exact_keys(value, ("failure_analysis", "recovery_plan"), "Takeover decision_detail")
        return {
            "failure_analysis": _validate_failure_analysis(value["failure_analysis"]),
            "recovery_plan": _validate_plan(value["recovery_plan"], plan_units, "recovery_plan"),
        }
    if decision == "End":
        _exact_keys(value, ("outcome",), "End decision_detail")
        if value["outcome"] not in {"completed", "terminated"}:
            raise V5ValidationError("End outcome must be completed or terminated")
        return {"outcome": value["outcome"]}
    raise V5ValidationError(f"unknown decision: {decision!r}")


def validate_target(
    target: Any,
    category: str,
    output_spec: Mapping[str, Any],
) -> dict[str, Any]:
    if category not in CATEGORIES:
        raise V5ValidationError(f"unknown category: {category!r}")
    spec = validate_output_spec(output_spec, category=category)
    if not isinstance(target, Mapping):
        raise V5ValidationError("target must be an object")
    if category == "initial_plan":
        _exact_keys(target, ("initial_plan",), "initial_plan target")
        return {"initial_plan": _validate_plan(target["initial_plan"], spec["plan_units"])}
    if category in {"takeover", "replan"}:
        expected_decision = "Takeover" if category == "takeover" else "Replan"
        _exact_keys(
            target,
            ("execution_decision", "decision_detail"),
            f"{category} target",
        )
        if target["execution_decision"] != expected_decision:
            raise V5ValidationError(
                f"{category} target must use {expected_decision} decision"
            )
        return {
            "execution_decision": expected_decision,
            "decision_detail": _validate_decision_detail(
                target["decision_detail"], expected_decision, spec["plan_units"]
            ),
        }
    _exact_keys(
        target,
        ("task_progress_percent", "predictions", "execution_decision", "decision_detail"),
        f"{category} target",
    )
    decision = target["execution_decision"]
    if decision not in EXECUTION_DECISIONS:
        raise V5ValidationError(f"invalid execution_decision: {decision!r}")
    allowed = {
        "ongoing": {"Continue", "Replan"},
        "takeover": {"Takeover"},
        "end": {"End"},
    }[category]
    if decision not in allowed:
        raise V5ValidationError(f"{category} cannot use decision {decision}")
    return {
        "task_progress_percent": _progress(
            target["task_progress_percent"], "task_progress_percent"
        ),
        "predictions": _validate_predictions(target["predictions"], spec),
        "execution_decision": decision,
        "decision_detail": _validate_decision_detail(
            target["decision_detail"], decision, spec["plan_units"]
        ),
    }


def parse_json_pointer(pointer: str) -> tuple[str, ...]:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise V5ValidationError(f"invalid JSON pointer: {pointer!r}")
    if pointer == "/":
        return ("",)
    return tuple(part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/"))


def _pointer(path: tuple[str, ...]) -> str:
    return "/" + "/".join(part.replace("~", "~0").replace("/", "~1") for part in path)


def dumps_with_mask_spans(
    target: Mapping[str, Any],
    category: str,
    output_spec: Mapping[str, Any],
    loss_mask_paths: Sequence[str] = (),
) -> tuple[str, list[tuple[int, int, str]]]:
    """Validate and compact-serialize a target while tracking masked values."""
    validated = validate_target(target, category, output_spec)
    requested = {parse_json_pointer(value) for value in loss_mask_paths}
    if len(requested) != len(tuple(loss_mask_paths)):
        raise V5ValidationError("loss_mask_paths must be unique")
    for left in requested:
        for right in requested:
            if left != right and len(left) < len(right) and right[: len(left)] == left:
                raise V5ValidationError("loss_mask_paths may not overlap by ancestry")

    chunks: list[str] = []
    size = 0
    found: dict[tuple[str, ...], tuple[int, int]] = {}

    def emit(text: str) -> None:
        nonlocal size
        chunks.append(text)
        size += len(text)

    def write(value: Any, path: tuple[str, ...]) -> None:
        start = size
        if isinstance(value, Mapping):
            emit("{")
            for index, (key, child) in enumerate(value.items()):
                if index:
                    emit(",")
                emit(json.dumps(str(key), ensure_ascii=False, separators=(",", ":")))
                emit(":")
                write(child, path + (str(key),))
            emit("}")
        elif isinstance(value, list):
            emit("[")
            for index, child in enumerate(value):
                if index:
                    emit(",")
                write(child, path + (str(index),))
            emit("]")
        else:
            emit(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        if path in requested:
            found[path] = (start, size)

    write(validated, ())
    missing = requested - set(found)
    if missing:
        raise V5ValidationError(
            "loss_mask_paths do not exist in target: " + ", ".join(sorted(_pointer(p) for p in missing))
        )
    spans = [(start, end, _pointer(path)) for path, (start, end) in found.items()]
    spans.sort()
    return "".join(chunks), spans


def dumps_assistant(
    target: Mapping[str, Any], category: str, output_spec: Mapping[str, Any]
) -> str:
    return dumps_with_mask_spans(target, category, output_spec)[0]


def _validate_image_reference(value: Any, index: int) -> None:
    if isinstance(value, str):
        if value.strip():
            return
    elif isinstance(value, Mapping):
        references = [
            value[key]
            for key in ("video", "path")
            if key in value
        ]
        if any(isinstance(reference, str) and reference.strip() for reference in references):
            return
    raise V5ValidationError(
        f"images[{index}] must be a non-empty string or an object with "
        "a non-empty video/path reference"
    )


def _validate_sample_v52(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable V5.2 wire format without rewriting old snapshots."""

    if not isinstance(sample, Mapping):
        raise V5ValidationError("sample must be an object")
    required = (
        "schema_version",
        "sample_id",
        "base_sample_id",
        "source",
        "category",
        "memory_variant",
        "output_spec",
        "output_profile_id",
        "task_instruction",
        "images",
        "prompt_context",
        "target",
        "supervision",
        "provenance",
    )
    _exact_keys(sample, required, "sample")
    if sample["schema_version"] != SCHEMA_VERSION:
        raise V5ValidationError("sample schema_version mismatch")
    category = sample["category"]
    memory_variant = sample["memory_variant"]
    if category not in LEGACY_CATEGORIES or memory_variant not in MEMORY_VARIANTS:
        raise V5ValidationError("invalid category or memory_variant")
    spec = validate_output_spec(sample["output_spec"], category=category)
    if sample["output_profile_id"] != output_profile_id(spec):
        raise V5ValidationError("sample output_profile_id mismatch")
    if category in {"initial_plan", "takeover"} and memory_variant != "no_memory":
        raise V5ValidationError(f"{category} must use no_memory")
    validate_model_visible_text(sample["task_instruction"], "task_instruction")
    if not isinstance(sample["images"], list) or not sample["images"]:
        raise V5ValidationError("images must be a non-empty array")
    for index, value in enumerate(sample["images"]):
        _validate_image_reference(value, index)
    # Local import avoids the schema/memory module cycle while keeping sample
    # validation authoritative for both model targets and causal prompt state.
    from .memory import validate_prompt_context

    validate_prompt_context(
        sample["prompt_context"],
        memory_variant=memory_variant,
        category=category,
        output_spec=spec,
    )
    supervision = sample["supervision"]
    if not isinstance(supervision, Mapping):
        raise V5ValidationError("supervision must be an object")
    _exact_keys(supervision, ("loss_mask_paths",), "supervision")
    paths = supervision["loss_mask_paths"]
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise V5ValidationError("loss_mask_paths must be an array")
    dumps_with_mask_spans(sample["target"], category, spec, tuple(paths))
    return dict(sample)


def _validate_context_noise(
    provenance: Mapping[str, Any], context_variant: str
) -> None:
    noise = provenance.get("context_noise")
    if context_variant not in NOISY_CONTEXT_VARIANTS:
        if noise is not None:
            raise V5ValidationError("clean context cannot declare context_noise")
        return
    if not isinstance(noise, Mapping) or noise.get("changed") is not True:
        raise V5ValidationError("noisy context requires changed context_noise metadata")
    seed = noise.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise V5ValidationError("context_noise.seed must be a non-negative integer")
    operations = noise.get("operations")
    if (
        isinstance(operations, (str, bytes))
        or not isinstance(operations, Sequence)
        or not operations
        or any(not isinstance(item, Mapping) or not item.get("kind") for item in operations)
    ):
        raise V5ValidationError("context_noise.operations must contain typed operations")


def _validate_task_instruction_provenance(
    source: str, provenance: Mapping[str, Any]
) -> None:
    """Fail closed for every production dataset that can enter V5.3 training."""

    allowed = _TASK_INSTRUCTION_SOURCES.get(source)
    if allowed is None:
        return
    instruction_source = provenance.get("task_instruction_source")
    if instruction_source not in allowed:
        raise V5ValidationError(
            "untrusted task_instruction_source for "
            f"{source}: {instruction_source!r}; allowed={sorted(allowed)}"
        )
    if source in _TASK_INSTRUCTION_PATH_REQUIRED:
        source_path = provenance.get("task_instruction_source_path")
        if not isinstance(source_path, str) or not source_path.strip():
            raise V5ValidationError(
                f"{source} task instruction must include its source JSON path"
            )
    if source == "takeover_q":
        source_field = provenance.get("task_instruction_source_field")
        if (
            not isinstance(source_field, str)
            or not str(instruction_source).endswith(f".{source_field}")
        ):
            raise V5ValidationError(
                "takeover_q task_instruction_source_field must match source suffix"
            )
        source_sha256 = provenance.get("task_instruction_source_sha256")
        if not isinstance(source_sha256, str) or not _SHA256_HEX.fullmatch(
            source_sha256
        ):
            raise V5ValidationError(
                "takeover_q task instruction must include source JSON sha256"
            )
        checked_paths = provenance.get("task_instruction_checked_paths")
        if (
            isinstance(checked_paths, (str, bytes))
            or not isinstance(checked_paths, Sequence)
            or provenance["task_instruction_source_path"] not in checked_paths
        ):
            raise V5ValidationError(
                "takeover_q selected instruction path must be in checked paths"
            )
        policy = provenance.get("task_instruction_policy")
        if (
            not isinstance(policy, str)
            or not policy.startswith("exact_episode_instruction_json_")
            or not policy.endswith("_v1")
        ):
            raise V5ValidationError(
                "takeover_q task instruction must include exact-entry field policy"
            )


def _validate_sample_v53(sample: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "schema_version",
        "sample_id",
        "base_sample_id",
        "source",
        "training_bucket",
        "category",
        "context_variant",
        "split",
        "output_spec",
        "output_profile_id",
        "task_instruction",
        "images",
        "prompt_context",
        "target",
        "supervision",
        "provenance",
    )
    _exact_keys(sample, required, "V5.3 sample")
    if sample["schema_version"] != SCHEMA_VERSION_V53:
        raise V5ValidationError("V5.3 sample schema_version mismatch")
    category = str(sample["category"])
    bucket = str(sample["training_bucket"])
    context_variant = str(sample["context_variant"])
    split = str(sample["split"])
    if category not in CATEGORIES:
        raise V5ValidationError(f"invalid V5.3 category: {category!r}")
    if bucket not in TRAINING_BUCKETS:
        raise V5ValidationError(f"invalid training_bucket: {bucket!r}")
    if context_variant not in CONTEXT_VARIANTS:
        raise V5ValidationError(f"invalid context_variant: {context_variant!r}")
    if split not in {"train", "test"}:
        raise V5ValidationError("V5.3 split must be train or test")
    expected_categories = {
        "initial_plan": {"initial_plan"},
        "ongoing": {"ongoing"},
        "end": {"end"},
        "robodojo": {"initial_plan", "ongoing", "end"},
        "takeover": {"takeover"},
        "replan_self": {"replan"},
        "replan_open": {"replan"},
    }
    if category not in expected_categories[bucket]:
        raise V5ValidationError(
            f"training_bucket {bucket!r} cannot contain category {category!r}"
        )
    if category == "initial_plan" and context_variant != "no_memory_no_initial":
        raise V5ValidationError("initial_plan must use no_memory_no_initial")
    spec = validate_output_spec(sample["output_spec"], category=category)
    if sample["output_profile_id"] != output_profile_id(spec):
        raise V5ValidationError("V5.3 sample output_profile_id mismatch")
    validate_model_visible_text(sample["task_instruction"], "task_instruction")
    images = sample["images"]
    if not isinstance(images, list) or not images:
        raise V5ValidationError("images must be a non-empty array")
    for index, value in enumerate(images):
        _validate_image_reference(value, index)

    from .memory import validate_prompt_context_variants

    validate_prompt_context_variants(
        sample["prompt_context"],
        context_variant=context_variant,
        category=category,
        output_spec=spec,
    )
    supervision = sample["supervision"]
    if not isinstance(supervision, Mapping):
        raise V5ValidationError("supervision must be an object")
    _exact_keys(supervision, ("loss_mask_paths",), "supervision")
    paths = supervision["loss_mask_paths"]
    if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
        raise V5ValidationError("loss_mask_paths must be an array")
    provenance = sample["provenance"]
    if not isinstance(provenance, Mapping):
        raise V5ValidationError("provenance must be an object")
    if provenance.get("split") not in {None, split}:
        raise V5ValidationError("provenance split differs from the V5.3 split")
    _validate_task_instruction_provenance(str(sample["source"]), provenance)
    _validate_context_noise(provenance, context_variant)
    dumps_with_mask_spans(sample["target"], category, spec, tuple(paths))
    return dict(sample)


def validate_sample(sample: Any) -> dict[str, Any]:
    if not isinstance(sample, Mapping):
        raise V5ValidationError("sample must be an object")
    version = sample.get("schema_version")
    if version == SCHEMA_VERSION:
        return _validate_sample_v52(sample)
    if version == SCHEMA_VERSION_V53:
        return _validate_sample_v53(sample)
    raise V5ValidationError(f"unsupported sample schema_version: {version!r}")


__all__ = [
    "CATEGORIES",
    "CONTEXT_VARIANTS",
    "EXECUTION_DECISIONS",
    "FAILURE_TYPE_BY_SOURCE_CODE",
    "FAILURE_TYPES",
    "MEMORY_VARIANTS",
    "NOISY_CONTEXT_VARIANTS",
    "OUTPUT_UNITS",
    "PROMPT_VERSION",
    "PROMPT_VERSION_V53",
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_V53",
    "SNAPSHOT_SCHEMA_VERSION",
    "SNAPSHOT_SCHEMA_VERSION_V53",
    "TRAINING_BUCKETS",
    "V5ValidationError",
    "dumps_assistant",
    "dumps_with_mask_spans",
    "parse_json_pointer",
    "output_profile_id",
    "validate_model_visible_text",
    "validate_sample",
    "validate_output_spec",
    "validate_target",
]
