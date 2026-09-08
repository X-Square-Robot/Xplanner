#!/usr/bin/env python3
"""Run clean-context V5.3 inference over one complete labelled video episode.

The rollout uses the training-time materializer and prompt/schema contracts, but
replaces teacher-forced memory with independent model-predicted memory state for
each context branch.  Ground truth is retained only for post-hoc scoring.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import stat
import sys
from typing import Any

from x_planner.data.event_states.materialize_episode import PROFILES, materialize_episode
from x_planner.data.event_states.inference import Generator, extract_json, parse_prediction, write_json
from x_planner.data.event_states.memory import LONG_MEMORY_LIMIT
from x_planner.data.event_states.prompt import render_user
from x_planner.data.event_states.schema import validate_sample


SCHEMA_VERSION = "v5_3_whole_episode_inference_v1"
EPISODE_SPEC_VERSION = "v5_3_frozen_episode_spec_v1"
CLEAN_CONTEXT_VARIANTS = (
    "no_memory_no_initial",
    "with_memory_no_initial",
    "with_memory_with_initial",
)
PROFILE_CONTEXT_VARIANTS = {
    "action_only": CLEAN_CONTEXT_VARIANTS,
    "segment_only": CLEAN_CONTEXT_VARIANTS[:2],
    "action_segment_joint": CLEAN_CONTEXT_VARIANTS,
}
ONGOING_TEMPORAL_PROMPT_SUFFIX = (
    "Temporal contract for this anchor: it is an ongoing, non-terminal observation. "
    "Do not choose End. Choose Continue, Replan, or Takeover. If Continue, return "
    "task_progress_percent below 100 and exactly two predictions containing every "
    "requested Action and Segment field; prediction 2 progress must be 0."
)
END_TEMPORAL_PROMPT_SUFFIX = (
    "Temporal contract for this anchor: it is the final terminal observation. "
    "Choose End and return the complete terminal response required by the schema."
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
MODEL_FILE_PATTERN = re.compile(r"^model(?:-[0-9]+-of-[0-9]+)?\.safetensors$")
PIN_SMALL_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "trainer_state.json",
    "video_preprocessor_config.json",
    "v10_checkpoint_meta.json",
    "vocab.json",
}


def execution_retry_prompt_suffix(
    *,
    schema_error: str,
    invalid_response: str,
    output_spec: Mapping[str, Any],
) -> str:
    """Build a grounded retry request with the exact requested Continue shape.

    Empty captions and string-valued progress placeholders deliberately remain
    schema-invalid if copied verbatim. The model must replace them using the
    current observation and causal memory; no supervision target is exposed.
    """

    predictions: list[dict[str, Any]] = []
    for index, role in ((1, "current"), (2, "next")):
        item: dict[str, Any] = {"index": index, "role": role}
        for unit in output_spec[f"prediction{index}_units"]:
            item[str(unit)] = {
                "available": True,
                "caption": "",
                "progress_percent": (
                    "REPLACE_WITH_INTEGER_0_TO_100" if index == 1 else 0
                ),
            }
        predictions.append(item)
    skeleton = {
        "task_progress_percent": "REPLACE_WITH_INTEGER_0_TO_99",
        "predictions": predictions,
        "execution_decision": "Continue",
        "decision_detail": None,
    }
    return (
        "Correction request: the previous model response failed strict schema "
        f"validation: {schema_error[:600]}. Previous invalid model JSON: "
        f"{invalid_response[:2000]}\n"
        "Regenerate the entire JSON object from the current observation and causal "
        "memory. For Continue, use exactly this key structure: "
        f"{json.dumps(skeleton, ensure_ascii=False, separators=(',', ':'))}. "
        "Replace every empty caption and every REPLACE_WITH value; do not copy a "
        "placeholder, omit a requested field, add keys, or use markdown. Prediction "
        "2 progress must remain integer 0. Follow the temporal contract above."
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        # V5 validates canonical object-key order for plan and memory nodes.
        # Preserve the already-validated insertion order across resume instead
        # of recursively sorting model objects on disk.
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def bounded_initial_plan_json_repair(
    text: str,
) -> tuple[str | None, dict[str, Any] | None]:
    """Repair only locally provable JSON punctuation damage.

    The model text values are never rewritten. The two allowed operations are
    removing a premature action-object close immediately before ``segments``
    and appending missing container closers at EOF when the tokenizer stopped
    between complete tokens. Unterminated strings and bracket mismatches are
    rejected.
    """
    value = extract_json(text)
    operations: list[dict[str, Any]] = []
    premature_action_close = re.compile(
        r'("caption"\s*:\s*"(?:[^"\\]|\\.)*")\s*}\s*("segments"\s*:)',
    )
    value, replacement_count = premature_action_close.subn(r"\1,\2", value)
    if replacement_count:
        operations.append({
            "operation": "remove_premature_action_close_before_segments",
            "count": replacement_count,
        })
    missing_segments_open_quote = re.compile(
        r'("caption"\s*:\s*"(?:[^"\\]|\\.)*")\s*}\s*segments"\s*:',
    )
    value, replacement_count = missing_segments_open_quote.subn(
        r'\1,"segments":', value
    )
    if replacement_count:
        operations.append({
            "operation": "restore_segments_open_quote_and_separator",
            "count": replacement_count,
        })

    stack: list[str] = []
    in_string = False
    escaped = False
    pairs = {"]": "[", "}": "{"}
    for character in value:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            stack.append(character)
        elif character in "]}":
            if not stack or stack[-1] != pairs[character]:
                return None, None
            stack.pop()
    if in_string or len(stack) > 8:
        return None, None
    if stack:
        suffix = "".join("}" if character == "{" else "]" for character in reversed(stack))
        value += suffix
        operations.append({
            "operation": "append_missing_eof_closers",
            "suffix": suffix,
        })
    if not operations:
        return None, None
    return value, {
        "policy": "bounded_initial_plan_json_punctuation_v1",
        "operations": operations,
        "original_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "repaired_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
    }


def bounded_initial_plan_schema_repair(
    text: str,
    output_spec: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    """Canonicalize model-only Initial Plan fields without inventing captions."""
    try:
        value = json.loads(extract_json(text))
    except json.JSONDecodeError:
        return None, None
    if not isinstance(value, Mapping):
        return None, None
    source_plan = value.get("initial_plan")
    if not isinstance(source_plan, list):
        return None, None
    plan_units = list(output_spec.get("plan_units") or [])
    if "action" not in plan_units:
        return None, None
    wants_segments = "segment" in plan_units
    canonical_plan: list[dict[str, Any]] = []
    dropped_items = 0
    dropped_segments = 0
    projected_sibling_segments = 0
    removed_extra_keys = 0
    for item in source_plan:
        if not isinstance(item, Mapping):
            dropped_items += 1
            continue
        action = item.get("action")
        if not isinstance(action, Mapping):
            dropped_items += 1
            continue
        caption = action.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            dropped_items += 1
            continue
        canonical_action: dict[str, Any] = {"caption": caption}
        removed_extra_keys += len(set(item) - {"index", "action", "segment"})
        removed_extra_keys += len(set(action) - {"caption", "segments"})
        if wants_segments:
            raw_segments = action.get("segments")
            if not isinstance(raw_segments, list):
                sibling = item.get("segment")
                raw_segments = (
                    [{"index": 1, "segment": sibling}]
                    if isinstance(sibling, Mapping)
                    else []
                )
                if raw_segments:
                    projected_sibling_segments += 1
            segments: list[dict[str, Any]] = []
            for segment_item in raw_segments:
                if not isinstance(segment_item, Mapping):
                    dropped_segments += 1
                    continue
                segment = segment_item.get("segment")
                if not isinstance(segment, Mapping):
                    dropped_segments += 1
                    continue
                segment_caption = segment.get("caption")
                if not isinstance(segment_caption, str) or not segment_caption.strip():
                    dropped_segments += 1
                    continue
                removed_extra_keys += len(set(segment_item) - {"index", "segment"})
                removed_extra_keys += len(set(segment) - {"caption"})
                segments.append({
                    "index": len(segments) + 1,
                    "segment": {"caption": segment_caption},
                })
            if not segments:
                dropped_items += 1
                continue
            canonical_action["segments"] = segments
        canonical_plan.append({
            "index": len(canonical_plan) + 1,
            "action": canonical_action,
        })
    if not canonical_plan:
        return None, None
    canonical = {"initial_plan": canonical_plan}
    canonical_text = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    source_text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if canonical_text == source_text:
        return None, None
    return canonical_text, {
        "policy": "bounded_initial_plan_model_fields_v1",
        "source_items": len(source_plan),
        "output_items": len(canonical_plan),
        "dropped_items": dropped_items,
        "dropped_segments": dropped_segments,
        "projected_sibling_segments": projected_sibling_segments,
        "removed_extra_keys": removed_extra_keys,
        "ground_truth_used": False,
        "original_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "repaired_sha256": hashlib.sha256(canonical_text.encode("utf-8")).hexdigest(),
    }


def compact_initial_plan_for_demo(
    prediction: Mapping[str, Any],
    *,
    maximum_actions: int = 12,
    maximum_segments_per_action: int = 4,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Order-preserving deduplication/capping without synthesizing captions."""
    source_plan = prediction.get("initial_plan")
    if not isinstance(source_plan, list):
        return dict(prediction), None
    compacted: list[dict[str, Any]] = []
    seen_actions: set[str] = set()
    dropped_duplicate_actions = 0
    dropped_duplicate_segments = 0
    dropped_action_cap = 0
    dropped_segment_cap = 0
    action_only_items = 0
    joint_items = 0
    for item in source_plan:
        action = item["action"]
        raw_segments = action.get("segments")
        if raw_segments is None:
            action_only_items += 1
        else:
            joint_items += 1
        action_caption = str(action["caption"])
        action_key = " ".join(action_caption.lower().split())
        if action_key in seen_actions:
            dropped_duplicate_actions += 1
            continue
        if len(compacted) >= maximum_actions:
            dropped_action_cap += 1
            continue
        seen_actions.add(action_key)
        if raw_segments is None:
            compacted.append({
                "index": len(compacted) + 1,
                "action": {"caption": action_caption},
            })
            continue
        segments: list[dict[str, Any]] = []
        seen_segments: set[str] = set()
        for segment_item in raw_segments:
            segment_caption = str(segment_item["segment"]["caption"])
            segment_key = " ".join(segment_caption.lower().split())
            if segment_key in seen_segments:
                dropped_duplicate_segments += 1
                continue
            if len(segments) >= maximum_segments_per_action:
                dropped_segment_cap += 1
                continue
            seen_segments.add(segment_key)
            segments.append({
                "index": len(segments) + 1,
                "segment": {"caption": segment_caption},
            })
        if not segments:
            continue
        compacted.append({
            "index": len(compacted) + 1,
            "action": {"caption": action_caption, "segments": segments},
        })
    report = {
        "policy": "model_caption_order_preserving_demo_compaction_v1",
        "source_actions": len(source_plan),
        "output_actions": len(compacted),
        "dropped_duplicate_actions": dropped_duplicate_actions,
        "dropped_duplicate_segments": dropped_duplicate_segments,
        "dropped_action_cap": dropped_action_cap,
        "dropped_segment_cap": dropped_segment_cap,
        "action_only_items": action_only_items,
        "joint_items": joint_items,
        "synthesized_captions": 0,
        "ground_truth_used": False,
    }
    if not any(
        report[key]
        for key in (
            "dropped_duplicate_actions",
            "dropped_duplicate_segments",
            "dropped_action_cap",
            "dropped_segment_cap",
        )
    ):
        return dict(prediction), None
    return {"initial_plan": compacted}, report


def normalize_ongoing_early_end(
    raw: str,
    sample: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Validate and preserve a model-selected early terminal transition.

    Dense inference anchors are label-derived, so a model can reasonably declare
    completion before the labelled episode's final frame.  Validate that response
    under a derived terminal output spec instead of forcing a fabricated next unit.
    """
    try:
        value = json.loads(extract_json(raw))
    except json.JSONDecodeError:
        return None, None
    if not isinstance(value, dict) or value.get("execution_decision") != "End":
        return None, None
    predictions = value.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 2:
        return None, None
    prediction2 = predictions[1]
    if not isinstance(prediction2, Mapping):
        return None, None
    source_spec = sample.get("output_spec")
    if not isinstance(source_spec, Mapping):
        return None, None
    requested_next = list(source_spec.get("prediction2_units") or [])
    observed_next = [unit for unit in requested_next if unit in prediction2]
    if observed_next not in ([], requested_next):
        return None, None
    effective_spec = {
        "prediction1_units": list(source_spec.get("prediction1_units") or []),
        "prediction2_units": observed_next,
        "plan_units": list(source_spec.get("plan_units") or []),
    }
    terminal_sample = copy.deepcopy(dict(sample))
    terminal_sample["category"] = "end"
    terminal_sample["output_spec"] = effective_spec
    prediction, error = parse_prediction(
        json.dumps(value, ensure_ascii=False), terminal_sample
    )
    if error is not None or not isinstance(prediction, dict):
        return None, None
    return prediction, {
        "policy": "ongoing_early_end_preserved_as_terminal_v2",
        "original_decision": "End",
        "effective_decision": "End",
        "source_output_spec": copy.deepcopy(dict(source_spec)),
        "effective_output_spec": effective_spec,
        "early_terminal_transition": True,
        "ground_truth_used": False,
    }


def normalize_completed_terminal_progress(
    prediction: Mapping[str, Any] | None,
    sample: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Make a completed terminal decision semantically consistent with 100%."""
    if not isinstance(prediction, Mapping):
        return None, None
    detail = prediction.get("decision_detail")
    if (
        prediction.get("execution_decision") != "End"
        or not isinstance(detail, Mapping)
        or detail.get("outcome") != "completed"
        or prediction.get("task_progress_percent") == 100
    ):
        return dict(prediction), None
    original_progress = prediction.get("task_progress_percent")
    candidate = copy.deepcopy(dict(prediction))
    candidate["task_progress_percent"] = 100
    normalized, error = parse_prediction(
        json.dumps(candidate, ensure_ascii=False), sample
    )
    if error is not None or not isinstance(normalized, dict):
        raise RuntimeError(
            "completed terminal progress normalization violated schema: "
            f"{error}"
        )
    return normalized, {
        "policy": "completed_terminal_progress_to_100_v1",
        "original_task_progress_percent": original_progress,
        "effective_task_progress_percent": 100,
        "ground_truth_used": False,
    }


def project_execution_prediction_to_output_spec(
    prediction: Mapping[str, Any],
    output_spec: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Project a prior Continue prediction by removing unrequested unit keys."""
    source_predictions = prediction.get("predictions")
    if not isinstance(source_predictions, list) or len(source_predictions) != 2:
        return None, None
    projected_predictions: list[dict[str, Any]] = []
    removed_units: list[dict[str, Any]] = []
    for offset, source in enumerate(source_predictions, 1):
        if not isinstance(source, Mapping):
            return None, None
        requested = tuple(output_spec.get(f"prediction{offset}_units") or ())
        if any(unit not in source for unit in requested):
            return None, None
        projected = {
            "index": offset,
            "role": "current" if offset == 1 else "next",
        }
        for unit in requested:
            projected[str(unit)] = copy.deepcopy(source[str(unit)])
        removed = sorted(
            unit for unit in ("action", "segment")
            if unit in source and unit not in requested
        )
        if removed:
            removed_units.append({"prediction": offset, "units": removed})
        projected_predictions.append(projected)
    candidate = copy.deepcopy(dict(prediction))
    candidate["predictions"] = projected_predictions
    candidate["execution_decision"] = "Continue"
    candidate["decision_detail"] = None
    return candidate, {
        "policy": "project_last_valid_model_prediction_to_current_output_spec_v1",
        "removed_units": removed_units,
        "ground_truth_used": False,
    }


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _validate_interval(value: Any, where: str, total_frames: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{where} must be an object")
    start = int(value["start_frame"])
    end = int(value["end_frame"])
    caption = str(value["caption"]).strip()
    if not 0 <= start < end <= total_frames:
        raise ValueError(f"{where} has invalid half-open interval [{start}, {end})")
    if not caption:
        raise ValueError(f"{where} caption is empty")
    return {"start_frame": start, "end_frame": end, "caption": caption}


def _annotation_intervals(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping) or not isinstance(value.get(field), Mapping):
        return []
    result: list[dict[str, Any]] = []
    for raw_interval, raw_caption in value[field].items():
        match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s*", str(raw_interval))
        caption = str(raw_caption or "").strip()
        if match is None or not caption:
            continue
        start, end = map(int, match.groups())
        result.append({"start_frame": start, "end_frame": end, "caption": caption})
    return sorted(
        result,
        key=lambda item: (item["start_frame"], item["end_frame"], item["caption"]),
    )


def load_episode_spec(
    path: Path,
    *,
    verify_sources: bool = True,
    minimum_intervals_per_unit: int = 4,
) -> dict[str, Any]:
    path = path.resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("episode spec must be an object")
    if value.get("schema_version") != EPISODE_SPEC_VERSION:
        raise ValueError("episode spec schema_version mismatch")
    total_frames = int(value.get("total_frames", 0))
    fps = float(value.get("fps", 0.0))
    if total_frames <= 0 or fps <= 0:
        raise ValueError("episode spec requires positive total_frames and fps")
    profiles = tuple(value.get("profiles") or ())
    if (
        not profiles
        or len(profiles) != len(set(profiles))
        or any(profile not in PROFILES for profile in profiles)
        or profiles != tuple(profile for profile in PROFILES if profile in profiles)
    ):
        raise ValueError(
            f"episode spec profiles must be a non-empty ordered subset of {PROFILES}"
        )
    videos = value.get("videos")
    if not isinstance(videos, Mapping) or tuple(sorted(videos)) != (
        "head", "left_wrist", "right_wrist"
    ):
        raise ValueError("episode spec requires head/left_wrist/right_wrist videos")
    for view, raw_path in videos.items():
        video = Path(str(raw_path))
        if not video.is_absolute() or not video.is_file():
            raise FileNotFoundError(f"missing absolute {view} video: {video}")
    actions = [
        _validate_interval(item, f"actions[{index}]", total_frames)
        for index, item in enumerate(value.get("actions") or ())
    ]
    segments = [
        _validate_interval(item, f"segments[{index}]", total_frames)
        for index, item in enumerate(value.get("segments") or ())
    ]
    if minimum_intervals_per_unit <= 0:
        raise ValueError("minimum_intervals_per_unit must be positive")
    required_units = {
        "action_only": ("Actions",),
        "segment_only": ("Segments",),
        "action_segment_joint": ("Actions", "Segments"),
    }
    active_units = {
        unit for profile in profiles for unit in required_units[profile]
    }
    counts = {"Actions": len(actions), "Segments": len(segments)}
    insufficient = {
        unit: counts[unit]
        for unit in sorted(active_units)
        if counts[unit] < minimum_intervals_per_unit
    }
    if insufficient:
        raise ValueError(
            "episode spec has too few intervals for active profiles: "
            f"minimum={minimum_intervals_per_unit}, observed={insufficient}"
        )
    if verify_sources:
        source_files = value.get("source_files")
        if (
            isinstance(source_files, (str, bytes))
            or not isinstance(source_files, Sequence)
            or not source_files
        ):
            raise ValueError("episode spec requires source_files for provenance verification")
        by_role: dict[str, Path] = {}
        for index, item in enumerate(source_files):
            if not isinstance(item, Mapping):
                raise TypeError(f"source_files[{index}] must be an object")
            source = Path(str(item.get("path") or ""))
            expected = str(item.get("sha256") or "")
            role = str(item.get("role") or "")
            if not role or role in by_role:
                raise ValueError(f"source_files[{index}] has a missing or duplicate role")
            if not source.is_file():
                raise FileNotFoundError(source)
            observed = sha256_file(source)
            if observed != expected:
                raise ValueError(
                    f"source sha256 mismatch for {source}: {observed} != {expected}"
                )
            by_role[role] = source
        expected_roles = {
            "v2v3umi_label_annotation",
            "episode_task_instruction",
            "benchmark3_holdout_manifest",
        }
        if set(by_role) != expected_roles:
            raise ValueError(f"source roles differ: {set(by_role)} != {expected_roles}")
        raw_episode_key = str(value["episode_key"]).rsplit("/", 1)[-1]
        label_map = json.loads(
            by_role["v2v3umi_label_annotation"].read_text(encoding="utf-8")
        )
        annotation = label_map.get(raw_episode_key) if isinstance(label_map, Mapping) else None
        if _annotation_intervals(annotation, "action_caption") != actions:
            raise ValueError("frozen Actions differ from the hashed source annotation")
        if _annotation_intervals(annotation, "human_segment_caption") != segments:
            raise ValueError("frozen Segments differ from the hashed source annotation")
        instruction_map = json.loads(
            by_role["episode_task_instruction"].read_text(encoding="utf-8")
        )
        instruction_row = (
            instruction_map.get(raw_episode_key)
            if isinstance(instruction_map, Mapping) else None
        )
        if (
            not isinstance(instruction_row, Mapping)
            or instruction_row.get("instruction") != value.get("task_instruction")
        ):
            raise ValueError("frozen task instruction differs from the hashed episode JSON")
        benchmark_matches: list[Mapping[str, Any]] = []
        with by_role["benchmark3_holdout_manifest"].open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, Mapping) and row.get("uid") == value.get("benchmark3_uid"):
                    benchmark_matches.append(row)
        if len(benchmark_matches) != 1:
            raise ValueError(
                f"frozen Benchmark3 UID has {len(benchmark_matches)} manifest matches"
            )
        episode_path = Path(str(benchmark_matches[0].get("existing_episode_path") or ""))
        if episode_path != Path(str(videos["head"])).parent:
            raise ValueError("Benchmark3 episode path differs from the frozen video directory")
    result = dict(value)
    result["actions"] = actions
    result["segments"] = segments
    result["videos"] = {str(key): str(raw) for key, raw in videos.items()}
    result["profiles"] = list(profiles)
    result["spec_path"] = str(path)
    result["spec_sha256"] = sha256_file(path)
    return result


@dataclass(frozen=True, slots=True)
class RolloutSlot:
    slot_id: str
    profile: str
    category: str
    context_variant: str
    anchor_frame: int
    base_sample: dict[str, Any]
    oracle_prompt_context: dict[str, Any] = field(default_factory=dict)


def build_rollout_slots(
    spec: Mapping[str, Any],
    *,
    anchor_stride_frames: int | None = None,
    anchor_frames: Sequence[int] | None = None,
    minimum_units_per_profile: int = 4,
    dense_gap_policy: str = "error",
) -> list[RolloutSlot]:
    samples, _missing = materialize_episode(
        episode_key=str(spec["episode_key"]),
        source=str(spec["source"]),
        source_group=str(spec["source_group"]),
        task_instruction=str(spec["task_instruction"]),
        actions=spec["actions"],
        segments=spec["segments"],
        videos=spec["videos"],
        total_frames=int(spec["total_frames"]),
        profiles=tuple(spec["profiles"]),
        split="test",
        provenance=dict(spec.get("provenance") or {}),
        anchor_stride_frames=anchor_stride_frames,
        anchor_frames=anchor_frames,
        minimum_units_per_profile=minimum_units_per_profile,
        dense_gap_policy=dense_gap_policy,
    )
    clean = [
        sample for sample in samples
        if sample["context_variant"] in CLEAN_CONTEXT_VARIANTS
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in clean:
        grouped[str(sample["provenance"]["label_profile"])].append(sample)
    slots: list[RolloutSlot] = []
    active_profiles = tuple(str(profile) for profile in spec["profiles"])
    for profile in active_profiles:
        profile_rows = grouped[profile]
        initial = [row for row in profile_rows if row["category"] == "initial_plan"]
        if profile == "segment_only":
            if initial:
                raise ValueError("segment_only unexpectedly materialized an initial plan")
        else:
            if len(initial) != 1:
                raise ValueError(f"{profile} requires exactly one initial plan")
            sample = initial[0]
            slots.append(RolloutSlot(
                slot_id=f"{profile}:initial_plan:no_memory_no_initial",
                profile=profile,
                category="initial_plan",
                context_variant="no_memory_no_initial",
                anchor_frame=0,
                base_sample=sample,
                oracle_prompt_context={},
            ))
        oracle_rows = {
            (str(row["base_sample_id"]), str(row["context_variant"])): row
            for row in profile_rows
            if row["category"] in {"ongoing", "end"}
        }
        anchors = {
            str(row["base_sample_id"]): row
            for row in profile_rows
            if row["category"] in {"ongoing", "end"}
            and row["context_variant"] == "no_memory_no_initial"
        }
        ordered = sorted(
            anchors.values(),
            key=lambda row: (
                int(row["provenance"]["anchor_frame"]),
                1 if row["category"] == "end" else 0,
                str(row["base_sample_id"]),
            ),
        )
        for variant in PROFILE_CONTEXT_VARIANTS[profile]:
            for sample in ordered:
                base_id = str(sample["base_sample_id"])
                oracle_sample = oracle_rows.get((base_id, variant))
                if oracle_sample is None:
                    raise ValueError(
                        f"missing materialized oracle context for {profile}/{base_id}/{variant}"
                    )
                slots.append(RolloutSlot(
                    slot_id=f"{profile}:{base_id}:{variant}",
                    profile=profile,
                    category=str(sample["category"]),
                    context_variant=variant,
                    anchor_frame=int(sample["provenance"]["anchor_frame"]),
                    base_sample=sample,
                    oracle_prompt_context=copy.deepcopy(
                        dict(oracle_sample.get("prompt_context") or {})
                    ),
                ))
    if anchor_stride_frames is None and anchor_frames is None:
        expected_count = int(spec.get("expected_slot_count", len(slots)))
        expected_profile_counts = {
            str(key): int(raw)
            for key, raw in dict(
                spec.get("expected_profile_slot_counts") or {}
            ).items()
            if str(key) in active_profiles
        }
    else:
        if anchor_stride_frames is not None and anchor_stride_frames <= 0:
            raise ValueError("anchor_stride_frames must be positive")
        if anchor_stride_frames is not None and anchor_frames is not None:
            raise ValueError("anchor_stride_frames and anchor_frames are mutually exclusive")
        execution_anchors = 1 + (
            len(range(0, int(spec["total_frames"]) - 1, anchor_stride_frames))
            if anchor_stride_frames is not None
            else len(tuple(sorted({int(value) for value in anchor_frames or ()})))
        )
        expected_profile_counts = {
            profile: (
                (0 if profile == "segment_only" else 1)
                + execution_anchors * len(PROFILE_CONTEXT_VARIANTS[profile])
            )
            for profile in active_profiles
        }
        expected_count = sum(expected_profile_counts.values())
    if len(slots) != expected_count:
        raise ValueError(f"rollout matrix has {len(slots)} slots, expected {expected_count}")
    observed = Counter(slot.profile for slot in slots)
    if expected_profile_counts and dict(observed) != expected_profile_counts:
        raise ValueError(
            f"profile slot counts differ: {dict(observed)} != {expected_profile_counts}"
        )
    return slots


def select_rollout_slots(
    slots: Sequence[RolloutSlot],
    *,
    profiles: Sequence[str],
    context_variants: Sequence[str],
) -> list[RolloutSlot]:
    """Select execution branches while retaining required initial-plan slots."""

    requested_profiles = tuple(dict.fromkeys(str(value) for value in profiles))
    requested_variants = tuple(
        dict.fromkeys(str(value) for value in context_variants)
    )
    if not requested_profiles:
        raise ValueError("at least one profile must be selected")
    if not requested_variants:
        raise ValueError("at least one context variant must be selected")
    unknown_profiles = set(requested_profiles) - set(PROFILES)
    if unknown_profiles:
        raise ValueError(f"unknown profiles: {sorted(unknown_profiles)}")
    unknown_variants = set(requested_variants) - set(CLEAN_CONTEXT_VARIANTS)
    if unknown_variants:
        raise ValueError(f"unknown context variants: {sorted(unknown_variants)}")
    for profile in requested_profiles:
        unsupported = set(requested_variants) - set(PROFILE_CONTEXT_VARIANTS[profile])
        if unsupported:
            raise ValueError(
                f"{profile} does not support context variants {sorted(unsupported)}"
            )

    selected = [
        slot
        for slot in slots
        if slot.profile in requested_profiles
        and (
            slot.category == "initial_plan"
            or slot.context_variant in requested_variants
        )
    ]
    if "with_memory_with_initial" in requested_variants:
        for profile in requested_profiles:
            initial = [
                slot
                for slot in selected
                if slot.profile == profile and slot.category == "initial_plan"
            ]
            if len(initial) != 1:
                raise ValueError(
                    f"{profile} requires exactly one initial plan for with-initial rollout"
                )
    if not selected:
        raise ValueError("rollout selection is empty")
    return selected


def _clone_sample(
    slot: RolloutSlot,
    *,
    prompt_context: Mapping[str, Any],
    context_variant_override: str | None = None,
) -> dict[str, Any]:
    sample = copy.deepcopy(slot.base_sample)
    model_context_variant = context_variant_override or slot.context_variant
    sample["context_variant"] = model_context_variant
    sample["sample_id"] = f"{sample['base_sample_id']}_{model_context_variant}"
    sample["prompt_context"] = copy.deepcopy(dict(prompt_context))
    sample["provenance"].pop("context_noise", None)
    return validate_sample(sample)


@dataclass(slots=True)
class PredictedMemoryState:
    initial_plan: list[dict[str, Any]] | None = None
    long_memory: list[dict[str, Any]] = field(default_factory=list)
    short_memory: dict[str, Any] | None = None

    def prompt_context(self, *, include_initial: bool) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if include_initial:
            if not self.initial_plan:
                raise ValueError("model-predicted initial plan is unavailable")
            result["initial_plan_memory"] = copy.deepcopy(self.initial_plan)
        result["long_memory"] = copy.deepcopy(self.long_memory)
        result["short_memory"] = copy.deepcopy(self.short_memory)
        return result

    def update(self, prediction: Mapping[str, Any] | None) -> dict[str, Any]:
        before = stable_json_sha256({
            "long_memory": self.long_memory,
            "short_memory": self.short_memory,
        })
        if prediction is None:
            return {
                "updated": False,
                "reason": "invalid_json_state_unchanged",
                "before_sha256": before,
                "after_sha256": before,
                "long_memory_size": len(self.long_memory),
            }
        predictions = prediction.get("predictions")
        if not isinstance(predictions, Sequence) or not predictions:
            return {
                "updated": False,
                "reason": "decision_without_predictions_state_unchanged",
                "before_sha256": before,
                "after_sha256": before,
                "long_memory_size": len(self.long_memory),
            }
        current = predictions[0]
        if not isinstance(current, Mapping):
            raise TypeError("validated prediction1 is not an object")
        item: dict[str, Any] = {"index": 0}
        short_prediction: dict[str, Any] = {}
        for unit in ("action", "segment"):
            raw = current.get(unit)
            if not isinstance(raw, Mapping):
                continue
            item[unit] = str(raw["caption"])
            short_prediction[unit] = copy.deepcopy(dict(raw))
        if not short_prediction:
            return {
                "updated": False,
                "reason": "prediction1_has_no_units_state_unchanged",
                "before_sha256": before,
                "after_sha256": before,
                "long_memory_size": len(self.long_memory),
            }
        identity = tuple(str(item.get(unit, "")).casefold() for unit in ("action", "segment"))
        seen = {
            tuple(str(prior.get(unit, "")).casefold() for unit in ("action", "segment"))
            for prior in self.long_memory
        }
        appended = identity not in seen
        if appended:
            self.long_memory.append(item)
            self.long_memory = self.long_memory[-LONG_MEMORY_LIMIT:]
            for index, prior in enumerate(self.long_memory, 1):
                prior["index"] = index
        self.short_memory = {
            "task_progress_percent": int(prediction["task_progress_percent"]),
            "prediction1": short_prediction,
        }
        after = stable_json_sha256({
            "long_memory": self.long_memory,
            "short_memory": self.short_memory,
        })
        return {
            "updated": True,
            "reason": "prediction1_committed" if appended else "prediction1_duplicate_short_only",
            "appended_long_memory": appended,
            "before_sha256": before,
            "after_sha256": after,
            "long_memory_size": len(self.long_memory),
        }


def _tokens(text: str) -> Counter[str]:
    return Counter(TOKEN_PATTERN.findall(str(text).casefold()))


def token_f1(prediction: str, target: str) -> float:
    predicted = _tokens(prediction)
    expected = _tokens(target)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = sum((predicted & expected).values())
    precision = overlap / sum(predicted.values())
    recall = overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def execution_caption_score(
    prediction: Mapping[str, Any] | None,
    target: Mapping[str, Any],
    output_spec: Mapping[str, Any],
) -> float:
    if prediction is None or not isinstance(prediction.get("predictions"), Sequence):
        return 0.0
    predicted = prediction["predictions"]
    expected = target["predictions"]
    scores: list[float] = []
    for index in range(2):
        for unit in output_spec[f"prediction{index + 1}_units"]:
            try:
                scores.append(token_f1(
                    str(predicted[index][unit]["caption"]),
                    str(expected[index][unit]["caption"]),
                ))
            except (IndexError, KeyError, TypeError):
                scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def raw_json_object(text: str) -> dict[str, Any] | None:
    """Parse the model JSON without applying category-specific schema rules."""

    try:
        value = json.loads(extract_json(text))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _score_fields(
    prediction: Mapping[str, Any] | None,
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    target = sample["target"]
    if sample["category"] == "initial_plan":
        return {
            "scoring_available": True,
            "caption_token_f1": None,
            "decision_correct": None,
            "task_progress_abs_error": None,
            "exact_match": prediction == target,
        }
    scoring_available = bool(
        sample.get("provenance", {}).get("ground_truth_available", True)
    )
    if not scoring_available:
        return {
            "scoring_available": False,
            "caption_token_f1": None,
            "decision_correct": None,
            "task_progress_abs_error": None,
            "task_progress_available": False,
            "exact_match": None,
        }
    has_progress = prediction is not None and isinstance(
        prediction.get("task_progress_percent"), int
    )
    progress_error = (
        abs(int(prediction["task_progress_percent"]) - int(target["task_progress_percent"]))
        if has_progress else 100
    )
    return {
        "scoring_available": True,
        "caption_token_f1": execution_caption_score(
            prediction, target, sample["output_spec"]
        ),
        "decision_correct": (
            prediction is not None
            and prediction.get("execution_decision") == target["execution_decision"]
        ),
        "task_progress_abs_error": progress_error,
        "task_progress_available": has_progress,
        "exact_match": prediction == target,
    }


def _partial_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".partial")


def _existing_rollout_rows(path: Path, resume: bool) -> tuple[list[dict[str, Any]], Path]:
    partial = _partial_path(path)
    if partial.is_file() and path.is_file():
        raise RuntimeError(f"both complete and partial prediction files exist for {path}")
    source = path if path.is_file() else partial if partial.is_file() else None
    if source is None:
        return [], partial
    if not resume:
        raise FileExistsError(f"rollout output already exists: {source}")
    return jsonl_rows(source), partial


def run_rollout(
    *,
    slots: Sequence[RolloutSlot],
    generator: Any,
    source_root: Path,
    output_path: Path,
    initial_max_new_tokens: int,
    execution_max_new_tokens: int,
    resume: bool,
    initial_prompt_suffix: str | None = None,
    compact_initial_plan: bool = False,
    enforce_temporal_decision_contract: bool = False,
    execution_schema_retries: int = 0,
    allow_execution_hold_fallback: bool = False,
    allow_initial_plan_json_repair: bool = True,
    allow_ongoing_early_end_normalization: bool = True,
    allow_terminal_progress_normalization: bool = True,
    memory_update_source: str = "effective",
    protocol: str = "legacy",
    context_source: str = "predicted",
    execution_context_policy: str = "configured",
) -> list[dict[str, Any]]:
    if execution_schema_retries < 0:
        raise ValueError("execution_schema_retries must be non-negative")
    if memory_update_source not in {"raw", "effective"}:
        raise ValueError("memory_update_source must be 'raw' or 'effective'")
    if protocol not in {"legacy", "blinded", "assisted"}:
        raise ValueError("protocol must be legacy, blinded, or assisted")
    if context_source not in {"predicted", "oracle"}:
        raise ValueError("context_source must be 'predicted' or 'oracle'")
    if execution_context_policy not in {"configured", "observations_only"}:
        raise ValueError(
            "execution_context_policy must be 'configured' or 'observations_only'"
        )
    if execution_context_policy == "observations_only" and context_source != "predicted":
        raise ValueError("observations_only execution requires predicted context source")
    active_profiles = tuple(dict.fromkeys(slot.profile for slot in slots))
    if not active_profiles or any(profile not in PROFILES for profile in active_profiles):
        raise ValueError("rollout slots require at least one known profile")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prior, partial = _existing_rollout_rows(output_path, resume)
    if len(prior) > len(slots):
        raise ValueError("partial rollout contains more rows than the matrix")
    for index, row in enumerate(prior):
        if row.get("slot_id") != slots[index].slot_id:
            raise ValueError(
                f"resume slot mismatch at {index}: {row.get('slot_id')} != {slots[index].slot_id}"
            )
    if output_path.is_file() and len(prior) != len(slots):
        raise ValueError("completed predictions file does not contain the complete matrix")

    initial_seen = {profile: False for profile in active_profiles}
    initial_plans: dict[str, list[dict[str, Any]] | None] = {
        profile: None for profile in active_profiles
    }
    memory_states = {
        (profile, variant): PredictedMemoryState()
        for profile in active_profiles
        for variant in PROFILE_CONTEXT_VARIANTS[profile]
        if variant != "no_memory_no_initial"
    } if context_source == "predicted" else {}
    rows: list[dict[str, Any]] = []
    for index, slot in enumerate(slots):
        is_initial = slot.category == "initial_plan"
        state = memory_states.get((slot.profile, slot.context_variant))
        visualization_memory_snapshot = (
            state.prompt_context(include_initial=False)
            if state is not None and not is_initial
            else {}
        )
        dependency_error: str | None = None
        if execution_context_policy == "observations_only" and not is_initial:
            if slot.context_variant == "with_memory_with_initial":
                if not initial_seen[slot.profile] or initial_plans[slot.profile] is None:
                    dependency_error = "model_initial_plan_invalid_or_unavailable"
                elif state is not None:
                    state.initial_plan = copy.deepcopy(initial_plans[slot.profile])
            context: dict[str, Any] = {}
        elif context_source == "oracle" and not is_initial:
            context = copy.deepcopy(slot.oracle_prompt_context)
        elif slot.context_variant == "no_memory_no_initial":
            context: dict[str, Any] = {}
        elif slot.context_variant == "with_memory_no_initial":
            assert state is not None
            context = state.prompt_context(include_initial=False)
        else:
            assert state is not None
            if not initial_seen[slot.profile] or initial_plans[slot.profile] is None:
                dependency_error = "model_initial_plan_invalid_or_unavailable"
                context = {}
            else:
                state.initial_plan = copy.deepcopy(initial_plans[slot.profile])
                context = state.prompt_context(include_initial=True)

        sample: dict[str, Any] | None = None
        prompt: str | None = None
        if dependency_error is None:
            sample = _clone_sample(
                slot,
                prompt_context=context,
                context_variant_override=(
                    "no_memory_no_initial"
                    if execution_context_policy == "observations_only" and not is_initial
                    else None
                ),
            )
            prompt = render_user(sample)
            if is_initial and initial_prompt_suffix:
                prompt = f"{prompt}\n{initial_prompt_suffix}"
            elif enforce_temporal_decision_contract and slot.category == "ongoing":
                prompt = f"{prompt}\n{ONGOING_TEMPORAL_PROMPT_SUFFIX}"
            elif enforce_temporal_decision_contract and slot.category == "end":
                prompt = f"{prompt}\n{END_TEMPORAL_PROMPT_SUFFIX}"

        if index < len(prior):
            record = prior[index]
            expected_status = "skipped_input_dependency" if dependency_error else "generated"
            if record.get("status") != expected_status:
                raise ValueError(
                    f"resume status mismatch for {slot.slot_id}: "
                    f"{record.get('status')} != {expected_status}"
                )
            if prompt is not None and record.get("prompt_sha256") != hashlib.sha256(
                prompt.encode("utf-8")
            ).hexdigest():
                raise ValueError(f"resume prompt mismatch for {slot.slot_id}")
        elif dependency_error is not None:
            record = {
                "schema_version": SCHEMA_VERSION,
                "protocol": protocol,
                "context_source": context_source,
                "execution_context_policy": execution_context_policy,
                "slot_id": slot.slot_id,
                "status": "skipped_input_dependency",
                "skip_reason": dependency_error,
                "profile": slot.profile,
                "category": slot.category,
                "context_variant": slot.context_variant,
                "model_context_variant": None,
                "base_sample_id": slot.base_sample["base_sample_id"],
                "anchor_frame": slot.anchor_frame,
                "output_spec": copy.deepcopy(slot.base_sample["output_spec"]),
                "task_instruction": slot.base_sample["task_instruction"],
                "prediction_raw": None,
                "raw_prediction": None,
                "raw_prediction_schema_valid": False,
                "raw_prediction_schema_error": dependency_error,
                "prediction": None,
                "prediction_schema_valid": False,
                "prediction_schema_error": dependency_error,
                "ground_truth": slot.base_sample["target"],
                "memory_input": None,
                "visualization_memory_snapshot": copy.deepcopy(
                    visualization_memory_snapshot
                ),
                "memory_update": {
                    "updated": False,
                    "reason": "skipped_input_dependency",
                },
                **_score_fields(None, slot.base_sample),
            }
            append_jsonl(partial, record)
        else:
            assert sample is not None and prompt is not None
            max_new_tokens = (
                initial_max_new_tokens if is_initial else execution_max_new_tokens
            )
            raw, generation = generator.generate(
                {
                    "v5_sample": sample,
                    "image": sample["images"],
                    "_rendered_prompt": prompt,
                },
                source_root,
                max_new_tokens=max_new_tokens,
            )
            prediction, schema_error = parse_prediction(raw, sample)
            first_prediction_raw = raw
            raw_prediction = (
                copy.deepcopy(prediction)
                if prediction is not None
                else raw_json_object(raw)
            )
            raw_schema_error = schema_error
            generation_attempts: list[dict[str, Any]] = [{
                "attempt": 1,
                "prompt_suffix": None,
                "prediction_raw": raw,
                "schema_error": schema_error,
                "output_tokens": generation.get("output_tokens"),
                "generation_seconds": generation.get("generation_seconds"),
            }]
            prediction_repair: dict[str, Any] | None = None
            if is_initial and schema_error is not None and allow_initial_plan_json_repair:
                repair_stages: list[dict[str, Any]] = []
                candidate_raw = raw
                punctuation_raw, punctuation_report = (
                    bounded_initial_plan_json_repair(candidate_raw)
                )
                if punctuation_raw is not None and punctuation_report is not None:
                    candidate_raw = punctuation_raw
                    repair_stages.append(punctuation_report)
                    repaired_prediction, repaired_error = parse_prediction(
                        candidate_raw, sample
                    )
                    if repaired_error is None:
                        prediction = repaired_prediction
                        schema_error = None
                        prediction_repair = punctuation_report
                if schema_error is not None:
                    canonical_raw, canonical_report = (
                        bounded_initial_plan_schema_repair(
                            candidate_raw, sample["output_spec"]
                        )
                    )
                    if canonical_raw is not None and canonical_report is not None:
                        repair_stages.append(canonical_report)
                        repaired_prediction, repaired_error = parse_prediction(
                            canonical_raw, sample
                        )
                        if repaired_error is None:
                            prediction = repaired_prediction
                            schema_error = None
                            prediction_repair = (
                                canonical_report
                                if len(repair_stages) == 1
                                else {
                                    "policy": "bounded_initial_plan_repair_pipeline_v1",
                                    "stages": repair_stages,
                                    "ground_truth_used": False,
                                }
                            )
            prediction_normalization: dict[str, Any] | None = None
            if (
                is_initial
                and compact_initial_plan
                and schema_error is None
                and isinstance(prediction, Mapping)
            ):
                compacted, prediction_normalization = compact_initial_plan_for_demo(
                    prediction
                )
                if prediction_normalization is not None:
                    normalized_prediction, normalized_error = parse_prediction(
                        json.dumps(compacted, ensure_ascii=False), sample
                    )
                    if normalized_error is not None:
                        raise RuntimeError(
                            "demo initial plan compaction violated schema: "
                            f"{normalized_error}"
                        )
                    prediction = normalized_prediction
            if (
                not is_initial
                and slot.category == "ongoing"
                and schema_error is not None
                and allow_ongoing_early_end_normalization
            ):
                normalized_early_end, early_end_report = normalize_ongoing_early_end(
                    raw, sample
                )
                if normalized_early_end is not None and early_end_report is not None:
                    prediction = normalized_early_end
                    schema_error = None
                    prediction_normalization = early_end_report
            selected_generation_prompt_suffix: str | None = None
            if not is_initial and schema_error is not None:
                for retry_index in range(execution_schema_retries):
                    retry_suffix = execution_retry_prompt_suffix(
                        schema_error=schema_error,
                        invalid_response=raw,
                        output_spec=sample["output_spec"],
                    )
                    retry_prompt = f"{prompt}\n{retry_suffix}"
                    retry_raw, retry_generation = generator.generate(
                        {
                            "v5_sample": sample,
                            "image": sample["images"],
                            "_rendered_prompt": retry_prompt,
                        },
                        source_root,
                        max_new_tokens=max_new_tokens,
                    )
                    retry_prediction, retry_error = parse_prediction(retry_raw, sample)
                    retry_raw_error = retry_error
                    retry_normalization: dict[str, Any] | None = None
                    if (
                        slot.category == "ongoing"
                        and retry_error is not None
                        and allow_ongoing_early_end_normalization
                    ):
                        normalized_early_end, early_end_report = (
                            normalize_ongoing_early_end(retry_raw, sample)
                        )
                        if (
                            normalized_early_end is not None
                            and early_end_report is not None
                        ):
                            retry_prediction = normalized_early_end
                            retry_error = None
                            retry_normalization = early_end_report
                            prediction_normalization = early_end_report
                    generation_attempts.append({
                        "attempt": retry_index + 2,
                        "prompt_suffix": retry_suffix,
                        "prediction_raw": retry_raw,
                        "schema_error": retry_error,
                        "raw_schema_error": retry_raw_error,
                        "normalization": retry_normalization,
                        "output_tokens": retry_generation.get("output_tokens"),
                        "generation_seconds": retry_generation.get("generation_seconds"),
                    })
                    raw = retry_raw
                    generation = retry_generation
                    prediction = retry_prediction
                    schema_error = retry_error
                    selected_generation_prompt_suffix = retry_suffix
                    if schema_error is None:
                        break
            prediction_fallback: dict[str, Any] | None = None
            if (
                not is_initial
                and slot.category == "ongoing"
                and schema_error is not None
                and allow_execution_hold_fallback
            ):
                for prior_row in reversed(rows):
                    prior_prediction = prior_row.get("prediction")
                    if (
                        prior_row.get("category") == "initial_plan"
                        or prior_row.get("profile") != slot.profile
                        or prior_row.get("context_variant") != slot.context_variant
                        or not prior_row.get("prediction_schema_valid")
                        or not isinstance(prior_prediction, Mapping)
                    ):
                        continue
                    candidate, projection_report = (
                        project_execution_prediction_to_output_spec(
                            prior_prediction, sample["output_spec"]
                        )
                    )
                    if candidate is None or projection_report is None:
                        continue
                    held, held_error = parse_prediction(
                        json.dumps(candidate, ensure_ascii=False), sample
                    )
                    if held_error is not None:
                        continue
                    prediction = held
                    schema_error = None
                    prediction_fallback = {
                        "policy": "hold_last_valid_model_prediction_v1",
                        "source_slot_id": prior_row["slot_id"],
                        "source_anchor_frame": prior_row.get("anchor_frame"),
                        "failed_attempt_count": len(generation_attempts),
                        "projection": projection_report,
                        "ground_truth_used": False,
                    }
                    break
            if (
                not is_initial
                and slot.category == "end"
                and schema_error is None
                and allow_terminal_progress_normalization
            ):
                normalized_terminal, terminal_report = (
                    normalize_completed_terminal_progress(prediction, sample)
                )
                prediction = normalized_terminal
                if terminal_report is not None:
                    prediction_normalization = terminal_report
            generation = dict(generation)
            generation["generation_seconds_all_attempts"] = sum(
                float(item.get("generation_seconds") or 0.0)
                for item in generation_attempts
            )
            memory_update: dict[str, Any] = {
                "updated": False,
                "reason": "not_a_memory_branch",
            }
            if (
                context_source == "oracle"
                and slot.context_variant != "no_memory_no_initial"
                and slot.category == "ongoing"
            ):
                memory_update = {
                    "updated": False,
                    "reason": "oracle_context_is_label_derived_and_stateless",
                    "source": "oracle_label",
                }
            elif state is not None and slot.category == "ongoing":
                memory_prediction = (
                    raw_prediction
                    if memory_update_source == "raw" and raw_schema_error is None
                    else prediction if memory_update_source == "effective"
                    else None
                )
                memory_update = state.update(
                    memory_prediction if isinstance(memory_prediction, Mapping) else None
                )
                memory_update["source"] = memory_update_source
            record = {
                "schema_version": SCHEMA_VERSION,
                "protocol": protocol,
                "context_source": context_source,
                "execution_context_policy": execution_context_policy,
                "slot_id": slot.slot_id,
                "status": "generated",
                "profile": slot.profile,
                "category": slot.category,
                "context_variant": slot.context_variant,
                "model_context_variant": sample["context_variant"],
                "sample_id": sample["sample_id"],
                "base_sample_id": sample["base_sample_id"],
                "anchor_frame": slot.anchor_frame,
                "output_spec": sample["output_spec"],
                "task_instruction": sample["task_instruction"],
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "inference_prompt_suffix": (
                    initial_prompt_suffix if is_initial else None
                ),
                "selected_generation_prompt_suffix": selected_generation_prompt_suffix,
                "memory_input": copy.deepcopy(sample["prompt_context"]),
                "visualization_memory_snapshot": copy.deepcopy(
                    visualization_memory_snapshot
                ),
                "prediction_raw": raw,
                "first_prediction_raw": first_prediction_raw,
                "raw_prediction": raw_prediction,
                "raw_prediction_schema_valid": raw_schema_error is None,
                "raw_prediction_schema_error": raw_schema_error,
                "prediction": prediction,
                "prediction_schema_valid": schema_error is None,
                "prediction_schema_error": schema_error,
                "prediction_repair": prediction_repair,
                "prediction_normalization": prediction_normalization,
                "prediction_attempts": generation_attempts,
                "prediction_fallback": prediction_fallback,
                "ground_truth": sample["target"],
                "memory_update": memory_update,
                "max_new_tokens": max_new_tokens,
                **_score_fields(prediction, sample),
                **generation,
            }
            append_jsonl(partial, record)

        rows.append(record)
        prediction = record.get("prediction")
        if is_initial:
            initial_seen[slot.profile] = True
            initial_plans[slot.profile] = (
                copy.deepcopy(prediction["initial_plan"])
                if isinstance(prediction, Mapping)
                and isinstance(prediction.get("initial_plan"), list)
                else None
            )
        elif (
            context_source == "predicted"
            and state is not None
            and slot.category == "ongoing"
            and index < len(prior)
        ):
            resumed_prediction = (
                record.get("raw_prediction")
                if memory_update_source == "raw"
                and record.get("raw_prediction_schema_valid")
                else prediction
                if memory_update_source == "effective"
                else None
            )
            state.update(
                resumed_prediction if isinstance(resumed_prediction, Mapping) else None
            )
        print(json.dumps({
            "slot": index + 1,
            "total": len(slots),
            "slot_id": slot.slot_id,
            "status": record["status"],
            "schema_valid": record.get("prediction_schema_valid"),
        }, ensure_ascii=False, sort_keys=True), flush=True)

    if len(rows) == len(slots) and partial.is_file():
        os.replace(partial, output_path)
    return rows


def _mean(values: Sequence[float | int]) -> float | None:
    return sum(float(value) for value in values) / len(values) if values else None


def summarize_rollout(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_slots: int,
    require_paired_baseline: bool = True,
) -> dict[str, Any]:
    if len(rows) != expected_slots:
        raise ValueError(f"summary requires {expected_slots} rows, got {len(rows)}")
    statuses = Counter(str(row["status"]) for row in rows)
    generated = [row for row in rows if row["status"] == "generated"]
    execution = [row for row in rows if row["category"] != "initial_plan"]
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in execution:
        groups[(str(row["profile"]), str(row["context_variant"]))].append(row)

    metrics: dict[str, Any] = {}
    for (profile, variant), group in sorted(groups.items()):
        scored = [row for row in group if row.get("scoring_available", True)]
        available_progress = sum(
            bool(row.get("task_progress_available")) for row in scored
        )
        metrics[f"{profile}/{variant}"] = {
            "slots": len(group),
            "scored_slots": len(scored),
            "ground_truth_coverage": len(scored) / len(group),
            "generated": sum(row["status"] == "generated" for row in group),
            "schema_valid_rate": _mean([
                int(bool(row.get("prediction_schema_valid"))) for row in group
            ]),
            "caption_token_f1_macro": _mean([
                float(row.get("caption_token_f1") or 0.0) for row in scored
            ]),
            "decision_accuracy": _mean([
                int(bool(row.get("decision_correct"))) for row in scored
            ]),
            "task_progress_mae_missing_as_100": _mean([
                int(row.get("task_progress_abs_error", 100)) for row in scored
            ]),
            "task_progress_coverage": (
                available_progress / len(scored) if scored else None
            ),
            "exact_match_rate": _mean([
                int(bool(row.get("exact_match"))) for row in scored
            ]),
        }

    indexed = {
        (str(row["profile"]), str(row["base_sample_id"]), str(row["context_variant"])): row
        for row in execution
    }
    paired: dict[str, Any] = {}
    for variant in ("with_memory_no_initial", "with_memory_with_initial"):
        deltas: list[float] = []
        by_profile: dict[str, list[float]] = defaultdict(list)
        for (profile, base_id, observed_variant), row in indexed.items():
            if observed_variant != variant:
                continue
            if not row.get("scoring_available", True):
                continue
            baseline = indexed.get((profile, base_id, "no_memory_no_initial"))
            if baseline is None:
                if require_paired_baseline:
                    raise ValueError(
                        f"missing no-memory pair for {profile}/{base_id}/{variant}"
                    )
                continue
            if not baseline.get("scoring_available", True):
                continue
            delta = float(row.get("caption_token_f1") or 0.0) - float(
                baseline.get("caption_token_f1") or 0.0
            )
            deltas.append(delta)
            by_profile[profile].append(delta)
        paired[variant] = {
            "pair_count": len(deltas),
            "caption_token_f1_delta_macro": _mean(deltas),
            "by_profile": {
                profile: {
                    "pair_count": len(values),
                    "caption_token_f1_delta_macro": _mean(values),
                }
                for profile, values in sorted(by_profile.items())
            },
        }

    memory_rows = [
        row for row in execution
        if str(row["context_variant"]).startswith("with_memory")
    ]
    memory_generated = [row for row in memory_rows if row["status"] == "generated"]
    memory_nonempty = [
        row for row in memory_generated
        if isinstance(row.get("memory_input"), Mapping)
        and (
            bool(row["memory_input"].get("long_memory"))
            or row["memory_input"].get("short_memory") is not None
        )
    ]
    memory_reasons = Counter(
        str((row.get("memory_update") or {}).get("reason", "missing"))
        for row in memory_rows
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "completed": len(rows) == expected_slots,
        "expected_slots": expected_slots,
        "observed_slots": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "model_calls": len(generated),
        "skipped_input_dependencies": statuses.get("skipped_input_dependency", 0),
        "generated_schema_valid": sum(
            bool(row.get("prediction_schema_valid")) for row in generated
        ),
        "generated_schema_valid_rate": _mean([
            int(bool(row.get("prediction_schema_valid"))) for row in generated
        ]),
        "primary_metric": "ongoing/end per-slot macro caption token-F1; invalid/skipped=0",
        "ground_truth_unavailable_execution_slots": sum(
            not row.get("scoring_available", True) for row in execution
        ),
        "metrics": metrics,
        "paired_memory_deltas": paired,
        "memory": {
            "slots": len(memory_rows),
            "generated": len(memory_generated),
            "nonempty_input_slots": len(memory_nonempty),
            "nonempty_input_coverage_over_generated": (
                len(memory_nonempty) / len(memory_generated) if memory_generated else None
            ),
            "update_reasons": dict(sorted(memory_reasons.items())),
        },
        "platform_success_definition": (
            "all slots recorded and requested videos rendered; independent of model quality"
        ),
    }


def _model_files(checkpoint: Path) -> list[Path]:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError(f"invalid safetensors index: {index_path}")
        names = sorted(set(str(value) for value in weight_map.values()))
        files = [checkpoint / name for name in names]
        if any(path.parent != checkpoint for path in files):
            raise ValueError(f"unsafe model shard path in {index_path}")
        return [index_path, *files]
    return sorted(
        path for path in checkpoint.iterdir()
        if path.is_file() and MODEL_FILE_PATTERN.fullmatch(path.name)
    )


def checkpoint_record(
    checkpoint: Path,
    *,
    minimum_model_bytes: int = 1_000_000_000,
) -> tuple[dict[str, Any] | None, str | None]:
    checkpoint = checkpoint.resolve()
    try:
        step = int(checkpoint.name.removeprefix("checkpoint-"))
    except ValueError:
        return None, "invalid checkpoint directory name"
    try:
        required = [
            checkpoint / "config.json",
            checkpoint / "trainer_state.json",
            checkpoint / "v10_checkpoint_meta.json",
        ]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            return None, f"missing required files: {missing}"
        model_files = _model_files(checkpoint)
        tensor_files = [path for path in model_files if path.suffix == ".safetensors"]
        if not tensor_files:
            return None, "missing model safetensors"
        model_bytes = sum(path.stat().st_size for path in tensor_files)
        if model_bytes < minimum_model_bytes:
            return None, f"model safetensors total is too small: {model_bytes}"
        trainer = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
        metadata = json.loads((checkpoint / "v10_checkpoint_meta.json").read_text(encoding="utf-8"))
        trainer_step = int(trainer["global_step"])
        metadata_step = int(metadata["global_step"])
        if trainer_step != step or metadata_step != step:
            return None, (
                f"step mismatch directory={step} trainer={trainer_step} metadata={metadata_step}"
            )
        if metadata.get("schema_version") != "v10_checkpoint_meta_v1":
            return None, "v10 checkpoint metadata schema mismatch"
        tokenizer_ok = (checkpoint / "tokenizer.json").is_file() and (
            checkpoint / "tokenizer_config.json"
        ).is_file()
        processor_ok = any(
            (checkpoint / name).is_file()
            for name in (
                "preprocessor_config.json",
                "processor_config.json",
                "video_preprocessor_config.json",
            )
        )
        if not tokenizer_ok or not processor_ok:
            return None, "checkpoint lacks tokenizer or processor assets"
        completion = (checkpoint / "v10_checkpoint_meta.json").stat()
        newest_model_mtime = max(path.stat().st_mtime_ns for path in tensor_files)
        if completion.st_mtime_ns < newest_model_mtime:
            return None, "completion metadata predates a model tensor"
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return {
        "path": str(checkpoint),
        "global_step": step,
        "model_bytes": model_bytes,
        "model_files": [path.name for path in model_files],
        "completion_mtime_ns": completion.st_mtime_ns,
        "manifest_digest": metadata.get("manifest_digest"),
        "data_config_digest": metadata.get("data_config_digest"),
    }, None


def stat_complete_checkpoint_record(
    checkpoint: Path,
    *,
    minimum_model_bytes: int = 1_000_000_000,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate a checkpoint while tolerating an unreadable completion payload.

    Some training pods create ``v10_checkpoint_meta.json`` as mode 0600 owned by
    root even though the actual inference assets are world-readable. This
    validator never changes permissions. It accepts that one file through
    stat-only evidence only when the trainer step matches the directory and the
    non-empty completion marker is newer than every model tensor and trainer
    state file. If the payload is readable, strict schema/step checks apply.
    """
    checkpoint = checkpoint.resolve()
    try:
        step = int(checkpoint.name.removeprefix("checkpoint-"))
    except ValueError:
        return None, "invalid checkpoint directory name"
    try:
        required = [
            checkpoint / "config.json",
            checkpoint / "trainer_state.json",
            checkpoint / "v10_checkpoint_meta.json",
        ]
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            return None, f"missing required files: {missing}"
        model_files = _model_files(checkpoint)
        tensor_files = [path for path in model_files if path.suffix == ".safetensors"]
        if not tensor_files:
            return None, "missing model safetensors"
        model_bytes = sum(path.stat().st_size for path in tensor_files)
        if model_bytes < minimum_model_bytes:
            return None, f"model safetensors total is too small: {model_bytes}"
        trainer_path = checkpoint / "trainer_state.json"
        trainer = json.loads(trainer_path.read_text(encoding="utf-8"))
        trainer_step = int(trainer["global_step"])
        if trainer_step != step:
            return None, f"step mismatch directory={step} trainer={trainer_step}"
        tokenizer_ok = (checkpoint / "tokenizer.json").is_file() and (
            checkpoint / "tokenizer_config.json"
        ).is_file()
        processor_ok = any(
            (checkpoint / name).is_file()
            for name in (
                "preprocessor_config.json",
                "processor_config.json",
                "video_preprocessor_config.json",
            )
        )
        if not tokenizer_ok or not processor_ok:
            return None, "checkpoint lacks tokenizer or processor assets"
        completion_path = checkpoint / "v10_checkpoint_meta.json"
        completion = completion_path.stat()
        if completion.st_size <= 0:
            return None, "completion metadata is empty"
        newest_required_mtime = max(
            trainer_path.stat().st_mtime_ns,
            *(path.stat().st_mtime_ns for path in tensor_files),
        )
        if completion.st_mtime_ns < newest_required_mtime:
            return None, "completion metadata predates a model tensor or trainer state"
        try:
            metadata = json.loads(completion_path.read_text(encoding="utf-8"))
        except PermissionError as exc:
            metadata = None
            completion_validation = "stat_only_unreadable_metadata"
            metadata_read_error = f"{type(exc).__name__}: {exc}"
        if metadata is not None:
            metadata_step = int(metadata["global_step"])
            if metadata_step != step:
                return None, f"step mismatch directory={step} metadata={metadata_step}"
            if metadata.get("schema_version") != "v10_checkpoint_meta_v1":
                return None, "v10 checkpoint metadata schema mismatch"
            completion_validation = "readable_metadata_schema_and_step"
            metadata_read_error = None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return {
        "path": str(checkpoint),
        "global_step": step,
        "model_bytes": model_bytes,
        "model_files": [path.name for path in model_files],
        "completion_mtime_ns": completion.st_mtime_ns,
        "completion_metadata_size": completion.st_size,
        "completion_metadata_mode": stat.filemode(completion.st_mode),
        "completion_metadata_uid": completion.st_uid,
        "completion_metadata_gid": completion.st_gid,
        "completion_validation": completion_validation,
        "completion_metadata_read_error": metadata_read_error,
        "manifest_digest": metadata.get("manifest_digest") if metadata else None,
        "data_config_digest": metadata.get("data_config_digest") if metadata else None,
    }, None


def _step_state_report(run_root: Path) -> dict[str, Any] | None:
    step_state_path = run_root / "prepared_data" / "v10_step_state.json"
    if not step_state_path.is_file():
        return None
    try:
        raw = json.loads(step_state_path.read_text(encoding="utf-8"))
        return {
            "path": str(step_state_path),
            "global_step": int(raw.get("global_step", 0)),
            "memory_noise_probability": raw.get("memory_noise_probability"),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"path": str(step_state_path), "error": f"{type(exc).__name__}: {exc}"}


def select_latest_complete_checkpoint(
    training_run_root: Path,
    *,
    minimum_model_bytes: int = 1_000_000_000,
) -> tuple[Path, dict[str, Any]]:
    run_root = training_run_root.resolve(strict=True)
    train_root = run_root / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(train_root)
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for path in sorted(train_root.glob("checkpoint-*")):
        record, reason = checkpoint_record(
            path, minimum_model_bytes=minimum_model_bytes
        )
        if record is None:
            rejected.append({"path": str(path), "reason": str(reason)})
        else:
            candidates.append(record)
    if not candidates:
        raise RuntimeError(f"no complete inference checkpoint under {train_root}")
    selected = max(
        candidates,
        key=lambda row: (int(row["global_step"]), int(row["completion_mtime_ns"])),
    )
    report = {
        "selected_at": utc_now(),
        "training_run_root": str(run_root),
        "selection_rule": "maximum global_step among complete V10 checkpoints at pod startup",
        "selected": selected,
        "complete_candidate_count": len(candidates),
        "rejected_candidate_count": len(rejected),
        "rejected": rejected,
        "training_step_state_at_selection": _step_state_report(run_root),
    }
    return Path(str(selected["path"])), report


def select_latest_stat_complete_checkpoint(
    training_run_root: Path,
    *,
    minimum_model_bytes: int = 1_000_000_000,
) -> tuple[Path, dict[str, Any]]:
    run_root = training_run_root.resolve(strict=True)
    train_root = run_root / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(train_root)
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for path in sorted(train_root.glob("checkpoint-*")):
        record, reason = stat_complete_checkpoint_record(
            path, minimum_model_bytes=minimum_model_bytes
        )
        if record is None:
            rejected.append({"path": str(path), "reason": str(reason)})
        else:
            candidates.append(record)
    if not candidates:
        raise RuntimeError(f"no stat-complete inference checkpoint under {train_root}")
    selected = max(
        candidates,
        key=lambda row: (int(row["global_step"]), int(row["completion_mtime_ns"])),
    )
    report = {
        "selected_at": utc_now(),
        "training_run_root": str(run_root),
        "selection_rule": (
            "maximum global_step with matching readable trainer state, complete "
            "readable inference assets, and a non-empty completion marker newer "
            "than model tensors; unreadable completion payload is never modified"
        ),
        "selected": selected,
        "complete_candidate_count": len(candidates),
        "rejected_candidate_count": len(rejected),
        "rejected": rejected,
        "training_step_state_at_selection": _step_state_report(run_root),
    }
    return Path(str(selected["path"])), report


def validate_explicit_checkpoint(
    checkpoint: Path,
    *,
    minimum_model_bytes: int = 1_000_000_000,
) -> tuple[Path, dict[str, Any]]:
    record, reason = checkpoint_record(
        checkpoint, minimum_model_bytes=minimum_model_bytes
    )
    if record is None:
        raise RuntimeError(f"explicit checkpoint is incomplete: {checkpoint}: {reason}")
    return checkpoint.resolve(), {
        "selected_at": utc_now(),
        "selection_rule": "explicit complete V10 checkpoint",
        "selected": record,
        "complete_candidate_count": 1,
        "rejected_candidate_count": 0,
        "rejected": [],
    }


def _pin_file(source: Path, target: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with source.open("rb") as reader, target.open("xb") as writer:
        for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b""):
            writer.write(chunk)
            digest.update(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    os.chmod(target, 0o444)
    return {
        "name": source.name,
        "size": target.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def pin_checkpoint(
    source: Path,
    target: Path,
    *,
    require_completion_metadata: bool = True,
) -> dict[str, Any]:
    source = source.resolve(strict=True)
    target = target.resolve()
    if target.exists():
        manifest_path = target / "pin_manifest.json"
        if not manifest_path.is_file():
            raise FileExistsError(f"checkpoint pin exists without manifest: {target}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("files", []):
            path = target / str(item["name"])
            if not path.is_file() or path.stat().st_size != int(item["size"]):
                raise RuntimeError(f"checkpoint pin file is missing or truncated: {path}")
        return manifest
    model_files = _model_files(source)
    selected = {path.name: path for path in model_files}
    skipped_unreadable: list[dict[str, str]] = []
    for name in PIN_SMALL_FILES:
        path = source / name
        if path.is_file():
            try:
                with path.open("rb") as handle:
                    handle.read(1)
            except PermissionError as exc:
                if name == "v10_checkpoint_meta.json" and not require_completion_metadata:
                    skipped_unreadable.append({
                        "name": name,
                        "reason": f"{type(exc).__name__}: {exc}",
                    })
                    continue
                raise
            selected[name] = path
    required_names = {"config.json", "trainer_state.json"}
    if require_completion_metadata:
        required_names.add("v10_checkpoint_meta.json")
    if not required_names.issubset(selected):
        raise RuntimeError(f"checkpoint pin selection lacks {required_names - set(selected)}")
    stage = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True, mode=0o755)
    try:
        files = [_pin_file(selected[name], stage / name) for name in sorted(selected)]
        manifest = {
            "schema_version": "v5_3_checkpoint_pin_v2",
            "created_at": utc_now(),
            "source_checkpoint": str(source),
            "pin_method": "byte_copy_to_independent_inode_then_atomic_directory_rename",
            "completion_metadata_required": require_completion_metadata,
            "skipped_unreadable_files": skipped_unreadable,
            "files": files,
            "model_sha256": {
                item["name"]: item["sha256"]
                for item in files if item["name"].endswith(".safetensors")
            },
        }
        write_json(stage / "pin_manifest.json", manifest)
        os.chmod(stage / "pin_manifest.json", 0o444)
        os.replace(stage, target)
        return manifest
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def run_fingerprint(
    *,
    spec: Mapping[str, Any],
    slots: Sequence[RolloutSlot],
    pin_manifest: Mapping[str, Any],
    initial_max_new_tokens: int,
    execution_max_new_tokens: int,
) -> str:
    return stable_json_sha256({
        "episode_spec_sha256": spec["spec_sha256"],
        "slot_ids": [slot.slot_id for slot in slots],
        "source_checkpoint": pin_manifest["source_checkpoint"],
        "model_sha256": pin_manifest["model_sha256"],
        "initial_max_new_tokens": initial_max_new_tokens,
        "execution_max_new_tokens": execution_max_new_tokens,
        "generation": {"do_sample": False, "dtype": "bfloat16", "attention": "sdpa"},
    })


def _code_hashes() -> dict[str, str]:
    module = Path(__file__).resolve()
    renderer = module.parent / "video.py"
    result = {str(module): sha256_file(module)}
    if renderer.is_file():
        result[str(renderer)] = sha256_file(renderer)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-spec", type=Path, required=True)
    parser.add_argument("--training-run-root", type=Path)
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--initial-max-new-tokens", type=int, default=2048)
    parser.add_argument("--execution-max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--anchor-stride-frames",
        type=int,
        help="Optional fixed whole-video inference stride; default uses label midpoints.",
    )
    parser.add_argument("--verify-sources", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.initial_max_new_tokens <= 0 or args.execution_max_new_tokens <= 0:
        raise ValueError("generation token limits must be positive")
    output = args.output_dir.resolve()
    if output.exists() and not args.resume and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = load_episode_spec(args.episode_spec, verify_sources=args.verify_sources)
    slots = build_rollout_slots(
        spec,
        anchor_stride_frames=args.anchor_stride_frames,
    )

    manifest_path = output / "run_manifest.json"
    previous = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file() else None
    )
    pin_path = output / "checkpoint_pin"
    if previous is not None:
        selection = dict(previous["checkpoint_selection"])
        pin_manifest = pin_checkpoint(pin_path, pin_path)
    else:
        if args.checkpoint == "auto":
            if args.training_run_root is None:
                raise ValueError("--training-run-root is required for --checkpoint auto")
            checkpoint, selection = select_latest_complete_checkpoint(args.training_run_root)
        else:
            checkpoint, selection = validate_explicit_checkpoint(Path(args.checkpoint))
        write_json(output / "checkpoint_selection.json", selection)
        pin_manifest = pin_checkpoint(checkpoint, pin_path)
    fingerprint = run_fingerprint(
        spec=spec,
        slots=slots,
        pin_manifest=pin_manifest,
        initial_max_new_tokens=args.initial_max_new_tokens,
        execution_max_new_tokens=args.execution_max_new_tokens,
    )
    if previous is not None:
        if previous.get("run_fingerprint") != fingerprint:
            raise ValueError("output directory belongs to a different rollout configuration")
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_now(),
            "run_fingerprint": fingerprint,
            "episode_spec": str(Path(args.episode_spec).resolve()),
            "episode_spec_sha256": spec["spec_sha256"],
            "episode_key": spec["episode_key"],
            "benchmark3_uid": spec["benchmark3_uid"],
            "split": "test",
            "total_frames": spec["total_frames"],
            "fps": spec["fps"],
            "profiles": list(PROFILES),
            "context_variants": list(CLEAN_CONTEXT_VARIANTS),
            "slot_count": len(slots),
            "sampling": {
                "mode": (
                    "label_midpoint"
                    if args.anchor_stride_frames is None
                    else "fixed_frame_stride"
                ),
                "anchor_stride_frames": args.anchor_stride_frames,
                "execution_anchor_frames": sorted({
                    slot.anchor_frame
                    for slot in slots
                    if slot.category in {"ongoing", "end"}
                }),
            },
            "checkpoint_selection": selection,
            "checkpoint_pin": str(pin_path),
            "checkpoint_pin_manifest": pin_manifest,
            "generation": {
                "do_sample": False,
                "dtype": "bfloat16",
                "attention": "sdpa",
                "use_cache": True,
                "initial_max_new_tokens": args.initial_max_new_tokens,
                "execution_max_new_tokens": args.execution_max_new_tokens,
            },
            "memory": {
                "source": "model_predictions_only",
                "branch_isolation": True,
                "invalid_json": "state_unchanged",
                "invalid_initial_plan": "with_initial_branch_skipped_no_gt_fallback",
                "long_memory_limit": LONG_MEMORY_LIMIT,
            },
            "environment": {
                "hostname": socket.gethostname(),
                "python": sys.version,
                "executable": sys.executable,
                "aihc_job_name": os.environ.get("AIHC_JOB_NAME"),
                "aihc_job_id": os.environ.get("AIHC_JOB_ID"),
            },
            "code_sha256": _code_hashes(),
        }
        write_json(manifest_path, manifest)

    generator = Generator(pin_path, processor_path=pin_path, device=args.device)
    predictions_path = output / "predictions.jsonl"
    rows = run_rollout(
        slots=slots,
        generator=generator,
        source_root=Path(args.episode_spec).resolve().parent,
        output_path=predictions_path,
        initial_max_new_tokens=args.initial_max_new_tokens,
        execution_max_new_tokens=args.execution_max_new_tokens,
        resume=args.resume,
    )
    summary = summarize_rollout(rows, expected_slots=len(slots))
    write_json(output / "summary.json", summary)

    videos: dict[str, Any] = {}
    if args.render:
        from .video import render_all_videos

        videos = render_all_videos(spec=spec, rows=rows, output_dir=output / "videos")
        summary["videos"] = videos
        summary["platform_succeeded"] = (
            summary["completed"]
            and all(
                int(report.get("source_frames", report["frames"]))
                == int(spec["total_frames"])
                for report in videos.values()
            )
        )
        write_json(output / "summary.json", summary)
    else:
        summary["videos"] = {}
        summary["platform_succeeded"] = summary["completed"]
        write_json(output / "summary.json", summary)
    result = {
        "output_dir": str(output),
        "platform_succeeded": summary["platform_succeeded"],
        "model_quality_gate": "reported_only_not_required_for_platform_success",
        "checkpoint": pin_manifest["source_checkpoint"],
        "pinned_checkpoint": str(pin_path),
        "summary": str(output / "summary.json"),
        "predictions": str(predictions_path),
        "videos": videos,
    }
    write_json(output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["platform_succeeded"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLEAN_CONTEXT_VARIANTS",
    "EPISODE_SPEC_VERSION",
    "PredictedMemoryState",
    "RolloutSlot",
    "build_rollout_slots",
    "checkpoint_record",
    "execution_caption_score",
    "load_episode_spec",
    "pin_checkpoint",
    "run_fingerprint",
    "run_rollout",
    "select_rollout_slots",
    "select_latest_complete_checkpoint",
    "select_latest_stat_complete_checkpoint",
    "stat_complete_checkpoint_record",
    "summarize_rollout",
    "token_f1",
]
