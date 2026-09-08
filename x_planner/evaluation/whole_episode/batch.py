#!/usr/bin/env python3
"""Run whole-episode rollouts with memory and initial plans in parallel.

The rollout profile defaults to joint Action+Segment. Sources annotated without
Segments, such as RoboDojo, must select ``--profile action_only`` so that the
inference output spec matches the profile the checkpoint was trained on.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import socket
import sys
import traceback
from typing import Any

from x_planner.data.event_states.materialize_episode import PROFILES
from x_planner.data.event_states.inference import Generator, write_json
from .rollout import (
    SCHEMA_VERSION,
    build_rollout_slots,
    jsonl_rows,
    load_episode_spec,
    pin_checkpoint,
    run_rollout,
    select_latest_complete_checkpoint,
    select_latest_stat_complete_checkpoint,
    select_rollout_slots,
    sha256_file,
    summarize_rollout,
    utc_now,
    validate_explicit_checkpoint,
)
from .video import LAYOUT_VERSION, render_focus_video


BATCH_SPEC_VERSION = "v5_3_whole_episode_batch_spec_v1"
BATCH_RUN_VERSION = "v5_3_whole_episode_batch_run_v1"
PROFILE = "action_segment_joint"
CONTEXT_VARIANT = "with_memory_with_initial"
INITIAL_PLAN_DEMO_CONSTRAINT = (
    "Demo plan constraint: produce 3 to 12 Actions, with 1 to 4 Segments per "
    "Action. Keep captions concise and do not repeat an Action or Segment caption."
)
INITIAL_PLAN_DEMO_CONSTRAINT_ACTION_ONLY = (
    "Demo plan constraint: produce 3 to 12 Actions. Keep captions concise and do "
    "not repeat an Action caption."
)


def _profile_emits_segments(profile: str) -> bool:
    if profile not in PROFILES:
        raise ValueError(f"unknown rollout profile: {profile}")
    return profile != "action_only"


def _initial_plan_demo_constraint(profile: str) -> str:
    return (
        INITIAL_PLAN_DEMO_CONSTRAINT
        if _profile_emits_segments(profile)
        else INITIAL_PLAN_DEMO_CONSTRAINT_ACTION_ONLY
    )


def _initial_plan_correction_suffix(profile: str, error: str) -> str:
    common = (
        f"{_initial_plan_demo_constraint(profile)}\n"
        "Correction request: the previous response failed schema validation: "
        f"{error[:600]}. Regenerate the complete JSON object from scratch. "
        "Every initial_plan item must use consecutive index values and exactly "
        "the keys index and action. "
    )
    if not _profile_emits_segments(profile):
        return common + (
            "Every action must contain exactly caption. Do not emit segments for "
            "any action. Return only one complete JSON object with no markdown."
        )
    return common + (
        "Every action must contain exactly caption and "
        "segments. Segments must be a non-empty array; every segment must use "
        "consecutive index values and exactly the keys index and segment, and each "
        "segment object must contain exactly caption. Do not omit segments from "
        "any action. Return only one complete JSON object with no markdown."
    )


def _stable_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_batch_spec(path: Path, *, expected_episodes: int) -> dict[str, Any]:
    path = path.resolve(strict=True)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("batch spec must be an object")
    if value.get("schema_version") != BATCH_SPEC_VERSION:
        raise ValueError("batch spec schema_version mismatch")
    raw_entries = value.get("episodes")
    if isinstance(raw_entries, (str, bytes)) or not isinstance(raw_entries, Sequence):
        raise TypeError("batch spec episodes must be an array")
    if len(raw_entries) != expected_episodes:
        raise ValueError(
            f"batch spec has {len(raw_entries)} episodes, expected {expected_episodes}"
        )
    entries: list[dict[str, Any]] = []
    names: set[str] = set()
    specs: set[str] = set()
    for index, raw in enumerate(raw_entries, 1):
        if not isinstance(raw, Mapping):
            raise TypeError(f"episodes[{index - 1}] must be an object")
        name = str(raw.get("name") or "")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", name):
            raise ValueError(f"episodes[{index - 1}] has an unsafe name: {name!r}")
        spec = Path(str(raw.get("spec") or ""))
        if not spec.is_absolute() or not spec.is_file():
            raise FileNotFoundError(f"episodes[{index - 1}] spec is missing: {spec}")
        selection = str(raw.get("selection") or "")
        if selection not in {"benchmark3_holdout", "supplemental_non_holdout"}:
            raise ValueError(f"episodes[{index - 1}] has invalid selection: {selection}")
        if name in names or str(spec.resolve()) in specs:
            raise ValueError(f"duplicate batch episode at index {index - 1}")
        frozen = json.loads(spec.read_text(encoding="utf-8"))
        total_frames = int(frozen.get("total_frames", 0))
        if total_frames <= 0:
            raise ValueError(f"episodes[{index - 1}] has invalid total_frames")
        names.add(name)
        specs.add(str(spec.resolve()))
        entries.append({
            "index": index,
            "name": name,
            "spec": str(spec.resolve()),
            "selection": selection,
            "total_frames": total_frames,
        })
    result = dict(value)
    result["episodes"] = entries
    result["spec_path"] = str(path)
    result["spec_sha256"] = sha256_file(path)
    return result


def _selected_slots(spec: Mapping[str, Any], stride: int, profile: str = PROFILE):
    return select_rollout_slots(
        build_rollout_slots(
            spec,
            anchor_stride_frames=stride,
            minimum_units_per_profile=1,
            dense_gap_policy="schema_proxy",
        ),
        profiles=(profile,),
        context_variants=(CONTEXT_VARIANT,),
    )


def _write_initial_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _initial_plan_quality_error(plan: Any, *, profile: str = PROFILE) -> str | None:
    expect_segments = _profile_emits_segments(profile)
    if isinstance(plan, (str, bytes)) or not isinstance(plan, Sequence) or not plan:
        return "initial plan is empty"
    if len(plan) > 16:
        return f"initial plan has {len(plan)} actions; maximum accepted is 16"
    action_captions: list[str] = []
    for offset, item in enumerate(plan, 1):
        action = item.get("action") if isinstance(item, Mapping) else None
        if not isinstance(action, Mapping):
            return f"initial plan action {offset} is malformed"
        action_captions.append(" ".join(str(action.get("caption", "")).lower().split()))
        segments = action.get("segments")
        if not expect_segments:
            if segments is not None:
                return (
                    f"initial plan action {offset} emitted segments under the "
                    f"{profile} profile"
                )
            continue
        if isinstance(segments, (str, bytes)) or not isinstance(segments, Sequence):
            return f"initial plan action {offset} has no segment array"
        if not 1 <= len(segments) <= 8:
            return (
                f"initial plan action {offset} has {len(segments)} segments; "
                "accepted range is 1 to 8"
            )
    if len(action_captions) >= 4:
        most_common = max(action_captions.count(value) for value in set(action_captions))
        if most_common * 2 > len(action_captions):
            return "more than half of action captions are exact repetitions"
    return None


def _initial_plan_gate(
    *,
    slots: Sequence[Any],
    generator: Any,
    source_root: Path,
    predictions_path: Path,
    attempt_dir: Path,
    initial_token_budgets: Sequence[int],
    profile: str = PROFILE,
) -> dict[str, Any]:
    partial = predictions_path.with_suffix(predictions_path.suffix + ".partial")
    if predictions_path.is_file() or partial.is_file():
        existing = jsonl_rows(predictions_path if predictions_path.is_file() else partial)
        if not existing:
            raise ValueError("existing rollout file is empty")
        row = existing[0]
        if row.get("category") != "initial_plan":
            raise ValueError("existing rollout does not begin with an initial plan")
        return {
            "resumed": True,
            "schema_valid": bool(row.get("prediction_schema_valid")),
            "plan_steps": len((row.get("prediction") or {}).get("initial_plan", [])),
            "max_new_tokens": row.get("max_new_tokens"),
            "prompt_suffix": row.get("inference_prompt_suffix"),
        }

    attempt_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    selected: Mapping[str, Any] | None = None
    selected_prompt_suffix: str | None = None
    prompt_suffix = _initial_plan_demo_constraint(profile)
    for attempt_index, budget in enumerate(initial_token_budgets):
        attempt_path = attempt_dir / f"max_tokens_{budget}.jsonl"
        rows = run_rollout(
            slots=slots[:1],
            generator=generator,
            source_root=source_root,
            output_path=attempt_path,
            initial_max_new_tokens=budget,
            execution_max_new_tokens=1,
            resume=True,
            initial_prompt_suffix=prompt_suffix,
            compact_initial_plan=True,
        )
        row = rows[0]
        prediction = row.get("prediction")
        plan = (
            prediction.get("initial_plan")
            if isinstance(prediction, Mapping)
            else None
        )
        attempt = {
            "max_new_tokens": budget,
            "schema_valid": bool(row.get("prediction_schema_valid")),
            "schema_error": row.get("prediction_schema_error"),
            "output_tokens": row.get("output_tokens"),
            "plan_steps": len(plan) if isinstance(plan, Sequence) else 0,
            "json_repair": row.get("prediction_repair"),
            "structural_stop": row.get("initial_plan_structural_stop"),
            "quality_error": _initial_plan_quality_error(plan, profile=profile),
            "correction_prompt": attempt_index > 0,
            "path": str(attempt_path),
        }
        attempts.append(attempt)
        if (
            attempt["schema_valid"]
            and attempt["plan_steps"] > 0
            and attempt["quality_error"] is None
        ):
            selected = row
            selected_prompt_suffix = prompt_suffix
            break
        error = str(
            attempt["quality_error"]
            or row.get("prediction_schema_error")
            or "unknown schema error"
        )
        prompt_suffix = _initial_plan_correction_suffix(profile, error)
    write_json(attempt_dir / "attempts.json", {"attempts": attempts})
    if selected is None:
        raise RuntimeError(f"initial plan gate failed: {attempts}")
    _write_initial_row(partial, selected)
    return {
        "resumed": False,
        "schema_valid": True,
        "plan_steps": len(selected["prediction"]["initial_plan"]),
        "max_new_tokens": selected["max_new_tokens"],
        "prompt_suffix": selected_prompt_suffix,
        "attempts": attempts,
    }


def _episode_run(
    *,
    entry: Mapping[str, Any],
    generator: Any,
    checkpoint_pin: Path,
    output_root: Path,
    stride: int,
    initial_token_budgets: Sequence[int],
    initial_plan_structure_limits: tuple[int, int] | None,
    execution_max_new_tokens: int,
    execution_schema_retries: int,
    allow_execution_hold_fallback: bool,
    execution_context_policy: str,
    initial_plan_page_seconds: float,
    end_hold_seconds: float,
    worker_index: int,
    device: str,
    profile: str = PROFILE,
) -> dict[str, Any]:
    episode_root = output_root / "episodes" / str(entry["name"])
    episode_root.mkdir(parents=True, exist_ok=True)
    spec_path = Path(str(entry["spec"]))
    verify_sources = entry["selection"] == "benchmark3_holdout"
    spec = load_episode_spec(
        spec_path,
        verify_sources=verify_sources,
        minimum_intervals_per_unit=1,
    )
    slots = _selected_slots(spec, stride, profile)
    expected_anchors = len(range(0, int(spec["total_frames"]) - 1, stride)) + 1
    if len(slots) != expected_anchors + 1:
        raise ValueError(
            f"selected slot count {len(slots)} != initial + {expected_anchors} anchors"
        )
    manifest_path = episode_root / "run_manifest.json"
    episode_fingerprint = _stable_sha256({
        "episode_spec_sha256": spec["spec_sha256"],
        "profile": profile,
        "slot_ids": [slot.slot_id for slot in slots],
        "checkpoint_pin_manifest": sha256_file(checkpoint_pin / "pin_manifest.json"),
        "initial_token_budgets": list(initial_token_budgets),
        "initial_plan_structure_limits": initial_plan_structure_limits,
        "execution_max_new_tokens": execution_max_new_tokens,
        "initial_plan_page_seconds": initial_plan_page_seconds,
        "end_hold_seconds": end_hold_seconds,
        "temporal_decision_contract": True,
        "execution_schema_retries": execution_schema_retries,
        "execution_hold_fallback": allow_execution_hold_fallback,
        "execution_context_policy": execution_context_policy,
    })
    if manifest_path.is_file():
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior_manifest.get("run_fingerprint") != episode_fingerprint:
            raise ValueError(f"episode output belongs to a different run: {episode_root}")
    else:
        write_json(manifest_path, {
            "schema_version": BATCH_RUN_VERSION,
            "created_at": utc_now(),
            "run_fingerprint": episode_fingerprint,
            "episode_index": entry["index"],
            "episode_name": entry["name"],
            "episode_spec": str(spec_path),
            "episode_spec_sha256": spec["spec_sha256"],
            "episode_key": spec["episode_key"],
            "selection": entry["selection"],
            "source_verification": verify_sources,
            "profile": profile,
            "context_variant": CONTEXT_VARIANT,
            "anchor_stride_frames": stride,
            "execution_anchor_count": expected_anchors,
            "slot_count": len(slots),
            "checkpoint_pin": str(checkpoint_pin),
            "initial_token_budgets": list(initial_token_budgets),
            "initial_plan_structure_limits": initial_plan_structure_limits,
            "execution_max_new_tokens": execution_max_new_tokens,
            "memory_source": "model_predictions_only",
            "execution_context_policy": execution_context_policy,
            "ground_truth_visible": False,
            "dense_gap_policy": "schema_proxy_unscored",
            "temporal_decision_contract": True,
            "execution_schema_retries": execution_schema_retries,
            "execution_hold_fallback": (
                "last_valid_model_prediction_only"
                if allow_execution_hold_fallback
                else "disabled"
            ),
            "worker_index": worker_index,
            "device": device,
        })

    completed_result_path = episode_root / "result.json"
    if completed_result_path.is_file():
        completed_result = json.loads(completed_result_path.read_text(encoding="utf-8"))
        completed_video = Path(str((completed_result.get("video") or {}).get("output") or ""))
        if (
            completed_result.get("platform_succeeded") is not True
            or not completed_video.is_file()
            or not Path(str(completed_result.get("predictions") or "")).is_file()
            or not Path(str(completed_result.get("summary") or "")).is_file()
        ):
            raise RuntimeError(f"completed episode artifacts are incomplete: {episode_root}")
        return completed_result

    predictions_path = episode_root / "predictions.jsonl"
    initial_gate = _initial_plan_gate(
        slots=slots,
        generator=generator,
        source_root=spec_path.parent,
        predictions_path=predictions_path,
        attempt_dir=episode_root / "initial_plan_attempts",
        initial_token_budgets=initial_token_budgets,
        profile=profile,
    )
    if not initial_gate["schema_valid"]:
        raise RuntimeError("resumed initial plan is invalid")
    rows = run_rollout(
        slots=slots,
        generator=generator,
        source_root=spec_path.parent,
        output_path=predictions_path,
        initial_max_new_tokens=max(initial_token_budgets),
        execution_max_new_tokens=execution_max_new_tokens,
        resume=True,
        initial_prompt_suffix=initial_gate.get("prompt_suffix"),
        compact_initial_plan=True,
        enforce_temporal_decision_contract=True,
        execution_schema_retries=execution_schema_retries,
        allow_execution_hold_fallback=allow_execution_hold_fallback,
        execution_context_policy=execution_context_policy,
    )
    skipped = [row for row in rows if row.get("status") != "generated"]
    if skipped:
        raise RuntimeError(f"episode contains {len(skipped)} skipped rollout slots")
    invalid = [row for row in rows if not row.get("prediction_schema_valid")]
    if invalid:
        raise RuntimeError(f"episode contains {len(invalid)} schema-invalid rollout slots")
    summary = summarize_rollout(
        rows,
        expected_slots=len(slots),
        require_paired_baseline=False,
    )
    summary["initial_plan_gate"] = initial_gate
    summary["profile"] = profile
    summary["context_variant"] = CONTEXT_VARIANT
    summary["execution_context_policy"] = execution_context_policy
    summary["selection"] = entry["selection"]
    summary["execution_schema_retries_used"] = sum(
        max(0, len(row.get("prediction_attempts") or []) - 1) for row in rows
    )
    summary["execution_hold_fallbacks"] = sum(
        row.get("prediction_fallback") is not None for row in rows
    )
    video_name = (
        f"{profile}_display_memory_observations_only_with_initial.mp4"
        if execution_context_policy == "observations_only"
        else f"{profile}_with_memory_with_initial.mp4"
    )
    video_path = episode_root / "videos" / video_name
    video = render_focus_video(
        spec=spec,
        rows=rows,
        output_path=video_path,
        profile=profile,
        context_variant=CONTEXT_VARIANT,
        initial_plan_page_seconds=initial_plan_page_seconds,
        end_hold_seconds=end_hold_seconds,
    )
    summary["video"] = video
    summary["platform_succeeded"] = (
        summary["completed"]
        and int(video["source_frames"]) == int(spec["total_frames"])
        and int(video["initial_plan_pages"]) >= 1
    )
    write_json(episode_root / "summary.json", summary)
    result = {
        "episode_index": entry["index"],
        "episode_name": entry["name"],
        "episode_key": spec["episode_key"],
        "task_instruction": spec["task_instruction"],
        "selection": entry["selection"],
        "platform_succeeded": summary["platform_succeeded"],
        "initial_plan_steps": initial_gate["plan_steps"],
        "prediction_rows": len(rows),
        "predictions": str(predictions_path),
        "summary": str(episode_root / "summary.json"),
        "video": video,
    }
    write_json(episode_root / "result.json", result)
    if not result["platform_succeeded"]:
        raise RuntimeError(f"episode platform gate failed: {entry['name']}")
    return result


def _worker_main(
    *,
    worker_index: int,
    device: str,
    entries: Sequence[Mapping[str, Any]],
    checkpoint_pin: str,
    output_root: str,
    stride: int,
    initial_token_budgets: Sequence[int],
    initial_plan_structure_limits: tuple[int, int] | None,
    execution_max_new_tokens: int,
    execution_schema_retries: int,
    allow_execution_hold_fallback: bool,
    execution_context_policy: str,
    initial_plan_page_seconds: float,
    end_hold_seconds: float,
    profile: str = PROFILE,
) -> None:
    root = Path(output_root)
    worker_root = root / "workers" / f"worker_{worker_index:02d}"
    worker_root.mkdir(parents=True, exist_ok=True)
    try:
        generator = Generator(
            Path(checkpoint_pin),
            processor_path=Path(checkpoint_pin),
            device=device,
            initial_plan_structure_limits=initial_plan_structure_limits,
        )
        results: list[dict[str, Any]] = []
        for entry in entries:
            print(json.dumps({
                "event": "episode_start",
                "worker": worker_index,
                "device": device,
                "episode": entry["name"],
            }, sort_keys=True), flush=True)
            results.append(_episode_run(
                entry=entry,
                generator=generator,
                checkpoint_pin=Path(checkpoint_pin),
                output_root=root,
                stride=stride,
                initial_token_budgets=initial_token_budgets,
                initial_plan_structure_limits=initial_plan_structure_limits,
                execution_max_new_tokens=execution_max_new_tokens,
                execution_schema_retries=execution_schema_retries,
                allow_execution_hold_fallback=allow_execution_hold_fallback,
                execution_context_policy=execution_context_policy,
                initial_plan_page_seconds=initial_plan_page_seconds,
                end_hold_seconds=end_hold_seconds,
                worker_index=worker_index,
                device=device,
                profile=profile,
            ))
        write_json(worker_root / "result.json", {
            "worker_index": worker_index,
            "device": device,
            "episodes": results,
        })
    except BaseException as exc:
        write_json(worker_root / "failure.json", {
            "worker_index": worker_index,
            "device": device,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        raise


def _assign_entries(
    entries: Sequence[Mapping[str, Any]], worker_count: int
) -> list[list[Mapping[str, Any]]]:
    assignments: list[list[Mapping[str, Any]]] = [[] for _ in range(worker_count)]
    loads = [0 for _ in range(worker_count)]
    for entry in sorted(entries, key=lambda item: int(item["total_frames"]), reverse=True):
        worker = min(range(worker_count), key=lambda index: (loads[index], index))
        assignments[worker].append(entry)
        loads[worker] += int(entry["total_frames"])
    return assignments


def _code_hashes() -> dict[str, str]:
    module = Path(__file__).resolve()
    data_root = module.parents[2] / "data" / "event_states"
    names = (
        module,
        data_root / "materialize_episode.py",
        data_root / "inference.py",
        data_root / "memory.py",
        data_root / "prompt.py",
        data_root / "schema.py",
        module.parent / "rollout.py",
        module.parent / "video.py",
    )
    return {str(path): sha256_file(path) for path in names}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-spec", type=Path, required=True)
    parser.add_argument("--training-run-root", type=Path)
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument(
        "--checkpoint-selection-mode",
        choices=("strict", "stat_complete"),
        default="strict",
        help=(
            "Use stat_complete only for local read-only fallback when the training "
            "pod's completion metadata exists but is permission-inaccessible."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument(
        "--profile",
        choices=PROFILES,
        default=PROFILE,
        help=(
            "Label profile to roll out. Use action_only for sources trained "
            "without Segment labels, such as RoboDojo."
        ),
    )
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument("--anchor-stride-frames", type=int, default=10)
    parser.add_argument("--initial-token-budgets", default="4096,8192")
    parser.add_argument("--initial-plan-max-actions", type=int, default=0)
    parser.add_argument(
        "--initial-plan-max-segments-per-action", type=int, default=0
    )
    parser.add_argument("--execution-max-new-tokens", type=int, default=512)
    parser.add_argument("--execution-schema-retries", type=int, default=1)
    parser.add_argument(
        "--execution-hold-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--execution-context-policy",
        choices=("configured", "observations_only"),
        default="configured",
        help=(
            "observations_only keeps predicted memory for visualization but sends "
            "neither memory nor Initial Plan to execution prompts"
        ),
    )
    parser.add_argument("--initial-plan-page-seconds", type=float, default=2.0)
    parser.add_argument("--end-hold-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)

    if args.expected_episodes <= 0 or args.anchor_stride_frames <= 0:
        raise ValueError("episode count and anchor stride must be positive")
    if args.execution_max_new_tokens <= 0:
        raise ValueError("execution_max_new_tokens must be positive")
    if args.execution_schema_retries < 0:
        raise ValueError("execution_schema_retries must be non-negative")
    if not math.isfinite(args.initial_plan_page_seconds) or args.initial_plan_page_seconds <= 0:
        raise ValueError("initial_plan_page_seconds must be positive and finite")
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("--devices must contain unique device names")
    initial_token_budgets = tuple(
        int(value) for value in args.initial_token_budgets.split(",") if value.strip()
    )
    if not initial_token_budgets or any(value <= 0 for value in initial_token_budgets):
        raise ValueError("initial token budgets must be positive")
    if tuple(sorted(set(initial_token_budgets))) != initial_token_budgets:
        raise ValueError("initial token budgets must be unique and increasing")
    structural_limit_values = (
        args.initial_plan_max_actions,
        args.initial_plan_max_segments_per_action,
    )
    if any(value < 0 for value in structural_limit_values):
        raise ValueError("initial plan structural limits cannot be negative")
    if (structural_limit_values[0] == 0) != (structural_limit_values[1] == 0):
        raise ValueError("initial plan structural limits must both be zero or positive")
    if structural_limit_values != (0, 0) and not _profile_emits_segments(args.profile):
        raise ValueError(
            "initial plan structural limits require a segment-bearing profile; "
            f"pass 0 for both under {args.profile}"
        )
    initial_plan_structure_limits = (
        None if structural_limit_values == (0, 0) else structural_limit_values
    )

    batch = _load_batch_spec(
        args.batch_spec, expected_episodes=args.expected_episodes
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    previous = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    pin_path = output / "checkpoint_pin"
    if previous is not None:
        selection = dict(previous["checkpoint_selection"])
        pin_manifest = pin_checkpoint(pin_path, pin_path)
    else:
        if args.checkpoint == "auto":
            if args.training_run_root is None:
                raise ValueError("--training-run-root is required for --checkpoint auto")
            selector = (
                select_latest_complete_checkpoint
                if args.checkpoint_selection_mode == "strict"
                else select_latest_stat_complete_checkpoint
            )
            checkpoint, selection = selector(args.training_run_root)
        else:
            checkpoint, selection = validate_explicit_checkpoint(
                Path(args.checkpoint)
            )
        write_json(output / "checkpoint_selection.json", selection)
        pin_manifest = pin_checkpoint(
            checkpoint,
            pin_path,
            require_completion_metadata=args.checkpoint_selection_mode == "strict",
        )

    code_hashes = _code_hashes()
    run_fingerprint = _stable_sha256({
        "batch_spec_sha256": batch["spec_sha256"],
        "model_sha256": pin_manifest["model_sha256"],
        "source_checkpoint": pin_manifest["source_checkpoint"],
        "checkpoint_selection_mode": args.checkpoint_selection_mode,
        "devices": devices,
        "profile": args.profile,
        "anchor_stride_frames": args.anchor_stride_frames,
        "initial_token_budgets": initial_token_budgets,
        "initial_plan_structure_limits": initial_plan_structure_limits,
        "execution_max_new_tokens": args.execution_max_new_tokens,
        "execution_schema_retries": args.execution_schema_retries,
        "execution_hold_fallback": args.execution_hold_fallback,
        "execution_context_policy": args.execution_context_policy,
        "initial_plan_page_seconds": args.initial_plan_page_seconds,
        "end_hold_seconds": args.end_hold_seconds,
        "code_sha256": code_hashes,
    })
    if previous is not None:
        if previous.get("run_fingerprint") != run_fingerprint:
            raise ValueError("output directory belongs to a different batch run")
    else:
        write_json(manifest_path, {
            "schema_version": BATCH_RUN_VERSION,
            "created_at": utc_now(),
            "run_fingerprint": run_fingerprint,
            "batch_spec": batch["spec_path"],
            "batch_spec_sha256": batch["spec_sha256"],
            "episode_count": len(batch["episodes"]),
            "episodes": batch["episodes"],
            "profile": args.profile,
            "context_variant": CONTEXT_VARIANT,
            "anchor_stride_frames": args.anchor_stride_frames,
            "checkpoint_selection_mode": args.checkpoint_selection_mode,
            "initial_token_budgets": list(initial_token_budgets),
            "initial_plan_structure_limits": initial_plan_structure_limits,
            "execution_max_new_tokens": args.execution_max_new_tokens,
            "execution_schema_retries": args.execution_schema_retries,
            "execution_hold_fallback": args.execution_hold_fallback,
            "execution_context_policy": args.execution_context_policy,
            "initial_plan_page_seconds": args.initial_plan_page_seconds,
            "end_hold_seconds": args.end_hold_seconds,
            "devices": list(devices),
            "checkpoint_selection": selection,
            "checkpoint_pin": str(pin_path),
            "checkpoint_pin_manifest": pin_manifest,
            "layout_version": LAYOUT_VERSION,
            "environment": {
                "hostname": socket.gethostname(),
                "python": sys.version,
                "executable": sys.executable,
                "aihc_job_name": os.environ.get("AIHC_JOB_NAME"),
                "aihc_job_id": os.environ.get("AIHC_JOB_ID"),
            },
            "code_sha256": code_hashes,
        })

    assignments = _assign_entries(batch["episodes"], len(devices))
    assignment_report = [
        {
            "worker_index": index,
            "device": devices[index],
            "total_source_frames": sum(int(item["total_frames"]) for item in entries),
            "episodes": [item["name"] for item in entries],
        }
        for index, entries in enumerate(assignments)
    ]
    write_json(output / "assignments.json", {"workers": assignment_report})
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    for worker_index, entries in enumerate(assignments):
        process = context.Process(
            target=_worker_main,
            kwargs={
                "worker_index": worker_index,
                "device": devices[worker_index],
                "entries": entries,
                "checkpoint_pin": str(pin_path),
                "output_root": str(output),
                "stride": args.anchor_stride_frames,
                "initial_token_budgets": initial_token_budgets,
                "initial_plan_structure_limits": initial_plan_structure_limits,
                "execution_max_new_tokens": args.execution_max_new_tokens,
                "execution_schema_retries": args.execution_schema_retries,
                "allow_execution_hold_fallback": args.execution_hold_fallback,
                "execution_context_policy": args.execution_context_policy,
                "initial_plan_page_seconds": args.initial_plan_page_seconds,
                "end_hold_seconds": args.end_hold_seconds,
                "profile": args.profile,
            },
            name=f"v53-batch-worker-{worker_index}",
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failed_workers = [
        {"name": process.name, "pid": process.pid, "exitcode": process.exitcode}
        for process in processes
        if process.exitcode != 0
    ]
    if failed_workers:
        write_json(output / "batch_failure.json", {
            "failed_workers": failed_workers,
            "assignments": assignment_report,
        })
        raise RuntimeError(f"batch workers failed: {failed_workers}")

    results = [
        json.loads(
            (output / "episodes" / str(entry["name"]) / "result.json").read_text(
                encoding="utf-8"
            )
        )
        for entry in batch["episodes"]
    ]
    completed = sum(bool(result["platform_succeeded"]) for result in results)
    batch_result = {
        "schema_version": BATCH_RUN_VERSION,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "platform_succeeded": completed == len(results) == args.expected_episodes,
        "profile": args.profile,
        "episode_count": len(results),
        "completed_episodes": completed,
        "checkpoint": pin_manifest["source_checkpoint"],
        "pinned_checkpoint": str(pin_path),
        "output_dir": str(output),
        "episodes": results,
    }
    write_json(output / "batch_result.json", batch_result)
    print(json.dumps(batch_result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if batch_result["platform_succeeded"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BATCH_RUN_VERSION",
    "BATCH_SPEC_VERSION",
    "CONTEXT_VARIANT",
    "INITIAL_PLAN_DEMO_CONSTRAINT",
    "INITIAL_PLAN_DEMO_CONSTRAINT_ACTION_ONLY",
    "PROFILE",
    "_assign_entries",
    "_initial_plan_demo_constraint",
    "_initial_plan_quality_error",
    "_load_batch_spec",
    "_profile_emits_segments",
    "_selected_slots",
]
