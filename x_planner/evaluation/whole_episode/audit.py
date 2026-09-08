#!/usr/bin/env python3
"""Run the phased V5.3 context, memory, plan, progress, and End audit."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import stat
import sys
import traceback
from typing import Any

from .metrics import (
    AUDIT_SCHEMA_VERSION,
    CONTEXT_VARIANTS,
    aggregate_execution_metrics,
    assist_event_counts,
    dense_stride_frames,
    duration_quantile_episode_names,
    end_audit,
    execution_row_score,
    initial_plan_metrics,
    label_boundary_midpoint_frames,
    memory_trace_record,
    paired_context_deltas,
    raw_effective_record,
)
from x_planner.data.event_states.inference import Generator, write_json
from .batch import _load_batch_spec
from .rollout import (
    RolloutSlot,
    append_jsonl,
    build_rollout_slots,
    jsonl_rows,
    load_episode_spec,
    pin_checkpoint,
    run_rollout,
    select_latest_complete_checkpoint,
    select_latest_stat_complete_checkpoint,
    select_rollout_slots,
    sha256_file,
    stat_complete_checkpoint_record,
    utc_now,
    validate_explicit_checkpoint,
)
from .video import render_focus_video


RUN_SCHEMA_VERSION = "v5_3_context_end_audit_run_v1"
PROFILE = "action_segment_joint"
INITIAL_PLAN_PROFILES = ("action_only", "action_segment_joint")
PROTOCOLS = ("blinded", "assisted")
BLINDED_FORBIDDEN_PROMPT_TEXT = (
    "Temporal contract for this anchor:",
    "Demo plan constraint:",
    "Correction request:",
    "Do not choose End",
    "final terminal observation",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_MANIFEST = Path(os.environ.get(
    "XPLANNER_BENCHMARK3_SNAPSHOT_MANIFEST",
    "/path/to/benchmark3/snapshot/manifest.json",
))
PREPARE_SUMMARY = Path(os.environ.get(
    "XPLANNER_BENCHMARK3_PREPARE_SUMMARY",
    "/path/to/benchmark3/prepared_data/prepare_summary.json",
))
EXPOSURE_PLAN = PREPARE_SUMMARY.with_name("exposure_plan.json")
CACHE_RUN_ROOT = Path(os.environ.get(
    "XPLANNER_BENCHMARK3_CACHE_ROOT", "/path/to/benchmark3/run"
))
LIVE_CODE_ROOT = PROJECT_ROOT / "data" / "event_states"
PORTABLE_BUNDLE = Path(os.environ.get(
    "XPLANNER_BENCHMARK3_PORTABLE_BUNDLE", "/path/to/benchmark3/code"
))


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for value in values:
        append_jsonl(path, value)


def _initial_constraint(profile: str) -> str:
    if profile == "action_only":
        return (
            "Demo plan constraint: produce 3 to 12 concise Actions. Do not repeat "
            "an Action caption and do not add Segment fields."
        )
    if profile == "action_segment_joint":
        return (
            "Demo plan constraint: produce 3 to 12 Actions, with 1 to 4 Segments "
            "per Action. Keep captions concise and do not repeat an Action or "
            "Segment caption."
        )
    raise ValueError(f"unsupported initial plan profile: {profile}")


def _initial_quality_error(profile: str, plan: Any) -> str | None:
    if isinstance(plan, (str, bytes)) or not isinstance(plan, Sequence) or not plan:
        return "initial plan is empty"
    if len(plan) > 16:
        return f"initial plan has {len(plan)} actions; maximum accepted is 16"
    captions: list[str] = []
    for offset, item in enumerate(plan, 1):
        action = item.get("action") if isinstance(item, Mapping) else None
        if not isinstance(action, Mapping):
            return f"initial plan action {offset} is malformed"
        captions.append(" ".join(str(action.get("caption") or "").casefold().split()))
        if profile == "action_segment_joint":
            segments = action.get("segments")
            if isinstance(segments, (str, bytes)) or not isinstance(segments, Sequence):
                return f"initial plan action {offset} has no segment array"
            if not 1 <= len(segments) <= 8:
                return (
                    f"initial plan action {offset} has {len(segments)} segments; "
                    "accepted range is 1 to 8"
                )
    if len(captions) >= 4:
        most_common = max(Counter(captions).values())
        if most_common * 2 > len(captions):
            return "more than half of action captions are exact repetitions"
    return None


def _correction_suffix(profile: str, error: str) -> str:
    base = _initial_constraint(profile)
    if profile == "action_only":
        contract = (
            "Every initial_plan item must use consecutive index values and exactly "
            "the keys index and action. Every action must contain exactly caption."
        )
    else:
        contract = (
            "Every initial_plan item must use consecutive index values and exactly "
            "the keys index and action. Every action must contain exactly caption "
            "and segments. Segments must be a non-empty array; every segment must "
            "use consecutive index values and exactly the keys index and segment, "
            "and each segment object must contain exactly caption."
        )
    return (
        f"{base}\nCorrection request: the previous response failed strict validation: "
        f"{error[:600]}. Regenerate the complete JSON object from scratch. {contract} "
        "Return only one complete JSON object with no markdown."
    )


def _probe_initial_plan(
    *,
    slot: RolloutSlot,
    protocol: str,
    generator: Any,
    source_root: Path,
    output_root: Path,
    token_budgets: Sequence[int],
) -> dict[str, Any]:
    profile = slot.profile
    root = output_root / protocol / profile
    result_path = root / "result.json"
    selected_path = root / "selected_row.jsonl"
    if result_path.is_file() and selected_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        rows = jsonl_rows(selected_path)
        if len(rows) != 1:
            raise ValueError(f"selected initial plan row count differs: {selected_path}")
        result["selected_row"] = rows[0]
        return result
    root.mkdir(parents=True, exist_ok=True)
    budgets = (int(token_budgets[0]),) if protocol == "blinded" else tuple(token_budgets)
    suffix = None if protocol == "blinded" else _initial_constraint(profile)
    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_suffix: str | None = None
    for attempt_index, budget in enumerate(budgets, 1):
        attempt_path = root / f"attempt_{attempt_index:02d}_{budget}.jsonl"
        rows = run_rollout(
            slots=[slot],
            generator=generator,
            source_root=source_root,
            output_path=attempt_path,
            initial_max_new_tokens=budget,
            execution_max_new_tokens=1,
            resume=True,
            initial_prompt_suffix=suffix,
            compact_initial_plan=protocol == "assisted",
            enforce_temporal_decision_contract=False,
            execution_schema_retries=0,
            allow_execution_hold_fallback=False,
            allow_initial_plan_json_repair=protocol == "assisted",
            allow_ongoing_early_end_normalization=False,
            allow_terminal_progress_normalization=False,
            memory_update_source="raw" if protocol == "blinded" else "effective",
            protocol=protocol,
        )
        row = rows[0]
        prediction = row.get("prediction")
        plan = prediction.get("initial_plan") if isinstance(prediction, Mapping) else None
        quality_error = _initial_quality_error(profile, plan)
        attempt = {
            "attempt": attempt_index,
            "max_new_tokens": budget,
            "prompt_suffix": suffix,
            "prompt_sha256": row.get("prompt_sha256"),
            "output_tokens": row.get("output_tokens"),
            "raw_schema_valid": row.get("raw_prediction_schema_valid"),
            "effective_schema_valid": row.get("prediction_schema_valid"),
            "schema_error": row.get("prediction_schema_error"),
            "quality_error": quality_error,
            "plan_steps": len(plan) if isinstance(plan, Sequence) else 0,
            "repair": row.get("prediction_repair"),
            "normalization": row.get("prediction_normalization"),
            "path": str(attempt_path),
        }
        attempts.append(attempt)
        selected = dict(row)
        selected_suffix = suffix
        if row.get("prediction_schema_valid") and quality_error is None:
            break
        suffix = _correction_suffix(
            profile,
            str(quality_error or row.get("prediction_schema_error") or "unknown error"),
        )
    if selected is None:
        raise RuntimeError("initial plan probe produced no attempts")
    plan = (
        selected["prediction"].get("initial_plan")
        if isinstance(selected.get("prediction"), Mapping)
        else None
    )
    quality_error = _initial_quality_error(profile, plan)
    passed = bool(selected.get("prediction_schema_valid") and quality_error is None)
    rollout_seed = dict(selected)
    if not passed:
        gate_error = (
            "initial_plan_quality_gate_failed: "
            + str(quality_error or selected.get("prediction_schema_error") or "unknown")
        )
        rollout_seed["raw_prediction_schema_valid"] = False
        rollout_seed["raw_prediction_schema_error"] = gate_error
        rollout_seed["prediction"] = None
        rollout_seed["prediction_schema_valid"] = False
        rollout_seed["prediction_schema_error"] = gate_error
        rollout_seed["quality_gate_failed"] = True
    _write_jsonl(selected_path, [rollout_seed])
    result = {
        "schema_version": RUN_SCHEMA_VERSION,
        "protocol": protocol,
        "profile": profile,
        "passed": passed,
        "failed_scores_zero": not passed,
        "selected_prompt_suffix": selected_suffix,
        "selected_max_new_tokens": selected.get("max_new_tokens"),
        "selected_row_path": str(selected_path),
        "quality_error": quality_error,
        "attempts": attempts,
        "raw_metrics": initial_plan_metrics(
            selected.get("raw_prediction"),
            selected["ground_truth"],
            schema_valid=bool(
                passed and selected.get("raw_prediction_schema_valid")
            ),
        ),
        "effective_metrics": initial_plan_metrics(
            selected.get("prediction"),
            selected["ground_truth"],
            schema_valid=bool(
                passed and selected.get("prediction_schema_valid")
            ),
        ),
    }
    write_json(result_path, result)
    result["selected_row"] = rollout_seed
    return result


def _seed_rollout(path: Path, row: Mapping[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    if path.is_file() or partial.is_file():
        return
    append_jsonl(partial, row)


def _protocol_kwargs(protocol: str) -> dict[str, Any]:
    if protocol == "blinded":
        return {
            "compact_initial_plan": False,
            "enforce_temporal_decision_contract": False,
            "execution_schema_retries": 0,
            "allow_execution_hold_fallback": False,
            "allow_initial_plan_json_repair": False,
            "allow_ongoing_early_end_normalization": False,
            "allow_terminal_progress_normalization": False,
            "memory_update_source": "raw",
            "protocol": protocol,
        }
    if protocol == "assisted":
        return {
            "compact_initial_plan": True,
            "enforce_temporal_decision_contract": True,
            "execution_schema_retries": 1,
            "allow_execution_hold_fallback": True,
            "allow_initial_plan_json_repair": True,
            "allow_ongoing_early_end_normalization": True,
            "allow_terminal_progress_normalization": True,
            "memory_update_source": "effective",
            "protocol": protocol,
        }
    raise ValueError(f"unknown protocol: {protocol}")


def _assert_blinded(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        prompt = str(row.get("prompt") or "")
        forbidden = [text for text in BLINDED_FORBIDDEN_PROMPT_TEXT if text in prompt]
        if forbidden:
            raise RuntimeError(
                f"blinded prompt leakage at {row.get('slot_id')}: {forbidden}"
            )
        if row.get("prediction_repair") or row.get("prediction_normalization"):
            raise RuntimeError(f"blinded postprocessing at {row.get('slot_id')}")
        if row.get("prediction_fallback") or len(row.get("prediction_attempts") or []) > 1:
            raise RuntimeError(f"blinded retry/fallback at {row.get('slot_id')}")
        if row.get("raw_prediction_schema_valid"):
            if row.get("raw_prediction") != row.get("prediction"):
                raise RuntimeError(f"blinded raw/effective drift at {row.get('slot_id')}")
        elif row.get("prediction") is not None:
            raise RuntimeError(f"blinded invalid output became effective at {row.get('slot_id')}")
        update = row.get("memory_update")
        if (
            isinstance(update, Mapping)
            and row.get("context_variant", "").startswith("with_memory")
            and row.get("category") == "ongoing"
            and row.get("status") == "generated"
            and update.get("source") != "raw"
        ):
            raise RuntimeError(f"blinded memory source drift at {row.get('slot_id')}")


def _annotate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    entry: Mapping[str, Any],
    anchor_mode: str,
) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            "episode_name": entry["name"],
            "selection": entry["selection"],
            "anchor_mode": anchor_mode,
        }
        for row in rows
    ]


def _run_schedule(
    *,
    spec: Mapping[str, Any],
    spec_path: Path,
    entry: Mapping[str, Any],
    anchor_mode: str,
    anchor_frames: Sequence[int],
    protocol: str,
    probe: Mapping[str, Any],
    generator: Any,
    output_root: Path,
    initial_token_budgets: Sequence[int],
    execution_max_new_tokens: int,
    render: bool,
) -> dict[str, Any]:
    root = output_root / anchor_mode / protocol
    root.mkdir(parents=True, exist_ok=True)
    all_slots = build_rollout_slots(
        spec,
        anchor_frames=anchor_frames,
        minimum_units_per_profile=1,
        dense_gap_policy="schema_proxy",
    )
    slots = select_rollout_slots(
        all_slots,
        profiles=(PROFILE,),
        context_variants=CONTEXT_VARIANTS,
    )
    expected_slots = 1 + (len(anchor_frames) + 1) * len(CONTEXT_VARIANTS)
    if len(slots) != expected_slots:
        raise ValueError(f"{anchor_mode}/{protocol} slot count {len(slots)} != {expected_slots}")
    predictions_path = root / "predictions.jsonl"
    _seed_rollout(predictions_path, probe["selected_row"])
    rows = run_rollout(
        slots=slots,
        generator=generator,
        source_root=spec_path.parent,
        output_path=predictions_path,
        initial_max_new_tokens=max(initial_token_budgets),
        execution_max_new_tokens=execution_max_new_tokens,
        resume=True,
        initial_prompt_suffix=probe.get("selected_prompt_suffix"),
        **_protocol_kwargs(protocol),
    )
    if len(rows) != expected_slots:
        raise RuntimeError(f"incomplete rollout: {len(rows)} != {expected_slots}")
    if protocol == "blinded":
        _assert_blinded(rows)
    annotated = _annotate_rows(rows, entry=entry, anchor_mode=anchor_mode)
    audit_rows_path = root / "audit_rows.jsonl"
    if not audit_rows_path.is_file():
        _write_jsonl(audit_rows_path, annotated)
    raw_metrics = aggregate_execution_metrics(annotated, view="raw")
    effective_metrics = aggregate_execution_metrics(annotated, view="effective")
    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "episode_name": entry["name"],
        "selection": entry["selection"],
        "anchor_mode": anchor_mode,
        "anchor_frames": list(anchor_frames),
        "protocol": protocol,
        "profile": PROFILE,
        "contexts": list(CONTEXT_VARIANTS),
        "slot_count": len(rows),
        "generated": sum(row.get("status") == "generated" for row in rows),
        "explicit_failures": sum(row.get("status") != "generated" for row in rows),
        "raw_metrics": raw_metrics,
        "effective_metrics": effective_metrics,
        "raw_paired_deltas": paired_context_deltas(annotated, view="raw"),
        "effective_paired_deltas": paired_context_deltas(annotated, view="effective"),
        "raw_end_audit": end_audit(annotated, view="raw"),
        "effective_end_audit": end_audit(annotated, view="effective"),
        "assist_events": assist_event_counts([
            row for row in annotated if row.get("category") != "initial_plan"
        ]),
        "predictions": str(predictions_path),
        "audit_rows": str(audit_rows_path),
    }
    videos: dict[str, Any] = {}
    if render:
        for context in CONTEXT_VARIANTS:
            if context == "with_memory_with_initial" and not probe.get("passed"):
                videos[context] = {
                    "status": "skipped_invalid_initial_plan",
                    "ground_truth_used": False,
                }
                continue
            video_path = root / "videos" / f"{context}.mp4"
            try:
                videos[context] = render_focus_video(
                    spec=spec,
                    rows=rows,
                    output_path=video_path,
                    profile=PROFILE,
                    context_variant=context,
                    initial_plan_page_seconds=2.0,
                    end_hold_seconds=2.0,
                )
            except Exception as exc:
                videos[context] = {
                    "status": "render_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "ground_truth_used": False,
                    "prediction_audit_preserved": True,
                }
    summary["videos"] = videos
    write_json(root / "summary.json", summary)
    return {
        "protocol": protocol,
        "anchor_mode": anchor_mode,
        "slot_count": len(rows),
        "audit_rows": str(audit_rows_path),
        "summary": str(root / "summary.json"),
        "videos": videos,
    }


def _episode_run(
    *,
    entry: Mapping[str, Any],
    generator: Any,
    output_root: Path,
    dense_names: set[str],
    initial_token_budgets: Sequence[int],
    execution_max_new_tokens: int,
    dense_stride: int,
) -> dict[str, Any]:
    episode_root = output_root / "episodes" / str(entry["name"])
    result_path = episode_root / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    episode_root.mkdir(parents=True, exist_ok=True)
    spec_path = Path(str(entry["spec"]))
    spec = load_episode_spec(
        spec_path,
        verify_sources=entry["selection"] == "benchmark3_holdout",
        minimum_intervals_per_unit=1,
    )
    label_frames = label_boundary_midpoint_frames(spec)
    initial_slots = {
        slot.profile: slot
        for slot in build_rollout_slots(
            spec,
            anchor_frames=label_frames,
            minimum_units_per_profile=1,
            dense_gap_policy="schema_proxy",
        )
        if slot.category == "initial_plan" and slot.profile in INITIAL_PLAN_PROFILES
    }
    if set(initial_slots) != set(INITIAL_PLAN_PROFILES):
        raise ValueError(f"missing initial plan profiles: {set(INITIAL_PLAN_PROFILES) - set(initial_slots)}")
    probes: dict[str, dict[str, Any]] = {}
    for protocol in PROTOCOLS:
        for profile in INITIAL_PLAN_PROFILES:
            key = f"{protocol}:{profile}"
            probes[key] = _probe_initial_plan(
                slot=initial_slots[profile],
                protocol=protocol,
                generator=generator,
                source_root=spec_path.parent,
                output_root=episode_root / "initial_plan_probe",
                token_budgets=initial_token_budgets,
            )
    schedules: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        schedules.append(_run_schedule(
            spec=spec,
            spec_path=spec_path,
            entry=entry,
            anchor_mode="label",
            anchor_frames=label_frames,
            protocol=protocol,
            probe=probes[f"{protocol}:{PROFILE}"],
            generator=generator,
            output_root=episode_root,
            initial_token_budgets=initial_token_budgets,
            execution_max_new_tokens=execution_max_new_tokens,
            render=False,
        ))
    dense_selected = str(entry["name"]) in dense_names
    if dense_selected:
        dense_frames = dense_stride_frames(spec, stride=dense_stride)
        for protocol in PROTOCOLS:
            schedules.append(_run_schedule(
                spec=spec,
                spec_path=spec_path,
                entry=entry,
                anchor_mode=f"dense_stride{dense_stride}",
                anchor_frames=dense_frames,
                protocol=protocol,
                probe=probes[f"{protocol}:{PROFILE}"],
                generator=generator,
                output_root=episode_root,
                initial_token_budgets=initial_token_budgets,
                execution_max_new_tokens=execution_max_new_tokens,
                render=True,
            ))
    result = {
        "schema_version": RUN_SCHEMA_VERSION,
        "episode_index": entry["index"],
        "episode_name": entry["name"],
        "episode_key": spec["episode_key"],
        "selection": entry["selection"],
        "total_frames": spec["total_frames"],
        "dense_selected": dense_selected,
        "label_anchor_count": len(label_frames) + 1,
        "initial_plan_probes": {
            key: {
                name: value
                for name, value in probe.items()
                if name != "selected_row"
            }
            for key, probe in probes.items()
        },
        "schedules": schedules,
        "audit_complete": True,
    }
    write_json(result_path, result)
    return result


def _worker_main(
    *,
    worker_index: int,
    device: str,
    entries: Sequence[Mapping[str, Any]],
    checkpoint_pin: str,
    output_root: str,
    dense_names: Sequence[str],
    initial_token_budgets: Sequence[int],
    execution_max_new_tokens: int,
    dense_stride: int,
) -> None:
    root = Path(output_root)
    worker_root = root / "workers" / f"worker_{worker_index:02d}"
    worker_root.mkdir(parents=True, exist_ok=True)
    try:
        generator = Generator(
            Path(checkpoint_pin),
            processor_path=Path(checkpoint_pin),
            device=device,
        )
        results: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for entry in entries:
            print(json.dumps({
                "event": "audit_episode_start",
                "worker": worker_index,
                "device": device,
                "episode": entry["name"],
            }, sort_keys=True), flush=True)
            try:
                results.append(_episode_run(
                    entry=entry,
                    generator=generator,
                    output_root=root,
                    dense_names=set(dense_names),
                    initial_token_budgets=initial_token_budgets,
                    execution_max_new_tokens=execution_max_new_tokens,
                    dense_stride=dense_stride,
                ))
            except Exception as exc:
                failure = {
                    "episode_index": entry["index"],
                    "episode_name": entry["name"],
                    "selection": entry["selection"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                episode_root = root / "episodes" / str(entry["name"])
                episode_root.mkdir(parents=True, exist_ok=True)
                write_json(episode_root / "failure.json", failure)
        write_json(worker_root / "result.json", {
            "worker_index": worker_index,
            "device": device,
            "episodes": results,
            "failures": failures,
        })
    except BaseException as exc:
        write_json(worker_root / "failure.json", {
            "worker_index": worker_index,
            "device": device,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        raise


def _audit_weight(entry: Mapping[str, Any], dense_names: set[str], dense_stride: int) -> int:
    spec = json.loads(Path(str(entry["spec"])).read_text(encoding="utf-8"))
    label_count = len(label_boundary_midpoint_frames(spec)) + 1
    dense_count = (
        len(dense_stride_frames(spec, stride=dense_stride)) + 1
        if str(entry["name"]) in dense_names else 0
    )
    return 2 * len(CONTEXT_VARIANTS) * (label_count + dense_count) + 8


def _assign_entries(
    entries: Sequence[Mapping[str, Any]],
    worker_count: int,
    *,
    dense_names: set[str],
    dense_stride: int,
) -> list[list[Mapping[str, Any]]]:
    assignments: list[list[Mapping[str, Any]]] = [[] for _ in range(worker_count)]
    loads = [0 for _ in range(worker_count)]
    weighted = [
        (_audit_weight(entry, dense_names, dense_stride), entry) for entry in entries
    ]
    for weight, entry in sorted(
        weighted, key=lambda value: (value[0], str(value[1]["name"])), reverse=True
    ):
        worker = min(range(worker_count), key=lambda index: (loads[index], index))
        assignments[worker].append(entry)
        loads[worker] += weight
    return assignments


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_record(path: Path, *, known_sha256: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path)}
    try:
        value = path.stat()
    except OSError as exc:
        return {**record, "exists": False, "error": f"{type(exc).__name__}: {exc}"}
    record.update({
        "exists": True,
        "type": "dir" if path.is_dir() else "file",
        "size": value.st_size,
        "mode": stat.filemode(value.st_mode),
        "uid": value.st_uid,
        "gid": value.st_gid,
        "mtime_ns": value.st_mtime_ns,
    })
    if known_sha256:
        record["sha256"] = known_sha256
        record["sha256_source"] = "authoritative_manifest"
    elif path.is_file() and value.st_size <= 64 * 1024 * 1024:
        try:
            record["sha256"] = sha256_file(path)
            record["sha256_source"] = "computed"
        except OSError as exc:
            record["sha256_error"] = f"{type(exc).__name__}: {exc}"
    elif path.is_file():
        record["sha256"] = None
        record["sha256_source"] = "omitted_large_file_use_bucket_manifest"
    return record


def _path_inventory(training_run_root: Path, checkpoint: Path) -> dict[str, Any]:
    snapshot = _read_json(SNAPSHOT_MANIFEST)
    leaf_hashes = {
        str(leaf["source_bucket_path"]): str(leaf["source_bucket_sha256"])
        for leaf in snapshot["leaves"]
    }
    paths = [
        LIVE_CODE_ROOT,
        PORTABLE_BUNDLE,
        PORTABLE_BUNDLE / "run_v5.3_baseline_scan_cache_json_train.sh",
        PORTABLE_BUNDLE / "entrypoints/train_v5.3_baseline_8gpu.sh",
        PORTABLE_BUNDLE / "tools/baseline_scan_v5_3.py",
        PORTABLE_BUNDLE / "tools/baseline_gate_v5_3.py",
        LIVE_CODE_ROOT / "scan_labels.py",
        LIVE_CODE_ROOT / "cache.py",
        LIVE_CODE_ROOT / "build_snapshot.py",
        LIVE_CODE_ROOT / "training.py",
        LIVE_CODE_ROOT / "prompt.py",
        LIVE_CODE_ROOT / "memory.py",
        LIVE_CODE_ROOT / "schema.py",
        CACHE_RUN_ROOT / "artifacts/label_scan/instruction_index.jsonl",
        CACHE_RUN_ROOT / "artifacts/label_scan/instruction_index_manifest.json",
        CACHE_RUN_ROOT / "artifacts/baseline/media_frame_cache.jsonl",
        CACHE_RUN_ROOT / "artifacts/baseline/quarantine.jsonl",
        CACHE_RUN_ROOT / "artifacts/baseline/manifest.json",
        SNAPSHOT_MANIFEST,
        PREPARE_SUMMARY,
        EXPOSURE_PLAN,
        training_run_root,
        checkpoint,
    ]
    paths.extend(Path(value) for value in leaf_hashes)
    records = [
        _path_record(path, known_sha256=leaf_hashes.get(str(path))) for path in paths
    ]
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "records": records,
        "stale_or_missing_claims": [{
            "path": str(
                PORTABLE_BUNDLE.parent
                / "runs/v5.3-baseline-smoke-guangzhou-20260826T142600Z"
            ),
            "status": "missing_at_audit_preflight",
        }],
    }


def _parity_report(checkpoint: Path) -> dict[str, Any]:
    snapshot = _read_json(SNAPSHOT_MANIFEST)
    prepared = _read_json(PREPARE_SUMMARY)
    exposure = _read_json(EXPOSURE_PLAN)
    trainer = _read_json(checkpoint / "trainer_state.json")
    total = int(snapshot["num_samples"])
    leaves = {str(leaf["training_bucket"]): leaf for leaf in snapshot["leaves"]}
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "created_at": utc_now(),
        "training": {
            "snapshot_manifest": str(SNAPSHOT_MANIFEST),
            "snapshot_content_digest": snapshot["content_digest"],
            "prompt_renderer_sha256": snapshot["prompt_renderer_sha256"],
            "total_physical_samples": total,
            "bucket_counts": {
                name: int(leaf["num_samples"]) for name, leaf in leaves.items()
            },
            "bucket_fractions": {
                name: int(leaf["num_samples"]) / total for name, leaf in leaves.items()
            },
            "context_variant_counts": {
                name: leaf["context_variant_counts"] for name, leaf in leaves.items()
            },
            "output_profile_counts": {
                name: leaf["output_profile_counts"] for name, leaf in leaves.items()
            },
            "coverage_semantics": exposure["coverage_semantics"],
            "benchmark3_overlap_samples": prepared["benchmark3_holdout"][
                "training_overlap_samples"
            ],
            "context_semantics": {
                "short_memory": "label-derived previous unit at materialization time",
                "long_memory": "label-derived prior unit history at materialization time",
                "initial_plan": "label-derived complete episode decomposition",
                "ongoing_anchor_schedule": "one midpoint per labelled profile unit",
            },
        },
        "checkpoint": {
            "path": str(checkpoint),
            "global_step": int(trainer["global_step"]),
            "epoch": trainer.get("epoch"),
            "training_completed": False,
            "intended_epochs": 2,
            "known_terminal_failure": (
                "training job failed after a three-view sample could not decode frame 505"
            ),
        },
        "inference": {
            "execution_profile": PROFILE,
            "contexts": list(CONTEXT_VARIANTS),
            "unsupported_context": "no_memory_with_initial",
            "protocols": {
                "blinded": {
                    "prompt": "training canonical render only",
                    "postprocessing": [],
                    "memory_update_source": "schema-valid raw prediction1",
                },
                "assisted": {
                    "prompt": [
                        "demo plan constraint",
                        "temporal ongoing/end suffix",
                        "schema correction prompt on one retry",
                    ],
                    "postprocessing": [
                        "bounded initial JSON punctuation repair",
                        "initial plan deduplication and caps",
                        "ongoing early End to Continue",
                        "one schema retry",
                        "same-context last-valid model fallback",
                        "terminal completed progress to 100",
                    ],
                    "memory_update_source": "effective prediction1",
                },
            },
            "generation": {
                "do_sample": False,
                "initial_blinded_max_new_tokens": 4096,
                "initial_assisted_token_budgets": [4096, 8192],
                "execution_max_new_tokens": 512,
            },
            "context_semantics": {
                "short_memory": "rolling schema-valid model prediction1",
                "long_memory": "rolling prior model prediction1 history",
                "initial_plan": "model-generated probe; invalid gate skips dependent branch",
                "label_anchor_schedule": "label starts, midpoints, and ends",
                "dense_anchor_schedule": "label anchors union fixed frame stride",
            },
        },
        "identified_mismatches": [
            "ongoing rows dominate physical training exposure",
            "joint execution rows are a small minority of ongoing training rows",
            "joint Initial Plan has only 2031 rows versus 327508 action-only rows",
            "training memory is oracle label history while inference memory is rolling model history",
            "training memory+Initial Plan uses a label-derived plan while inference uses a model-generated plan",
            "training includes noisy-memory contexts that are not a natural rollout context in this audit",
            "training ongoing anchors are unit midpoints while this audit additionally tests label boundaries and dense stride anchors",
            "assisted temporal/category suffixes and correction prompts do not occur in training",
            "checkpoint-80500 represents an incomplete training run",
        ],
    }


def _collect_rows(
    output: Path,
    entries: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    for entry in entries:
        episode_root = output / "episodes" / str(entry["name"])
        result_path = episode_root / "result.json"
        if not result_path.is_file():
            continue
        result = _read_json(result_path)
        for key, probe in result["initial_plan_probes"].items():
            probes.append({
                "episode_name": entry["name"],
                "selection": entry["selection"],
                "probe": key,
                **probe,
            })
        for schedule in result["schedules"]:
            rows.extend(jsonl_rows(Path(schedule["audit_rows"])))
    return rows, probes


def _aggregate_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocols: Sequence[str] = PROTOCOLS,
    dense_stride: int = 10,
) -> dict[str, Any]:
    strata = {
        "label_holdout": ("label", "benchmark3_holdout"),
        "label_supplemental": ("label", "supplemental_non_holdout"),
        "dense_holdout": (
            f"dense_stride{dense_stride}",
            "benchmark3_holdout",
        ),
    }
    result: dict[str, Any] = {}
    for name, (anchor_mode, selection) in strata.items():
        stratum = [
            row for row in rows
            if row.get("anchor_mode") == anchor_mode
            and row.get("selection") == selection
        ]
        result[name] = {}
        for protocol in protocols:
            selected = [row for row in stratum if row.get("protocol") == protocol]
            if not selected:
                continue
            result[name][protocol] = {
                "rows": len(selected),
                "episodes": len({row["episode_name"] for row in selected}),
                "raw": aggregate_execution_metrics(selected, view="raw"),
                "effective": aggregate_execution_metrics(selected, view="effective"),
                "raw_paired_deltas": paired_context_deltas(selected, view="raw"),
                "effective_paired_deltas": paired_context_deltas(
                    selected, view="effective"
                ),
                "assist_events": assist_event_counts(selected),
            }
    return result


def _aggregate_initial_plan_probes(
    probes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for probe_name in sorted({str(probe["probe"]) for probe in probes}):
        selected = [probe for probe in probes if probe["probe"] == probe_name]
        raw_scores = [
            float(probe["raw_metrics"]["strict_action_alignment_score"])
            for probe in selected
        ]
        effective_scores = [
            float(probe["effective_metrics"]["strict_action_alignment_score"])
            for probe in selected
        ]
        selected_attempts = [
            probe["attempts"][-1] for probe in selected if probe.get("attempts")
        ]
        failed = [probe for probe in selected if not probe.get("passed")]
        failed_score_violations = [
            {
                "episode_name": probe.get("episode_name"),
                "raw_strict_score": probe["raw_metrics"][
                    "strict_action_alignment_score"
                ],
                "effective_strict_score": probe["effective_metrics"][
                    "strict_action_alignment_score"
                ],
            }
            for probe in failed
            if (
                float(probe["raw_metrics"]["strict_action_alignment_score"]) != 0.0
                or float(
                    probe["effective_metrics"]["strict_action_alignment_score"]
                ) != 0.0
            )
        ]
        attempts = [
            attempt
            for probe in selected
            for attempt in probe.get("attempts") or []
        ]
        result[probe_name] = {
            "episodes": len(selected),
            "passed": sum(bool(probe.get("passed")) for probe in selected),
            "pass_rate": (
                sum(bool(probe.get("passed")) for probe in selected) / len(selected)
                if selected else None
            ),
            "failed": len(failed),
            "failed_scores_zero": not failed_score_violations,
            "failed_score_violations": failed_score_violations,
            "attempt_count": len(attempts),
            "retry_count": sum(
                max(0, len(probe.get("attempts") or []) - 1)
                for probe in selected
            ),
            "repair_attempt_count": sum(bool(attempt.get("repair")) for attempt in attempts),
            "normalization_attempt_count": sum(
                bool(attempt.get("normalization")) for attempt in attempts
            ),
            "raw_action_alignment_mean": (
                sum(raw_scores) / len(raw_scores) if raw_scores else None
            ),
            "effective_action_alignment_mean": (
                sum(effective_scores) / len(effective_scores)
                if effective_scores else None
            ),
            "effective_action_count_mean": (
                sum(
                    int(probe["effective_metrics"]["action_count"])
                    for probe in selected
                ) / len(selected)
                if selected else None
            ),
            "effective_segment_alignment_mean": (
                sum(
                    float(probe["effective_metrics"]["segment_alignment_mean"])
                    for probe in selected
                ) / len(selected)
                if selected else None
            ),
            "selected_full_token_budget_rate": (
                sum(
                    int(attempt.get("output_tokens") or 0)
                    >= int(attempt["max_new_tokens"])
                    for attempt in selected_attempts
                ) / len(selected_attempts)
                if selected_attempts else None
            ),
        }
    return result


def _failure_scoring_audit(
    rows: Sequence[Mapping[str, Any]],
    probes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify that every invalid model branch receives the declared worst score."""

    violations: list[dict[str, Any]] = []
    invalid_execution_views = 0
    for row in rows:
        if row.get("category") not in {"ongoing", "end"}:
            continue
        for view, validity_field in (
            ("raw", "raw_prediction_schema_valid"),
            ("effective", "prediction_schema_valid"),
        ):
            if row.get(validity_field):
                continue
            invalid_execution_views += 1
            score = execution_row_score(row, view=view)
            unit_failure = any(
                float(unit["caption_token_f1"]) != 0.0
                or float(unit["progress_abs_error"]) != 100.0
                for unit in score["unit_scores"].values()
            )
            caption_failure = (
                score["caption_scoring_available"]
                and float(score["caption_token_f1"]) != 0.0
            )
            if (
                float(score["task_progress_abs_error"]) != 100.0
                or bool(score["decision_correct"])
                or caption_failure
                or unit_failure
            ):
                violations.append({
                    "type": "execution_failure_not_worst_scored",
                    "episode_name": row.get("episode_name"),
                    "slot_id": row.get("slot_id"),
                    "view": view,
                    "score": score,
                })
    failed_plan_probes = 0
    for probe in probes:
        if probe.get("passed"):
            continue
        failed_plan_probes += 1
        raw_score = float(
            probe["raw_metrics"]["strict_action_alignment_score"]
        )
        effective_score = float(
            probe["effective_metrics"]["strict_action_alignment_score"]
        )
        if raw_score != 0.0 or effective_score != 0.0:
            violations.append({
                "type": "initial_plan_failure_not_zero_scored",
                "episode_name": probe.get("episode_name"),
                "probe": probe.get("probe"),
                "raw_strict_score": raw_score,
                "effective_strict_score": effective_score,
            })
    return {
        "passed": not violations,
        "policy": {
            "invalid_execution_task_and_unit_abs_error": 100,
            "invalid_execution_caption_and_decision_credit": 0,
            "invalid_initial_plan_strict_alignment_credit": 0,
            "invalid_signed_bias": "excluded_no_invented_direction",
        },
        "invalid_execution_views_checked": invalid_execution_views,
        "failed_initial_plan_probes_checked": failed_plan_probes,
        "violations": violations,
    }


def _code_hashes() -> dict[str, str]:
    module = Path(__file__).resolve()
    data_root = module.parents[2] / "data" / "event_states"
    paths = (
        module,
        module.with_name("metrics.py"),
        data_root / "materialize_episode.py",
        data_root / "inference.py",
        data_root / "memory.py",
        data_root / "prompt.py",
        data_root / "schema.py",
        module.parent / "rollout.py",
        module.parent / "video.py",
    )
    return {str(path): sha256_file(path) for path in paths}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-spec", type=Path, required=True)
    parser.add_argument("--training-run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-selection-mode",
        choices=("strict", "stat_complete"),
        default="strict",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--devices", default="cuda:0")
    parser.add_argument("--expected-episodes", type=int, default=20)
    parser.add_argument("--dense-stride", type=int, default=10)
    parser.add_argument("--initial-token-budgets", default="4096,8192")
    parser.add_argument("--execution-max-new-tokens", type=int, default=512)
    args = parser.parse_args(argv)
    if args.expected_episodes <= 0 or args.dense_stride <= 0:
        raise ValueError("episode count and dense stride must be positive")
    if args.execution_max_new_tokens <= 0:
        raise ValueError("execution max token count must be positive")
    devices = tuple(value.strip() for value in args.devices.split(",") if value.strip())
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("devices must be a non-empty unique list")
    budgets = tuple(
        int(value) for value in args.initial_token_budgets.split(",") if value.strip()
    )
    if budgets != tuple(sorted(set(budgets))) or any(value <= 0 for value in budgets):
        raise ValueError("initial token budgets must be unique, positive, and increasing")
    if len(budgets) < 2:
        raise ValueError("assisted protocol requires two initial token budgets")

    batch = _load_batch_spec(args.batch_spec, expected_episodes=args.expected_episodes)
    dense_names = set(duration_quantile_episode_names(batch["episodes"]))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)
    if args.checkpoint == "auto":
        selector = (
            select_latest_complete_checkpoint
            if args.checkpoint_selection_mode == "strict"
            else select_latest_stat_complete_checkpoint
        )
        checkpoint_path, selection = selector(args.training_run_root)
    else:
        if args.checkpoint_selection_mode == "strict":
            checkpoint_path, selection = validate_explicit_checkpoint(checkpoint_path)
        else:
            record, error = stat_complete_checkpoint_record(checkpoint_path)
            if record is None:
                raise RuntimeError(f"explicit stat-complete checkpoint failed: {error}")
            checkpoint_path = checkpoint_path.resolve()
            selection = {
                "selected_at": utc_now(),
                "selection_rule": "explicit stat-complete V10 checkpoint",
                "selected": record,
                "complete_candidate_count": 1,
                "rejected_candidate_count": 0,
                "rejected": [],
            }
    if checkpoint_path.name != "checkpoint-80500":
        raise ValueError(f"audit is pinned to checkpoint-80500, got {checkpoint_path}")
    write_json(output / "checkpoint_selection.json", selection)
    pin_path = output / "checkpoint_pin"
    pin_manifest = pin_checkpoint(
        checkpoint_path,
        pin_path,
        require_completion_metadata=args.checkpoint_selection_mode == "strict",
    )
    code_hashes = _code_hashes()
    fingerprint = _stable_sha256({
        "batch_spec_sha256": batch["spec_sha256"],
        "source_checkpoint": pin_manifest["source_checkpoint"],
        "model_sha256": pin_manifest["model_sha256"],
        "code_sha256": code_hashes,
        "protocols": PROTOCOLS,
        "contexts": CONTEXT_VARIANTS,
        "dense_names": sorted(dense_names),
        "dense_stride": args.dense_stride,
        "initial_token_budgets": budgets,
        "execution_max_new_tokens": args.execution_max_new_tokens,
        "devices": devices,
    })
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        previous = _read_json(manifest_path)
        if previous.get("run_fingerprint") != fingerprint:
            raise ValueError("output directory belongs to another audit configuration")
    else:
        write_json(manifest_path, {
            "schema_version": RUN_SCHEMA_VERSION,
            "created_at": utc_now(),
            "run_fingerprint": fingerprint,
            "batch_spec": batch["spec_path"],
            "batch_spec_sha256": batch["spec_sha256"],
            "episode_count": len(batch["episodes"]),
            "primary_split": "benchmark3_holdout",
            "supplemental_split": "supplemental_non_holdout",
            "dense_episode_names": sorted(dense_names),
            "dense_selection": "nearest duration quartiles among 18 holdout episodes",
            "protocols": list(PROTOCOLS),
            "contexts": list(CONTEXT_VARIANTS),
            "unsupported_context": "no_memory_with_initial",
            "profile": PROFILE,
            "generation": {
                "do_sample": False,
                "initial_token_budgets": list(budgets),
                "execution_max_new_tokens": args.execution_max_new_tokens,
            },
            "checkpoint_selection_mode": args.checkpoint_selection_mode,
            "checkpoint_selection": selection,
            "checkpoint_pin": str(pin_path),
            "checkpoint_pin_manifest": pin_manifest,
            "devices": list(devices),
            "environment": {
                "hostname": socket.gethostname(),
                "python": sys.version,
                "executable": sys.executable,
                "aihc_job_name": os.environ.get("AIHC_JOB_NAME"),
                "aihc_job_id": os.environ.get("AIHC_JOB_ID"),
            },
            "code_sha256": code_hashes,
        })
    inventory = _path_inventory(args.training_run_root.resolve(), checkpoint_path)
    write_json(output / "path_inventory.json", inventory)
    inventory_lines = ["# V5.3 path inventory", ""]
    for record in inventory["records"]:
        inventory_lines.append(
            f"- {record['path']}: exists={record.get('exists')} "
            f"type={record.get('type')} size={record.get('size')} "
            f"mode={record.get('mode')} sha256={record.get('sha256')}"
        )
    (output / "path_inventory.md").write_text(
        "\n".join(inventory_lines) + "\n", encoding="utf-8"
    )
    parity = _parity_report(checkpoint_path)
    write_json(output / "train_infer_parity.json", parity)
    (output / "train_infer_parity.md").write_text(
        "\n".join([
            "# V5.3 train/inference parity",
            "",
            f"- Physical training samples: {parity['training']['total_physical_samples']}",
            f"- Training prompt renderer SHA-256: {parity['training']['prompt_renderer_sha256']}",
            f"- Checkpoint: {parity['checkpoint']['path']} at step {parity['checkpoint']['global_step']}",
            "- Primary inference protocol: blinded canonical prompt with no repair or temporal suffix.",
            "- Secondary protocol: assisted; every repair, retry, fallback, and normalization is reported separately.",
            "- Unsupported context: no_memory_with_initial.",
            "",
            "## Identified mismatches",
            "",
            *[f"- {value}" for value in parity["identified_mismatches"]],
        ]) + "\n",
        encoding="utf-8",
    )

    assignments = _assign_entries(
        batch["episodes"],
        len(devices),
        dense_names=dense_names,
        dense_stride=args.dense_stride,
    )
    assignment_report = [
        {
            "worker_index": index,
            "device": devices[index],
            "estimated_model_calls": sum(
                _audit_weight(entry, dense_names, args.dense_stride)
                for entry in entries
            ),
            "episodes": [entry["name"] for entry in entries],
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
                "dense_names": sorted(dense_names),
                "initial_token_budgets": budgets,
                "execution_max_new_tokens": args.execution_max_new_tokens,
                "dense_stride": args.dense_stride,
            },
            name=f"v53-audit-worker-{worker_index}",
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    worker_failures = [
        {"name": process.name, "pid": process.pid, "exitcode": process.exitcode}
        for process in processes if process.exitcode != 0
    ]
    if worker_failures:
        write_json(output / "batch_failure.json", {
            "failed_workers": worker_failures,
            "assignments": assignment_report,
        })
        raise RuntimeError(f"audit workers failed before bounded episode handling: {worker_failures}")

    rows, probes = _collect_rows(output, batch["episodes"])
    episode_results = [
        _read_json(output / "episodes" / str(entry["name"]) / "result.json")
        for entry in batch["episodes"]
        if (output / "episodes" / str(entry["name"]) / "result.json").is_file()
    ]
    episode_failures = [
        _read_json(output / "episodes" / str(entry["name"]) / "failure.json")
        for entry in batch["episodes"]
        if (output / "episodes" / str(entry["name"]) / "failure.json").is_file()
    ]
    _write_jsonl(output / "predictions.jsonl", rows)
    _write_jsonl(output / "initial_plan_probe.jsonl", probes)
    _write_jsonl(
        output / "raw_vs_effective.jsonl",
        [raw_effective_record(row) for row in rows],
    )
    _write_jsonl(
        output / "memory_trace.jsonl",
        [
            memory_trace_record(row) for row in rows
            if str(row.get("context_variant", "")).startswith("with_memory")
        ],
    )
    prompt_samples: dict[str, Any] = {}
    for row in rows:
        key = ":".join((
            str(row.get("protocol")),
            str(row.get("context_variant")),
            str(row.get("category")),
        ))
        if key in prompt_samples or not row.get("prompt"):
            continue
        prompt_samples[key] = {
            "episode_name": row.get("episode_name"),
            "anchor_mode": row.get("anchor_mode"),
            "prompt": row.get("prompt"),
            "prompt_sha256": row.get("prompt_sha256"),
            "output_spec": row.get("output_spec"),
        }
    write_json(output / "prompt_samples.json", prompt_samples)
    failure_scoring = _failure_scoring_audit(rows, probes)
    episodes_complete = len(episode_results) == len(batch["episodes"])
    summary = {
        "schema_version": RUN_SCHEMA_VERSION,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "platform_succeeded": not worker_failures,
        "audit_complete": episodes_complete and failure_scoring["passed"],
        "episode_count": len(batch["episodes"]),
        "completed_episodes": len(episode_results),
        "failed_episodes": episode_failures,
        "checkpoint": pin_manifest["source_checkpoint"],
        "pinned_checkpoint": str(pin_path),
        "dense_episode_names": sorted(dense_names),
        "row_count": len(rows),
        "initial_plan": _aggregate_initial_plan_probes(probes),
        "strata": _aggregate_report(rows, dense_stride=args.dense_stride),
        "assist_events": assist_event_counts([
            row for row in rows if row.get("category") != "initial_plan"
        ]),
        "failure_scoring_audit": failure_scoring,
    }
    write_json(output / "summary.json", summary)
    end_report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "created_at": utc_now(),
        "strata": {},
    }
    for stratum, anchor_mode, selection in (
        ("label_holdout", "label", "benchmark3_holdout"),
        ("label_supplemental", "label", "supplemental_non_holdout"),
        ("dense_holdout", f"dense_stride{args.dense_stride}", "benchmark3_holdout"),
    ):
        selected = [
            row for row in rows
            if row.get("anchor_mode") == anchor_mode
            and row.get("selection") == selection
        ]
        end_report["strata"][stratum] = {
            protocol: {
                "raw": end_audit(
                    [row for row in selected if row.get("protocol") == protocol],
                    view="raw",
                ),
                "effective": end_audit(
                    [row for row in selected if row.get("protocol") == protocol],
                    view="effective",
                ),
            }
            for protocol in PROTOCOLS
        }
    write_json(output / "end_audit.json", end_report)
    batch_result = {
        "schema_version": RUN_SCHEMA_VERSION,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "platform_succeeded": not worker_failures,
        "audit_complete": summary["audit_complete"],
        "model_failures_are_scored_zero": failure_scoring["passed"],
        "episode_count": len(batch["episodes"]),
        "completed_episodes": len(episode_results),
        "failed_episode_count": len(episode_failures),
        "checkpoint": pin_manifest["source_checkpoint"],
        "output_dir": str(output),
        "summary": str(output / "summary.json"),
        "end_audit": str(output / "end_audit.json"),
        "train_infer_parity": str(output / "train_infer_parity.json"),
    }
    write_json(output / "batch_result.json", batch_result)
    print(json.dumps(batch_result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if batch_result["audit_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
