"""Materialize the two reviewed Replan sources into V5.3 decision-only rows."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .holdout import EvaluationHoldout, DEFAULT_EVALUATION_MANIFEST, DEFAULT_EVALUATION_SHA256
from .prompt import prompt_renderer_digest_v53, render_user
from .schema import (
    SCHEMA_VERSION_V53,
    dumps_assistant,
    output_profile_id,
    validate_model_visible_text,
    validate_sample,
)
from .task_instruction import TaskInstructionError, select_record_instruction


SELF_ROOT = Path("/data/takeover_recovery/samples_v2")
OPEN_ROOT = Path("/data/takeover_recovery/vifailback_v1")
_SPACE = re.compile(r"\s+")
_FORBIDDEN_SUBTASK = re.compile(r"\bsubtasks?\b", re.IGNORECASE)
_RAW_FAILURE_PREFIX = re.compile(r"^\s*(?:[1-8]\.[1-5])\s*[-:—–]*\s*")
_SAFE_TASK = re.compile(r"[^a-z0-9]+")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(*parts: object) -> str:
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()[:24]


def _clean_english(value: Any, where: str) -> str:
    text = _SPACE.sub(" ", str(value or "")).strip()
    text = _FORBIDDEN_SUBTASK.sub(
        lambda match: "task steps" if match.group(0).lower().endswith("s") else "task step",
        text,
    )
    text = _RAW_FAILURE_PREFIX.sub("", text)
    return validate_model_visible_text(text, where)


def _task_instruction(
    record: Mapping[str, Any], *, source_path: Path
) -> tuple[str, str, str, str]:
    """Read only explicit source labels; task/path slugs are never supervision."""

    text, source, resolved_path, field = select_record_instruction(
        record,
        source_prefix="samples_json",
        source_path=source_path,
    )
    assert resolved_path is not None
    return _clean_english(text, "replan instruction"), source, resolved_path, field


def _episode_key(record: Mapping[str, Any], dataset_id: str) -> str:
    if dataset_id == "replan_self":
        value = record.get("episode_id") or record.get("video_path")
    else:
        value = record.get("hdf5") or record.get("video") or record.get("case_id")
    value = str(value or "").strip()
    if not value:
        raise ValueError("replan record has no episode identity")
    return f"{dataset_id}/{value}"


def _images(record: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    frames = record.get("frames")
    values = frames.get("cot_frames") if isinstance(frames, Mapping) else None
    if not isinstance(values, list) or not values:
        raise ValueError("replan record has no cot_frames")
    result: list[dict[str, Any]] = []
    for item in values:
        relative = item.get("path") if isinstance(item, Mapping) else None
        if not isinstance(relative, str) or not relative:
            raise ValueError("invalid cot frame reference")
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as exc:
            raise ValueError("cot frame escapes its dataset root") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        reference: dict[str, Any] = {"path": str(path), "view": "head"}
        if isinstance(item, Mapping):
            frame = item.get("frame")
            if isinstance(frame, int) and not isinstance(frame, bool) and frame >= 0:
                reference["source_frame"] = frame
        result.append(reference)
    return result


def _raw_video_paths(record: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("video_path", "video", "hdf5"):
        value = record.get(name)
        if isinstance(value, str) and value:
            result[name] = value
    return result


def adapt_record(
    record: Mapping[str, Any],
    *,
    dataset_id: str,
    root: Path,
) -> dict[str, Any]:
    if dataset_id not in {"replan_self", "replan_open"}:
        raise ValueError(f"unknown Replan dataset: {dataset_id}")
    case_id = str(record.get("case_id") or "").strip()
    if not case_id:
        raise ValueError("Replan record has no case_id")
    cot = record.get("cot_label")
    if not isinstance(cot, Mapping):
        raise ValueError("Replan record has no cot_label")
    reason = _clean_english(cot.get("reasoning"), "Replan reason")
    plan_caption = _clean_english(cot.get("plan_en"), "Replan updated plan")
    instruction, instruction_source, instruction_source_path, instruction_source_field = (
        _task_instruction(record, source_path=root / "samples.json")
    )
    episode_key = _episode_key(record, dataset_id)
    images = _images(record, root)
    output_spec = {
        "prediction1_units": ["action"],
        "prediction2_units": ["action"],
        "plan_units": ["action"],
    }
    base_id = f"{dataset_id}_{_stable_id(case_id, episode_key)}"
    task_name = _SAFE_TASK.sub("_", str(record.get("task") or "task").lower()).strip("_")
    task_name = (task_name[:64] or "task") + "_" + _stable_id(record.get("task"))[:8]
    sample = {
        "schema_version": SCHEMA_VERSION_V53,
        "sample_id": f"{base_id}_no_memory_no_initial",
        "base_sample_id": base_id,
        "source": dataset_id,
        "training_bucket": dataset_id,
        "category": "replan",
        "context_variant": "no_memory_no_initial",
        "split": "train",
        "output_spec": output_spec,
        "output_profile_id": output_profile_id(output_spec),
        "task_instruction": instruction,
        "images": images,
        "prompt_context": {},
        "target": {
            "execution_decision": "Replan",
            "decision_detail": {
                "reason": reason,
                "updated_plan": [{
                    "index": 1,
                    "action": {"caption": plan_caption},
                }],
            },
        },
        "supervision": {"loss_mask_paths": []},
        "provenance": {
            "canonical_source": dataset_id,
            "episode_key": episode_key,
            "case_id": case_id,
            "task_name": task_name,
            "split": "train",
            "instruction_source": instruction_source,
            "task_instruction_source": instruction_source,
            "task_instruction_source_path": instruction_source_path,
            "task_instruction_source_field": instruction_source_field,
            "task_instruction_policy": "explicit_task_caption_or_instruction_only_v1",
            "fail_category": str(record.get("fail_category") or ""),
            "raw_video_paths": _raw_video_paths(record),
            "source_index": str((root / "samples.json").resolve()),
            "memory_pair_eligible": False,
        },
    }
    return validate_sample(sample)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return count


def materialize_dataset(
    *,
    dataset_id: str,
    root: Path,
    output_root: Path,
    holdout: EvaluationHoldout,
) -> dict[str, Any]:
    source_path = root / "samples.json"
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{source_path} must contain a list")
    samples: list[dict[str, Any]] = []
    contracts: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    benchmark_excluded = 0
    for index, record in enumerate(raw):
        try:
            sample = adapt_record(record, dataset_id=dataset_id, root=root)
            matches = holdout.match_sample(sample)
            if matches:
                benchmark_excluded += 1
                quarantined.append({
                    "index": index,
                    "case_id": record.get("case_id"),
                    "reason": "evaluation_holdout_overlap",
                    "matches": matches,
                })
                continue
            samples.append(sample)
            contracts.append({
                "sample_id": sample["sample_id"],
                "training_bucket": dataset_id,
                "category": "replan",
                "context_variant": sample["context_variant"],
                "output_profile_id": sample["output_profile_id"],
                "split": "train",
                "images": sample["images"],
                "source_record": dict(record),
                "input_json": {
                    "task_instruction": sample["task_instruction"],
                    "prompt_context": sample["prompt_context"],
                    "output_spec": sample["output_spec"],
                },
                "prompt": render_user(sample),
                "output_json": sample["target"],
                "output_text": dumps_assistant(
                    sample["target"], sample["category"], sample["output_spec"]
                ),
                "provenance": sample["provenance"],
            })
        except Exception as exc:
            reason = type(exc).__name__
            if isinstance(exc, TaskInstructionError):
                reason = "missing_task_instruction"
            if "non-empty" in str(exc) and "Replan" in str(exc):
                reason = "missing_replan_cot"
            reasons[reason] += 1
            quarantined.append({
                "index": index,
                "case_id": record.get("case_id") if isinstance(record, Mapping) else None,
                "reason": reason,
                "detail": str(exc),
            })

    bucket_root = output_root / "buckets" / dataset_id
    count = _atomic_jsonl(
        bucket_root / "train.jsonl",
        ({"data_id": sample["sample_id"], "v5_sample": sample, "image": sample["images"]} for sample in samples),
    )
    _atomic_jsonl(bucket_root / "contracts.jsonl", contracts)
    _atomic_jsonl(bucket_root / "quarantine.jsonl", quarantined)
    manifest = {
        "schema_version": "v10_action_segment_v5_3_bucket_v1",
        "complete": True,
        "training_bucket": dataset_id,
        "source_path": str(source_path.resolve()),
        "source_sha256": _sha256(source_path),
        "prompt_renderer_sha256": prompt_renderer_digest_v53(),
        "raw_records": len(raw),
        "train_records": count,
        "quarantined_records": len(quarantined),
        "quarantine_reasons": dict(sorted(reasons.items())),
        "evaluation_holdout_excluded": benchmark_excluded,
        "evaluation_holdout": holdout.metadata(),
        "files": {
            "train": {"path": "train.jsonl", "sha256": _sha256(bucket_root / "train.jsonl")},
            "contracts": {"path": "contracts.jsonl", "sha256": _sha256(bucket_root / "contracts.jsonl")},
            "quarantine": {"path": "quarantine.jsonl", "sha256": _sha256(bucket_root / "quarantine.jsonl")},
        },
    }
    manifest_path = bucket_root / "manifest.json"
    _atomic_jsonl(manifest_path, (manifest,))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-root", type=Path, default=SELF_ROOT)
    parser.add_argument("--open-root", type=Path, default=OPEN_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_EVALUATION_MANIFEST)
    parser.add_argument("--evaluation-sha256", default=DEFAULT_EVALUATION_SHA256)
    args = parser.parse_args(argv)
    holdout = EvaluationHoldout.load(
        args.evaluation_holdout,
        expected_sha256=args.evaluation_holdout_sha256,
    )
    reports = {
        "replan_self": materialize_dataset(
            dataset_id="replan_self", root=args.self_root, output_root=args.output_root, holdout=holdout
        ),
        "replan_open": materialize_dataset(
            dataset_id="replan_open", root=args.open_root, output_root=args.output_root, holdout=holdout
        ),
    }
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["adapt_record", "materialize_dataset"]
