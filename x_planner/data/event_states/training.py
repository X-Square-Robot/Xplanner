"""Evaluation-holdout-fenced event-state training wrapper."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from . import dataset as _dataset  # noqa: F401
from .holdout import (
    EvaluationHoldout,
    DEFAULT_EVALUATION_MANIFEST,
    DEFAULT_EVALUATION_SHA256,
    audit_artifact,
)
from ..pipeline.training import main as _train_main


def _argument_value(arguments: list[str], name: str) -> str:
    for index, value in enumerate(arguments):
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    raise ValueError(f"event-state launcher argument {name} is required")


def _optional_argument_value(arguments: list[str], name: str) -> str | None:
    try:
        return _argument_value(arguments, name)
    except ValueError:
        return None


def _argument_flag(arguments: list[str], name: str) -> bool:
    return name in arguments


def _generation_authorization(
    manifest: dict[str, object], arguments: list[str]
) -> dict[str, object]:
    content_digest = manifest.get("content_digest")
    if not isinstance(content_digest, str) or not content_digest:
        raise ValueError("training snapshot has no content digest")
    expected = _optional_argument_value(arguments, "--expected-content-digest")
    if expected is not None and expected != content_digest:
        raise ValueError("snapshot content digest differs from launcher pin")
    partial = manifest.get("partial") is True
    allowed = _argument_flag(arguments, "--allow-partial-generation")
    if partial and not allowed:
        raise PermissionError(
            "partial snapshot requires --allow-partial-generation"
        )
    if partial and expected is None:
        raise ValueError(
            "partial snapshot requires --expected-content-digest "
            "(expected_content_digest)"
        )
    return {
        "partial": partial,
        "partial_authorized": partial and allowed,
        "content_digest": content_digest,
        "expected_content_digest": expected,
    }


def _strip_release_arguments(arguments: list[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value == "--allow-partial-generation":
            index += 1
            continue
        if value == "--expected-content-digest":
            if index + 1 >= len(arguments):
                raise ValueError("--expected-content-digest requires a value")
            index += 2
            continue
        if value.startswith("--expected-content-digest="):
            index += 1
            continue
        result.append(value)
        index += 1
    return result


def evaluation_preflight(
    arguments: list[str],
    *,
    full_audit: bool = True,
) -> dict[str, object]:
    snapshot = Path(_argument_value(arguments, "--snapshot")).resolve(strict=True)
    data_config = Path(_argument_value(arguments, "--data_config")).resolve(strict=True)
    holdout = EvaluationHoldout.load(
        DEFAULT_EVALUATION_MANIFEST,
        expected_sha256=DEFAULT_EVALUATION_SHA256,
    )
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    generation = _generation_authorization(manifest, arguments)
    fence = manifest.get("evaluation_holdout")
    if not isinstance(fence, dict):
        raise ValueError("training snapshot lacks an evaluation-holdout fence")
    if fence.get("manifest_sha256") != holdout.manifest_sha256:
        raise ValueError("training snapshot evaluation-holdout digest mismatch")
    prepare_path = data_config.parent / "prepare_summary.json"
    if not prepare_path.is_file():
        raise FileNotFoundError(
            f"data config lacks its evaluation-holdout prepare summary: {prepare_path}"
        )
    prepared = json.loads(prepare_path.read_text(encoding="utf-8"))
    prepared_fence = prepared.get("evaluation_holdout")
    if (
        not isinstance(prepared_fence, dict)
        or prepared_fence.get("manifest_sha256") != holdout.manifest_sha256
        or prepared_fence.get("training_overlap_samples") != 0
    ):
        raise ValueError("prepared data is not evaluation-holdout fenced")
    if full_audit:
        audit = audit_artifact(snapshot, holdout, fail_on_match=True)
        audit_scope = "full_snapshot_rank0"
    else:
        if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
            raise ValueError("only a distributed nonzero rank may reuse rank-0 audit")
        audit = prepared_fence.get("independent_audit")
        if (
            not isinstance(audit, dict)
            or audit.get("passed") is not True
            or audit.get("excluded_samples") != 0
        ):
            raise ValueError("prepared data lacks a passing independent audit")
        audit_scope = "verified_prepare_audit_nonzero_rank"
    return {
        "snapshot": str(snapshot),
        "data_config": str(data_config),
        "evaluation_manifest_sha256": holdout.manifest_sha256,
        "checked_samples": audit["checked_samples"],
        "overlap_samples": audit["excluded_samples"],
        "audit_scope": audit_scope,
        "generation": generation,
    }


def main() -> None:
    rank = int(os.environ.get("RANK", "0"))
    preflight = evaluation_preflight(sys.argv[1:], full_audit=rank == 0)
    preflight["rank"] = rank
    print(
        "XPLANNER_EVALUATION_PREFLIGHT "
        + json.dumps(preflight, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    sys.argv = [sys.argv[0], *_strip_release_arguments(sys.argv[1:])]
    _train_main()


if __name__ == "__main__":
    main()


__all__ = [
    "_generation_authorization",
    "_strip_release_arguments",
    "evaluation_preflight",
    "main",
]
