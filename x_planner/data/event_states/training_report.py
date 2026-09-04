"""Persist finite-loss, step, resource, snapshot, and checkpoint V5 smoke evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from ..context.common import file_sha256, write_json
from ..context.training_report import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-steps", type=int, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--trained-checkpoint", type=Path)
    parser.add_argument("--require-loss-decrease", action="store_true")
    args = parser.parse_args()
    result = summarize(
        args.log,
        args.monitor,
        args.output,
        task="v5_mixed",
        resize_mode="auto_near_640",
        required_steps=args.required_steps,
        exit_code=args.exit_code,
    )
    model_identity = args.checkpoint / "model.safetensors"
    if not model_identity.is_file():
        model_identity = args.checkpoint / "model.safetensors.index.json"
    if not model_identity.is_file():
        raise FileNotFoundError(f"missing model identity in {args.checkpoint}")
    root_manifest = args.dataset_root / "manifest.json"
    result.update({
        "schema_version": "v10_action_segment_v5_gpu_smoke_report_v1",
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_manifest_sha256": file_sha256(root_manifest) if root_manifest.is_file() else None,
        "initial_checkpoint": str(args.checkpoint.resolve()),
        "initial_checkpoint_model_identity_file": model_identity.name,
        "initial_checkpoint_model_sha256": file_sha256(model_identity),
        "gpu_id": str(args.gpu_id),
        "batch_size": 1,
        "max_length": args.max_length,
        "formal_training": False,
    })
    losses = [float(value) for value in result.get("losses", [])]
    window = min(5, max(1, len(losses) // 3))
    first_mean = sum(losses[:window]) / window if losses else None
    last_mean = sum(losses[-window:]) / window if losses else None
    loss_decreased = bool(
        first_mean is not None
        and last_mean is not None
        and math.isfinite(first_mean)
        and math.isfinite(last_mean)
        and last_mean < first_mean
    )
    result.update({
        "loss_trend_window": window if losses else 0,
        "loss_first_window_mean": first_mean,
        "loss_last_window_mean": last_mean,
        "loss_last_to_first_ratio": (
            last_mean / first_mean
            if first_mean not in (None, 0.0) and last_mean is not None
            else None
        ),
        "loss_decreased": loss_decreased,
        "loss_decrease_required": bool(args.require_loss_decrease),
    })
    if args.trained_checkpoint is not None:
        checkpoint = args.trained_checkpoint.resolve()
        trained_identity = checkpoint / "model.safetensors"
        if not trained_identity.is_file():
            trained_identity = checkpoint / "model.safetensors.index.json"
        metadata = checkpoint / "v10_checkpoint_meta.json"
        if not checkpoint.is_dir() or not trained_identity.is_file() or not metadata.is_file():
            result.update({
                "trained_checkpoint": str(checkpoint),
                "trained_checkpoint_complete": False,
                "trained_checkpoint_error": "missing model identity or v10 metadata",
            })
            result["passed"] = False
        else:
            checkpoint_meta = json.loads(metadata.read_text(encoding="utf-8"))
            checkpoint_step = int(checkpoint_meta.get("global_step", -1))
            result.update({
                "trained_checkpoint": str(checkpoint),
                "trained_checkpoint_complete": checkpoint_step >= args.required_steps,
                "trained_checkpoint_model_identity_file": trained_identity.name,
                "trained_checkpoint_model_sha256": file_sha256(trained_identity),
                "trained_checkpoint_meta_sha256": file_sha256(metadata),
                "trained_checkpoint_global_step": checkpoint_step,
            })
            if checkpoint_step < args.required_steps:
                result["trained_checkpoint_error"] = (
                    "checkpoint predates required smoke steps"
                )
                result["passed"] = False
    if args.require_loss_decrease and not loss_decreased:
        result["passed"] = False
    write_json(str(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
