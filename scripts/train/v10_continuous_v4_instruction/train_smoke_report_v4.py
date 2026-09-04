"""Persist finite-loss, timing, memory, and provenance evidence for a V4 GPU smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..v10_continuous_v3_memory.common_v3 import file_sha256, write_json
from ..v10_continuous_v3_memory.train_smoke_report_v3 import summarize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--resize-mode", required=True)
    parser.add_argument("--required-steps", type=int, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    result = summarize(
        args.log,
        args.monitor,
        args.output,
        task=args.task,
        resize_mode=args.resize_mode,
        required_steps=args.required_steps,
        exit_code=args.exit_code,
    )
    manifest = json.loads((args.snapshot / "manifest.json").read_text(encoding="utf-8"))
    model_identity = args.checkpoint / "model.safetensors"
    if not model_identity.is_file():
        model_identity = args.checkpoint / "model.safetensors.index.json"
    if not model_identity.is_file():
        raise FileNotFoundError(f"missing model weight identity in {args.checkpoint}")
    result.update({
        "schema_version": "memory_v4_gpu_smoke_report_v1",
        "snapshot": str(args.snapshot.resolve()),
        "snapshot_content_digest": manifest["content_digest"],
        "snapshot_manifest_sha256": file_sha256(args.snapshot / "manifest.json"),
        "initial_checkpoint": str(args.checkpoint.resolve()),
        "initial_checkpoint_model_identity_file": model_identity.name,
        "initial_checkpoint_model_sha256": file_sha256(model_identity),
        "gpu_id": str(args.gpu_id),
        "batch_size": args.batch_size,
        "assistant_outputs_l3": False,
        "formal_training": False,
    })
    write_json(str(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
