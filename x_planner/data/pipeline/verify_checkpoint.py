#!/usr/bin/env python3
"""Verify V10 finite loss, masks evidence, and complete checkpoint state."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .branch_checkpoint import latest_checkpoint
from .snapshot import atomic_write_json


def verify(output_dir: Path, *, min_step: int, log_file: Path | None) -> dict[str, object]:
    checkpoint = latest_checkpoint(output_dir.resolve())
    required = {
        "trainer_state": checkpoint / "trainer_state.json",
        "optimizer": checkpoint / "optimizer.pt",
        "scheduler": checkpoint / "scheduler.pt",
        "metadata": checkpoint / "v10_checkpoint_meta.json",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    model_files = list(checkpoint.glob("model*.safetensors"))
    rng_files = list(checkpoint.glob("rng_state*.pth"))
    if not model_files:
        missing.append("model_safetensors")
    if not rng_files:
        missing.append("rng_state")
    if missing:
        raise FileNotFoundError(f"incomplete V10 checkpoint {checkpoint}: {missing}")
    state = json.loads(required["trainer_state"].read_text(encoding="utf-8"))
    metadata = json.loads(required["metadata"].read_text(encoding="utf-8"))
    losses = [
        float(row["loss"])
        for row in state.get("log_history", [])
        if "loss" in row
    ]
    values = losses + [
        float(row["grad_norm"])
        for row in state.get("log_history", [])
        if "grad_norm" in row
    ]
    global_step = int(state.get("global_step", 0))
    if global_step < min_step:
        raise RuntimeError(f"global_step={global_step} is below required {min_step}")
    if int(metadata.get("global_step", -1)) != global_step:
        raise RuntimeError("checkpoint metadata global_step does not match trainer_state")
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("loss/grad_norm is empty or contains NaN/Inf")
    batch_reported = None
    if log_file is not None:
        batch_reported = (
            log_file.is_file()
            and "[v10-first-batch]" in log_file.read_text(encoding="utf-8", errors="replace")
        )
        if not batch_reported:
            raise RuntimeError(f"V10 mask/shape first-batch report missing from {log_file}")
    result = {
        "passed": True,
        "checkpoint": str(checkpoint),
        "global_step": global_step,
        "losses": losses,
        "last_learning_rate": next(
            (row.get("learning_rate") for row in reversed(state.get("log_history", [])) if "learning_rate" in row),
            None,
        ),
        "model_files": [path.name for path in model_files],
        "rng_files": [path.name for path in rng_files],
        "batch_mask_shape_reported": batch_reported,
        "manifest_digest": metadata.get("manifest_digest"),
    }
    atomic_write_json(output_dir.resolve() / "v10_verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-step", type=int, default=1)
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args()
    result = verify(args.output_dir, min_step=args.min_step, log_file=args.log_file)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
