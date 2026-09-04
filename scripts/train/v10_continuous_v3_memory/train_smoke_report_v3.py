"""Summarize finite-loss, timing, and peak-memory evidence for a V3 GPU smoke."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from .common_v3 import write_json


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _numbers(pattern: str, text: str) -> list[float]:
    return [float(value) for value in re.findall(pattern, text)]


def summarize(
    log: Path,
    monitor: Path,
    output: Path,
    *,
    task: str,
    resize_mode: str,
    required_steps: int,
    exit_code: int,
) -> dict[str, Any]:
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    losses = _numbers(rf"['\"]loss['\"]\s*:\s*['\"]?({NUMBER})", text)
    grad_norms = _numbers(rf"['\"]grad_norm['\"]\s*:\s*['\"]?({NUMBER})", text)
    runtimes = _numbers(rf"['\"]train_runtime['\"]\s*:\s*['\"]?({NUMBER})", text)
    memory_used_mib: list[float] = []
    gpu_utils: list[float] = []
    if monitor.is_file():
        for line in monitor.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = [value.strip() for value in line.split(",")]
            if len(fields) >= 2:
                try:
                    memory_used_mib.append(float(fields[0]))
                    gpu_utils.append(float(fields[1]))
                except ValueError:
                    continue
    finite = bool(losses) and all(math.isfinite(value) for value in losses + grad_norms)
    runtime = runtimes[-1] if runtimes else None
    completed_steps = len(losses)
    result = {
        "schema_version": "memory_v3_gpu_smoke_report_v1",
        "task": task,
        "resize_mode": resize_mode,
        "exit_code": int(exit_code),
        "required_steps": int(required_steps),
        "completed_logged_steps": completed_steps,
        "losses": losses,
        "grad_norms": grad_norms,
        "finite_loss_and_grad_norm": finite,
        "loss_improved_below_first": (
            len(losses) > 1 and min(losses[1:]) < losses[0]
        ),
        "train_runtime_seconds": runtime,
        "mean_step_time_seconds": (
            runtime / completed_steps if runtime is not None and completed_steps else None
        ),
        "peak_gpu_memory_mib": max(memory_used_mib, default=None),
        "max_gpu_utilization_percent": max(gpu_utils, default=None),
        "first_batch_mask_shape_reported": "[v10-first-batch]" in text,
        "oom_detected": "out of memory" in text.lower(),
        "log": str(log.resolve()),
        "monitor": str(monitor.resolve()),
    }
    result["passed"] = bool(
        exit_code == 0
        and completed_steps >= required_steps
        and finite
        and result["first_batch_mask_shape_reported"]
        and result["peak_gpu_memory_mib"] is not None
    )
    write_json(str(output), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--resize-mode", required=True)
    parser.add_argument("--required-steps", type=int, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args()
    result = summarize(
        args.log, args.monitor, args.output,
        task=args.task, resize_mode=args.resize_mode,
        required_steps=args.required_steps, exit_code=args.exit_code,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
