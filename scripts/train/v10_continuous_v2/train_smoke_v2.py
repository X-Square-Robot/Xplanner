#!/usr/bin/env python3
"""Preflight and run a guarded one-GPU V10 V2 training smoke."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..v10_continuous.prepare_training import DEFAULT_MODEL, prepare
from ..v10_continuous.validate_schema import validate_snapshot
from ..v10_continuous.verify_checkpoint import verify


def _gpu_rows() -> list[tuple[int, int, int, int]]:
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], check=True, text=True, capture_output=True)
    rows = []
    for line in result.stdout.splitlines():
        values = [int(value.strip()) for value in line.split(",")]
        if len(values) == 4:
            rows.append(tuple(values))
    return rows


def _choose_gpu(requested: int | None) -> int:
    rows = _gpu_rows()
    candidates = [row for row in rows if row[1] <= 2048 and row[2] >= 220000 and row[3] <= 10]
    if requested is not None:
        match = next((row for row in rows if row[0] == requested), None)
        if match is None:
            raise ValueError(f"GPU {requested} does not exist")
        if match not in candidates:
            raise RuntimeError(
                f"GPU {requested} is not idle enough: used={match[1]}MiB "
                f"free={match[2]}MiB util={match[3]}%"
            )
        return requested
    if not candidates:
        raise RuntimeError(
            "no idle GPU satisfies memory.used<=2GiB, memory.free>=220GB, util<=10%"
        )
    return candidates[0][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--model-max-length", type=int, default=4096)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    snapshot = args.snapshot.resolve()
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    validation = validate_snapshot(snapshot)
    prepared = prepare(
        snapshot, work_dir, args.model_path,
        max_length=args.model_max_length, allow_small=True, enable_memory_noise=True,
    )
    smoke_output = work_dir / "data_smoke.json"
    subprocess.run([
        sys.executable, "-m", "scripts.train.v10_continuous.smoke_data",
        "--data-config", str(work_dir / "data.yml"),
        "--model-path", str(args.model_path.resolve()),
        "--output", str(smoke_output),
    ], check=True)
    if args.preflight_only:
        print(json.dumps({
            "passed": True, "mode": "preflight", "snapshot": str(snapshot),
            "validation": validation, "prepare": prepared, "data_smoke": str(smoke_output),
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return
    gpu = _choose_gpu(args.gpu_id)
    output_dir = work_dir / "train"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "launch.log"
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "X2ROBOT_AV_SEQUENTIAL_SPAN_MAX": "128",
        "V10_SKIP_FINAL_MODEL_SAVE": "1",
    })
    command = [
        sys.executable, "-m", "torch.distributed.run", "--nnodes", "1",
        "--nproc_per_node", "1", "--master_addr", "127.0.0.1",
        "--master_port", str(20000 + os.getpid() % 20000),
        "-m", "scripts.train.v10_continuous.train_v10",
        "--snapshot", str(snapshot), "--step-state", str(work_dir / "v10_step_state.json"),
        "--resume-mode", "exact", "--model_path", str(args.model_path.resolve()),
        "--data_config", str(work_dir / "data.yml"), "--attn_implementation", "sdpa",
        "--model_max_length", str(args.model_max_length), "--image_min_pixels", "1024",
        "--image_max_pixels", "589824", "--bf16", "true", "--tf32", "true",
        "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "1",
        "--max_steps", str(args.max_steps), "--num_train_epochs", "100",
        "--learning_rate", "1e-5", "--llm_lr", "1e-5", "--weight_decay", "0.0",
        "--warmup_steps", "0", "--lr_scheduler_type", "cosine",
        "--gradient_checkpointing", "true", "--ddp_find_unused_parameters", "false",
        "--loss_reduction_scope", "sample", "--lm_head_loss_only_on_labels", "false",
        "--save_strategy", "steps", "--save_steps", str(args.max_steps),
        "--save_total_limit", "1", "--logging_steps", "1",
        "--dataloader_num_workers", "0", "--ignore_data_skip", "true",
        "--seed", "42", "--data_seed", "42", "--disable_tqdm", "true",
        "--output_dir", str(output_dir), "--report_to", "none",
        "--run_name", f"v10_v2_smoke_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    ]
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    verified = verify(output_dir, min_step=args.max_steps, log_file=log_path)
    print(json.dumps({
        "passed": True, "gpu": gpu, "snapshot": str(snapshot),
        "data_smoke": str(smoke_output), "verification": verified,
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
