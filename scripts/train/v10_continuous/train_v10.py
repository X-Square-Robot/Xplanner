#!/usr/bin/env python3
"""V10 wrapper around the proven B300 Qwen3.5 launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from .runtime import (
    DATA_CONFIG_DIGEST_ENV,
    MANIFEST_DIGEST_ENV,
    MANIFEST_PATH_ENV,
    RESUME_MODE_ENV,
    STEP_STATE_ENV,
    apply_b30z_compatibility,
    apply_trainer_patches,
    write_step_state,
)
from .snapshot import atomic_write_json, sha256_file


def _argument_value(arguments: list[str], name: str) -> str:
    for index, value in enumerate(arguments):
        if value == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    raise ValueError(f"launcher argument {name} is required")


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.split("-")[-1]), path))
        except ValueError:
            continue
    return max(checkpoints, default=(0, None))[1]


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--step-state", type=Path, required=True)
    parser.add_argument(
        "--resume-mode",
        choices=("exact", "refresh-data", "weights-only"),
        default="exact",
    )
    known, launcher_args = parser.parse_known_args()
    output_dir = Path(_argument_value(launcher_args, "--output_dir")).resolve()
    data_config = Path(_argument_value(launcher_args, "--data_config")).resolve()
    manifest_path = known.snapshot.resolve() / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = str(manifest["content_digest"])
    latest = _latest_checkpoint(output_dir)
    if known.resume_mode == "weights-only" and latest is not None:
        raise ValueError(
            "weights-only initialization requires a fresh output_dir; found "
            f"existing training checkpoint {latest}"
        )
    if latest is not None:
        meta_path = latest / "v10_checkpoint_meta.json"
        if not meta_path.is_file():
            raise FileNotFoundError(f"V10 checkpoint metadata missing: {meta_path}")
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        previous_digest = previous.get("manifest_digest")
        if known.resume_mode == "exact" and previous_digest != digest:
            raise ValueError(
                f"exact resume manifest mismatch: checkpoint={previous_digest} requested={digest}"
            )
        if known.resume_mode == "refresh-data" and previous_digest != digest:
            atomic_write_json(output_dir / "manifest_transition.json", {
                "from": previous_digest,
                "to": digest,
                "checkpoint": str(latest),
            })
        initial_step = int(previous.get("global_step", 0))
    else:
        initial_step = 0
    write_step_state(known.step_state, initial_step)

    os.environ[STEP_STATE_ENV] = str(known.step_state.resolve())
    os.environ[MANIFEST_PATH_ENV] = str(manifest_path)
    os.environ[MANIFEST_DIGEST_ENV] = digest
    os.environ[DATA_CONFIG_DIGEST_ENV] = sha256_file(data_config)
    os.environ[RESUME_MODE_ENV] = known.resume_mode
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "v10_run_config.json", {
        "snapshot": str(known.snapshot.resolve()),
        "manifest_digest": digest,
        "data_config": str(data_config),
        "data_config_digest": sha256_file(data_config),
        "resume_mode": known.resume_mode,
        "launcher_args": launcher_args,
        "initial_global_step": initial_step,
    })
    apply_b30z_compatibility()
    apply_trainer_patches()
    sys.argv = [sys.argv[0], *launcher_args]
    from qwenvl.train.launcher import train

    train()


if __name__ == "__main__":
    main()
