#!/usr/bin/env python3
"""Create a recoverable checkpoint branch for training on a newer snapshot."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from .snapshot import atomic_write_json, sha256_file


def latest_checkpoint(output_dir: Path) -> Path:
    values: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            values.append((int(path.name.rsplit("-", 1)[1]), path))
        except ValueError:
            continue
    if not values:
        raise FileNotFoundError(f"no checkpoint-* found in {output_dir}")
    return max(values)[1]


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def branch(
    source_output: Path,
    target_output: Path,
    snapshot: Path,
    data_config: Path,
) -> dict[str, object]:
    source = latest_checkpoint(source_output.resolve())
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint is incomplete, missing {missing}: {source}")
    manifest_path = snapshot.resolve() / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_output = target_output.resolve()
    target_output.mkdir(parents=True, exist_ok=True)
    destination = target_output / source.name
    if destination.exists():
        raise FileExistsError(f"checkpoint branch already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{source.name}-", dir=target_output))
    try:
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            if relative.as_posix() in {"x2_sampler_state.json", "v10_checkpoint_meta.json"}:
                continue
            target = temporary / relative
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                _link_or_copy(path, target)
        old_meta_path = source / "v10_checkpoint_meta.json"
        old_meta = (
            json.loads(old_meta_path.read_text(encoding="utf-8"))
            if old_meta_path.is_file() else {}
        )
        trainer_state = json.loads((source / "trainer_state.json").read_text(encoding="utf-8"))
        metadata = {
            **old_meta,
            "schema_version": "v10_checkpoint_meta_v1",
            "global_step": int(trainer_state["global_step"]),
            "manifest_path": str(manifest_path),
            "manifest_digest": str(manifest["content_digest"]),
            "data_config_digest": sha256_file(data_config.resolve()),
            "resume_mode": "refresh-data",
            "parent_checkpoint": str(source),
            "branched_at_unix": time.time(),
            "sampler_state_reset": True,
        }
        atomic_write_json(temporary / "v10_checkpoint_meta.json", metadata)
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    result = {
        "source_checkpoint": str(source),
        "branch_checkpoint": str(destination),
        "global_step": metadata["global_step"],
        "manifest_digest": metadata["manifest_digest"],
        "optimizer_scheduler_rng_preserved": True,
        "sampler_state_reset": True,
    }
    atomic_write_json(target_output / "branch_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--target-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    args = parser.parse_args()
    result = branch(
        args.source_output,
        args.target_output,
        args.snapshot,
        args.data_config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
