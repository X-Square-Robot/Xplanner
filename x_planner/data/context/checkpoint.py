"""Read-only V2 checkpoint audit and guarded V3 checkpoint branching."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from ..pipeline.branch_checkpoint import _link_or_copy
from ..pipeline.snapshot import atomic_write_json, sha256_file
from .common import write_json


DEFAULT_SOURCE_RUN = Path(os.environ.get(
    "XPLANNER_SOURCE_RUN", "/path/to/source_training_run"
))


def _read_parent_metadata(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    if not path.is_file():
        return {}, {"status": "missing", "path": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8")), {
            "status": "readable", "path": str(path)
        }
    except Exception as exc:
        return {}, {
            "status": "unreadable",
            "path": str(path),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _reset_scheduler_stage(checkpoint: Path) -> dict[str, Any]:
    """Restart the LR schedule for the first V3 stage without touching optimizer moments."""
    import torch

    path = checkpoint / "scheduler.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not isinstance(state.get("base_lrs"), list):
        raise ValueError(f"unsupported scheduler state: {path}")
    previous = {
        "last_epoch": int(state.get("last_epoch", -1)),
        "step_count": int(state.get("_step_count", 0)),
        "last_lrs": [float(value) for value in state.get("_last_lr", ())],
    }
    base_lrs = [float(value) for value in state["base_lrs"]]
    state.update({
        "last_epoch": 0,
        "_step_count": 1,
        "_is_initial": False,
        "_get_lr_called_within_step": False,
        "_last_lr": [0.0 for _ in base_lrs],
    })
    temporary = path.with_name(path.name + ".v3-stage.tmp")
    try:
        torch.save(state, temporary)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "policy": "restart_first_v3_stage_preserve_optimizer_moments",
        "previous": previous,
        "current": {
            "last_epoch": 0,
            "step_count": 1,
            "last_lrs": [0.0 for _ in base_lrs],
            "base_lrs": base_lrs,
        },
    }
def branch_selected(
    source: Path,
    target_output: Path,
    snapshot: Path,
    data_config: Path,
    *,
    reset_scheduler_stage: bool = False,
) -> dict[str, Any]:
    """Branch the exact already-audited checkpoint, never a moving latest glob."""
    source = source.resolve()
    target_output = target_output.resolve()
    manifest_path = snapshot.resolve() / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trainer_state = json.loads((source / "trainer_state.json").read_text(encoding="utf-8"))
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
        scheduler_stage = (
            _reset_scheduler_stage(temporary) if reset_scheduler_stage else None
        )
        old_meta_path = source / "v10_checkpoint_meta.json"
        old_meta, parent_metadata = _read_parent_metadata(old_meta_path)
        metadata = {
            **old_meta,
            "schema_version": "v10_checkpoint_meta_v1",
            "global_step": int(trainer_state["global_step"]),
            "manifest_path": str(manifest_path),
            "manifest_digest": str(manifest["content_digest"]),
            "data_config_digest": sha256_file(data_config.resolve()),
            "resume_mode": "branch" if reset_scheduler_stage else "refresh-data",
            "parent_checkpoint": str(source),
            "parent_metadata": parent_metadata,
            "branched_at_unix": time.time(),
            "sampler_state_reset": True,
            "scheduler_stage": scheduler_stage or {
                "policy": "preserve_within_v3_refresh_data"
            },
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
        "optimizer_rng_preserved": True,
        "scheduler_preserved": not reset_scheduler_stage,
        "scheduler_stage_reset": reset_scheduler_stage,
        "scheduler_stage": metadata["scheduler_stage"],
        "sampler_state_reset": True,
    }
    atomic_write_json(target_output / "branch_manifest.json", result)
    return result


def audit(source_run: Path) -> dict[str, Any]:
    source_run = source_run.resolve()
    nested_output = source_run / "train"
    training_output = (
        nested_output
        if nested_output.is_dir() and any(nested_output.glob("checkpoint-*"))
        else source_run
    )
    checkpoints = []
    for path in training_output.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        required = {
            "trainer_state.json", "optimizer.pt", "scheduler.pt",
            "v10_checkpoint_meta.json",
        }
        names = {item.name for item in path.iterdir() if item.is_file()}
        missing = sorted(required - names)
        if not any(path.glob("model*.safetensors")):
            missing.append("model*.safetensors")
        if not any(path.glob("rng_state*.pth")):
            missing.append("rng_state*.pth")
        trainer_step = None
        _, metadata_status = _read_parent_metadata(path / "v10_checkpoint_meta.json")
        try:
            trainer_step = int(json.loads(
                (path / "trainer_state.json").read_text(encoding="utf-8")
            )["global_step"])
        except Exception:
            if "trainer_state.json" not in missing:
                missing.append("readable trainer_state.json")
        checkpoints.append({
            "path": str(path), "directory_step": step,
            "trainer_global_step": trainer_step,
            "missing": missing,
            "metadata_status": metadata_status,
            "complete": not missing and trainer_step == step,
        })
    checkpoints.sort(key=lambda item: int(item["directory_step"]))
    launch_log = training_output / "launch.log"
    step_state = source_run / "v10_step_state.json"
    step_state_record: dict[str, Any] = {
        "path": str(step_state), "exists": step_state.exists(), "readable": False,
    }
    try:
        step_state_record.update({
            "readable": True,
            "value": json.loads(step_state.read_text(encoding="utf-8")),
        })
    except Exception as exc:
        step_state_record["error"] = f"{type(exc).__name__}: {exc}"
    latest = checkpoints[-1] if checkpoints else None
    launch_log_age_seconds = (
        max(0.0, time.time() - launch_log.stat().st_mtime)
        if launch_log.is_file() else None
    )
    source_likely_active = bool(
        launch_log_age_seconds is not None and launch_log_age_seconds < 120
    )
    result = {
        "schema_version": "memory_v3_source_checkpoint_audit_v1",
        "source_run": str(source_run),
        "training_output": str(training_output),
        "launch_log": {
            "path": str(launch_log), "exists": launch_log.is_file(),
            "bytes": launch_log.stat().st_size if launch_log.is_file() else None,
            "age_seconds": launch_log_age_seconds,
            "source_likely_active": source_likely_active,
        },
        "step_state": step_state_record,
        "checkpoints": checkpoints,
        "latest_directory": latest,
        "selected_checkpoint": latest["path"] if latest and latest["complete"] else None,
        "branchable": bool(latest and latest["complete"] and launch_log.is_file()),
        "live_source_branch_policy": (
            "copy exact audited complete checkpoint; never re-resolve moving latest"
        ),
    }
    return result


def complete_generation(
    train_dir: Path,
    run_state: Path,
    global_run_state: Path,
) -> dict[str, Any]:
    """Advance pointers only after a new checkpoint is complete and auditable."""
    audited = audit(train_dir)
    selected = audited.get("selected_checkpoint")
    latest = audited.get("latest_directory")
    if not selected or not latest or not latest.get("complete"):
        raise RuntimeError(f"training generation has no complete checkpoint: {train_dir}")
    state = json.loads(run_state.read_text(encoding="utf-8"))
    state.update({
        "current_train_dir": str(train_dir.resolve()),
        "current_checkpoint": str(Path(str(selected)).resolve()),
        "global_step": int(latest["trainer_global_step"]),
        "training_completed_at_unix": time.time(),
    })
    write_json(str(run_state), state)
    global_state = {
        **state,
        "schema_version": "memory_v3_current_run_v1",
        "work_dir": str(run_state.resolve().parent),
        "generation_state": str(run_state.resolve()),
    }
    write_json(str(global_run_state), global_state)
    return {
        "train_dir": str(train_dir.resolve()),
        "current_checkpoint": state["current_checkpoint"],
        "global_step": state["global_step"],
        "snapshot": state["snapshot"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    audit_parser.add_argument("--output", type=Path)
    branch_parser = sub.add_parser("branch")
    branch_parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    branch_parser.add_argument("--target-output", type=Path, required=True)
    branch_parser.add_argument("--snapshot", type=Path, required=True)
    branch_parser.add_argument("--data-config", type=Path, required=True)
    branch_parser.add_argument("--audit-output", type=Path)
    branch_parser.add_argument("--run-state", type=Path)
    branch_parser.add_argument("--global-run-state", type=Path)
    branch_parser.add_argument("--reset-scheduler-stage", action="store_true")
    complete_parser = sub.add_parser("complete-generation")
    complete_parser.add_argument("--train-dir", type=Path, required=True)
    complete_parser.add_argument("--run-state", type=Path, required=True)
    complete_parser.add_argument("--global-run-state", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "complete-generation":
        result = complete_generation(
            args.train_dir, args.run_state, args.global_run_state
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    result = audit(args.source_run)
    output = getattr(args, "output", None) or getattr(args, "audit_output", None)
    if output is not None:
        write_json(str(output), result)
    if args.command == "branch":
        if not result["branchable"]:
            raise RuntimeError("latest V2 checkpoint directory is not safely branchable")
        training_output = Path(str(result["training_output"]))
        result = {
            "audit": result,
            "branch": branch_selected(
                Path(str(result["selected_checkpoint"])),
                args.target_output, args.snapshot, args.data_config,
                reset_scheduler_stage=args.reset_scheduler_stage,
            ),
        }
        if args.run_state is not None:
            generation_state = {
                "schema_version": "memory_v3_training_generation_v1",
                "current_train_dir": str(args.target_output.resolve()),
                "snapshot": str(args.snapshot.resolve()),
                "source_training_output": str(training_output.resolve()),
                "branch_checkpoint": result["branch"]["branch_checkpoint"],
                "global_step": result["branch"]["global_step"],
                "optimizer_rng_preserved": True,
                "scheduler_preserved": result["branch"]["scheduler_preserved"],
                "scheduler_stage_reset": result["branch"]["scheduler_stage_reset"],
                "sampler_state_reset": True,
            }
            write_json(str(args.run_state), generation_state)
            if args.global_run_state is not None:
                write_json(str(args.global_run_state), {
                    **generation_state,
                    "schema_version": "memory_v3_current_run_v1",
                    "work_dir": str(args.run_state.resolve().parent),
                    "generation_state": str(args.run_state.resolve()),
                })
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
