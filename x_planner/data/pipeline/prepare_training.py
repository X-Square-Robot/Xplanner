#!/usr/bin/env python3
"""Validate a V10 snapshot and generate the reference deployment dataset/runtime config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

from .snapshot import atomic_write_json, sha256_file
from .validate_schema import validate_snapshot


DEFAULT_MODEL = Path(os.environ.get("XPLANNER_MODEL_PATH", "/path/to/Qwen3.5-9B"))


def prepare(
    snapshot: Path,
    work_dir: Path,
    model_path: Path,
    *,
    max_length: int,
    allow_small: bool,
    enable_memory_noise: bool,
) -> dict[str, object]:
    snapshot = snapshot.resolve()
    work_dir = work_dir.resolve()
    model_path = model_path.resolve()
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = validate_snapshot(snapshot)
    train = manifest["splits"]["train"]
    profiles = manifest.get("train_profile_datasets", {})
    if not allow_small:
        if int(train["episodes"]) < 100:
            raise ValueError(f"formal training requires >=100 train Episodes, got {train['episodes']}")
        if int(manifest["splits"]["validation"]["episodes"]) < 10:
            raise ValueError("formal training requires >=10 validation Episodes")
        if len(profiles) < 2:
            raise ValueError(f"formal training requires >=2 Profiles, got {sorted(profiles)}")
    if not profiles:
        raise ValueError("snapshot has no train_profile_datasets")
    # Transformers may save a complete checkpoint either as sharded weights
    # plus an index or as one model.safetensors file.  The previous sharded-only
    # gate incorrectly rejected the valid single-file checkpoints produced by
    # the V10 trainer after full model consolidation.
    has_sharded_weights = (model_path / "model.safetensors.index.json").is_file()
    has_single_weights = (model_path / "model.safetensors").is_file()
    if not (has_sharded_weights or has_single_weights):
        raise FileNotFoundError(f"incomplete model directory: {model_path}")
    work_dir.mkdir(parents=True, exist_ok=True)
    step_state = work_dir / "v10_step_state.json"
    quarantine_path = work_dir / "media_quarantine.json"
    bad_sample_report = work_dir / "bad_samples_runtime.jsonl"
    media_failure_report = work_dir / "media_failures_runtime.jsonl"
    atomic_write_json(step_state, {"global_step": 0, "memory_noise_probability": 0.0})
    if not quarantine_path.exists():
        atomic_write_json(quarantine_path, {
            "schema_version": "v10_media_quarantine_v1",
            "sample_ids": [],
        })

    paths = []
    for profile in sorted(profiles):
        dataset_path = snapshot / "train_profiles" / profile
        paths.append({
            "path": str(dataset_path),
            "episode_type": "x2_multimodal",
            "task_name": f"v10_{profile}",
        })
    config = {
        "dataset": {
            "train_test_split": 1.0,
            "multimodal_chunk_size": 200,
            "bad_sample_tolerance": {
                "enabled": True,
                "report_path": str(bad_sample_report),
                "include_traceback": True,
                "max_traceback_chars": 8000,
            },
            "sampler": {
                "seed": 42,
                "type": "default",
                "batch_size": 1,
                "task_balance": {
                    "type": "power_law",
                    "params": {"alpha": 0.5},
                    "unit": "frames",
                },
                "task_balance_report_path": str(work_dir / "task_balance_report.json"),
            },
            "pipeline": ["vision", "text", "metadata"],
            "cache": {"enabled": False, "dir": str(work_dir / "dataset_cache")},
            "processors": {
                "vision": {
                    "type": "v10_resilient_video_frame",
                    "params": {
                        "image_factor": 32,
                        "min_pixels": 1024,
                        "max_pixels": 589824,
                        "max_pixels_split_by_images": True,
                        "decoder_backend": "av",
                        "quarantine_path": str(quarantine_path),
                        "media_failure_report_path": str(media_failure_report),
                        "drop_failed_views": True,
                    },
                },
                "text": {
                    "type": "v10_video_frame_qwen3_5",
                    "params": {
                        "step_state_path": str(step_state),
                        "enable_memory_noise": bool(enable_memory_noise),
                        "memory_seed": 42,
                        "short_memory_k": 1,
                        "visible_long_memory_limit": 8,
                    },
                },
                "epilogue": {
                        "type": "v10_multimodal_qwen3_5",
                    "params": {
                        "processor_path": str(model_path),
                        "max_seq_length": max_length,
                        "padding_side": "right",
                        "packing": False,
                    },
                },
            },
            "sources": [{
                "name": "event_states",
                "source_type": "multimodal",
                "paths": paths,
            }],
        }
    }
    config_path = work_dir / "data.yml"
    temporary = config_path.with_suffix(".yml.tmp")
    temporary.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    os.replace(temporary, config_path)
    summary = {
        "snapshot": str(snapshot),
        "manifest_path": str(manifest_path),
        "manifest_digest": manifest["content_digest"],
        "manifest_file_sha256": sha256_file(manifest_path),
        "data_config": str(config_path),
        "data_config_digest": sha256_file(config_path),
        "step_state": str(step_state),
        "quarantine_path": str(quarantine_path),
        "bad_sample_report": str(bad_sample_report),
        "media_failure_report": str(media_failure_report),
        "profiles": sorted(profiles),
        "train_episodes": train["episodes"],
        "train_samples": train["samples"],
        "validation": validation,
        "memory_noise_enabled": bool(enable_memory_noise),
    }
    atomic_write_json(work_dir / "prepare_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--allow-small", action="store_true")
    parser.add_argument("--disable-memory-noise", action="store_true")
    args = parser.parse_args()
    result = prepare(
        args.snapshot,
        args.work_dir,
        args.model_path,
        max_length=args.max_length,
        allow_small=args.allow_small,
        enable_memory_noise=not args.disable_memory_noise,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
