"""V5 dataset registrations and exact-count training configuration builder."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import zlib
from typing import Any

import yaml

from x2robot_dataset_v2.processors.text.base import register_text_processor
from x2robot_dataset_v2.processors.text.multimodal_jsonl_qwen3_5_text_processor import (
    QWEN_DIALOGUES_KEY,
    MultimodalJsonlQwen3_5TextProcessor,
)
from x2robot_dataset_v2.processors.vision.multimodal_jsonl_vision_processor import (
    _resolve_multimodal_sample_idx,
)
from x2robot_dataset_v2.readers.multimodal_jsonl_reader import load_indexed_jsonl_item
from x2robot_dataset_v2.utils.multimodal_utils import process_dialogue

from ..discovery.common.atomic import atomic_write
from ..context import dataset as _dataset  # noqa: F401
from ..context.common import file_sha256, write_json
from . import epilogue as _epilogue  # noqa: F401
from .exposure_plan import plan_exposure
from .holdout import (
    EvaluationHoldout,
    DEFAULT_EVALUATION_MANIFEST,
    DEFAULT_EVALUATION_SHA256,
    audit_artifact,
)
from .prompt import prompt_renderer_digest, prompt_renderer_digest_v53, render_user
from .schema import (
    SNAPSHOT_SCHEMA_VERSION,
    SNAPSHOT_SCHEMA_VERSION_V53,
    dumps_with_mask_spans,
    validate_sample,
)
from .bucket_mix import (
    DEFAULT_WEIGHTS as V53_DEFAULT_WEIGHTS,
    allocate_exposures as allocate_exposures_v53,
    allocate_weighted_exposures,
)
from .build_snapshot import (
    BASELINE_ONLY_BUCKETS,
    BASELINE_ONLY_WEIGHTS,
    LEAF_SCHEMA_VERSION_V53,
)


TEXT_PROCESSOR_NAME = "v10_action_segment_v5_qwen3_5"
EPILOGUE_NAME = "v10_action_segment_v5_qwen3_5"
LEAF_SCHEMA_VERSION = "v10_action_segment_v5_leaf_v2"
if LEAF_SCHEMA_VERSION_V53 != "v10_action_segment_v5_3_bucket_leaf_v1":
    raise RuntimeError("V5.3 leaf schema literal drifted from build_snapshot")


def _bad_sample_tolerance_config(work_dir: Path) -> dict[str, Any]:
    """Return the DDP-safe decode-failure policy used by the data loader."""
    return {
        "enabled": True,
        "report_path": str(work_dir / "bad_samples_runtime.jsonl"),
        "include_traceback": True,
        "max_traceback_chars": 8000,
    }


def dialogues_from_sample(sample: dict[str, Any]) -> list[dict[str, Any]]:
    validated = validate_sample(sample)
    assistant, spans = dumps_with_mask_spans(
        validated["target"],
        validated["category"],
        validated["output_spec"],
        validated["supervision"]["loss_mask_paths"],
    )
    return [
        {"role": "user", "text": render_user(validated)},
        {
            "role": "assistant",
            "text": assistant,
            "loss_mask_char_spans": [list(value) for value in spans],
        },
    ]


@register_text_processor(TEXT_PROCESSOR_NAME)
class ActionSegmentV5TextProcessor(MultimodalJsonlQwen3_5TextProcessor):
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        episode = data.get("_episode")
        if episode is None:
            return data
        sample_idx = _resolve_multimodal_sample_idx(
            episode, data.get("_frame_local_idx", 0)
        )
        item = load_indexed_jsonl_item(episode.path, sample_idx)
        raw = item.get("v5_sample")
        if not isinstance(raw, dict):
            raise ValueError("V5 indexed row is missing v5_sample")
        sample = copy.deepcopy(raw)
        kept = data.get("_v10_kept_image_positions")
        if kept is not None:
            positions = tuple(int(value) for value in kept)
            images = list(sample["images"])
            if not positions or len(set(positions)) != len(positions):
                raise ValueError("V5 retained image positions are invalid")
            if min(positions) < 0 or max(positions) >= len(images):
                raise IndexError("V5 retained image position is out of range")
            sample["images"] = [images[position] for position in positions]
        dialogues = dialogues_from_sample(sample)
        rng = data.get("_rng") or random.Random(
            zlib.crc32(f"{episode.path}|{sample_idx}".encode())
        )
        processed = process_dialogue(
            dialogues,
            seed=rng.getrandbits(64),
            num_images=len(sample["images"]),
        )
        # Our V5 prompt is not a legacy augmentation target.  The generic
        # processor therefore returns a deepcopy and preserves mask spans.
        assistant = [turn for turn in processed if turn.get("role") == "assistant"]
        if len(assistant) != 1 or "loss_mask_char_spans" not in assistant[0]:
            raise ValueError("V5 assistant loss-mask metadata was not preserved")
        data[QWEN_DIALOGUES_KEY] = json.dumps(processed, ensure_ascii=False)
        image_tokens = sum(
            (turn.get("text", "") or "").count("<image>") for turn in processed
        )
        if image_tokens != len(sample["images"]):
            raise ValueError(
                f"V5 image placeholder mismatch: {image_tokens} != {len(sample['images'])}"
            )
        data["_expected_image_count"] = image_tokens
        return data


def discover_leaf_datasets(root: Path, split: str = "train") -> list[dict[str, Any]]:
    # Keep this literal local: the legacy materializer contract test executes
    # this dependency-free function directly from its AST.  The module-level
    # assertion above keeps it pinned to build_snapshot in normal imports.
    v53_leaf_schema_version = "v10_action_segment_v5_3_bucket_leaf_v1"
    root = root.resolve()
    leaves: list[dict[str, Any]] = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        try:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        leaf_version = value.get("schema_version")
        if leaf_version not in {LEAF_SCHEMA_VERSION, v53_leaf_schema_version}:
            continue
        if value.get("split") != split:
            continue
        leaf = manifest_path.parent
        for required in ("data.jsonl", "data.index", "episodes.jsonl"):
            if not (leaf / required).is_file():
                raise FileNotFoundError(f"V5 leaf is incomplete: {leaf / required}")
        count = int(value.get("num_samples", -1))
        if count <= 0:
            raise ValueError(f"V5 leaf has no samples: {leaf}")
        is_v53 = leaf_version == v53_leaf_schema_version
        memory_pair_eligible = value.get("memory_pair_eligible", False)
        if not isinstance(memory_pair_eligible, bool):
            raise ValueError(f"V5 leaf memory_pair_eligible must be boolean: {leaf}")
        context_axis = str(
            value.get("context_variant") if is_v53 else value.get("memory_variant")
        )
        item = {
            "path": str(leaf),
            "source": str(value["source"]),
            "category": str(value["category"]),
            # Keep the historical key because sampler task naming is an
            # internal storage concern.  For V5.3 it carries the explicit
            # context axis (often "mixed" for one physical bucket leaf).
            "memory_variant": context_axis,
            "context_variant": context_axis,
            "output_profile_id": str(value["output_profile_id"]),
            "task_name": str(value["task_name"]),
            "split": split,
            "num_samples": count,
            "memory_pair_eligible": memory_pair_eligible,
            "leaf_schema_version": leaf_version,
            "manifest_sha256": file_sha256(manifest_path),
        }
        expected_task = (
            "v5__{source}__{memory_variant}__{category}__"
            "{output_profile_id}__{task_name}"
        ).format(**item)
        item["sampler_task_name"] = expected_task.replace("/", "_")
        leaves.append(item)
    if not leaves:
        raise ValueError(f"no V5 {split} leaf datasets found under {root}")
    names = [item["sampler_task_name"] for item in leaves]
    if len(names) != len(set(names)):
        raise ValueError("V5 leaf task names are not globally unique")
    return leaves


def prepare(
    dataset_root: Path,
    work_dir: Path,
    model_path: Path,
    *,
    max_length: int,
    max_budget: int | None = None,
    split: str = "train",
    full_coverage: bool = False,
    source_weights: dict[str, int] | None = None,
    evaluation_holdout: EvaluationHoldout | None = None,
    allow_partial_generation: bool = False,
    expected_content_digest: str | None = None,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    work_dir = work_dir.resolve()
    model_path = model_path.resolve()
    root_manifest_path = dataset_root / "manifest.json"
    if not root_manifest_path.is_file():
        raise ValueError("V5 training root has no manifest")
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    root_schema_version = root_manifest.get("schema_version")
    is_v53_snapshot = root_schema_version == SNAPSHOT_SCHEMA_VERSION_V53
    if root_schema_version not in {SNAPSHOT_SCHEMA_VERSION, SNAPSHOT_SCHEMA_VERSION_V53}:
        raise ValueError(f"unsupported V5 snapshot schema: {root_schema_version!r}")
    observed_content_digest = root_manifest.get("content_digest")
    if not isinstance(observed_content_digest, str) or not observed_content_digest:
        raise ValueError("V5 training root has no content digest")
    if expected_content_digest is not None and (
        observed_content_digest != expected_content_digest
    ):
        raise ValueError("V5 training root content digest differs from the pinned value")
    generation_partial = root_manifest.get("partial") is True
    if generation_partial and not allow_partial_generation:
        raise PermissionError(
            "partial V5 generation requires --allow-partial-generation"
        )
    if generation_partial and expected_content_digest is None:
        raise ValueError(
            "partial V5 generation requires --expected-content-digest"
        )
    holdout_audit: dict[str, Any] | None = None
    if evaluation_holdout is not None:
        root_holdout = root_manifest.get("evaluation_holdout")
        if not isinstance(root_holdout, dict):
            raise ValueError("V5 training root is not Evaluation holdout-fenced")
        if root_holdout.get("manifest_sha256") != evaluation_holdout.manifest_sha256:
            raise ValueError("V5 training root uses a different Evaluation holdout holdout")
        relative = Path(str(root_holdout.get("report_relative_path") or ""))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("V5 training root has an unsafe Evaluation holdout report path")
        report_path = dataset_root / relative
        if (
            not report_path.is_file()
            or file_sha256(report_path) != root_holdout.get("report_sha256")
        ):
            raise ValueError("V5 training root Evaluation holdout report digest mismatch")
        holdout_audit = audit_artifact(
            dataset_root,
            evaluation_holdout,
            fail_on_match=True,
        )
    if max_length not in {8192, 16384, 32768, 65536, 131072, 262144}:
        raise ValueError("V5 max_length must use an approved context bucket")
    leaves = discover_leaf_datasets(dataset_root, split=split)
    present_sources = {item["source"] for item in leaves}
    if source_weights is None and present_sources == {"robodojo", "takeover_q"}:
        # The reviewed RT snapshot is intentionally a two-source
        # specialization artifact.  The formal three-source contract remains
        # 70/15/15; RT uses an explicit balanced exposure instead of failing
        # because the still-migrating Baseline source is absent.
        source_weights = {"robodojo": 50, "takeover_q": 50}
    if source_weights is not None:
        leaves = [item for item in leaves if item["source"] in source_weights]
        if not leaves:
            raise ValueError("requested V5 source subset has no leaves")
    if split == "train" and is_v53_snapshot:
        available = {item["source"]: item["num_samples"] for item in leaves}
        if len(available) != len(leaves):
            raise ValueError("V5.3 snapshot requires one physical leaf per bucket")
        requested_total = max_budget or sum(available.values())
        baseline_only = set(available) == set(BASELINE_ONLY_BUCKETS)
        if baseline_only:
            weights = dict(BASELINE_ONLY_WEIGHTS)
            if source_weights is not None:
                weights.update({name: float(weight) for name, weight in source_weights.items()})
            bucket_counts = allocate_weighted_exposures(
                available,
                total=requested_total,
                weights=weights,
            )
        else:
            weights = dict(V53_DEFAULT_WEIGHTS)
            if source_weights is not None:
                weights.update({name: float(weight) for name, weight in source_weights.items()})
            bucket_counts = allocate_exposures_v53(
                available,
                total=requested_total,
                weights=weights,
            )
        virtual_tasks = []
        counts = {}
        for item in leaves:
            desired = bucket_counts[item["source"]]
            remaining = desired
            replica = 0
            while remaining > 0:
                replica += 1
                count = min(item["num_samples"], remaining)
                task_name = f"{item['sampler_task_name']}__replica_{replica:04d}"
                virtual_tasks.append({
                    "source": item["source"],
                    "path": item["path"],
                    "task_name": task_name,
                    "count": count,
                    "physical_leaf_task": item["sampler_task_name"],
                    "replica_index": replica,
                })
                counts[task_name] = count
                remaining -= count
        exposure: dict[str, Any] = {
            "schema_version": "v10_action_segment_v5_3_training_exposure_v1",
            "snapshot_profile": "baseline_only" if baseline_only else "seven_bucket",
            "full_coverage": full_coverage,
            "total_exposures": sum(counts.values()),
            "counts_per_task": counts,
            "virtual_tasks": virtual_tasks,
            "source_weights": weights,
            "source_exposures": bucket_counts,
            "replay_enabled": any(
                bucket_counts[name] > available[name] for name in available
            ),
        }
        if baseline_only:
            exposure["excluded_sources"] = [
                "robodojo", "takeover", "replan_self", "replan_open"
            ]
        else:
            exposure.update({
                "robodojo_fraction": bucket_counts["robodojo"] / requested_total,
                "robodojo_fraction_exact_policy": 0.20,
            })
    elif split == "train":
        exposure = plan_exposure(
            leaves,
            full_coverage=full_coverage,
            requested_total=max_budget,
            source_weights=source_weights,
        )
        virtual_tasks = exposure["virtual_tasks"]
        counts = exposure["counts_per_task"]
    else:
        if full_coverage is False:
            # Validation is always full physical coverage and never replayed;
            # the flag is intentionally ignored for this split.
            full_coverage = True
        virtual_tasks = [
            {
                "source": item["source"],
                "path": item["path"],
                "task_name": item["sampler_task_name"],
                "count": item["num_samples"],
                "physical_leaf_task": item["sampler_task_name"],
                "replica_index": 1,
            }
            for item in leaves
        ]
        counts = {item["task_name"]: item["count"] for item in virtual_tasks}
        exposure = {
            "schema_version": "v10_action_segment_v5_validation_exposure_v1",
            "full_coverage": True,
            "total_exposures": sum(counts.values()),
            "counts_per_task": counts,
            "virtual_tasks": virtual_tasks,
            "replay_enabled": False,
        }
    grouped: dict[str, list[dict[str, str]]] = {}
    for item in virtual_tasks:
        grouped.setdefault(item["source"], []).append({
            "path": item["path"],
            "episode_type": "x2_multimodal",
            "task_name": item["task_name"],
        })
    work_dir.mkdir(parents=True, exist_ok=True)
    quarantine = work_dir / "media_quarantine.json"
    if not quarantine.exists():
        write_json(str(quarantine), {"schema_version": "v10_media_quarantine_v1", "sample_ids": []})
    sources = [
        {"name": source, "source_type": "multimodal", "paths": paths}
        for source, paths in sorted(grouped.items())
    ]
    config = {
        "dataset": {
            "train_test_split": 1.0,
            "multimodal_chunk_size": 64,
            "bad_sample_tolerance": _bad_sample_tolerance_config(work_dir),
            "sampler": {
                "seed": 42,
                "type": "default",
                "batch_size": 1,
                "task_balance": {
                    "type": "static_count",
                    "params": {"counts_per_task": counts},
                    "unit": "frames",
                },
                "task_balance_report_path": str(work_dir / "task_balance_report.json"),
            },
            "pipeline": ["vision", "text", "metadata"],
            "cache": {"enabled": False, "dir": str(work_dir / "dataset_cache")},
            "processors": {
                "vision": {
                    "type": "v10_memory_video_frame",
                    "params": {
                        "image_factor": 32,
                        "min_pixels": 1024,
                        "max_pixels": 589824,
                        "pixel_cap": 589824,
                        "target_long_edge": 640,
                        "resize_policy_id": "auto_near_640_no_upscale_v1",
                        "decoder_backend": "av",
                        "quarantine_path": str(quarantine),
                        "media_failure_report_path": str(work_dir / "media_failures_runtime.jsonl"),
                        "drop_failed_views": True,
                    },
                },
                "text": {"type": TEXT_PROCESSOR_NAME, "params": {}},
                "epilogue": {
                    "type": EPILOGUE_NAME,
                    "params": {
                        "processor_path": str(model_path),
                        "max_seq_length": max_length,
                        "padding_side": "right",
                        "packing": False,
                    },
                },
            },
            "sources": sources,
        }
    }
    config_path = work_dir / "data.yml"
    with atomic_write(str(config_path)) as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
    exposure_path = work_dir / "exposure_plan.json"
    write_json(str(exposure_path), exposure)
    summary = {
        "schema_version": "v10_action_segment_v5_prepare_v1",
        "snapshot_schema_version": root_schema_version,
        "dataset_root": str(dataset_root),
        "data_config": str(config_path),
        "data_config_sha256": file_sha256(config_path),
        "prompt_renderer_sha256": (
            prompt_renderer_digest_v53() if is_v53_snapshot else prompt_renderer_digest()
        ),
        "max_length": max_length,
        "physical_samples": sum(item["num_samples"] for item in leaves),
        "selected_samples": sum(counts.values()),
        "full_coverage": full_coverage,
        "source_weights": exposure.get("source_weights"),
        "leaf_count": len(leaves),
        "physical_source_counts": {
            source: sum(item["num_samples"] for item in leaves if item["source"] == source)
            for source in sorted(grouped)
        },
        "exposure_plan": str(exposure_path),
        "exposure_source_counts": exposure.get("source_exposures"),
        "counts_per_task": counts,
        "generation_partial": generation_partial,
        "partial_generation_authorized": (
            generation_partial and allow_partial_generation
        ),
        "content_digest": observed_content_digest,
    }
    if evaluation_holdout is not None:
        summary["evaluation_holdout"] = {
            **evaluation_holdout.metadata(),
            "independent_audit": holdout_audit,
            "training_overlap_samples": 0,
        }
    write_json(str(work_dir / "prepare_summary.json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=65536)
    parser.add_argument("--max-budget", type=int)
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--full-coverage", action="store_true")
    parser.add_argument("--allow-partial-generation", action="store_true")
    parser.add_argument("--expected-content-digest")
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        default=DEFAULT_EVALUATION_MANIFEST,
    )
    parser.add_argument(
        "--evaluation-expected-sha256",
        default=DEFAULT_EVALUATION_SHA256,
    )
    parser.add_argument(
        "--source-weights",
        help="Comma-separated integer percentages, e.g. robodojo=50,takeover_q=50.",
    )
    args = parser.parse_args()
    evaluation_holdout = EvaluationHoldout.load(
        args.evaluation_manifest,
        expected_sha256=args.evaluation_expected_sha256,
    )
    source_weights = None
    if args.source_weights:
        source_weights = {}
        for item in args.source_weights.split(","):
            name, separator, raw_value = item.partition("=")
            if not separator or not name or name in source_weights:
                raise ValueError(f"invalid --source-weights item: {item!r}")
            source_weights[name] = int(raw_value)
    print(json.dumps(prepare(
        args.dataset_root,
        args.work_dir,
        args.model_path,
        max_length=args.max_length,
        max_budget=args.max_budget,
        split=args.split,
        full_coverage=args.full_coverage,
        source_weights=source_weights,
        evaluation_holdout=evaluation_holdout,
        allow_partial_generation=args.allow_partial_generation,
        expected_content_digest=args.expected_content_digest,
    ), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
