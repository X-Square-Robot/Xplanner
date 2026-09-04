#!/usr/bin/env python3
"""Smoke driver for final_vqa_dedup_train_v4 on wall-vlm.

This file intentionally keeps the compatibility code outside both existing
repositories' implementation modules.  Importing it registers the missing
``video_frame`` vision processor, whose input contract is the dataset's
``{"video": ..., "frame": ...}`` image reference.

Modes:
  self-test  - synthetic decode/order check for the runtime processor
  prepare    - build a deterministic, media-validated indexed-JSONL subset
  train      - delegate to qwenvl.train.launcher after runtime registration
  verify     - check trainer_state.json for finite, decreasing loss
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import struct
import sys
import tempfile
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image

from x2robot_dataset_v2.processors.text.base import register_text_processor
from x2robot_dataset_v2.processors.text.multimodal_jsonl_qwen3_5_text_processor import (
    QWEN_DIALOGUES_KEY,
    MultimodalJsonlQwen3_5TextProcessor,
)
from x2robot_dataset_v2.processors.vision.base import register_vision_processor
from x2robot_dataset_v2.processors.vision.multimodal_jsonl_vision_processor import (
    VIDEO_META_KEY,
    VIDEO_OBSERVATIONS_KEY,
    MultimodalJsonlVisionProcessor,
    _resolve_multimodal_sample_idx,
)
from x2robot_dataset_v2.readers.multimodal_jsonl_reader import (
    load_indexed_jsonl_item,
    load_jsonl_image,
    resolve_jsonl_image_path,
)
from x2robot_dataset_v2.utils import multimodal_schema as _multimodal_schema
from x2robot_dataset_v2.utils.multimodal_schema import normalize_multimodal_dialogues
from x2robot_dataset_v2.utils.multimodal_utils import process_dialogue


DEFAULT_DATA_ROOT = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/luhao/ACode/final_vqa_dedup_train_v4"
)
DEFAULT_MODEL_PATH = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/luhao/models/Qwen3.5-9B"
)
SCENARIOS = {
    "assess_active": ("ASSESS_ACTIVE", "ASSESS"),
    "plan_init": ("PLAN_INIT", "PLAN"),
    "plan_replan": ("PLAN_REPLAN", "PLAN"),
}


def iter_multimodal_image_refs(value: Any) -> list[Any]:
    """Normalize the image field using one rule for vision, text, and lengths."""
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        values = [value]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise TypeError(
            f"image field must be a string, dict, list, or null; got {type(value).__name__}"
        )
    return [
        ref
        for ref in values
        if ref is not None and not (isinstance(ref, str) and not ref.strip())
    ]


# wall-vlm_B300's length estimator imports this symbol, while the paired B300
# data snapshot predates it. Inject it before any sampler subprocess is born.
_multimodal_schema.iter_multimodal_image_refs = iter_multimodal_image_refs


def iter_raw_video_refs(item: dict[str, Any]) -> list[Any]:
    """Normalize the optional top-level video field for length estimation."""
    value = item.get("video")
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        return [value]
    if isinstance(value, (list, tuple)):
        return [ref for ref in value if ref is not None]
    raise TypeError(
        f"video field must be a string, dict, list, or null; got {type(value).__name__}"
    )


def parse_video_ref(value: Any) -> tuple[str, int | None, int | None] | None:
    """Parse the video-reference contract used by wall-vlm's estimator."""
    if isinstance(value, str):
        path = value.strip()
        return (path, None, None) if path else None
    if not isinstance(value, dict):
        return None
    path = value.get("path", value.get("video"))
    if not isinstance(path, str) or not path.strip():
        return None
    start = value.get("start_frame")
    end = value.get("end_frame")
    if isinstance(start, bool) or (start is not None and not isinstance(start, int)):
        return None
    if isinstance(end, bool) or (end is not None and not isinstance(end, int)):
        return None
    return path.strip(), start, end


def resolve_clip_window(
    total_frames: int,
    duration: float,
    native_fps: float,
    start_frame: int | None,
    end_frame: int | None,
) -> tuple[int, int, int, float]:
    """Resolve a bounded frame interval with the paired repo's expected tuple."""
    start = max(0, int(start_frame or 0))
    end = int(end_frame) if end_frame is not None else int(total_frames)
    if total_frames > 0:
        start = min(start, int(total_frames))
        end = min(max(start, end), int(total_frames))
    else:
        end = max(start, end)
    count = max(0, end - start)
    clip_duration = (
        count / native_fps
        if native_fps > 0 and count > 0
        else float(duration)
    )
    return start, end, count, float(clip_duration)


# wall-vlm_B300 also imports a newer utility module that is absent from the
# paired B300 dataset checkout. Expose only its three required pure helpers at
# runtime, keeping both existing repositories untouched.
_video_module_name = "x2robot_dataset_v2.utils.multimodal_video"
if _video_module_name not in sys.modules:
    _video_module = ModuleType(_video_module_name)
    _video_module.iter_raw_video_refs = iter_raw_video_refs
    _video_module.parse_video_ref = parse_video_ref
    _video_module.resolve_clip_window = resolve_clip_window
    sys.modules[_video_module_name] = _video_module


@register_text_processor("video_frame_qwen3_5")
class VideoFrameQwen3_5TextProcessor(MultimodalJsonlQwen3_5TextProcessor):
    """Qwen3.5 dialogue processor whose image count includes frame dictionaries."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        episode = data.get("_episode")
        if episode is None:
            return data
        frame_local_idx = data.get("_frame_local_idx", 0)
        sample_idx = _resolve_multimodal_sample_idx(episode, frame_local_idx)
        item = load_indexed_jsonl_item(episode.path, sample_idx)
        dialogues = normalize_multimodal_dialogues(item)
        num_images = len(iter_multimodal_image_refs(item.get("image")))
        rng = data.get("_rng") or random.Random(
            zlib.crc32(f"{episode.path}|{sample_idx}".encode())
        )
        processed = process_dialogue(
            dialogues,
            seed=rng.getrandbits(64),
            num_images=num_images,
        )
        data[QWEN_DIALOGUES_KEY] = json.dumps(processed, ensure_ascii=False)
        image_token_count = sum(
            (turn.get("text", "") or "").count("<image>") for turn in processed
        )
        if image_token_count > 0:
            data["_expected_image_count"] = image_token_count
        return data


@register_vision_processor("video_frame")
class VideoFrameVisionProcessor(MultimodalJsonlVisionProcessor):
    """Decode ``image`` entries that reference individual video frames.

    String image references remain supported.  Dictionary references are
    grouped by resolved video path so two temporal observations from one video
    share one decoder call, while the original cross-camera order is restored
    before the inherited multimodal image processing runs.
    """

    @staticmethod
    def _parse_frame_ref(ref: dict[str, Any], jsonl_path: str) -> tuple[str, int]:
        video = ref.get("video")
        frame = ref.get("frame")
        if not isinstance(video, str) or not video.strip():
            raise TypeError(f"video-frame ref has invalid video path: {ref!r}")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise TypeError(f"video-frame ref has invalid frame index: {ref!r}")
        return resolve_jsonl_image_path(video.strip(), jsonl_path), frame

    def load_image_refs(
        self, refs: Iterable[Any], jsonl_path: str
    ) -> list[Image.Image]:
        """Load mixed string and video-frame refs without changing their order."""
        refs = list(refs)
        images: list[Image.Image | None] = [None] * len(refs)
        grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for position, ref in enumerate(refs):
            if isinstance(ref, str):
                if not ref.strip():
                    raise TypeError(f"empty image ref at position {position}")
                images[position] = load_jsonl_image(ref.strip(), jsonl_path)
            elif isinstance(ref, dict):
                path, frame = self._parse_frame_ref(ref, jsonl_path)
                grouped[path].append((position, frame))
            else:
                raise TypeError(
                    f"image ref at position {position} must be string or dict, "
                    f"got {type(ref).__name__}"
                )

        for video_path, requests in grouped.items():
            if not os.path.isfile(video_path):
                raise FileNotFoundError(video_path)
            indices = [frame for _position, frame in requests]
            decoded = self.decoder.decode_frames(video_path, indices)
            if len(decoded) != len(requests):
                raise RuntimeError(
                    f"decoder returned {len(decoded)} frames for {len(requests)} "
                    f"requests from {video_path}: {indices}"
                )
            for (position, _frame), array in zip(requests, decoded):
                if not isinstance(array, np.ndarray) or array.ndim != 3:
                    raise TypeError(
                        f"decoder returned invalid frame at position {position}: "
                        f"{type(array).__name__}"
                    )
                images[position] = Image.fromarray(array).convert("RGB")

        if any(image is None for image in images):
            raise RuntimeError("internal error: not every media reference was decoded")
        return [image for image in images if image is not None]

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        episode = data.get("_episode")
        if episode is None:
            return data

        frame_local_idx = data.get("_frame_local_idx", 0)
        sample_idx = _resolve_multimodal_sample_idx(episode, frame_local_idx)
        data["frame_idx"] = sample_idx
        data["uid"] = episode.path
        item = load_indexed_jsonl_item(episode.path, sample_idx)

        image_refs = list(iter_multimodal_image_refs(item.get("image")))
        images = self.load_image_refs(image_refs, episode.path)

        is_train = data.get("_is_train", True)
        rng = data.get("_augmentation_rng") or data.get("_rng")
        self._aug_seed = rng.getrandbits(32) if rng is not None else None
        saved_max_pixels = self.max_pixels
        if self.max_pixels_split_by_images and len(images) > 1:
            self.max_pixels = max(self.min_pixels, self.max_pixels // len(images))
        try:
            result = self.process_multimodal(
                images,
                episode_type=episode.episode_type,
                is_train=is_train,
            )
        finally:
            self.max_pixels = saved_max_pixels
            self._aug_seed = None
        data.update(result)

        if result.get("orig_height", 0) > 0 and result.get("orig_width", 0) > 0:
            data["_grounding_resize_info"] = {
                "orig_height": result["orig_height"],
                "orig_width": result["orig_width"],
                "resized_height": result["resized_height"],
                "resized_width": result["resized_width"],
            }

        videos, video_metas = self._process_videos(episode, item)
        data[VIDEO_OBSERVATIONS_KEY] = videos
        data[VIDEO_META_KEY] = json.dumps(video_metas)
        return data


def _read_raw_row(topic_dir: Path, row: int) -> tuple[bytes, dict[str, Any]]:
    with (topic_dir / "data.index").open("rb") as index_file:
        index_file.seek(row * 8)
        raw_offset = index_file.read(8)
    if len(raw_offset) != 8:
        raise IndexError(f"row {row} has no index entry in {topic_dir}")
    offset = struct.unpack("<Q", raw_offset)[0]
    with (topic_dir / "data.jsonl").open("rb") as data_file:
        data_file.seek(offset)
        raw_line = data_file.readline()
    if not raw_line:
        raise ValueError(f"row {row} is empty in {topic_dir}")
    return raw_line, json.loads(raw_line)


def _validate_semantics(
    item: dict[str, Any], expected_scenario: str, expected_task: str
) -> list[Any]:
    if item.get("training_scenario") != expected_scenario:
        raise ValueError(
            f"scenario={item.get('training_scenario')!r}, expected {expected_scenario!r}"
        )
    if item.get("training_task") != expected_task:
        raise ValueError(
            f"task={item.get('training_task')!r}, expected {expected_task!r}"
        )
    dialogue = item.get("text")
    if not isinstance(dialogue, list) or len(dialogue) != 2:
        raise ValueError("text must be a two-turn dialogue")
    if dialogue[0].get("role") != "user" or dialogue[1].get("role") != "assistant":
        raise ValueError("dialogue roles must be user then assistant")
    user_text = dialogue[0].get("text")
    assistant_text = dialogue[1].get("text")
    if not isinstance(user_text, str) or not isinstance(assistant_text, str):
        raise TypeError("dialogue text values must be strings")
    target = json.loads(assistant_text)
    if expected_task == "ASSESS":
        if target.get("state") not in {"EXECUTING", "SUCCEEDED", "FAILED"}:
            raise ValueError(f"invalid ASSESS target: {target!r}")
    elif not isinstance(target.get("subtasks"), list) or not target["subtasks"]:
        raise ValueError(f"invalid PLAN target: {target!r}")

    refs = list(iter_multimodal_image_refs(item.get("image")))
    if not refs:
        raise ValueError("sample has no media")
    if user_text.count("<image>") != len(refs):
        raise ValueError(
            f"placeholder/media mismatch: {user_text.count('<image>')} != {len(refs)}"
        )
    for ref in refs:
        if isinstance(ref, dict):
            video = ref.get("video")
            if not isinstance(video, str) or not os.path.isabs(video):
                raise ValueError(f"smoke requires absolute video paths: {ref!r}")
        elif isinstance(ref, str):
            if not os.path.isabs(ref):
                raise ValueError(f"smoke requires absolute image paths: {ref!r}")
        else:
            raise TypeError(f"unsupported media ref: {ref!r}")
    return refs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_runtime_config(
    work_dir: Path, dataset_dir: Path, model_path: Path, max_length: int
) -> Path:
    config = {
        "dataset": {
            "train_test_split": 1.0,
            "multimodal_chunk_size": 200,
            "sampler": {
                "length_cache_dir": str(work_dir / "length_cache"),
                "seed": 42,
                "type": "knapsack_packed",
                "cutoff": max_length,
                "pad_to_cutoff": False,
            },
            "pipeline": ["vision", "text", "metadata"],
            "cache": {"enabled": False, "dir": str(work_dir / "dataset_cache")},
            "processors": {
                "vision": {
                    "type": "video_frame",
                    "params": {
                        "image_factor": 32,
                        "min_pixels": 1024,
                        "max_pixels": 589824,
                        "max_pixels_split_by_images": True,
                        "decoder_backend": "av",
                    },
                },
                "text": {"type": "video_frame_qwen3_5"},
                "epilogue": {
                    "type": "multimodal_qwen3_5",
                    "params": {
                        "processor_path": str(model_path),
                        "max_seq_length": max_length,
                        "padding_side": "right",
                        "packing": True,
                    },
                },
            },
            "sources": [
                {
                    "name": "final_vqa_dedup_train_v4_smoke",
                    "source_type": "multimodal",
                    "paths": [
                        {
                            "path": str(dataset_dir),
                            "episode_type": "x2_multimodal",
                            "task_name": "final_vqa_smoke",
                        }
                    ],
                }
            ],
        }
    }
    config_path = work_dir / "data.yml"
    with config_path.open("w", encoding="utf-8") as output:
        yaml.safe_dump(config, output, allow_unicode=True, sort_keys=False)
    return config_path


def prepare(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    model_path = args.model_path.resolve()
    work_dir = args.work_dir.resolve()
    dataset_dir = work_dir / "dataset"
    if dataset_dir.exists():
        raise FileExistsError(
            f"refusing to reuse existing generated dataset: {dataset_dir}"
        )
    if not (model_path / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"incomplete model directory: {model_path}")
    root_manifest = json.loads((data_root / "manifest.json").read_text())

    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir()
    processor = VideoFrameVisionProcessor(
        image_factor=32,
        min_pixels=1024,
        max_pixels=589824,
        decoder_backend="av",
        max_pixels_split_by_images=True,
    )
    selected: list[tuple[str, int, bytes, dict[str, Any]]] = []
    rejected: dict[str, Counter[str]] = {}

    for split_name, (expected_scenario, expected_task) in SCENARIOS.items():
        topic_dir = data_root / split_name / "train"
        count = int(root_manifest["topic_manifests"][f"{split_name}/train"]["num_samples"])
        rng = random.Random(f"{args.seed}:{split_name}")
        candidates = rng.sample(range(count), min(args.max_scan, count))
        reasons: Counter[str] = Counter()
        accepted = 0
        for row in candidates:
            try:
                raw_line, item = _read_raw_row(topic_dir, row)
                refs = _validate_semantics(item, expected_scenario, expected_task)
                images = processor.load_image_refs(refs, str(topic_dir / "data.jsonl"))
                if len(images) != len(refs):
                    raise RuntimeError("decoded image count mismatch")
                for image in images:
                    if image.width <= 0 or image.height <= 0:
                        raise ValueError("decoded an empty image")
                    image.close()
                selected.append((split_name, row, raw_line, item))
                accepted += 1
                if accepted == args.per_scenario:
                    break
            except Exception as exc:  # candidate rejection is recorded, never hidden
                key = f"{type(exc).__name__}: {str(exc)[:180]}"
                reasons[key] += 1
        rejected[split_name] = reasons
        if accepted != args.per_scenario:
            raise RuntimeError(
                f"only found {accepted}/{args.per_scenario} valid rows for "
                f"{split_name} after {len(candidates)} candidates; reasons={dict(reasons)}"
            )

    data_path = dataset_dir / "data.jsonl"
    index_path = dataset_dir / "data.index"
    offsets: list[int] = []
    with data_path.open("wb") as data_output:
        for _split, _row, raw_line, _item in selected:
            offsets.append(data_output.tell())
            data_output.write(raw_line.rstrip(b"\r\n") + b"\n")
    with index_path.open("wb") as index_output:
        for offset in offsets:
            index_output.write(struct.pack("<Q", offset))

    selection_rows = [
        {
            "split": split_name,
            "row": row,
            "data_id": item.get("data_id"),
            "training_scenario": item.get("training_scenario"),
            "training_task": item.get("training_task"),
            "media_count": len(list(iter_multimodal_image_refs(item.get("image")))),
        }
        for split_name, row, _raw, item in selected
    ]
    manifest = {
        "schema_version": "high_policy_sft_smoke_v1",
        "jsonl_file": "data.jsonl",
        "index_file": "data.index",
        "num_samples": len(selected),
        "source_root": str(data_root),
        "seed": args.seed,
        "selected_rows": selection_rows,
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    config_path = _write_runtime_config(
        work_dir, dataset_dir, model_path, args.max_length
    )
    summary = {
        "data_root": str(data_root),
        "model_path": str(model_path),
        "work_dir": str(work_dir),
        "config_path": str(config_path),
        "num_samples": len(selected),
        "selected_rows": selection_rows,
        "excluded_scenarios": {
            "assess_whole": "media unavailable on this host (planning audit: 0/32 sampled paths valid)"
        },
        "rejections": {name: dict(counts) for name, counts in rejected.items()},
        "data_sha256": _sha256(data_path),
        "index_sha256": _sha256(index_path),
    }
    summary_path = work_dir / "prepare_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def self_test(_args: argparse.Namespace) -> None:
    import av

    with tempfile.TemporaryDirectory(prefix="final_vqa_video_frame_") as tmp:
        root = Path(tmp)
        video_path = root / "colors.mp4"
        container = av.open(str(video_path), mode="w")
        stream = container.add_stream("mpeg4", rate=5)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        colors = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255)]
        for color in colors:
            array = np.zeros((64, 64, 3), dtype=np.uint8)
            array[:, :] = color
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

        still_path = root / "still.png"
        Image.new("RGB", (64, 64), (255, 255, 255)).save(still_path)
        item = {
            "image": [
                {"video": str(video_path), "frame": 0, "view": "front"},
                str(still_path),
                {"video": str(video_path), "frame": 3, "view": "wrist"},
            ],
            "text": [
                {"role": "user", "text": "<image><image><image>colors"},
                {"role": "assistant", "text": '{"state":"EXECUTING"}'},
            ],
        }
        raw = json.dumps(item).encode("utf-8") + b"\n"
        (root / "data.jsonl").write_bytes(raw)
        (root / "data.index").write_bytes(struct.pack("<Q", 0))
        (root / "manifest.json").write_text(json.dumps({"num_samples": 1}))

        processor = VideoFrameVisionProcessor(
            image_factor=16,
            min_pixels=4096,
            max_pixels=4096,
            decoder_backend="av",
        )
        episode = SimpleNamespace(
            path=str(root / "data.jsonl"),
            st_index=0,
            no_static_frames=None,
            iter_st=0,
            episode_type="x2_multimodal",
        )
        result = processor(
            {"_episode": episode, "_frame_local_idx": 0, "_is_train": False}
        )
        observations = result["image_observations"]
        if len(observations) != 3:
            raise AssertionError(f"expected 3 ordered observations, got {len(observations)}")
        pixels = [obs[0].getpixel((32, 32)) for obs in observations]
        if not (pixels[0][0] > pixels[0][1] + 80 and pixels[0][0] > pixels[0][2] + 80):
            raise AssertionError(f"first decoded frame is not red: {pixels[0]}")
        if min(pixels[1]) < 240:
            raise AssertionError(f"middle string image is not white: {pixels[1]}")
        if not (pixels[2][2] > pixels[2][0] + 80 and pixels[2][2] > pixels[2][1] + 80):
            raise AssertionError(f"last decoded frame is not blue: {pixels[2]}")
        print(
            json.dumps(
                {"status": "passed", "observations": len(observations), "pixels": pixels}
            ),
            flush=True,
        )


def train() -> None:
    """Delegate to the existing launcher with smoke-only observability hooks."""
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    # B30Z reports compute capability 10.3, while this torch 2.10.0+cu128
    # build's runtime reduction compiler only advertises sm_100/sm_120. The
    # upstream Qwen3.5 implementation uses a three-element CUDA prod solely to
    # obtain image split sizes, which consequently asks NVRTC for unsupported
    # compute_103. Keep the exact upstream method, replacing that one reduction
    # with equivalent elementwise multiplication (precompiled CUDA kernels).
    if torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3):
        def get_image_features_b30z(
            self,
            pixel_values,
            image_grid_thw=None,
            **kwargs,
        ):
            kwargs.pop("return_dict", None)
            pixel_values = pixel_values.type(self.visual.dtype)
            vision_output = self.visual(
                pixel_values,
                grid_thw=image_grid_thw,
                return_dict=True,
                **kwargs,
            )
            image_embeds = vision_output.pooler_output
            grid_products = (
                image_grid_thw[:, 0]
                * image_grid_thw[:, 1]
                * image_grid_thw[:, 2]
            )
            split_sizes = (
                grid_products // self.visual.spatial_merge_size**2
            ).tolist()
            vision_output.pooler_output = torch.split(image_embeds, split_sizes)
            return vision_output

        Qwen3_5Model.get_image_features = get_image_features_b30z
        print(
            "[final-vqa-smoke] enabled B30Z CC 10.3 image-grid compatibility",
            flush=True,
        )

    from qwenvl.train.trainer import QwenVLTrainer

    original_compute_loss = QwenVLTrainer.compute_loss

    def compute_loss_with_batch_report(self, model, inputs, *args, **kwargs):
        if not getattr(self, "_final_vqa_smoke_batch_reported", False):
            fields: dict[str, Any] = {}
            for key, value in inputs.items():
                shape = getattr(value, "shape", None)
                if shape is not None:
                    fields[key] = list(shape)
                elif isinstance(value, (list, tuple)):
                    fields[key] = f"{type(value).__name__}[{len(value)}]"
                else:
                    fields[key] = type(value).__name__
            labels = inputs.get("labels")
            supervised = (
                int((labels != -100).sum().detach().cpu()) if labels is not None else 0
            )
            report = {
                "fields": fields,
                "supervised_tokens": supervised,
                "trainable_parameters": sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                ),
                "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            }
            print(
                "[final-vqa-smoke-batch] " + json.dumps(report, sort_keys=True),
                flush=True,
            )
            self._final_vqa_smoke_batch_reported = True
        return original_compute_loss(self, model, inputs, *args, **kwargs)

    QwenVLTrainer.compute_loss = compute_loss_with_batch_report

    if os.environ.get("FINAL_VQA_SMOKE_SKIP_MODEL_SAVE") == "1":
        def skip_model_save(self, output_dir=None, _internal_call=False):
            target = output_dir or self.args.output_dir
            print(
                f"[final-vqa-smoke] skipping final model weight save to {target}",
                flush=True,
            )

        QwenVLTrainer.save_model = skip_model_save

    from qwenvl.train.launcher import train as launcher_train

    launcher_train()


def verify(args: argparse.Namespace) -> None:
    state_path = args.output_dir.resolve() / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = json.loads(state_path.read_text())
    losses = [
        float(entry["loss"])
        for entry in state.get("log_history", [])
        if "loss" in entry
    ]
    grad_norms = [
        float(entry["grad_norm"])
        for entry in state.get("log_history", [])
        if "grad_norm" in entry
    ]
    if len(losses) < 10:
        raise RuntimeError(f"expected at least 10 logged losses, got {len(losses)}")
    if not all(math.isfinite(value) for value in losses + grad_norms):
        raise RuntimeError("loss or grad_norm contains NaN/Inf")

    first = statistics.median(losses[:5])
    last = statistics.median(losses[-5:])
    ratio = last / first if first else math.inf
    global_step = int(state.get("global_step", 0))
    passed = global_step >= args.expected_steps and ratio <= (1.0 - args.min_drop_fraction)
    batch_marker = "[final-vqa-smoke-batch]"
    batch_reported = args.log_file.is_file() and batch_marker in args.log_file.read_text(
        encoding="utf-8", errors="replace"
    )
    result = {
        "passed": passed and batch_reported,
        "global_step": global_step,
        "logged_loss_count": len(losses),
        "first_5_loss_median": first,
        "last_5_loss_median": last,
        "last_to_first_ratio": ratio,
        "required_drop_fraction": args.min_drop_fraction,
        "batch_reported": batch_reported,
        "losses": losses,
        "grad_norms": grad_norms,
    }
    verification_path = args.output_dir.parent / "verification.json"
    verification_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("self-test")

    prep = subparsers.add_parser("prepare")
    prep.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    prep.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    prep.add_argument("--work-dir", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=20260804)
    prep.add_argument("--per-scenario", type=int, default=2)
    prep.add_argument("--max-scan", type=int, default=512)
    prep.add_argument("--max-length", type=int, default=4096)

    check = subparsers.add_parser("verify")
    check.add_argument("--output-dir", type=Path, required=True)
    check.add_argument("--log-file", type=Path, required=True)
    check.add_argument("--expected-steps", type=int, default=30)
    check.add_argument("--min-drop-fraction", type=float, default=0.5)
    return parser


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        del sys.argv[1]
        train()
        return
    args = build_parser().parse_args()
    if args.mode == "self-test":
        self_test(args)
    elif args.mode == "prepare":
        prepare(args)
    elif args.mode == "verify":
        verify(args)
    else:  # pragma: no cover - argparse prevents this
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
