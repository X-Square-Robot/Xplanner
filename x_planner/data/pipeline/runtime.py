"""reference deployment runtime registration, noisy-memory curriculum and checkpoint hooks."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

from . import epilogue as _v10_epilogue  # noqa: F401

# Importing the proven smoke adapter registers video_frame and installs the two
# small compatibility utilities missing from the paired reference deployment dataset checkout.
from x_planner.data.video_frames import iter_multimodal_image_refs  # noqa: F401
from . import vision as _v10_vision  # noqa: F401
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

from .memory import MemoryAugmentor, MemoryCodec
from .models import V10Sample
from .prompt import render_user_text
from .schema import dumps_target


STEP_STATE_ENV = "V10_STEP_STATE_PATH"
MANIFEST_PATH_ENV = "V10_MANIFEST_PATH"
MANIFEST_DIGEST_ENV = "V10_MANIFEST_DIGEST"
DATA_CONFIG_DIGEST_ENV = "V10_DATA_CONFIG_DIGEST"
RESUME_MODE_ENV = "V10_RESUME_MODE"
SKIP_FINAL_SAVE_ENV = "XPLANNER_SKIP_FINAL_MODEL_SAVE"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_step_state(path: str | Path, global_step: int) -> None:
    probability = min(0.4, 0.4 * max(0, global_step) / 60.0)
    _atomic_json(Path(path), {
        "global_step": int(global_step),
        "memory_noise_probability": probability,
        "updated_at_unix": time.time(),
    })


def read_step(path: str | Path) -> int:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return int(value.get("global_step", 0))
    except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def _image_token_id_from_model(model: Any) -> int | None:
    """Find the HF image token through DDP/DeepSpeed wrapper layers."""
    candidate = model
    seen: set[int] = set()
    for _ in range(8):
        if candidate is None or id(candidate) in seen:
            break
        seen.add(id(candidate))
        config = getattr(candidate, "config", None)
        token_id = getattr(config, "image_token_id", None)
        if token_id is None and config is not None:
            token_id = getattr(getattr(config, "text_config", None), "image_token_id", None)
        if token_id is not None:
            return int(token_id)
        candidate = getattr(candidate, "module", None)
    return None


@register_text_processor("v10_video_frame_qwen3_5")
class V10VideoFrameQwen35TextProcessor(MultimodalJsonlQwen3_5TextProcessor):
    """Rebuild the V10 dialogue at runtime, applying noise to Long before Short."""

    def __init__(
        self,
        *args: Any,
        step_state_path: str | None = None,
        enable_memory_noise: bool = True,
        memory_seed: int = 42,
        short_memory_k: int = 1,
        visible_long_memory_limit: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.step_state_path = step_state_path or os.environ.get(STEP_STATE_ENV, "")
        self.enable_memory_noise = bool(enable_memory_noise)
        self.codec = MemoryCodec(
            short_memory_k=short_memory_k,
            visible_long_memory_limit=visible_long_memory_limit,
        )
        self.augmentor = MemoryAugmentor(self.codec, seed=memory_seed)
        self._noise_reports = 0

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        episode = data.get("_episode")
        if episode is None:
            return data
        frame_local_idx = data.get("_frame_local_idx", 0)
        sample_idx = _resolve_multimodal_sample_idx(episode, frame_local_idx)
        item = load_indexed_jsonl_item(episode.path, sample_idx)
        raw_sample = item.get("v10_sample")
        if not isinstance(raw_sample, dict):
            raise ValueError("V10 row is missing v10_sample")
        sample = V10Sample.from_dict(raw_sample)
        kept_positions = data.get("_v10_kept_image_positions")
        if kept_positions is not None:
            if not isinstance(kept_positions, (list, tuple)):
                raise TypeError("_v10_kept_image_positions must be a sequence")
            positions = tuple(int(position) for position in kept_positions)
            if not positions:
                raise ValueError("V10 requires at least one retained real image")
            if len(set(positions)) != len(positions):
                raise ValueError("V10 retained image positions contain duplicates")
            if min(positions) < 0 or max(positions) >= len(sample.images):
                raise IndexError(
                    f"V10 retained image positions out of range: {positions} / {len(sample.images)}"
                )
            sample = replace(
                sample,
                images=tuple(sample.images[position] for position in positions),
            )
        long_memory = sample.long_memory
        short_memory = self.codec.short_from_long(long_memory)
        operation = "gt"
        if self.enable_memory_noise and data.get("_is_train", True):
            step = read_step(self.step_state_path) if self.step_state_path else 0
            augmented = self.augmentor.augment(
                long_memory, sample_id=sample.sample_id, global_step=step
            )
            long_memory = augmented.long_memory
            short_memory = augmented.short_memory
            operation = augmented.operation
            if augmented.applied and (self._noise_reports < 3 or self._noise_reports % 1000 == 0):
                print(
                    f"[v10-memory-noise] sample={sample.sample_id} step={step} "
                    f"p={augmented.probability:.4f} operation={operation}",
                    flush=True,
                )
                self._noise_reports += 1

        dialogues = [
            {
                "role": "user",
                "text": render_user_text(
                    sample,
                    codec=self.codec,
                    long_memory=long_memory,
                    short_memory=short_memory,
                ),
            },
            {
                "role": "assistant",
                "text": dumps_target(sample.target, sample.profile),
            },
        ]
        num_images = len(sample.images)
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
        if image_token_count != num_images:
            raise ValueError(
                f"V10 image placeholder mismatch after processing: {image_token_count} != {num_images}"
            )
        data["_expected_image_count"] = image_token_count
        return data


class V10CheckpointCallback(TrainerCallback):
    def __init__(self, step_state_path: str) -> None:
        self.step_state_path = step_state_path

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            write_step_state(self.step_state_path, int(state.global_step))

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            write_step_state(self.step_state_path, int(state.global_step))

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        metadata = {
            "schema_version": "v10_checkpoint_meta_v1",
            "global_step": int(state.global_step),
            "epoch": state.epoch,
            "manifest_path": os.environ.get(MANIFEST_PATH_ENV, ""),
            "manifest_digest": os.environ.get(MANIFEST_DIGEST_ENV, ""),
            "data_config_digest": os.environ.get(DATA_CONFIG_DIGEST_ENV, ""),
            "resume_mode": os.environ.get(RESUME_MODE_ENV, "exact"),
            "memory_noise_probability": min(0.4, 0.4 * int(state.global_step) / 60.0),
            "training_args": args.to_dict(),
        }
        _atomic_json(checkpoint / "v10_checkpoint_meta.json", metadata)
        write_step_state(self.step_state_path, int(state.global_step))


_PATCHED = False


def apply_trainer_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from x_planner.trainer.trainer import QwenVLTrainer

    original_init = QwenVLTrainer.__init__
    original_compute_loss = QwenVLTrainer.compute_loss
    original_save_model = QwenVLTrainer.save_model

    def init_with_v10(self, *args: Any, **kwargs: Any):
        original_init(self, *args, **kwargs)
        step_path = os.environ.get(STEP_STATE_ENV)
        if not step_path:
            raise RuntimeError(f"{STEP_STATE_ENV} is required for V10 training")
        self.add_callback(V10CheckpointCallback(step_path))
        model = kwargs.get("model") or (args[0] if args else None)
        self._v10_visual_calls = 0
        self._v10_batch_calls = 0
        self._v10_bad_grad_reports = 0
        if model is not None:
            visual = getattr(getattr(model, "model", None), "visual", None)
            if visual is not None:
                def _visual_hook(_module, _inputs, output):
                    self._v10_visual_calls += 1
                    value = getattr(output, "pooler_output", None)
                    if isinstance(value, torch.Tensor):
                        finite = bool(torch.isfinite(value).all().detach().cpu())
                        audit_limit = int(os.environ.get("V10_VISUAL_AUDIT_LIMIT", "2"))
                        if self._v10_visual_calls <= audit_limit or not finite:
                            print("[v10-visual-forward] " + json.dumps({
                                "calls": self._v10_visual_calls,
                                "local_rank": int(os.environ.get("LOCAL_RANK", "-1")),
                                "shape": list(value.shape),
                                "finite": finite,
                                "mean_abs": float(value.detach().float().abs().mean().cpu()),
                            }, sort_keys=True), flush=True)
                        if not finite:
                            raise FloatingPointError(
                                f"V10 visual output is not finite at call {self._v10_visual_calls}"
                            )
                visual.register_forward_hook(_visual_hook)
            if os.environ.get("V10_GRAD_AUDIT") == "1":
                for parameter_name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    def _grad_hook(gradient, name=parameter_name):
                        if gradient is not None and not bool(torch.isfinite(gradient).all()):
                            if self._v10_bad_grad_reports < 32:
                                finite = gradient.detach()[torch.isfinite(gradient)]
                                print("[v10-nonfinite-gradient] " + json.dumps({
                                    "name": name,
                                    "shape": list(gradient.shape),
                                    "nan": int(torch.isnan(gradient).sum().cpu()),
                                    "posinf": int(torch.isposinf(gradient).sum().cpu()),
                                    "neginf": int(torch.isneginf(gradient).sum().cpu()),
                                    "finite_max_abs": float(finite.float().abs().max().cpu()) if finite.numel() else None,
                                }, sort_keys=True), flush=True)
                            self._v10_bad_grad_reports += 1
                        return gradient
                    parameter.register_hook(_grad_hook)

    def compute_loss_with_v10_assertions(self, model, inputs, *args: Any, **kwargs: Any):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        if input_ids is None or attention_mask is None or labels is None:
            shape_report = {
                key: list(value.shape) if hasattr(value, "shape") else type(value).__name__
                for key, value in inputs.items()
            }
            raise ValueError(
                "V10 batch must contain input_ids, attention_mask and labels; "
                f"received={json.dumps(shape_report, sort_keys=True)}"
            )
        if input_ids.shape != attention_mask.shape or input_ids.shape != labels.shape:
            raise ValueError(
                f"V10 tensor shape mismatch: ids={tuple(input_ids.shape)} "
                f"attention={tuple(attention_mask.shape)} labels={tuple(labels.shape)}"
            )
        supervised = int((labels != -100).sum().detach().cpu())
        if supervised <= 0:
            raise ValueError("V10 batch has no Assistant JSON supervision tokens")
        if bool((labels[attention_mask == 0] != -100).any().detach().cpu()):
            raise ValueError("V10 padding token participates in loss")
        self._v10_batch_calls += 1
        batch_audit_limit = int(os.environ.get("V10_BATCH_AUDIT_LIMIT", "0"))
        if self._v10_batch_calls <= batch_audit_limit:
            pixels = inputs.get("pixel_values")
            grid = inputs.get("image_grid_thw")
            batch_report = {
                "calls": self._v10_batch_calls,
                "local_rank": int(os.environ.get("LOCAL_RANK", "-1")),
                "input_shape": list(input_ids.shape),
                "supervised_tokens": supervised,
                "pixel_values_shape": list(pixels.shape)
                if isinstance(pixels, torch.Tensor) else None,
                "pixels_finite": bool(torch.isfinite(pixels).all().detach().cpu())
                if isinstance(pixels, torch.Tensor) else None,
                "pixels_mean": float(pixels.detach().float().mean().cpu())
                if isinstance(pixels, torch.Tensor) else None,
                "pixels_std": float(pixels.detach().float().std().cpu())
                if isinstance(pixels, torch.Tensor) else None,
                "image_grid_thw": grid.detach().cpu().tolist()
                if isinstance(grid, torch.Tensor) else None,
            }
            print(
                "[v10-batch] " + json.dumps(batch_report, sort_keys=True),
                flush=True,
            )
        if not getattr(self, "_v10_first_batch_reported", False):
            supervised_mask = labels != -100
            supervised_positions = supervised_mask[0].nonzero(as_tuple=False).flatten()
            spans = 0
            previous = -2
            for position in supervised_positions.detach().cpu().tolist():
                if position != previous + 1:
                    spans += 1
                previous = position
            pixels = inputs.get("pixel_values")
            grid = inputs.get("image_grid_thw")
            image_token_id = _image_token_id_from_model(model)
            report = {
                "input_shape": list(input_ids.shape),
                "attention_shape": list(attention_mask.shape),
                "label_shape": list(labels.shape),
                "supervised_tokens": supervised,
                "shifted_supervised_tokens": int((labels[:, 1:] != -100).sum().detach().cpu()),
                "masked_tokens": int((labels == -100).sum().detach().cpu()),
                "pixel_values_shape": list(inputs["pixel_values"].shape)
                if "pixel_values" in inputs else None,
                "attention_ones": int(attention_mask.sum().detach().cpu()),
                "padding_tokens": int((attention_mask == 0).sum().detach().cpu()),
                "supervision_spans": spans,
                "supervised_first": int(supervised_positions[0].detach().cpu()),
                "supervised_last": int(supervised_positions[-1].detach().cpu()),
                "labels_match_input": bool(torch.equal(labels[supervised_mask], input_ids[supervised_mask])),
                "image_token_id": image_token_id,
                "image_token_count": int((input_ids == image_token_id).sum().detach().cpu()) if image_token_id is not None else None,
                "image_grid_thw": grid.detach().cpu().tolist() if isinstance(grid, torch.Tensor) else None,
                "pixels_finite": bool(torch.isfinite(pixels).all().detach().cpu()) if isinstance(pixels, torch.Tensor) else None,
                "pixels_mean": float(pixels.detach().float().mean().cpu()) if isinstance(pixels, torch.Tensor) else None,
                "pixels_std": float(pixels.detach().float().std().cpu()) if isinstance(pixels, torch.Tensor) else None,
            }
            print("[v10-first-batch] " + json.dumps(report, sort_keys=True), flush=True)
            self._v10_first_batch_reported = True
        result = original_compute_loss(self, model, inputs, *args, **kwargs)
        loss = result[0] if isinstance(result, tuple) else result
        if not bool(torch.isfinite(loss).all().detach().cpu()):
            raise FloatingPointError(f"V10 loss is not finite: {loss}")
        if (
            os.environ.get("V10_IMAGE_ABLATION") == "1"
            and not getattr(self, "_v10_image_ablation_reported", False)
            and isinstance(inputs.get("pixel_values"), torch.Tensor)
        ):
            ablated_inputs = dict(inputs)
            ablated_inputs["labels"] = labels
            ablated_inputs["pixel_values"] = torch.zeros_like(inputs["pixel_values"])
            with torch.no_grad():
                ablated_result = original_compute_loss(
                    self, model, ablated_inputs, *args, **kwargs
                )
            ablated_loss = ablated_result[0] if isinstance(ablated_result, tuple) else ablated_result
            print("[v10-image-ablation] " + json.dumps({
                "real_loss": float(loss.detach().float().cpu()),
                "zero_image_loss": float(ablated_loss.detach().float().cpu()),
                "absolute_delta": float((loss.detach() - ablated_loss.detach()).float().abs().cpu()),
                "visual_forward_calls": self._v10_visual_calls,
            }, sort_keys=True), flush=True)
            self._v10_image_ablation_reported = True
        return result

    def save_model_with_smoke_guard(self, output_dir=None, _internal_call=False):
        if os.environ.get(SKIP_FINAL_SAVE_ENV) == "1" and not _internal_call:
            print(
                f"[v10-train] skipping redundant final model export to {output_dir or self.args.output_dir}",
                flush=True,
            )
            return None
        return original_save_model(self, output_dir, _internal_call)

    QwenVLTrainer.__init__ = init_with_v10
    QwenVLTrainer.compute_loss = compute_loss_with_v10_assertions
    QwenVLTrainer.save_model = save_model_with_smoke_guard
    _PATCHED = True


def _needs_image_grid_compatibility(
    torch_version: str, capability: tuple[int, int]
) -> bool:
    """Return whether Qwen3.5's CUDA int64 ``prod`` path must be avoided.

    B30Z (CC 10.3) needed the original workaround.  another deployment's torch
    2.10.0+cu128 build also raises ``CUDA driver error: invalid argument`` for
    a minimal CUDA int64 reduction on A800 (CC 8.0), while the equivalent
    element-wise multiplication is healthy.  Keep the workaround narrowly
    version/capability gated instead of changing model math globally.
    """
    normalized_version = torch_version.split("+", 1)[0]
    return capability == (10, 3) or normalized_version == "2.10.0"


def apply_b30z_compatibility() -> None:
    if not torch.cuda.is_available():
        return
    capability = torch.cuda.get_device_capability()
    if not _needs_image_grid_compatibility(torch.__version__, capability):
        return
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    def get_image_features_compat(self, pixel_values, image_grid_thw=None, **kwargs):
        kwargs.pop("return_dict", None)
        pixel_values = pixel_values.type(self.visual.dtype)
        vision_output = self.visual(
            pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs
        )
        image_embeds = vision_output.pooler_output
        grid_products = image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
        split_sizes = (grid_products // self.visual.spatial_merge_size**2).tolist()
        vision_output.pooler_output = torch.split(image_embeds, split_sizes)
        return vision_output

    Qwen3_5Model.get_image_features = get_image_features_compat
    print(
        "[v10-train] enabled image-grid compatibility "
        f"for torch={torch.__version__} capability={capability}",
        flush=True,
    )
