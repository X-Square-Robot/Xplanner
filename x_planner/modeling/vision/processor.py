# Copyright (c) 2026
"""Custom image processor for the pluggable vision path (data side).

Mirrors Qwen's ``(pixel_values, image_grid_thw)`` output contract so the Qwen3.5
processor + dataset epilogue stay untouched, but:

* normalizes with the **encoder's own** mean/std (pulled from its HF image processor),
* rounds H/W to a multiple of the encoder ``patch_size`` **preserving aspect ratio**
  (native-AR; factor-32 inputs from the dataset vision processor pass through unchanged),
* emits **flattened patches** ``[sum_patches, 3*patch*patch]`` + ``grid_thw=[1,gh,gw]``
  with ``merge_size = 1`` -- so ``Qwen3VLProcessor`` expands each ``<|image_pad|>`` to
  ``gh*gw`` tokens and the tower produces exactly that many embeds (per-patch, no merge).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature

from x_planner.modeling.vision.tower import patchify_image

# ImageNet defaults (DINOv3 uses these) -- overridden from the encoder's own processor.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _round_to_patch(h: int, w: int, patch: int, max_pixels: Optional[int]) -> "tuple[int, int]":
    """Nearest ``(h, w)`` that are multiples of ``patch`` (AR-preserving), optional cap."""
    rh = max(patch, round(h / patch) * patch)
    rw = max(patch, round(w / patch) * patch)
    if max_pixels is not None and rh * rw > max_pixels:
        scale = math.sqrt(max_pixels / (h * w))
        # Floor (not round) when capping: nearest-rounding both dims can land
        # above max_pixels (e.g. 40x1000 @ cap 20000 -> 32x704 = +12.6%), and
        # the cap exists as an OOM/row-length guard.
        rh = max(patch, int(h * scale / patch) * patch)
        rw = max(patch, int(w * scale / patch) * patch)
    return rh, rw


class PluggableImageProcessor(BaseImageProcessor):
    """Encoder-native normalize + patch-16 flatten -> ``pixel_values`` / ``image_grid_thw``."""

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        patch_size: int = 16,
        image_mean: Optional[list] = None,
        image_std: Optional[list] = None,
        rescale_factor: float = 1.0 / 255.0,
        max_pixels: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.image_mean = list(image_mean) if image_mean is not None else _IMAGENET_MEAN
        self.image_std = list(image_std) if image_std is not None else _IMAGENET_STD
        self.rescale_factor = rescale_factor
        self.max_pixels = max_pixels
        # Consumed by Qwen3VLProcessor to expand <|image_pad|>: gh*gw // merge_size**2.
        self.merge_size = 1

    def _to_chw_float(self, image) -> torch.Tensor:
        """Any (PIL / ndarray / tensor) RGB image -> float ``[3, H, W]`` in [0, 255]."""
        if isinstance(image, torch.Tensor):
            t = image.detach().float()
            if t.ndim == 3 and t.shape[0] not in (1, 3):  # HWC -> CHW
                t = t.permute(2, 0, 1)
        else:
            arr = np.asarray(image.convert("RGB") if hasattr(image, "convert") else image)
            t = torch.from_numpy(np.array(arr, dtype=np.float32))
            if t.ndim == 2:
                t = t.unsqueeze(-1).repeat(1, 1, 3)
            t = t.permute(2, 0, 1)  # HWC -> CHW
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        return t

    def _preprocess_one(self, image) -> "tuple[torch.Tensor, tuple[int, int]]":
        img = self._to_chw_float(image)  # [3, H, W] in [0, 255]
        _, h, w = img.shape
        rh, rw = _round_to_patch(h, w, self.patch_size, self.max_pixels)
        if (rh, rw) != (h, w):
            img = F.interpolate(
                img.unsqueeze(0), size=(rh, rw), mode="bicubic", align_corners=False
            ).squeeze(0)
        img = img * self.rescale_factor
        mean = torch.tensor(self.image_mean, dtype=img.dtype).view(3, 1, 1)
        std = torch.tensor(self.image_std, dtype=img.dtype).view(3, 1, 1)
        img = (img - mean) / std
        return patchify_image(img, self.patch_size)  # ([gh*gw, 3*p*p], (gh, gw))

    def preprocess(self, images, return_tensors=None, **kwargs) -> BatchFeature:
        if images is None:
            return BatchFeature(data={}, tensor_type=return_tensors)
        if not isinstance(images, (list, tuple)):
            images = [images]

        patch_blocks, grids = [], []
        for image in images:
            patches, (gh, gw) = self._preprocess_one(image)
            patch_blocks.append(patches)
            grids.append([1, gh, gw])

        pixel_values = torch.cat(patch_blocks, dim=0)  # [sum_patches, 3*p*p]
        image_grid_thw = torch.tensor(grids, dtype=torch.long)  # [N, 3]
        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )

    def __call__(self, images=None, **kwargs) -> BatchFeature:
        return self.preprocess(images, **kwargs)


def build_pluggable_processor(qwen_processor, ckpt: str, max_pixels: Optional[int] = None):
    """Swap the loaded Qwen3.5 processor's ``image_processor`` for a pluggable one.

    Reuses the Qwen tokenizer + chat_template + video_processor untouched (so the
    epilogue's ``apply_chat_template`` / ``<|image_pad|>`` expansion behave identically),
    pulling normalization stats + patch size from the encoder's own HF image processor.
    Returns the same processor instance (mutated in place).
    """
    from transformers import AutoConfig, AutoImageProcessor

    try:
        enc_ip = AutoImageProcessor.from_pretrained(ckpt)
        image_mean = getattr(enc_ip, "image_mean", None)
        image_std = getattr(enc_ip, "image_std", None)
        rescale_factor = getattr(enc_ip, "rescale_factor", 1.0 / 255.0)
    except Exception:  # noqa: BLE001 -- some encoders ship no image processor
        image_mean = image_std = None
        rescale_factor = 1.0 / 255.0

    enc_cfg = AutoConfig.from_pretrained(ckpt)
    vision_cfg = getattr(enc_cfg, "vision_config", enc_cfg)
    patch_size = int(getattr(vision_cfg, "patch_size", 16))

    qwen_processor.image_processor = PluggableImageProcessor(
        patch_size=patch_size,
        image_mean=image_mean,
        image_std=image_std,
        rescale_factor=rescale_factor,
        max_pixels=max_pixels,
    )
    return qwen_processor
