# Copyright (c) 2026
"""Pluggable vision tower: encoder + per-patch projector + factory (model side).

Swaps ``model.model.visual`` for a foreign encoder (DINOv3) + a per-patch MLP
projector while keeping the LM, MRoPE, packing and the x2robot_dataset_v2 image path
untouched. Bundles three tightly-coupled pieces:

* ``patchify_image`` / ``unpatchify_image`` -- lossless image <-> flat-patch conversion
  (Qwen's flattened-patch ``pixel_values`` contract, so different-H/W images ``cat``);
* projectors (only ``per_patch_mlp`` shipped) -- encoder features -> LM hidden;
* :class:`PluggableVisualTower` + ``build_vision_backbone*`` -- the ``.visual`` drop-in
  and its factory (``qwen`` is intentionally absent -- that path keeps the stock tower).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPooling


# ----------------------------------------------------------------------
# Image <-> flat-patch conversion (strict inverses; unit-tested)
# ----------------------------------------------------------------------
# Each image -> a [gh*gw, 3*patch*patch] raster block so per-image tensors of
# different H/W still torch.cat(dim=0) cleanly in the epilogue (native aspect ratio);
# the tower inverts the exact same rearrange to recover [1, 3, H, W].


def patchify_image(img: torch.Tensor, patch: int) -> "tuple[torch.Tensor, tuple[int, int]]":
    """``[C, H, W]`` (H,W multiples of ``patch``) -> ``([gh*gw, C*patch*patch], (gh, gw))``."""
    c, h, w = img.shape
    if h % patch or w % patch:
        raise ValueError(f"patchify: H,W ({h},{w}) must be multiples of patch {patch}")
    gh, gw = h // patch, w // patch
    x = img.reshape(c, gh, patch, gw, patch)  # [C, gh, p, gw, p]
    x = x.permute(1, 3, 0, 2, 4).contiguous()  # [gh, gw, C, p, p]
    return x.reshape(gh * gw, c * patch * patch), (gh, gw)


def unpatchify_image(patches: torch.Tensor, gh: int, gw: int, patch: int, channels: int = 3) -> torch.Tensor:
    """Inverse of :func:`patchify_image`: ``[gh*gw, C*patch*patch]`` -> ``[1, C, H, W]``."""
    x = patches.reshape(gh, gw, channels, patch, patch)  # [gh, gw, C, p, p]
    x = x.permute(2, 0, 3, 1, 4).contiguous()  # [C, gh, p, gw, p]
    return x.reshape(channels, gh * patch, gw * patch).unsqueeze(0)  # [1, C, H, W]


# ----------------------------------------------------------------------
# Projectors (vision-encoder features -> LM hidden)
# ----------------------------------------------------------------------
# Only ``per_patch_mlp`` is shipped (one LM token per encoder patch, no spatial merge
# -- LM sequence length is governed purely by input resolution). Behind a tiny registry
# so richer projectors (pixel-shuffle, perceiver-resampler) can be added later.

PROJECTOR_REGISTRY: "dict[str, type[nn.Module]]" = {}


def register_projector(name: str):
    def deco(cls):
        PROJECTOR_REGISTRY[name] = cls
        return cls
    return deco


@register_projector("per_patch_mlp")
class PerPatchMLPProjector(nn.Module):
    """Plain 2-layer MLP applied independently to every encoder patch token.

    ``[N, enc_hidden] -> [N, lm_hidden]``.  An input LayerNorm stabilizes the
    (self-supervised, non-language-aligned) encoder features before the LM sees them.
    """

    def __init__(self, enc_hidden: int, lm_hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(enc_hidden, eps=1e-6)
        self.fc1 = nn.Linear(enc_hidden, lm_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(lm_hidden, lm_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x))))


def build_projector(name: str, enc_hidden: int, lm_hidden: int) -> nn.Module:
    if name not in PROJECTOR_REGISTRY:
        raise ValueError(
            f"unknown projector_type {name!r}; available: {sorted(PROJECTOR_REGISTRY)}"
        )
    return PROJECTOR_REGISTRY[name](enc_hidden, lm_hidden)


# ----------------------------------------------------------------------
# PluggableVisualTower -- drop-in replacement for model.model.visual
# ----------------------------------------------------------------------
# Exposes exactly the contract Qwen3_5Model.get_image_features relies on: ``.dtype``,
# ``.spatial_merge_size`` (== 1 here) and ``forward(pixel_values, grid_thw) ->
# BaseModelOutputWithPooling`` whose pooler_output is a flat [sum_tokens, lm_hidden].
# Native-AR + dense encoders -> forward PER IMAGE; a video grid [t, gh, gw] is encoded
# PER FRAME (2D encoder, no temporal fusion; the LM gets timestamps via MRoPE + text).


class PluggableVisualTower(nn.Module):
    def __init__(self, encoder, encoder_kind: str, lm_hidden: int, projector_type: str = "per_patch_mlp"):
        super().__init__()
        self.encoder = encoder
        self.encoder_kind = encoder_kind
        self.spatial_merge_size = 1  # per-patch, no merge

        enc_cfg = encoder.config
        vision_cfg = getattr(enc_cfg, "vision_config", enc_cfg)
        self.patch_size = int(getattr(vision_cfg, "patch_size", 16))
        enc_hidden = int(getattr(vision_cfg, "hidden_size"))
        # DINOv3 prepends 1 CLS + N register tokens; strip them to keep only patches.
        self.num_prefix_tokens = 1 + int(getattr(vision_cfg, "num_register_tokens", 0))

        # Named ``merger`` so the trainer's optimizer buckets it as the projector group.
        self.merger = build_projector(projector_type, enc_hidden, lm_hidden)
        self.lm_hidden = int(lm_hidden)  # projector-agnostic empty-batch width

    @property
    def dtype(self) -> torch.dtype:
        return next(self.encoder.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.encoder.parameters()).device

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(**kwargs)

    def _encode_patches(self, imgs: torch.Tensor) -> torch.Tensor:
        """``[B, 3, H, W]`` -> patch features ``[B*gh*gw, enc_hidden]`` (prefix stripped)."""
        out = self.encoder(pixel_values=imgs)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        patches = hidden[:, self.num_prefix_tokens :, :]
        return patches.reshape(-1, patches.shape[-1])

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor = None, **kwargs):
        if grid_thw is None:
            raise ValueError(
                "PluggableVisualTower.forward requires grid_thw ([N, 3] t/h/w per image)."
            )
        pixel_values = pixel_values.to(self.dtype)
        p = self.patch_size
        embeds = []
        offset = 0
        for t, gh, gw in grid_thw.tolist():
            n = t * gh * gw
            block = pixel_values[offset : offset + n]  # [t*gh*gw, 3*p*p]
            offset += n
            per_frame = gh * gw
            # t == 1: one image. t > 1: a video encoded PER FRAME (the 2D
            # encoder has no temporal path); frames share a size, so they run
            # as one encoder batch. Token order stays frame-major, matching the
            # per-frame pad blocks in the text.
            imgs = torch.cat(
                [
                    unpatchify_image(block[f * per_frame : (f + 1) * per_frame], gh, gw, p)
                    for f in range(t)
                ],
                dim=0,
            ).to(self.dtype)  # [t, 3, H, W]
            feats = self._encode_patches(imgs)  # [t*gh*gw, enc_hidden]
            embeds.append(self.merger(feats))  # [t*gh*gw, lm_hidden]

        flat = torch.cat(embeds, dim=0) if embeds else pixel_values.new_zeros((0, self.lm_hidden))
        return BaseModelOutputWithPooling(last_hidden_state=flat, pooler_output=flat)


# ----------------------------------------------------------------------
# Factory: build a PluggableVisualTower for a named backbone
# ----------------------------------------------------------------------
# ``qwen`` is intentionally NOT here -- that path keeps the stock Qwen tower untouched.

# name -> encoder_kind passed to the tower (currently 1:1, room for aliases/quirks).
BACKBONES = {
    "dinov3": "dinov3",
    # "vjepa2": "vjepa2",  # follow-up: needs get_position_ids grid patch + tubelet handling
}


def _load_encoder(ckpt: str, dtype):
    from transformers import AutoModel

    return AutoModel.from_pretrained(ckpt, torch_dtype=dtype)


def _encoder_from_config(encoder_config: dict, dtype):
    from transformers import AutoConfig, AutoModel

    params = {k: v for k, v in encoder_config.items() if k != "model_type"}
    cfg = AutoConfig.for_model(encoder_config["model_type"], **params)
    return AutoModel.from_config(cfg).to(dtype)


def build_vision_backbone(
    name: str,
    ckpt: str,
    lm_hidden: int,
    projector_type: str = "per_patch_mlp",
    dtype=torch.bfloat16,
) -> PluggableVisualTower:
    """Load the pretrained encoder + fresh projector, wrapped as a ``.visual`` tower."""
    if name not in BACKBONES:
        raise ValueError(f"unknown vision_backbone {name!r}; available: {sorted(BACKBONES)}")
    encoder = _load_encoder(ckpt, dtype)
    return PluggableVisualTower(encoder, BACKBONES[name], lm_hidden, projector_type).to(dtype)


def build_vision_backbone_from_config(
    name: str,
    encoder_config: dict,
    lm_hidden: int,
    projector_type: str = "per_patch_mlp",
    dtype=torch.bfloat16,
) -> PluggableVisualTower:
    """Same tower, but the encoder is built from a saved config (weights loaded later)."""
    if name not in BACKBONES:
        raise ValueError(f"unknown vision_backbone {name!r}; available: {sorted(BACKBONES)}")
    encoder = _encoder_from_config(encoder_config, dtype)
    return PluggableVisualTower(encoder, BACKBONES[name], lm_hidden, projector_type).to(dtype)
