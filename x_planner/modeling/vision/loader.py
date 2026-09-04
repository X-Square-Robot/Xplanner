# Copyright (c) 2026
"""Reload a checkpoint whose ``.visual`` was swapped for a pluggable tower.

Stock ``AutoModel.from_pretrained`` would rebuild the *Qwen* vision tower and silently
mismatch the saved ``model.visual.*`` weights.  ``load_pluggable_qwen35`` reads the
custom keys we stamped into ``config.json`` (see :func:`stamp_pluggable_config`),
rebuilds the tower architecture from the saved encoder config, swaps it in, then loads
the trained tower weights.  Used by fresh inference / lmms-eval; training-resume does
the swap itself before ``trainer.train``.
"""

from __future__ import annotations

import glob
import json
import os

import torch


def stamp_pluggable_config(model, name: str, ckpt: str, projector_type: str, max_pixels=None):
    """Record the pluggable-vision selection on ``model.config`` so it lands in config.json."""
    cfg = model.config
    cfg.vision_backbone = name
    cfg.vision_ckpt = ckpt
    cfg.vision_projector_type = projector_type
    cfg.vision_max_pixels = max_pixels
    enc = model.model.visual.encoder
    cfg.vision_encoder_config = enc.config.to_dict()


def _assemble_state_dict(path: str) -> dict:
    # Prefer the shard index (authoritative); else match model*.safetensors only --
    # merging every *.safetensors would let a stray adapter/partial file in the
    # dir silently win by sort order.
    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards = [os.path.join(path, s) for s in sorted(set(weight_map.values()))]
    else:
        shards = sorted(glob.glob(os.path.join(path, "model*.safetensors")))
    if shards:
        from safetensors.torch import load_file

        sd = {}
        for s in shards:
            sd.update(load_file(s))
        return sd
    bin_path = os.path.join(path, "pytorch_model.bin")
    if os.path.isfile(bin_path):
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"no model*.safetensors or pytorch_model.bin under {path}")


def load_pluggable_qwen35(path: str, dtype=torch.bfloat16, attn_implementation: str = "sdpa"):
    """Rebuild a saved pluggable-vision Qwen3.5-VL model (encoder + projector + LM)."""
    from transformers import AutoModelForImageTextToText

    from x_planner.modeling.vision.tower import build_vision_backbone_from_config

    with open(os.path.join(path, "config.json")) as f:
        cfg_json = json.load(f)
    name = cfg_json.get("vision_backbone")
    if not name or name == "qwen":
        # Not a pluggable checkpoint -- stock load.
        return AutoModelForImageTextToText.from_pretrained(
            path, torch_dtype=dtype, attn_implementation=attn_implementation
        )

    # ignore_mismatched_sizes: the checkpoint's model.visual.* are TOWER weights;
    # from_pretrained builds the stock Qwen visual first and would otherwise choke on
    # the name-colliding-but-differently-sized merger.norm. We overwrite the whole
    # visual with the rebuilt tower below and load its weights explicitly.
    model = AutoModelForImageTextToText.from_pretrained(
        path, torch_dtype=dtype, attn_implementation=attn_implementation,
        ignore_mismatched_sizes=True,
    )
    lm_hidden = model.get_input_embeddings().weight.shape[1]
    tower = build_vision_backbone_from_config(
        name,
        cfg_json["vision_encoder_config"],
        lm_hidden,
        cfg_json.get("vision_projector_type", "per_patch_mlp"),
        dtype=dtype,
    )
    model.model.visual = tower
    model.config.vision_config.spatial_merge_size = 1

    sd = _assemble_state_dict(path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    tower_missing = [k for k in missing if ".visual." in k]
    if tower_missing:
        raise RuntimeError(f"pluggable tower weights missing from checkpoint: {tower_missing[:8]}")
    return model


def load_pluggable_processor(path: str):
    """Rebuild the full processor for a (possibly pluggable) checkpoint.

    ``AutoProcessor.from_pretrained(ckpt)`` fails on a pluggable checkpoint:
    its ``preprocessor_config.json`` names ``PluggableImageProcessor``, which is
    not a transformers Auto class.  This helper reconstructs the image processor
    from the saved json and assembles the Qwen processor around it; for a stock
    (``vision_backbone: qwen`` / unstamped) checkpoint it defers to AutoProcessor.
    Pair with :func:`load_pluggable_qwen35` in inference / lmms-eval glue.
    """
    from transformers import AutoProcessor

    cfg_file = os.path.join(path, "config.json")
    name = None
    if os.path.isfile(cfg_file):
        with open(cfg_file) as f:
            name = json.load(f).get("vision_backbone")
    if not name or name == "qwen":
        return AutoProcessor.from_pretrained(path)

    from transformers import AutoTokenizer

    from x_planner.modeling.vision.processor import PluggableImageProcessor

    with open(os.path.join(path, "preprocessor_config.json")) as f:
        ip_cfg = json.load(f)
    image_processor = PluggableImageProcessor(
        **{
            k: ip_cfg[k]
            for k in ("patch_size", "image_mean", "image_std", "rescale_factor", "max_pixels")
            if k in ip_cfg
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    try:
        from transformers import AutoVideoProcessor

        video_processor = AutoVideoProcessor.from_pretrained(path)
    except Exception:  # noqa: BLE001 -- video processor is optional (image-only ckpt)
        video_processor = None

    try:
        from transformers.models.qwen3_5.processing_qwen3_5 import Qwen3_5Processor as _Proc
    except ImportError:  # older layout shared with qwen3_vl
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor as _Proc
    return _Proc(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=getattr(tokenizer, "chat_template", None),
    )
