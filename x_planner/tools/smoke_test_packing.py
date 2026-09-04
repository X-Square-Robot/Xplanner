#!/usr/bin/env python
# Copyright (c) 2026
"""GPU smoke test for Qwen3.5-VL neat-packing.

Two checks on the REAL model (needs a GPU):

1. **No cross-document leakage** (the correctness proof): pack N documents into one
   row and compare the packed logits, per document, against that document run
   *standalone* (packed alone). With the GDN patch + explicit FA2 cu_seqlens the
   two must agree (top-1 argmax ~100%, tiny logit delta). Run with ``--no-patch``
   to see the contrast: later documents diverge because the GatedDeltaNet
   recurrence leaks across boundaries.

2. **Forward + backward finiteness**: a packed training step produces a finite
   loss and finite gradients.

Usage:
    python -m x_planner.tools.smoke_test_packing --model_path /data/Models/Qwen3.5-9B
    python -m x_planner.tools.smoke_test_packing --model_path ... --no_patch     # contrast
    python -m x_planner.tools.smoke_test_packing --model_path ... --with_action  # +RVQ action doc

Requires flash-linear-attention>=0.4.1 + FlashAttention-2 and a CUDA device.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

RVQ_CKPT = os.environ.get(
    "RVQ_CKPT", "",
)
RVQ_CFG = os.environ.get(
    "RVQ_CFG", "",
)


def _img(h, w):
    return Image.fromarray(np.uint8(np.random.rand(h, w, 3) * 255))


def encode_doc(processor, messages, images):
    """Tokenize one chat sample into a packing 'doc' dict (mirrors the epilogue)."""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    inputs = processor(
        text=[text], images=images or None, padding=False, return_tensors="pt"
    )
    ids = inputs["input_ids"][0]
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    mm = torch.zeros_like(ids, dtype=torch.int)
    mm = mm.masked_fill(ids == image_id, 1)
    mm = mm.masked_fill(ids == video_id, 2)
    doc = {"input_ids": ids, "labels": ids.clone(), "mm_token_type_ids": mm}
    if inputs.get("pixel_values") is not None:
        doc["pixel_values"] = inputs["pixel_values"]
        doc["image_grid_thw"] = inputs["image_grid_thw"]
    return doc


def encode_video_doc(processor, vproc, video_path, question, answer):
    """Tokenize one VIDEO chat sample into a packing 'doc' dict.

    Mirrors ``Qwen3_5MultimodalEpilogueProcessor._encode_one_video``: decode +
    LF-cap frames (via the merged vision processor), patchify with
    ``do_sample_frames=False``, interleave per-temporal-token timestamps, then
    tokenize the materialized text.
    """
    class _Ep:
        path = "/tmp"

    videos, metas = vproc._process_videos(_Ep(), {"video": [video_path]})
    frames, meta = videos[0], metas[0]
    vp = processor.video_processor
    out = vp(videos=[frames], do_sample_frames=False,
             return_metadata=True, return_tensors="pt")
    pvv = out["pixel_values_videos"]
    grid = out["video_grid_thw"]
    T, H, W = (int(x) for x in grid[0].tolist())
    merge = vp.merge_size
    ts = processor._calculate_timestamps(
        meta["frames_indices"], meta["native_fps"], merge
    )
    seqlen = (H * W) // (merge * merge)
    block = "".join(
        f"<{float(ts[t]):.1f} seconds>"
        f"<|vision_start|>{'<|video_pad|>' * seqlen}<|vision_end|>"
        for t in range(T)
    )
    messages = [
        {"role": "user", "content": block + "\n" + question},
        {"role": "assistant", "content": answer},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    ids = processor.tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    )["input_ids"][0]
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    mm = torch.zeros_like(ids, dtype=torch.int)
    mm = mm.masked_fill(ids == image_id, 1)
    mm = mm.masked_fill(ids == video_id, 2)
    return {
        "input_ids": ids, "labels": ids.clone(), "mm_token_type_ids": mm,
        "pixel_values_videos": pvv, "video_grid_thw": grid,
    }


def to_device(batch, device, dtype):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if v.dtype.is_floating_point:
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def forward_logits(model, packed, device, dtype):
    inp = to_device(packed, device, dtype)
    inp_for_model = {k: v for k, v in inp.items() if k != "labels"}
    return model(**inp_for_model).logits  # [1, T, V]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="/data/Models/Qwen3.5-9B")
    ap.add_argument("--no_patch", action="store_true",
                    help="skip the GDN packing patch (expect leakage in later docs)")
    ap.add_argument("--with_action", action="store_true",
                    help="add an RVQ-action document (registers <rvq_*> tokens + resize)")
    ap.add_argument("--with_video", action="store_true",
                    help="add a real video document (vsi590k) via the epilogue video flow")
    ap.add_argument("--video_path",
                    default="/open_data/Multi-Modal-dataset/VSI-590K/scannet/scene0191_00.mp4",
                    help="mp4 used for the --with_video document")
    # A doc is leak-free if its packed-vs-standalone logits agree closely. bf16 +
    # shape-dependent flash/triton kernels flip a few low-confidence argmaxes even
    # with zero leakage (worse for image docs), so we require BOTH a high argmax
    # agreement AND a bounded max logit delta. Real leakage is far outside these
    # (argmax ~0.3-0.75, max|Δ| ~16-25) -- see --no_patch.
    ap.add_argument("--argmax_thresh", type=float, default=0.90)
    ap.add_argument("--maxdiff_thresh", type=float, default=8.0)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "this smoke test needs a CUDA device"
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)
    np.random.seed(0)  # reproducible docs so --patch vs --no_patch compare like-for-like

    from x_planner.data.packing import pack_sequences
    from x_planner.data.packing import apply_qwen3_5_packing_patch, make_packed_rope_index_fn

    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as _AutoModel
    except ImportError:
        from transformers import Qwen3_5ForConditionalGeneration as _AutoModel

    print(f"[load] {args.model_path} (bf16, flash_attention_2)", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = _AutoModel.from_pretrained(
        args.model_path, torch_dtype=dtype, attn_implementation="flash_attention_2",
    ).to(device)
    model.config.use_cache = False

    action_tokenizer = None
    if args.with_action:
        from x_planner.data.rvq_tokenizer import RVQActionTokenizer
        action_tokenizer = RVQActionTokenizer(
            checkpoint_path=RVQ_CKPT, config_dir=RVQ_CFG, device="cpu", rvq_version="v3_2",
        )
        n_added = processor.tokenizer.add_tokens(action_tokenizer.get_special_tokens())
        if n_added:
            model.resize_token_embeddings(len(processor.tokenizer))
        print(f"[rvq] +{n_added} tokens, vocab -> {len(processor.tokenizer)}", flush=True)

    if not args.no_patch:
        apply_qwen3_5_packing_patch()
        print("[patch] GDN/decoder packing patch applied", flush=True)
    else:
        print("[patch] SKIPPED (--no_patch): expect later-doc divergence", flush=True)

    rope = make_packed_rope_index_fn(model)

    # --- Build documents (text-only, image, text-only [, action]) ---
    docs = [
        encode_doc(processor, [
            {"role": "user", "content": "Name three primary colors."},
            {"role": "assistant", "content": "Red, green, and blue."},
        ], None),
        encode_doc(processor, [
            {"role": "user", "content": [
                {"type": "image", "image": _img(224, 320)},
                {"type": "text", "text": "What is in this image?"}]},
            {"role": "assistant", "content": "A randomly generated test pattern."},
        ], [_img(224, 320)]),
        encode_doc(processor, [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4."},
        ], None),
    ]
    if args.with_action and action_tokenizer is not None:
        aar = torch.randn(1, 32, action_tokenizer.action_dim) * 0.05
        dof = torch.ones(1, 32, action_tokenizer.action_dim)
        rvq_str = "".join(action_tokenizer.encode_to_tokens(aar, dof_mask=dof)[0])
        docs.append(encode_doc(processor, [
            {"role": "user", "content": [
                {"type": "image", "image": _img(224, 320)},
                {"type": "text", "text": "Predict the next action."}]},
            {"role": "assistant", "content": rvq_str},
        ], [_img(224, 320)]))

    if args.with_video:
        from x2robot_dataset_v2.processors.vision.multimodal_jsonl_vision_processor import (
            MultimodalJsonlVisionProcessor,
        )
        vproc = MultimodalJsonlVisionProcessor.from_config({
            "type": "multimodal_jsonl",
            "video_fps": 2, "video_maxlen": 128,
            "video_max_pixels": 256 * 256, "video_min_pixels": 16 * 16,
        })
        # Insert the video doc in the MIDDLE so a later text doc must survive the
        # GDN recurrence crossing the long video span (the strongest leakage test).
        docs.insert(2, encode_video_doc(
            processor, vproc, args.video_path,
            "What kind of room is shown in this video?", "A living room.",
        ))
        print(f"[video] added doc from {args.video_path}", flush=True)

    model.eval()
    packed_all = pack_sequences(docs, rope)
    # Non-dropping: every doc must be present (length-balanced packing relies on
    # the sampler, not pack_sequences, to bound the row -- nothing is dropped here).
    total_doc_len = sum(int(d["input_ids"].shape[0]) for d in docs)
    assert packed_all["input_ids"].shape[1] == total_doc_len, (
        f"pack_sequences dropped tokens: {packed_all['input_ids'].shape[1]} != {total_doc_len}"
    )
    cu = packed_all["cu_seq_lens_q"].tolist()
    logits_all = forward_logits(model, packed_all, device, dtype)
    print(f"\n[equiv] packed T={logits_all.shape[1]}, docs at cu={cu}\n", flush=True)

    ok = True
    for i, doc in enumerate(docs):
        ref = forward_logits(model, pack_sequences([doc], rope), device, dtype)
        seg = logits_all[:, cu[i]:cu[i + 1], :].float()
        ref = ref.float()
        agree = (seg.argmax(-1) == ref.argmax(-1)).float().mean().item()
        maxdiff = (seg - ref).abs().max().item()
        doc_ok = agree >= args.argmax_thresh and maxdiff <= args.maxdiff_thresh
        flag = "OK " if doc_ok else "BAD"
        has_img = "image_grid_thw" in doc
        has_vid = "video_grid_thw" in doc
        print(f"  [{flag}] doc{i} len={cu[i+1]-cu[i]:4d} img={int(has_img)} "
              f"vid={int(has_vid)} argmax_agree={agree:.4f} "
              f"max|Δlogit|={maxdiff:.3f}", flush=True)
        ok = ok and doc_ok

    print(f"\n[equiv] {'PASS — no cross-document leakage' if ok else 'FAIL — leakage detected'}",
          flush=True)

    # --- forward + backward finiteness on a packed training step ---
    model.train()
    # Gradient checkpointing keeps the backward within memory on shared GPUs.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    inp = to_device(packed_all, device, dtype)
    out = model(**inp)
    loss = out.loss
    assert loss is not None and torch.isfinite(loss), f"loss not finite: {loss}"
    loss.backward()
    gnorm = torch.sqrt(sum((p.grad.detach().float() ** 2).sum()
                           for p in model.parameters() if p.grad is not None))
    assert torch.isfinite(gnorm), f"grad norm not finite: {gnorm}"
    print(f"[bwd] loss={loss.item():.4f}  grad_norm={gnorm.item():.3f}  (finite)", flush=True)

    if not ok and not args.no_patch:
        sys.exit(1)
    print("\nSMOKE TEST DONE", flush=True)


if __name__ == "__main__":
    main()
