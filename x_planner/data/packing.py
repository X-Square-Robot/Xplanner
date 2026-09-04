# Copyright (c) 2026
"""Correct sequence packing (neat-packing) for Qwen3.5-VL.

Qwen3.5's text backbone is hybrid: 3/4 of the layers are linear-attention
(GatedDeltaNet). Naively packing several samples into one row leaks the conv
state + delta-rule recurrence across document boundaries in those layers, so the
standard attention-mask / block-diagonal packing trick (which only fixes the
full-attention layers) is NOT sufficient here.

Two boundaries must be enforced:

1. **Linear-attention layers** -- ``apply_qwen3_5_packing_patch()`` ports the
   LLaMA-Factory / Axolotl GPU patch: it threads per-document ``position_ids``
   into the GatedDeltaNet and switches it to the FLA *varlen* kernels
   (``causal_conv1d`` / ``chunk_gated_delta_rule`` with ``cu_seqlens`` derived
   from ``position_ids[0]``), so conv + recurrence reset at every document
   boundary.
2. **Full-attention layers** -- FlashAttention-2 needs the cumulative seqlens.
   The model passes the **3D MRoPE** ``position_ids`` ([3, B, L]) to FA2, and
   transformers' ``prepare_fa_kwargs_from_position_ids`` would flatten all three
   axes and mis-segment the row.  So the packer emits explicit
   ``cu_seq_lens_q/k`` + ``max_length_q/k`` (FlashAttentionKwargs); FA2 uses
   those directly and ignores the (broken) 3D derivation.

Usage:
    from x_planner.data.packing import apply_qwen3_5_packing_patch, make_packed_rope_index_fn
    apply_qwen3_5_packing_patch()                     # monkey-patch the model classes
    rope_index_fn = make_packed_rope_index_fn(model)  # picklable, config-only get_rope_index
    # build_dataset_v2 injects rope_index_fn AND pack_sequences (below) into the
    # dataset_v2 Qwen3.5 epilogue, which does the actual packing at collate time --
    # the library itself never imports this project.

Requires ``flash-linear-attention>=0.4.1`` and FlashAttention-2.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

_PATCHED = False


def apply_qwen3_5_packing_patch() -> None:
    """Monkey-patch Qwen3_5 decoder + GatedDeltaNet for cu_seqlens-correct packing.

    Idempotent. Raises ImportError if the FLA varlen kernels are unavailable.
    """
    global _PATCHED
    if _PATCHED:
        return

    try:
        from fla.modules.convolution import causal_conv1d as fla_causal_conv1d
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Qwen3.5 packing requires flash-linear-attention>=0.4.1 "
            "(fla.modules.convolution.causal_conv1d + fla.ops.gated_delta_rule.chunk_gated_delta_rule)."
        ) from exc

    from transformers.modeling_flash_attention_utils import prepare_fa_kwargs_from_position_ids
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5GatedDeltaNet,
        apply_mask_to_padding_states,
    )

    def _decoder_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                position_ids=position_ids,  # <-- thread position_ids into GDN
            )
        elif self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def _gdn_forward(
        self,
        hidden_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
        position_ids=None,
    ):
        # NOTE: a cache OBJECT being present is normal even in training -- the
        # text model creates an (empty, never-updated) Qwen3_5DynamicCache
        # whenever text_config.use_cache resolves True. That is harmless here:
        # this patched forward neither reads nor writes it, and a full-sequence
        # pass (cache_position starting at 0) is computed correctly. What IS
        # broken is *incremental decode* -- a step continuing from previously
        # cached state (cache_position[0] > 0) would silently see only its own
        # tokens. Guard exactly that.
        if (
            cache_params is not None
            and cache_position is not None
            and int(cache_position[0]) > 0
        ):
            raise NotImplementedError(
                "x_planner packing patch: incremental decode is not supported once "
                "apply_qwen3_5_packing_patch() ran in this process -- run "
                "generation in a process without the patch."
            )
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        # MRoPE position_ids arrive as [3, B, L]; the temporal axis carries the
        # per-document resets we need to derive cu_seqlens from. Using [0] (a 2D
        # [B, L] tensor) is required -- feeding the 3D tensor to
        # prepare_fa_kwargs_from_position_ids would flatten all three axes.
        if position_ids is not None and position_ids.ndim == 3:
            position_ids = position_ids[0]
        # Varlen cu_seqlens only when the batch is folded into a single row (packing).
        if position_ids is not None and batch_size == 1:
            cu_seqlens = prepare_fa_kwargs_from_position_ids(position_ids)[0][0]
        else:
            cu_seqlens = None

        # FLA kernels use [B, T, D] layout (no transpose, unlike the stock path).
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        mixed_qkv, _ = fla_causal_conv1d(
            x=mixed_qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            cu_seqlens=cu_seqlens,
        )

        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        core_attn_out, _ = chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            **({"cu_seqlens": cu_seqlens} if cu_seqlens is not None else {}),
        )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z).reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)

    Qwen3_5DecoderLayer.forward = _decoder_forward
    Qwen3_5GatedDeltaNet.forward = _gdn_forward
    _PATCHED = True


# ---------------------------------------------------------------------------
# Config-only get_rope_index (picklable across DataLoader workers)
# ---------------------------------------------------------------------------


class _RopeIndexShim:
    """Holds only the model ``config`` so ``get_rope_index`` can run standalone.

    Current ``Qwen3_5Model.get_rope_index`` needs ``self.config`` plus
    ``self.get_vision_position_ids`` (a pure helper, no weights). Binding both
    onto this tiny shim lets the packing collator/epilogue compute per-document
    MRoPE positions without dragging the whole model into multiprocessing
    pickling when ``dataloader_num_workers > 0``.
    """

    __slots__ = ("config",)

    def __init__(self, config: Any) -> None:
        self.config = config

    def get_vision_position_ids(self, *args, **kwargs):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

        return Qwen3_5Model.get_vision_position_ids(self, *args, **kwargs)


def make_packed_rope_index_fn(model: Any) -> Callable:
    """Return a picklable ``get_rope_index`` bound to the model's config only."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    shim = _RopeIndexShim(model.config)
    return functools.partial(Qwen3_5Model.get_rope_index, shim)


# ---------------------------------------------------------------------------
# Packing collate helpers (injected into the dataset_v2 Qwen3.5 epilogue)
# ---------------------------------------------------------------------------


def build_cu_seqlens(doc_lengths: Sequence[int], device=None):
    """Cumulative seqlens (int32, ``[n_docs + 1]``) + max doc length (int)."""
    cu = torch.zeros(len(doc_lengths) + 1, dtype=torch.int32, device=device)
    if doc_lengths:
        cu[1:] = torch.tensor(doc_lengths, dtype=torch.int32, device=device).cumsum(0)
    max_len = int(max(doc_lengths)) if doc_lengths else 0
    return cu, max_len


def build_mm_token_type_ids(
    input_ids: torch.Tensor,
    image_token_id: int,
    video_token_id: int,
) -> torch.Tensor:
    """Mark text=0 / image=1 / video=2 (same contract as the HF processor)."""
    mm = torch.zeros_like(input_ids, dtype=torch.int)
    mm = mm.masked_fill(input_ids == image_token_id, 1)
    mm = mm.masked_fill(input_ids == video_token_id, 2)
    return mm


def _vision_token_ids_from_rope_fn(get_rope_index: Callable) -> tuple[int, int]:
    """Read ``image_token_id`` / ``video_token_id`` off the config-only rope shim."""
    cfg = None
    if isinstance(get_rope_index, functools.partial) and get_rope_index.args:
        cfg = getattr(get_rope_index.args[0], "config", None)
    if cfg is None:
        raise ValueError(
            "pack_sequences needs image/video token ids from the rope-index "
            "shim config (make_packed_rope_index_fn), or each doc must carry "
            "mm_token_type_ids."
        )
    return int(cfg.image_token_id), int(cfg.video_token_id)


def _doc_mm_token_type_ids(
    doc: Dict[str, Any],
    ids: torch.Tensor,
    image_token_id: int,
    video_token_id: int,
) -> torch.Tensor:
    """Prefer the epilogue-provided tensor; otherwise derive from ``input_ids``."""
    mm = doc.get("mm_token_type_ids")
    if mm is None:
        return build_mm_token_type_ids(ids, image_token_id, video_token_id)
    if not torch.is_tensor(mm):
        mm = torch.as_tensor(mm, dtype=torch.int)
    if mm.ndim == 2:
        mm = mm.squeeze(0)
    if int(mm.shape[0]) != int(ids.shape[0]):
        raise ValueError(
            f"mm_token_type_ids length {mm.shape[0]} != input_ids length "
            f"{ids.shape[0]}"
        )
    return mm.to(dtype=torch.int)


def pack_sequences(
    docs: Sequence[Dict[str, Any]],
    get_rope_index: Callable,
    max_length: Optional[int] = None,
    *,
    pad_to_cutoff: bool = False,
    pad_token_id: int = 0,
) -> Dict[str, Any]:
    """Fold per-document tokenized samples into one packed row (bsz == 1).

    Each ``doc`` is a dict with ``input_ids`` / ``labels`` (1D LongTensors) and
    optional ``mm_token_type_ids`` / ``image_grid_thw`` / ``video_grid_thw`` /
    ``pixel_values`` / ``pixel_values_videos``.  When ``mm_token_type_ids`` is
    absent it is derived from ``input_ids`` via the model config's image/video
    token ids (required by current ``Qwen3_5Model.get_rope_index``).

    **Non-dropping**: all non-empty documents are concatenated into one varlen
    row with no padding.  Per-document MRoPE ``position_ids`` are built via
    ``get_rope_index`` (each re-based to 0) and concatenated; explicit
    ``cu_seq_lens_q/k`` + ``max_length_q/k`` are emitted so FlashAttention-2 packs
    the full-attention layers correctly (the 3D-MRoPE position-id derivation is
    unreliable).  ``attention_mask`` is ``None`` (no padding inside the row).

    The row length is controlled by the length-aware sampler upstream, not here;
    ``max_length`` is therefore advisory and used only by the optional
    ``pad_to_cutoff`` path below.

    Lives in THIS project (not dataset_v2) on purpose: it is Qwen3.5-VL training
    policy. ``build_dataset_v2`` injects it into the epilogue as
    ``pack_sequences_fn`` alongside ``get_rope_index``, keeping dataset_v2 free
    of any x_planner import.  Returns a model-ready batch dict.
    """
    image_token_id, video_token_id = _vision_token_ids_from_rope_fn(get_rope_index)

    input_ids: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    mm_list: List[torch.Tensor] = []
    pos_list: List[torch.Tensor] = []
    doc_lengths: List[int] = []
    kept_docs: List[Dict[str, Any]] = []

    for doc in docs:
        ids = doc["input_ids"]
        lab = doc["labels"]
        length = int(ids.shape[0])
        if length == 0:
            continue
        mm = _doc_mm_token_type_ids(doc, ids, image_token_id, video_token_id)
        pos, _ = get_rope_index(
            ids.unsqueeze(0),
            mm_token_type_ids=mm.unsqueeze(0),
            image_grid_thw=doc.get("image_grid_thw"),
            video_grid_thw=doc.get("video_grid_thw"),
            attention_mask=None,
        )  # [3, 1, length], re-based to 0 per document
        input_ids.append(ids)
        labels.append(lab)
        mm_list.append(mm)
        pos_list.append(pos)
        doc_lengths.append(length)
        kept_docs.append(doc)

    if not input_ids:
        raise ValueError("pack_sequences received no non-empty documents.")

    # --- BEGIN knapsack pad-to-cutoff (Path B') -----------------------------
    # Strategy B' (KnapsackPackedSampler with ``pad_to_cutoff: true``) wants every
    # row padded to a fixed ``cutoff`` length so tensor shapes are constant
    # (compile / CUDA-graph friendly, fully cross-rank equal).  The pad span is a
    # trailing masked "document": labels=-100, its own position_ids (so the GDN
    # patch resets there), and its own cu_seqlens segment (so FA2 attends within
    # the pad only).  Default is OFF -- strategies A and B never enter this block,
    # so deleting it leaves their behaviour untouched (see LENGTH_BALANCED_PACKING
    # PLAN section 6.6 for the one-step removal recipe).
    if pad_to_cutoff:
        if max_length is None:
            raise ValueError("pad_to_cutoff=True requires max_length (cutoff).")
        total = sum(doc_lengths)
        pad_len = int(max_length) - total
        if pad_len < 0:
            raise ValueError(
                f"pad_to_cutoff: packed length {total} exceeds cutoff "
                f"{max_length}; the knapsack sampler must keep bins <= cutoff."
            )
        if pad_len > 0:
            ref_ids = input_ids[0]
            pad_ids_tensor = torch.full(
                (pad_len,), pad_token_id, dtype=ref_ids.dtype
            )
            pad_mm = torch.zeros(pad_len, dtype=torch.int)  # text pads
            pad_pos, _ = get_rope_index(
                pad_ids_tensor.unsqueeze(0),
                mm_token_type_ids=pad_mm.unsqueeze(0),
                image_grid_thw=None,
                video_grid_thw=None,
                attention_mask=None,
            )
            input_ids.append(pad_ids_tensor)
            labels.append(torch.full((pad_len,), -100, dtype=labels[0].dtype))
            mm_list.append(pad_mm)
            pos_list.append(pad_pos)
            doc_lengths.append(pad_len)
    # --- END knapsack pad-to-cutoff (Path B') -------------------------------

    cu_seqlens, max_len = build_cu_seqlens(doc_lengths)
    batch: Dict[str, Any] = {
        "input_ids": torch.cat(input_ids).unsqueeze(0),       # [1, T]
        "labels": torch.cat(labels).unsqueeze(0),             # [1, T]
        "mm_token_type_ids": torch.cat(mm_list).unsqueeze(0),  # [1, T]
        "position_ids": torch.cat(pos_list, dim=-1),          # [3, 1, T]
        "attention_mask": None,
        # Explicit FA2 varlen kwargs for the full-attention layers.
        "cu_seq_lens_q": cu_seqlens,
        "cu_seq_lens_k": cu_seqlens,
        "max_length_q": max_len,
        "max_length_k": max_len,
    }

    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        vals = [d[key] for d in kept_docs if d.get(key) is not None]
        if vals:
            batch[key] = torch.cat(vals, dim=0)
    return batch


__all__ = [
    "apply_qwen3_5_packing_patch",
    "make_packed_rope_index_fn",
    "build_cu_seqlens",
    "build_mm_token_type_ids",
    "pack_sequences",
]
