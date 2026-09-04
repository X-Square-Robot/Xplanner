# Copyright (c) 2026
"""Cut Cross-Entropy (CCE) loss for Qwen3.5-VL full SFT.

Standard VLM-SFT spends its single largest activation on the LM-head logits: a
``[B, T, vocab]`` tensor (Qwen3.5's vocab is ~150k) that is *also* upcast to fp32
for a numerically-stable cross-entropy.  For a packed 16k-token row that is
several GB just for the logits, dwarfing every other activation.

Cut Cross-Entropy (Apple, https://github.com/apple/ml-cross-entropy) computes the
*same* cross-entropy loss and gradients **without ever materializing the
logits**: a fused Triton kernel streams the ``hidden @ lm_head.weight.T`` matmul,
the log-sum-exp, and the gather of the target logit, keeping only per-token
scalars resident.  Loss/grad match ``F.cross_entropy`` up to bf16 rounding
(validated: loss rel-err ~1e-5; grad diff is the bf16 ULP).

This module monkey-patches ``Qwen3_5ForConditionalGeneration.forward`` so that,
when ``labels`` are supplied, the loss is produced by CCE straight from the
post-final-norm hidden states and ``lm_head.weight`` -- the ``lm_head`` matmul
and the logits tensor are skipped entirely.  When ``labels`` is ``None``
(generation / logits-only eval) the stock path runs unchanged.

Opt-in via ``--use_cce`` (applied in the launcher, next to the packing patch).
Supersedes ``--lm_head_loss_only_on_labels``: CCE is the stronger memory win, as
it also avoids the ``[N_supervised, vocab]`` fp32 logits that the masked-head
path still materializes.  The grad-accumulation normalization matches the rest
of the trainer (token mean or sample mean via ``loss_reduction_scope``; see
:meth:`QwenVLTrainer._cce_loss` and :mod:`x_planner.loss_reduce`).
"""
from __future__ import annotations

from x_planner.constants import IGNORE_INDEX
from x_planner.loss_reduce import is_sample_loss_scope, reduce_lm_losses

_CCE_PATCHED = False


def _import_linear_cross_entropy():
    """Import ``cut_cross_entropy.linear_cross_entropy`` with an actionable error."""
    try:
        from cut_cross_entropy import linear_cross_entropy
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(
            "--use_cce requires the cut_cross_entropy package. Install it with:\n"
            "    pip install cut-cross-entropy\n"
            "(needs a Triton-capable GPU; see https://github.com/apple/ml-cross-entropy)."
        ) from exc
    return linear_cross_entropy


def _logit_softcap(config):
    """Return the model's final-logit softcap (Qwen3.5 has none -> ``None``)."""
    text_cfg = config.get_text_config() if hasattr(config, "get_text_config") else config
    return getattr(text_cfg, "final_logit_softcapping", None)


def cce_loss_from_hidden(
    hidden_states,
    lm_head_weight,
    labels,
    num_items_in_batch=None,
    ignore_index: int = IGNORE_INDEX,
    softcap=None,
    impl: str = "cce",
    upcast: bool = False,
    cu_seq_lens=None,
    loss_reduction_scope: str = "batch",
):
    """Grad-accum-correct CCE loss from post-norm hidden states + classifier.

    Mirrors transformers' ``fixed_cross_entropy`` for token mean: SUM the
    supervised-token losses and divide by ``num_items_in_batch`` -- the
    supervised-token count over the whole accumulation window (and across ranks
    under ``--average_tokens_across_devices``) -- so summing the per-microbatch
    losses yields the exact global token mean instead of an average-of-means.

    With ``loss_reduction_scope in {"sample","sequence"}``, uses
    ``reduction="none"`` then :func:`x_planner.loss_reduce.reduce_lm_losses`
    so each packed document (or batch row) contributes its *token-mean* equally;
    ``num_items_in_batch`` is then the global sample count. Falls back to a local
    mean only when the count is unavailable.

    ``shift=True`` performs the causal next-token shift inside the kernel
    (``e[..., :-1, :]`` / ``targets[..., 1:]``) without materializing a shifted
    copy.

    **Precision/memory tier** (``impl`` / ``upcast``) -- the gradient w.r.t. the
    hidden states is ``E_softmax[W] - W[target]``, a small residual of large
    vectors, so it loses precision in bf16 (catastrophic cancellation, worst when
    the softmax is near-uniform). Measured ~9B-scale extra peak memory / realistic
    d/dhidden L2 error:

    * ``impl="cce"`` (Triton, default): never materializes logits -- ~1.1 GB,
      ~2-4% error (grows on flat softmax). Lowest memory, fixed bf16 precision.
    * ``impl="torch_compile"``: chunked ``e@c.T`` + fp32 cross-entropy fused by
      inductor -- ~3.3 GB, ~0.5% error. Best accuracy/memory trade.
    * ``impl="torch_compile", upcast=True``: fp32 ``e``/``c`` -- ~6.5 GB, ~1e-7
      error (matches the stock fp32-logits path). For when bit-parity matters.

    All three beat the stock fp32-logits path (~15 GB at T=8192). ``upcast`` is
    invalid with the Triton kernel (it requires bf16/fp16 ``e``).
    """
    linear_cross_entropy = _import_linear_cross_entropy()
    e, c = hidden_states, lm_head_weight
    if upcast:
        # The Triton CCE kernel asserts bf16/fp16 embeddings; only torch_compile
        # (pure chunked torch ops) can run in fp32. Caller is validated upstream.
        e, c = e.float(), c.float()

    shift_labels = labels[..., 1:].contiguous()
    if is_sample_loss_scope(loss_reduction_scope):
        per_token = linear_cross_entropy(
            e,
            c,
            labels,
            ignore_index=ignore_index,
            softcap=softcap,
            reduction="none",
            shift=True,
            impl=impl,
        )
        return reduce_lm_losses(
            per_token,
            shift_labels=shift_labels,
            loss_reduction_scope=loss_reduction_scope,
            num_items_in_batch=num_items_in_batch,
            cu_seq_lens=cu_seq_lens,
        )

    loss = linear_cross_entropy(
        e,
        c,
        labels,
        ignore_index=ignore_index,
        softcap=softcap,
        reduction="sum",
        shift=True,
        impl=impl,
    )
    if num_items_in_batch is not None:
        return loss / num_items_in_batch
    n = (shift_labels != ignore_index).sum().clamp(min=1)
    return loss / n


def apply_cce_patch(impl: str = "cce", upcast: bool = False):
    """Monkey-patch ``Qwen3_5ForConditionalGeneration.forward`` to use CCE.

    ``impl`` / ``upcast`` pick the precision/memory tier (see
    :func:`cce_loss_from_hidden`): ``"cce"`` (default, lowest memory),
    ``"torch_compile"`` (higher precision), and ``upcast=True`` (fp32, near-exact;
    ``torch_compile`` only). The choice is captured in the patched forward, so the
    trainer needs no extra plumbing.

    Idempotent (first call wins).  Raises ``ImportError`` if cut_cross_entropy is
    unavailable and ``ValueError`` for ``upcast`` with the Triton kernel.  Only the
    loss path changes: with ``labels`` the loss is computed by CCE and ``logits``
    is ``None`` (no projection, no logits tensor); without ``labels`` the stock
    ``lm_head`` + ``logits_to_keep`` path is preserved verbatim so generation and
    logits-only eval are unaffected.
    """
    global _CCE_PATCHED
    if _CCE_PATCHED:
        return
    if upcast and impl != "torch_compile":
        raise ValueError(
            "cce_upcast=True requires cce_impl='torch_compile': the Triton CCE "
            "kernel requires bf16/fp16 embeddings and cannot run in fp32."
        )
    _import_linear_cross_entropy()  # fail fast before mutating the class

    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    out_cls = m.Qwen3_5CausalLMOutputWithPast

    def _cce_vl_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        logits_to_keep=0,
        num_items_in_batch=None,
        loss_reduction_scope="batch",
        **kwargs,
    ):
        # Trainer-only kwarg: do not forward into the backbone.
        cu_seq_lens = kwargs.get("cu_seq_lens_q")
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]  # post-final-norm last hidden state (input to lm_head)

        loss = None
        logits = None
        if labels is not None:
            # Fused: loss straight from hidden_states @ lm_head.weight.T, no logits.
            loss = cce_loss_from_hidden(
                hidden_states,
                self.lm_head.weight,
                labels,
                num_items_in_batch=num_items_in_batch,
                ignore_index=IGNORE_INDEX,
                softcap=_logit_softcap(self.config),
                impl=impl,
                upcast=upcast,
                cu_seq_lens=cu_seq_lens,
                loss_reduction_scope=loss_reduction_scope,
            )
        else:
            # Stock path (generation / logits-only eval): unchanged.
            slice_indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        return out_cls(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    m.Qwen3_5ForConditionalGeneration.forward = _cce_vl_forward
    _CCE_PATCHED = True
