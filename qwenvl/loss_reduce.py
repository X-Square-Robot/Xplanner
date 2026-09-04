# Copyright (c) 2026
"""Loss reduction helpers for token-mean vs sample-mean SFT.

``loss_reduction_scope``:

* ``"batch"`` (default) -- sum of supervised-token CE / global supervised-token
  count (stock HF token mean; pair with ``--average_tokens_across_devices``).
* ``"sample"`` / ``"sequence"`` -- sum of *per-sample* mean CE / global sample
  count (true sample average: each document contributes equally regardless of
  answer length). ``"sequence"`` is kept as an alias for backward compatibility.

Grad-accum / multi-GPU correctness mirrors transformers' ``fixed_cross_entropy``:
each micro-step returns a *sum* over the local samples (of per-sample means),
divided by the global count over the accum window (and across ranks when
``--average_tokens_across_devices``). Summing those micro-step losses then
yields the exact global sample mean.

Shared by the trainer masked-head path and Cut Cross-Entropy
(:mod:`qwenvl.model.cce`).
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from qwenvl.constants import IGNORE_INDEX

SampleCountBatches = Sequence[dict]


def is_sample_loss_scope(scope: Optional[str]) -> bool:
    """True for true sample-mean reduction (``sample`` or legacy ``sequence``)."""
    return (scope or "batch") in ("sample", "sequence")


def count_supervised_samples_in_batch(batch: dict) -> int:
    """Count samples that contribute at least one causal-LM supervised target.

    Must match :func:`sum_of_per_sample_means`: only labels that enter the
    shifted CE (``labels[..., 1:]``) count. ``labels[..., 0]`` never contributes
    to causal LM loss, so a doc whose only non-``-100`` label sits at absolute
    position 0 is *not* counted (avoids numerator 0 / divisor > 0).

    Packed rows (``cu_seq_lens_q`` + ``B == 1``): one count per document whose
    shift-aligned span has a supervised target (pad-to-cutoff all-``-100``
    segments are skipped). Non-packed rows: one count per batch row with any
    supervised target after the causal shift.
    """
    labels = batch.get("labels")
    if labels is None or not torch.is_tensor(labels) or labels.numel() == 0:
        return 0

    cu = batch.get("cu_seq_lens_q")
    if cu is not None and labels.dim() == 2 and labels.shape[0] == 1:
        shift = labels[0, 1:]
        supervised = shift != IGNORE_INDEX
        t_shift = int(shift.shape[0])
        total = 0
        cu_list = cu.tolist() if torch.is_tensor(cu) else list(cu)
        for start, end in zip(cu_list[:-1], cu_list[1:]):
            start_i, end_i = int(start), int(end)
            # Same mapping as _sum_of_packed_doc_means.
            lo = max(start_i - 1, 0)
            hi = min(end_i - 1, t_shift)
            if lo < hi and bool(supervised[lo:hi].any().item()):
                total += 1
        return total

    if labels.dim() == 1:
        return int((labels[1:] != IGNORE_INDEX).any().item())
    # [B, T]: a row contributes iff it has any supervised next-token target.
    return int((labels[:, 1:] != IGNORE_INDEX).any(dim=-1).sum().item())


def count_supervised_samples(batch_samples: SampleCountBatches) -> int:
    """Sum of :func:`count_supervised_samples_in_batch` over a grad-accum window."""
    return sum(count_supervised_samples_in_batch(batch) for batch in batch_samples)


def sum_of_per_sample_means(
    per_token_loss: torch.Tensor,
    *,
    shift_labels: torch.Tensor,
    cu_seq_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sum of per-sample mean CE over samples that have supervised tokens.

    ``per_token_loss`` and ``shift_labels`` are aligned with the causal-LM shift
    (length ``T - 1`` per row):

    * Packed ``B == 1``: both 1-D ``[T - 1]`` (or ``[1, T - 1]``); segments come
      from ``cu_seq_lens`` over the *unshifted* sequence length ``T``. Target at
      shift index ``i`` belongs to the document containing absolute position
      ``i + 1``.
    * Non-packed ``B >= 1``: both ``[B, T - 1]``; each row is one sample.
      ``cu_seq_lens`` is ignored.
    """
    if per_token_loss.numel() == 0:
        return per_token_loss.new_zeros(())

    loss = per_token_loss
    labels = shift_labels
    if loss.dim() == 1:
        loss = loss.unsqueeze(0)
        labels = labels.unsqueeze(0)

    if loss.shape != labels.shape:
        raise ValueError(
            f"per_token_loss shape {tuple(per_token_loss.shape)} must match "
            f"shift_labels shape {tuple(shift_labels.shape)}"
        )

    supervised = labels != IGNORE_INDEX
    # CCE / CE may leave garbage or zeros on ignore positions; always mask.
    loss = loss * supervised.to(dtype=loss.dtype)

    bsz = loss.shape[0]
    if cu_seq_lens is not None and bsz == 1:
        return _sum_of_packed_doc_means(loss[0], supervised[0], cu_seq_lens)

    # One sample per row.
    total = loss.new_zeros(())
    for b in range(bsz):
        mask_b = supervised[b]
        if bool(mask_b.any().item()):
            total = total + loss[b][mask_b].mean()
    return total


def _sum_of_packed_doc_means(
    per_token_loss: torch.Tensor,
    supervised: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Sum of per-document means for a single packed row (shift-aligned)."""
    total = per_token_loss.new_zeros(())
    cu_list = cu_seq_lens.tolist() if torch.is_tensor(cu_seq_lens) else list(cu_seq_lens)
    # shift index i predicts absolute position i + 1.
    t_shift = per_token_loss.shape[0]
    for start, end in zip(cu_list[:-1], cu_list[1:]):
        start_i, end_i = int(start), int(end)
        # Targets in [start, end) <-> shift indices in [start - 1, end - 1).
        lo = max(start_i - 1, 0)
        hi = min(end_i - 1, t_shift)
        if lo >= hi:
            continue
        mask = supervised[lo:hi]
        if not bool(mask.any().item()):
            continue
        total = total + per_token_loss[lo:hi][mask].mean()
    return total


def reduce_lm_losses(
    per_token_loss: torch.Tensor,
    *,
    shift_labels: torch.Tensor,
    loss_reduction_scope: str = "batch",
    num_items_in_batch: Optional[Union[int, torch.Tensor]] = None,
    cu_seq_lens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reduce per-token CE to a micro-step scalar for the trainer.

    Token scope: ``sum(loss[supervised]) / num_items`` (global token count).
    Sample scope: ``sum_of_per_sample_means(...) / num_items`` (global sample count).
    """
    supervised = shift_labels != IGNORE_INDEX
    if is_sample_loss_scope(loss_reduction_scope):
        loss_sum = sum_of_per_sample_means(
            per_token_loss,
            shift_labels=shift_labels,
            cu_seq_lens=cu_seq_lens,
        )
        if num_items_in_batch is not None:
            denom = num_items_in_batch if not _is_zero_count(num_items_in_batch) else 1
            return loss_sum / denom
        # Local fallback: mean over contributing samples.
        if cu_seq_lens is not None and shift_labels.dim() == 1:
            n = 0
            cu_list = cu_seq_lens.tolist()
            t_shift = shift_labels.shape[0]
            for start, end in zip(cu_list[:-1], cu_list[1:]):
                lo = max(int(start) - 1, 0)
                hi = min(int(end) - 1, t_shift)
                if lo < hi and bool(supervised[lo:hi].any().item()):
                    n += 1
            return loss_sum / max(n, 1)
        if shift_labels.dim() == 1:
            n = int(supervised.any().item())
        else:
            n = int(supervised.any(dim=-1).sum().item())
        return loss_sum / max(n, 1)

    # Token mean (stock).
    loss_sum = (per_token_loss * supervised.to(dtype=per_token_loss.dtype)).sum()
    if num_items_in_batch is not None:
        denom = num_items_in_batch if not _is_zero_count(num_items_in_batch) else 1
        return loss_sum / denom
    n = int(supervised.sum().item())
    return loss_sum / max(n, 1)


def _is_zero_count(num_items_in_batch: Union[int, torch.Tensor]) -> bool:
    if torch.is_tensor(num_items_in_batch):
        return bool((num_items_in_batch == 0).all().item())
    return int(num_items_in_batch) == 0


__all__ = [
    "is_sample_loss_scope",
    "count_supervised_samples",
    "count_supervised_samples_in_batch",
    "sum_of_per_sample_means",
    "reduce_lm_losses",
]
