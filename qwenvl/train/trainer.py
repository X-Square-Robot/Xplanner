# Copyright (c) 2026
"""Trainer for Qwen3.5-VL full SFT.

Thin subclass of ``transformers.Trainer`` that adds:

1. optional decoupled learning rates for the vision tower / patch merger vs.
   the language model (unchanged behaviour when the extra LRs are unset);
2. integration with the **x2robot_dataset_v2** ``X2RobotSampler`` -- a custom
   distributed sampler that owns per-rank sharding, global task-balance, and
   O(1) checkpoint/resume (``state_dict``/``load_state_dict``/``advance``).

When an ``x2_sampler`` is supplied, ``_get_train_sampler`` returns it and
``get_train_dataloader`` builds the loader with ``num_processes=1`` so accelerate
does **not** re-shard on top of the sampler's own per-rank sharding (which would
drop/duplicate data on multi-GPU).  When ``x2_sampler`` is ``None`` the trainer
behaves exactly like stock ``transformers.Trainer``.
"""

from __future__ import annotations

import os
import warnings
from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from qwenvl.constants import IGNORE_INDEX
from qwenvl.loss_reduce import (
    count_supervised_samples,
    is_sample_loss_scope,
    reduce_lm_losses,
)

try:
    from transformers.trainer_utils import seed_worker
except Exception:  # pragma: no cover - older/newer transformers
    seed_worker = None


class X2SamplerResumeCallback(TrainerCallback):
    """Persist + advance the ``X2RobotSampler`` state across checkpoints.

    - ``on_save``: dumps ``sampler.state_dict()`` into the checkpoint folder so a
      later run can fast-forward (O(1)) instead of replaying batches.
    - ``on_step_end``: advances the per-rank ``consumed`` counter by the number
      of samples one optimizer step consumed, so the *next* checkpoint records
      how far we got within the epoch.

    Pair with ``--ignore_data_skip True`` so HF does not *also* skip batches
    (the sampler's restored ``consumed`` already fast-forwards the stream).
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def on_step_end(self, args, state, control, **kwargs):
        per_rank_step_samples = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps
        )
        if hasattr(self.sampler, "advance"):
            self.sampler.advance(per_rank_step_samples)

    def on_save(self, args, state, control, **kwargs):
        # The sampler state (consumed/epoch) is advanced identically on every rank,
        # so it is rank-agnostic: only GLOBAL rank 0 writes it. Otherwise all
        # world_size ranks race on the same file on a shared FS (multi-node ->
        # torn/corrupt JSON -> resume crash). Write atomically (tmp + replace).
        if not state.is_world_process_zero:
            return
        import json
        import os

        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(ckpt, exist_ok=True)
        dst = os.path.join(ckpt, "x2_sampler_state.json")
        try:
            tmp = dst + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.sampler.state_dict(), f)
            os.replace(tmp, dst)
        except Exception as exc:
            # Never fail the checkpoint over sampler state, but never be silent
            # either: without this file a resume replays the epoch from its start.
            warnings.warn(
                f"x2_sampler state NOT saved to {dst} ({exc!r}); resuming from "
                "this checkpoint will replay the current epoch from the beginning."
            )


class _MFUMeter:
    """Per-GPU Model FLOPs Utilization meter (collective-free).

    NOT a TrainerCallback on purpose: HF's ``Trainer.log`` snapshots ``logs`` into
    ``log_history`` and then runs callbacks **in registration order** -- the stock
    ``WandbCallback`` / ``PrinterCallback`` are registered before any we add, so a
    trailing ``on_log`` callback would inject ``mfu`` *after* wandb/console already
    emitted the dict (it silently never shows up). Instead ``QwenVLTrainer.log``
    calls :meth:`inject` to mutate ``logs`` *before* ``super().log`` snapshots/reports
    it.

    Why collective-free: HF's own token counter (``--include_num_input_tokens_seen``)
    all-gathers + host-syncs the count every step (``accelerator.gather(...).item()``),
    which **hangs NCCL** under uneven packing (a rank whose bin stream ends early
    stops participating). So we never set that flag. This rank's tokens are counted
    locally in ``compute_loss`` (``input_ids.numel()`` -- no gather, no ``.item()``)
    into ``trainer._mfu_tokens``; MFU is per-accelerator anyway::

        MFU = 6 * N * (this_gpu_tokens / sec) / peak_flops_per_gpu   # fwd 2ND + bwd 4ND

    Reported on rank 0 as that rank's own MFU (representative under a balanced
    sampler); time is the wall clock between logs (no cuda events / barrier).

    Estimate caveats: LM-only (ViT FLOPs are extra -> image-heavy steps under-report)
    and counts ``lm_head`` on every token though ``logits_to_keep`` runs it on fewer
    (small over-count that roughly offsets). With gradient checkpointing the achieved
    HFU is ~1.3x this MFU (recompute is overhead, not useful FLOPs).
    """

    def __init__(self, num_params: int, peak_flops_per_gpu: float):
        self.flops_per_token = 6.0 * num_params
        self.peak = max(peak_flops_per_gpu, 1.0)
        self._last_tokens = None
        self._last_time = None

    def inject(self, tokens: int, logs: dict) -> None:
        import time

        now = time.monotonic()
        if (
            self._last_tokens is not None
            and self._last_time is not None
            and tokens > self._last_tokens
        ):
            dt = now - self._last_time
            if dt > 0:
                tok_per_sec = (tokens - self._last_tokens) / dt
                logs["mfu"] = round(self.flops_per_token * tok_per_sec / self.peak, 4)
                logs["tokens_per_sec_per_gpu"] = round(tok_per_sec)
        self._last_tokens = tokens
        self._last_time = now


class SelectiveGradientCheckpointingCallback(TrainerCallback):
    """Trade freed activation memory for fewer recompute FLOPs (higher MFU).

    Full gradient checkpointing recomputes every layer's forward in the backward
    pass -- ~33% extra FLOPs (the gap between the 6ND MFU and the achieved HFU).
    When memory allows, keeping a subset of layers *resident* (un-checkpointed)
    skips their recompute. transformers>=5 makes this per-layer: each decoder/vision
    block is a ``GradientCheckpointingLayer`` that self-checkpoints in ``__call__``
    iff its own ``gradient_checkpointing`` bool is set. HF's
    ``gradient_checkpointing_enable`` sets every layer's bool to ``True``; this
    callback (run in ``on_train_begin``, *after* that enable but before the loop)
    flips a chosen subset back to ``False``.

    * ``keep_lm_layers`` -- number of LM decoder layers to keep resident, spread
      evenly across depth (peak activation memory ~ sum of resident layers, so the
      position barely matters; even spacing is the neutral default).
    * ``checkpoint_vision`` -- when ``False`` the whole vision tower stays resident
      (it runs on every image step; its activations are small relative to the LM).

    No-op unless a knob is set. Holds the *unwrapped* module: DeepSpeed wraps the
    model after construction but the underlying layer objects persist, and the bool
    is read at forward time, so flipping it before the first step takes effect.
    """

    def __init__(self, model, keep_lm_layers: int = 0, checkpoint_vision: bool = True):
        self.model = model
        self.keep_lm_layers = int(keep_lm_layers)
        self.checkpoint_vision = bool(checkpoint_vision)

    def _locate_layers(self):
        base = getattr(self.model, "model", self.model)          # Qwen3_5Model
        lm = getattr(base, "language_model", None)
        lm_layers = list(getattr(lm, "layers", []) or [])
        visual = getattr(base, "visual", None)
        vis_blocks = list(getattr(visual, "blocks", []) or [])
        return lm_layers, vis_blocks

    def on_train_begin(self, args, state, control, **kwargs):
        lm_layers, vis_blocks = self._locate_layers()
        n = len(lm_layers)
        keep = max(0, min(self.keep_lm_layers, n))
        # Evenly-spaced resident layers: centers of `keep` equal depth bands.
        kept = {min(int((i + 0.5) * n / keep), n - 1) for i in range(keep)} if keep else set()
        for i, layer in enumerate(lm_layers):
            if i in kept and getattr(layer, "gradient_checkpointing", False):
                layer.gradient_checkpointing = False
        if not self.checkpoint_vision:
            for blk in vis_blocks:
                if getattr(blk, "gradient_checkpointing", False):
                    blk.gradient_checkpointing = False
        if state.is_world_process_zero:
            print(
                f"[selective-gc] LM decoder layers checkpointed "
                f"{n - len(kept)}/{n} (resident idx={sorted(kept)}); "
                f"vision tower checkpointing={self.checkpoint_vision} "
                f"({len(vis_blocks)} blocks)",
                flush=True,
            )


class QwenVLTrainer(Trainer):
    def __init__(self, *args, peak_flops_per_gpu=None, x2_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)

        # Gradient-accumulation loss fix (https://huggingface.co/blog/gradient_accumulation).
        # Qwen3.5-VL sets ``accepts_loss_kwargs = False`` and never threads
        # ``num_items_in_batch`` into its ``loss_function``, so HF's per-token
        # normalization is dead: ``_get_num_items_in_batch`` returns ``None`` ->
        # the loss degrades to ``reduction="mean"`` -> ``training_step`` divides by
        # ``gradient_accumulation_steps`` -> *average of per-microbatch means*,
        # which != the global token mean when supervised-token counts differ across
        # rows/ranks. Registering our own ``compute_loss_func`` flips every HF gate
        # ON (tokens get counted; ``training_step`` stops dividing by grad-accum)
        # and fixes the non-packed path for free; the packed ``B == 1`` path is
        # normalized to match in ``_masked_lm_head_loss``.
        if self.compute_loss_func is None:
            self.compute_loss_func = self._causal_lm_loss_func

        self.x2_sampler = x2_sampler
        if x2_sampler is not None:
            self.add_callback(X2SamplerResumeCallback(x2_sampler))
        self._dummy_vision_cache = None  # lazily-built (pixel_values, image_grid_thw)

        # Live per-GPU MFU logging (--log_mfu, default off) -- collective-free (this
        # rank's tokens are counted locally in compute_loss; injected into the log
        # dict by self.log, see _MFUMeter), so it does NOT need
        # --include_num_input_tokens_seen (whose per-step gather hangs NCCL under
        # uneven packing). The denominator comes from --peak_flops_per_gpu
        # (default A800/A100 bf16, 312 TFLOPS); the ctor kwarg overrides it.
        self._log_mfu = bool(getattr(self.args, "log_mfu", False))
        self._mfu_tokens = 0  # this rank's running input-token count (no gather)
        self._mfu_meter = None
        if self._log_mfu:
            if peak_flops_per_gpu is None:
                peak_flops_per_gpu = float(
                    getattr(self.args, "peak_flops_per_gpu", 312e12)
                )
            n_params = sum(p.numel() for p in self.model.parameters())
            self._mfu_meter = _MFUMeter(n_params, peak_flops_per_gpu)

        # Selective gradient checkpointing (MFU): keep some layers resident to skip
        # their recompute. No-op unless --gc_keep_lm_layers>0 or --gc_checkpoint_vision False.
        if self.args.gradient_checkpointing and (
            getattr(self.args, "gc_keep_lm_layers", 0)
            or not getattr(self.args, "gc_checkpoint_vision", True)
        ):
            self.add_callback(
                SelectiveGradientCheckpointingCallback(
                    self.model,
                    keep_lm_layers=getattr(self.args, "gc_keep_lm_layers", 0),
                    checkpoint_vision=getattr(self.args, "gc_checkpoint_vision", True),
                )
            )

    # ------------------------------------------------------------------
    # per-GPU MFU logging (inject before HF snapshots/reports the log dict)
    # ------------------------------------------------------------------

    def log(self, logs, *args, **kwargs):
        # Mutate `logs` BEFORE super().log so mfu reaches log_history + every callback
        # (wandb/printer). Only for training logs (skip eval) and only on rank 0.
        if (
            self._mfu_meter is not None
            and self.is_world_process_zero()
            and "loss" in logs
            and "eval_loss" not in logs
        ):
            self._mfu_meter.inject(self._mfu_tokens, logs)
        return super().log(logs, *args, **kwargs)

    # ------------------------------------------------------------------
    # dataset_v2 sampler integration
    # ------------------------------------------------------------------

    def _get_train_sampler(self, train_dataset=None):
        if self.x2_sampler is not None:
            return self.x2_sampler
        return super()._get_train_sampler(train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        if self.x2_sampler is None:
            return super().get_train_dataloader()
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        from accelerate.data_loader import prepare_data_loader

        params = dict(
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
        )
        # Sampler shape depends on ``yields_batches`` (see
        # docs/qwenvl/length_balanced_packing.md). Keyed off the generic flag, not
        # the concrete class, so a plain per-index sampler still works:
        #  - per-index sampler (``yields_batches=False``): DataLoader uses
        #    ``sampler=`` + ``batch_size`` and the collate packs that many docs;
        #  - batch (bin) sampler (knapsack_packed, ``yields_batches=True``): each
        #    iteration is one bin -> one packed row, so use ``batch_sampler=``.
        if getattr(self.x2_sampler, "yields_batches", False):
            if self._train_batch_size != 1:
                raise ValueError(
                    "batch-sampler mode (knapsack_packed): the DataLoader ignores "
                    f"--per_device_train_batch_size={self._train_batch_size}, but "
                    "the resume callback advances `consumed` by that factor, so "
                    "every checkpoint would over-count bins and resume would "
                    "silently skip data. Set --per_device_train_batch_size 1."
                )
            params["batch_sampler"] = self.x2_sampler
        else:
            params["batch_size"] = self._train_batch_size
            params["sampler"] = self.x2_sampler
            params["drop_last"] = self.args.dataloader_drop_last
        if self.args.dataloader_prefetch_factor is not None:
            params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        if self.args.dataloader_num_workers > 0 and seed_worker is not None:
            params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            )

        loader = DataLoader(self.train_dataset, **params)

        # X2RobotSampler already yields this rank's shard (rank::world_size).
        # Passing num_processes=1 makes accelerate skip BatchSamplerShard, so we
        # do NOT shard a second time, while DataLoaderShard still moves batches
        # to device and wires the gradient-accumulation end-of-loader signal.
        return prepare_data_loader(
            loader,
            device=self.accelerator.device,
            num_processes=1,
            process_index=0,
            put_on_device=True,
        )

    # ------------------------------------------------------------------
    # per-component learning rates (uniform: a component trains iff its LR is set)
    # ------------------------------------------------------------------

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        args = self.args
        base_lr = args.learning_rate
        vision_lr = getattr(args, "vision_lr", None)
        # projector_lr is canonical; merger_lr kept as a backward-compat alias.
        projector_lr = getattr(args, "projector_lr", None)
        if projector_lr is None:
            projector_lr = getattr(args, "merger_lr", None)
        llm_lr = getattr(args, "llm_lr", None)
        embedding_lr = getattr(args, "embedding_lr", None)

        # Single uniform rule for both backbones (no legacy/gated split): each
        # component's LR comes from its knob. requires_grad was already set per
        # component by builders.apply_lr_gating, so a component whose knob is unset is
        # frozen and skipped by the grouping loop -- its base_lr fallback is never used.
        lr_of = {
            "vision": vision_lr if vision_lr is not None else base_lr,
            "merger": projector_lr if projector_lr is not None else base_lr,
            "embedding": embedding_lr if embedding_lr is not None else (
                llm_lr if llm_lr is not None else base_lr
            ),
            "llm": llm_lr if llm_lr is not None else base_lr,
        }

        def bucket(name: str) -> str:
            # Freshly-resized input-embedding / lm_head rows (e.g. the RVQ action
            # tokens, trained from scratch) get their own LR when --embedding_lr is
            # set; otherwise they stay in the "llm" bucket at the base LR. Matches
            # "embed_tokens" specifically so the vision patch-embed isn't caught.
            if embedding_lr is not None and ("embed_tokens" in name or "lm_head" in name):
                return "embedding"
            if ".visual." in name and ".merger." in name:
                return "merger"
            if ".visual." in name:
                return "vision"
            return "llm"

        # Canonical weight-decay set: every parameter that is NOT a (Layer/RMS)Norm
        # and not a bias. More exact than the old substring heuristic -- e.g. it
        # keeps weight decay on the vision patch-embed projection that an "embed"
        # substring would have wrongly dropped.
        decay_parameters = set(get_parameter_names(self.model, ALL_LAYERNORM_LAYERS))
        decay_parameters = {n for n in decay_parameters if "bias" not in n}

        groups: dict = {}
        counts: dict = {}
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            b = bucket(name)
            no_decay = name not in decay_parameters
            groups.setdefault((b, no_decay), []).append(p)
            counts[b] = counts.get(b, 0) + 1

        optimizer_grouped_parameters = [
            {
                "params": params,
                "lr": lr_of[b],
                "weight_decay": 0.0 if no_decay else args.weight_decay,
            }
            for (b, no_decay), params in groups.items()
        ]

        # Visibility: a set LR silently no-ops if its bucket caught nothing (frozen or
        # renamed submodule). Print the trainable counts and warn on an empty override.
        if args.local_rank in (-1, 0):
            print(f"[optim] trainable param buckets: {counts}", flush=True)
            for knob, b in (
                (vision_lr, "vision"), (projector_lr, "merger"),
                (embedding_lr, "embedding"), (llm_lr, "llm"),
            ):
                if knob is not None and counts.get(b, 0) == 0:
                    warnings.warn(
                        f"a LR was set for the '{b}' group but it has no trainable "
                        "parameters (frozen or renamed?); the override has no effect."
                    )

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    # ------------------------------------------------------------------
    # loss normalization (cross-device + optional sample scope)
    # ------------------------------------------------------------------

    def _get_num_items_in_batch(self, batch_samples, device):
        """Normalizer for the grad-accum window's summed loss.

        ``loss_reduction_scope == "batch"`` (default) delegates to stock HF, which
        counts supervised tokens (``labels != -100``) and -- with
        ``--average_tokens_across_devices`` -- gathers them across ranks. This is
        what matters for packing: each rank's packed row holds a different number
        of tokens, so without the cross-device gather DDP averages differently
        normalized losses.

        ``loss_reduction_scope == "sample"`` (alias ``"sequence"``) instead counts
        *samples* with at least one supervised token -- packed documents via
        ``cu_seq_lens_q`` (skipping all-``-100`` pad-to-cutoff segments), or batch
        rows on the non-packed path -- so each document contributes equally to the
        loss regardless of answer length. The matching numerator is the sum of
        per-sample mean CE (see :mod:`qwenvl.loss_reduce`).
        """
        scope = getattr(self.args, "loss_reduction_scope", "batch")
        if not is_sample_loss_scope(scope):
            return super()._get_num_items_in_batch(batch_samples, device)

        # Only meaningful if the model actually consumes num_items_in_batch.
        if not (self.model_accepts_loss_kwargs or self.compute_loss_func is not None):
            return super()._get_num_items_in_batch(batch_samples, device)
        if not batch_samples or "labels" not in batch_samples[0]:
            return None

        total = count_supervised_samples(batch_samples)
        num_items_in_batch = torch.tensor(float(total), device=device, dtype=torch.float)
        if self.args.average_tokens_across_devices and self.args.world_size > 1:
            num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()
        elif self.args.n_gpu > 1:
            num_items_in_batch = num_items_in_batch / self.args.n_gpu
        # After the cross-device reduce: avoid /0 on an all-ignore accum window.
        # (Loss paths already return 0 when there are no supervised tokens.)
        return torch.clamp(num_items_in_batch.to(device), min=1.0)

    # ------------------------------------------------------------------
    # memory: project only the loss-bearing tokens through the LM head
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute the SFT loss with three adaptations over stock ``Trainer``:

        0. **Cut Cross-Entropy** (``use_cce``, takes precedence when set): the
           patched model forward returns the loss fused from hidden states +
           ``lm_head.weight`` with no logits tensor materialized at all -- the
           strongest of the three memory wins. See :meth:`_cce_loss` and
           :mod:`qwenvl.model.cce`.
        1. **LM-head only on loss-bearing tokens** (``lm_head_loss_only_on_labels``):
           project just the causally shifted supervised positions through ``lm_head``
           via the model's ``logits_to_keep`` hook. The masked-out positions are
           ignored by cross-entropy anyway, so this is numerically identical to the
           stock loss/gradients -- it only avoids the ``[1, T, vocab]`` fp32 logits
           tensor (the dominant VLM-SFT activation). Packed single-row (``B == 1``)
           training only; otherwise the base loss falls back to stock.
        2. **Dummy vision pass on text-only steps**: if this step's batch carries no
           real pixel input, run a tiny image through the vision tower (zero-weighted)
           so every rank's DeepSpeed gradient-reduction set is identical -- otherwise
           image-free ranks skip the vision-param reductions and the NCCL collectives
           desync/hang under ZeRO. Adds 0 to the loss value. Always on (training
           cannot proceed without it once any rank gets an image-free packed row).
        """
        # Per-GPU MFU bookkeeping: accumulate THIS rank's processed tokens locally
        # (no gather, no host sync) for the MFU meter. Counts every micro-step, so the
        # grad-accum window is summed correctly; packed rows have no padding so
        # numel() is the true token count.
        if self._log_mfu and getattr(model, "training", True):
            ids = inputs.get("input_ids")
            if ids is not None:
                self._mfu_tokens += int(ids.numel())

        labels = inputs.get("labels")
        # Stash packing boundaries for the stock ``compute_loss_func`` fallback
        # (HF only passes outputs/labels there; sample-mean needs cu_seq_lens).
        self._pending_cu_seq_lens = inputs.get("cu_seq_lens_q")
        try:
            # Cut Cross-Entropy: fused loss straight from hidden states, no logits
            # materialized (supersedes the masked-head path; needs the --use_cce
            # forward patch from qwenvl.model.cce). Any batch shape -- ``shift=True``
            # shifts per row, so packed B==1 and padded B>1 are both correct.
            use_cce = (
                getattr(self.args, "use_cce", False)
                and labels is not None
                and getattr(model, "training", True)
                and labels.dim() == 2
            )
            use_masked = (
                not use_cce
                and getattr(self.args, "lm_head_loss_only_on_labels", True)
                and labels is not None
                and getattr(model, "training", True)
                and labels.dim() == 2
                and labels.shape[0] == 1
            )
            if use_cce:
                loss, outputs = self._cce_loss(model, inputs, num_items_in_batch)
            elif use_masked:
                loss, outputs = self._masked_lm_head_loss(model, inputs, num_items_in_batch)
            else:
                loss, outputs = super().compute_loss(
                    model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch,
                )

            # Keep the vision tower in the autograd graph on text-only steps (see (2)).
            if (
                getattr(model, "training", True)
                and inputs.get("pixel_values") is None
                and inputs.get("pixel_values_videos") is None
            ):
                loss = loss + self._dummy_vision_touch(model, inputs["input_ids"].device)
        finally:
            self._pending_cu_seq_lens = None

        return (loss, outputs) if return_outputs else loss

    def _cce_loss(self, model, inputs, num_items_in_batch):
        """Base loss via Cut Cross-Entropy (``--use_cce``).

        The CCE forward patch (``qwenvl.model.cce.apply_cce_patch``) makes the
        model return the loss computed *directly* from the post-norm hidden states
        and ``lm_head.weight`` -- no ``[*, vocab]`` logits tensor is ever
        materialized. ``labels`` stays in ``inputs`` (the patched forward consumes
        it) and ``num_items_in_batch`` / ``loss_reduction_scope`` are threaded
        through so grad-accum normalization (token mean or sample mean) happens
        inside the kernel wrapper, mirroring ``fixed_cross_entropy``.

        Here we only reproduce stock ``Trainer.compute_loss``'s cross-device tail
        (trainer.py:2057) so this path stays byte-identical to the non-packed
        ``compute_loss_func`` path under DDP: each rank divides by the *global*
        count, then DDP averages, so multiply back by the device count.
        """
        scope = getattr(self.args, "loss_reduction_scope", "batch")
        outputs = model(
            **inputs,
            num_items_in_batch=num_items_in_batch,
            loss_reduction_scope=scope,
        )
        # Patch-effectiveness guard: the CCE forward returns logits=None on the
        # labels path. Logits here mean apply_cce_patch never took effect (or a
        # different model class swallowed num_items_in_batch into kwargs) -- its
        # mean-reduced loss would then be scaled by world_size below, silently
        # corrupting the gradient magnitude.
        if getattr(outputs, "logits", None) is not None:
            raise RuntimeError(
                "--use_cce is set but the model forward returned logits: the CCE "
                "class patch is not in effect. Call qwenvl.model.cce.apply_cce_patch "
                "before training and check the model class is "
                "Qwen3_5ForConditionalGeneration."
            )
        loss = outputs.loss
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss = loss * (
                self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu
            )
        return loss, outputs

    def _masked_lm_head_loss(self, model, inputs, num_items_in_batch):
        """Base loss for (1): cross-entropy over only the supervised positions."""
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        cu_seq_lens = inputs.get("cu_seq_lens_q")
        shift_labels = labels[0, 1:]                    # hidden position i predicts token i+1
        mask = shift_labels != IGNORE_INDEX
        keep = mask.nonzero(as_tuple=True)[0]           # hidden positions to project

        if keep.numel() == 0:
            # No supervised token in this row: keep lm_head in the graph at 0 loss.
            outputs = model(**inputs, logits_to_keep=1)
            return outputs.logits.float().sum() * 0.0, outputs

        outputs = model(**inputs, logits_to_keep=keep)  # lm_head runs on `keep` only
        logits = outputs.logits[0].float()              # [N, vocab]
        target = shift_labels[mask]                     # [N]

        # Per-token CE, then reduce by scope. Token scope mirrors transformers'
        # ``fixed_cross_entropy`` (sum / global supervised-token count). Sample
        # scope sums per-document means and divides by the global sample count
        # (see :mod:`qwenvl.loss_reduce`). Either way, summing micro-step
        # losses over the accum window yields the exact global mean.
        scope = getattr(self.args, "loss_reduction_scope", "batch")
        if is_sample_loss_scope(scope):
            # Full shift-length tensor so document boundaries map cleanly.
            per_token = shift_labels.new_zeros(shift_labels.shape, dtype=logits.dtype)
            per_token[mask] = F.cross_entropy(
                logits, target, ignore_index=IGNORE_INDEX, reduction="none"
            )
            loss = reduce_lm_losses(
                per_token,
                shift_labels=shift_labels,
                loss_reduction_scope=scope,
                num_items_in_batch=num_items_in_batch,
                cu_seq_lens=cu_seq_lens,
            )
        else:
            loss = F.cross_entropy(
                logits, target, ignore_index=IGNORE_INDEX, reduction="sum"
            )
            if num_items_in_batch is not None:
                loss = loss / num_items_in_batch
            else:
                loss = loss / target.numel()

        # Reproduce stock Trainer.compute_loss's cross-device tail (trainer.py:2057)
        # so this packed path stays byte-identical to the non-packed compute_loss_func
        # path under DDP (each rank divides by the *global* count, then DDP averages).
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss = loss * (
                self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu
            )
        return loss, outputs

    def _causal_lm_loss_func(self, outputs, labels, num_items_in_batch=None):
        """Gradient-accumulation-correct causal-LM loss for the non-packed path.

        Registered as ``self.compute_loss_func`` (see ``__init__``). Stock
        ``Trainer.compute_loss`` calls this after running the model *without*
        labels, so normalization happens here. Token scope delegates to
        ``ForCausalLMLoss`` (``sum / num_items_in_batch``). Sample scope reduces
        each row to its token mean, then sums those means and divides by the
        global sample count. Model-agnostic: ``vocab_size`` is read from the
        logits.
        """
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        scope = getattr(self.args, "loss_reduction_scope", "batch")
        if not is_sample_loss_scope(scope):
            return ForCausalLMLoss(
                logits, labels, vocab_size=logits.size(-1), num_items_in_batch=num_items_in_batch
            )

        # Sample mean: non-packed rows (B >= 1) or packed B==1 fallback when the
        # masked/CCE paths are off. Packed boundaries come from compute_loss's
        # stash -- without them a packed row would be treated as one sample while
        # num_items_in_batch still counts documents.
        if logits.shape[:2] != labels.shape[:2]:
            # HF sometimes passes logits already shifted; ForCausalLMLoss handles
            # the common case -- fall back rather than guess the layout.
            return ForCausalLMLoss(
                logits, labels, vocab_size=logits.size(-1), num_items_in_batch=num_items_in_batch
            )
        shift_logits = logits[..., :-1, :].contiguous().float()
        shift_labels = labels[..., 1:].contiguous()
        vocab = shift_logits.size(-1)
        per_token = F.cross_entropy(
            shift_logits.view(-1, vocab),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view_as(shift_labels)
        cu = getattr(self, "_pending_cu_seq_lens", None)
        return reduce_lm_losses(
            per_token,
            shift_labels=shift_labels,
            loss_reduction_scope=scope,
            num_items_in_batch=num_items_in_batch,
            cu_seq_lens=cu,
        )

    # ------------------------------------------------------------------
    # dummy vision pass (text-only steps) -- ports penguinvl's trick
    # ------------------------------------------------------------------

    def _dummy_vision_touch(self, model, device):
        """Run a tiny image through the vision tower; return a 0-valued scalar.

        Forces (zero) gradients onto every vision-tower + merger parameter so the
        DeepSpeed reduction set matches image-bearing ranks. Numerically adds 0 to
        the loss. No-op (literal 0) if the model exposes no image API.
        """
        unwrapped = self.accelerator.unwrap_model(model)
        if not hasattr(unwrapped, "get_image_features"):
            return torch.zeros((), device=device)
        pixel_values, image_grid_thw = self._dummy_vision_inputs(device)
        out = unwrapped.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        feats = getattr(out, "pooler_output", None)
        if feats is None:
            feats = getattr(out, "last_hidden_state", out)
        s = sum(f.sum() for f in feats) if isinstance(feats, (list, tuple)) else feats.sum()
        if not torch.is_tensor(s):
            return torch.zeros((), device=device)
        return s * 0.0

    def _dummy_vision_inputs(self, device):
        """Build + cache one tiny black-image (pixel_values, image_grid_thw)."""
        if self._dummy_vision_cache is None:
            from PIL import Image

            proc = self.processing_class
            image_processor = getattr(proc, "image_processor", proc)
            enc = image_processor(images=Image.new("RGB", (64, 64), (0, 0, 0)), return_tensors="pt")
            self._dummy_vision_cache = (enc["pixel_values"], enc["image_grid_thw"])
        pixel_values, image_grid_thw = self._dummy_vision_cache
        return pixel_values.to(device), image_grid_thw.to(device)
