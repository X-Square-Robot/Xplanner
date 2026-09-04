# Copyright (c) 2026
"""Entry point for Qwen3.5-VL full supervised fine-tuning.

Data backend: **x2robot_dataset_v2** -- pass ``--data_config path/to.yml``. The
YAML drives ``X2RobotDataset.from_config`` (vision/text/epilogue pipeline); the
Qwen3.5 epilogue renders the official chat template and produces
``input_ids/labels/attention_mask/pixel_values/image_grid_thw``. The dataset's
``X2RobotSampler`` is plugged into the Trainer for distributed sharding + O(1)
resume.

    torchrun ... -m qwenvl.train.launcher --model_path Qwen/Qwen3.5-9B --data_config vqa_qwen3_5.yml ...
"""

import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import AutoProcessor, HfArgumentParser

from qwenvl.data import make_bad_sample_fallback_collator
from qwenvl.train.builders import (
    apply_lr_gating,
    build_dataset_v2,
    load_data_cfg,
    load_model,
    maybe_register_rvq_action_tokenizer,
    maybe_setup_packing,
    maybe_swap_vision_tower,
)
from qwenvl.train.trainer import QwenVLTrainer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


@dataclass
class ModelArguments:
    model_path: str = field(default="Qwen/Qwen3.5-9B")
    attn_implementation: str = field(default="flash_attention_2")
    # Pluggable vision encoder. "qwen" (default) keeps the stock native ViT untouched;
    # "dinov3" swaps model.model.visual for the encoder at --vision_ckpt + a per-patch
    # MLP projector (images, native aspect ratio). See qwenvl/model/vision.
    vision_backbone: str = field(default="qwen")
    vision_ckpt: Optional[str] = field(default=None)
    vision_projector_type: str = field(default="per_patch_mlp")
    # Optional pixel-count clamp for the pluggable image processor (None = passthrough;
    # resolution is normally controlled by processors.vision.resolution in the yaml).
    vision_max_pixels: Optional[int] = field(default=None)


@dataclass
class DataArguments:
    # x2robot_dataset_v2 backend:
    data_config: Optional[str] = field(
        default=None, metadata={"help": "x2robot_dataset_v2 YAML config path."}
    )
    # Optional processor resolution overrides (applied to the image processor):
    image_min_pixels: Optional[int] = field(default=None)
    image_max_pixels: Optional[int] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=16384)
    # optional decoupled LRs (None -> use --learning_rate for everything)
    vision_lr: Optional[float] = field(default=None)
    merger_lr: Optional[float] = field(default=None)
    # Per-component "set-LR-to-train" knobs (see builders.apply_lr_gating +
    # trainer.create_optimizer). Uniform rule: a component (LM / encoder /
    # projector) trains iff its LR knob is set, else it is frozen; at least one
    # knob is required (a --learning_rate-only run raises). --learning_rate
    # still seeds the LR scheduler and any knob left at None.
    #   llm_lr        -- LR for the language model (+ embeddings / lm_head)
    #   projector_lr  -- LR for the pluggable projector (the "merger" group)
    llm_lr: Optional[float] = field(default=None)
    projector_lr: Optional[float] = field(default=None)
    # separate LR for the input-embedding / lm_head rows (e.g. the from-scratch
    # RVQ action tokens in the discrete-action co-train); None -> base learning_rate.
    embedding_lr: Optional[float] = field(default=None)
    # loss normalization over the grad-accum window:
    #   "batch"    -- token mean (stock HF; pair with --average_tokens_across_devices
    #                 for correct multi-GPU/packing scaling). Default.
    #   "sample"   -- true sample mean: each packed document (or batch row) contributes
    #                 its per-token mean equally, so long answers don't dominate.
    #   "sequence" -- alias of "sample" (kept for backward compatibility).
    # See QwenVLTrainer._get_num_items_in_batch and qwenvl.loss_reduce.
    loss_reduction_scope: str = field(default="batch")
    # Memory: project only the (causally shifted) supervised positions through the
    # LM head instead of the whole packed row. The masked-out positions are ignored
    # by cross-entropy anyway, so this is numerically identical to the stock loss --
    # it just avoids the [1, T, vocab] fp32 logits tensor (the dominant VLM-SFT
    # activation). Packed single-row (B==1) training only; else falls back to stock.
    lm_head_loss_only_on_labels: bool = field(default=True)
    # Cut Cross-Entropy (qwenvl.model.cce): fuse the LM head + cross-entropy into a
    # Triton kernel that NEVER materializes the [*, vocab] logits (Qwen3.5 vocab is
    # ~150k -> several GB of fp32 logits per packed row). Loss/grad match stock
    # cross-entropy up to bf16 rounding. Strictly stronger than
    # --lm_head_loss_only_on_labels (which it supersedes when both are set);
    # requires the cut-cross-entropy package. Training only (eval uses logits).
    use_cce: bool = field(default=False)
    # CCE precision/memory tier (the loss d/dhidden loses bf16 precision via
    # catastrophic cancellation; pick how much memory to trade back for accuracy):
    #   "cce"            -- Triton, never materializes logits (~1.1GB@9B), ~2-4% grad
    #                       error in the realistic peaked-softmax SFT regime (default).
    #   "torch_compile"  -- chunked e@cT + fp32 cross-entropy (~3.3GB), ~0.5% error.
    # With --cce_upcast True (torch_compile only) e/c run in fp32 (~6.5GB) -> matches
    # the stock fp32-logits path (~1e-7). All three beat stock logits (~15GB@T=8192).
    cce_impl: str = field(default="cce")
    cce_upcast: bool = field(default=False)
    # Selective gradient checkpointing (MFU): with --gradient_checkpointing True, HF
    # checkpoints *every* layer (full recompute ~= +33% FLOPs). These keep a subset
    # resident (no recompute) when memory allows, trading freed VRAM for throughput.
    #   gc_keep_lm_layers   -- number of the 32 LM decoder layers to keep resident,
    #                          spread evenly across depth (0 = checkpoint all, stock).
    #   gc_checkpoint_vision -- checkpoint the 27-layer vision tower (False = keep it
    #                          resident; it runs every image step, small activations).
    # Tune up gc_keep_lm_layers while watching peak memory; pair with
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. See QwenVLTrainer.
    gc_keep_lm_layers: int = field(default=0)
    gc_checkpoint_vision: bool = field(default=True)
    # Live per-GPU MFU logging (QwenVLTrainer._MFUMeter): collective-free
    # tokens/sec + model-FLOPs-utilization injected into every training log.
    # Off by default; enable to watch tokens/sec + MFU (one numel() per micro-step).
    log_mfu: bool = field(default=False)
    # MFU denominator: bf16 peak FLOPs of ONE accelerator. Default is A100/A800
    # (312e12); H800 dense ~791e12, H100 dense ~989e12.
    peak_flops_per_gpu: float = field(default=312e12)


def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)

    processor = AutoProcessor.from_pretrained(model_args.model_path)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = training_args.model_max_length

    # Per-image pixel budget (Qwen3.5 defaults are used when left unset).
    if data_args.image_min_pixels is not None:
        processor.image_processor.min_pixels = data_args.image_min_pixels
    if data_args.image_max_pixels is not None:
        processor.image_processor.max_pixels = data_args.image_max_pixels

    # Load the dataset_v2 config up-front so RVQ token registration + embedding
    # resize can run *before* gradient-checkpointing wires its input-grad hook.
    data_cfg = None
    if data_args.data_config is not None:
        data_cfg = load_data_cfg(data_args.data_config)

    model = load_model(model_args.model_path, dtype, model_args.attn_implementation)
    model.config.use_cache = False
    # The text model resolves use_cache from its OWN sub-config, not the top
    # one -- leaving it True makes every training forward build a (never-used)
    # Qwen3_5DynamicCache.
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False

    # Cut Cross-Entropy: patch the LM forward to fuse loss without logits (opt-in).
    # Class-level patch -> order vs. model load is irrelevant; do it before train().
    if training_args.use_cce:
        from qwenvl.model.cce import apply_cce_patch

        apply_cce_patch(impl=training_args.cce_impl, upcast=training_args.cce_upcast)
        if training_args.local_rank in (-1, 0):
            print(
                f"[cce] Cut Cross-Entropy enabled (fused loss; no logits tensor) "
                f"impl={training_args.cce_impl} upcast={training_args.cce_upcast}",
                flush=True,
            )

    # Pluggable vision encoder: swap model.model.visual for DINOv3 (etc.) + a
    # per-patch projector, and swap the processor's image_processor to the matching
    # native-AR patchifier. No-op for --vision_backbone qwen.
    if maybe_swap_vision_tower(model_args, model):
        from qwenvl.model.vision import build_pluggable_processor

        build_pluggable_processor(
            processor, model_args.vision_ckpt, max_pixels=model_args.vision_max_pixels
        )
        if (
            data_args.image_min_pixels is not None
            or data_args.image_max_pixels is not None
        ) and training_args.local_rank in (-1, 0):
            print(
                "[vision] NOTE: --image_min_pixels/--image_max_pixels were applied "
                "to the Qwen image processor, which the pluggable path just "
                "replaced -- they have no effect here (resolution comes from the "
                "data yaml; cap via --vision_max_pixels).",
                flush=True,
            )

    # Discrete-action co-train: register RVQ tokens + resize embeddings BEFORE
    # enable_input_require_grads (so the resized embedding keeps the grad hook).
    action_tokenizer = None
    rope_index_fn = None
    if data_cfg is not None:
        action_tokenizer = maybe_register_rvq_action_tokenizer(data_cfg, processor, model)
        # Sequence packing: patch the model + build a config-only get_rope_index.
        rope_index_fn = maybe_setup_packing(data_cfg, model)

    if training_args.gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # Per-component training: each component trains iff its LR is set (uniform for both
    # backbones; --vision_lr/--projector_lr/--llm_lr, +optional --embedding_lr). Raises
    # if nothing would train. Only when it actually FREEZES part of the model does DDP
    # need find_unused_parameters (harmless/ignored under DeepSpeed; a full-train run
    # freezes nothing and doesn't trip it).
    if apply_lr_gating(model, training_args) and training_args.ddp_find_unused_parameters is None:
        training_args.ddp_find_unused_parameters = True

    # ---- data backend (x2robot_dataset_v2) ----
    if data_cfg is None:
        raise ValueError(
            "No --data_config given. This launcher uses the x2robot_dataset_v2 "
            "backend; pass --data_config <yaml> (the legacy --data_path "
            "LazySupervisedDataset backend has been removed)."
        )
    train_dataset, x2_sampler = build_dataset_v2(
        data_cfg, processor, action_tokenizer, training_args.model_max_length,
        get_rope_index=rope_index_fn,
    )
    # Wrap the dataset collate so an all-bad packed bin substitutes a good
    # sample instead of raising (DDP-safe). No-op unless bad_sample_tolerance
    # is enabled AND a whole bin fails; see qwenvl.data.bad_sample_fallback.
    data_collator = make_bad_sample_fallback_collator(train_dataset)

    # O(1) resume: restore the sampler stream position from the checkpoint
    # we are about to resume from (pair with --ignore_data_skip True).
    ckpts = sorted(
        pathlib.Path(training_args.output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if ckpts:
        import json
        state_file = ckpts[-1] / "x2_sampler_state.json"
        if state_file.is_file():
            with open(state_file) as f:
                x2_sampler.load_state_dict(json.load(f))

    trainer = QwenVLTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        x2_sampler=x2_sampler,
    )

    # Force the DataLoader workers to fork. `import deepspeed` sets the global
    # multiprocessing start method to 'spawn' at import time, and HF only overrides it
    # to 'fork' on MPS (trainer.py get_train_dataloader) -- so on CUDA the loader would
    # inherit 'spawn' and try to pickle the (closure) bad-sample fallback collator,
    # which fails: "Can't pickle local object ...collate". Workers only decode
    # images/video on CPU, so fork is correct here. Done AFTER the dataset build so the
    # length estimator kept its own explicit spawn context (needed once CUDA is init).
    import multiprocessing as _mp
    try:
        _mp.set_start_method("fork", force=True)
    except (RuntimeError, ValueError):
        pass

    resume = bool(list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")))
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.save_state()

    model.config.use_cache = True
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = True
    trainer.save_model(training_args.output_dir)
    if training_args.local_rank in (-1, 0):
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
