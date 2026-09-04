# Copyright (c) 2026
"""Assembly helpers that wire the model + x2robot_dataset_v2 backend together.

Kept out of :mod:`qwenvl.train.launcher` so the entry point reads as a flat
sequence of steps. Each helper is a no-op unless its feature is configured:

* :func:`load_model`                       -- load Qwen3.5-VL regardless of auto-mapping;
* :func:`load_data_cfg`                     -- read the dataset_v2 YAML config;
* :func:`maybe_register_rvq_action_tokenizer` -- discrete-action co-train tokens;
* :func:`maybe_setup_packing`               -- neat-packing model patch + rope index;
* :func:`build_dataset_v2`                  -- build (dataset, sampler) from the YAML.
"""

import os


def load_model(model_path: str, dtype, attn_implementation: str):
    """Load the Qwen3.5-VL multimodal model regardless of the exact auto-mapping."""
    kwargs = dict(torch_dtype=dtype, attn_implementation=attn_implementation)
    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
    except (ImportError, ValueError, KeyError):
        from transformers import Qwen3_5ForConditionalGeneration
        return Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **kwargs)


def maybe_swap_vision_tower(model_args, model):
    """Replace ``model.model.visual`` with a pluggable encoder + projector.

    No-op for ``vision_backbone == "qwen"`` (the stock tower stays). Otherwise:
    build the tower (pretrained encoder + fresh projector), set it, force
    ``vision_config.spatial_merge_size = 1`` (per-patch; used by MRoPE's
    ``get_rope_index``), and stamp the selection onto ``model.config`` so the
    checkpoint is self-describing for :func:`load_pluggable_qwen35`.

    Returns ``True`` when a swap happened (so the caller uses the pluggable processor).
    """
    name = getattr(model_args, "vision_backbone", "qwen")
    if name == "qwen":
        return False
    if not getattr(model_args, "vision_ckpt", None):
        raise ValueError(f"--vision_backbone {name} requires --vision_ckpt <hf id or path>")

    from qwenvl.model.vision import build_vision_backbone, stamp_pluggable_config

    lm_hidden = model.get_input_embeddings().weight.shape[1]
    dtype = next(model.parameters()).dtype
    projector_type = getattr(model_args, "vision_projector_type", "per_patch_mlp")
    tower = build_vision_backbone(
        name, model_args.vision_ckpt, lm_hidden, projector_type, dtype=dtype
    )
    model.model.visual = tower
    model.config.vision_config.spatial_merge_size = 1
    stamp_pluggable_config(
        model, name, model_args.vision_ckpt, projector_type,
        max_pixels=getattr(model_args, "vision_max_pixels", None),
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"[vision] swapped visual -> {name} ({model_args.vision_ckpt}); "
            f"enc_hidden={tower.merger.norm.normalized_shape[0]} -> lm_hidden={lm_hidden}; "
            f"projector={projector_type}, spatial_merge_size=1",
            flush=True,
        )
    return True


def apply_lr_gating(model, training_args):
    """Uniform per-component training: a component trains iff its LR knob is set.

    One rule for every run/backbone -- no legacy vs gated split. Knobs:
    ``--vision_lr`` (encoder), ``--projector_lr`` (the ``merger`` group; ``--merger_lr``
    is a backward-compat alias), ``--llm_lr`` (language model). ``--embedding_lr``
    optionally lets the input-embedding / lm_head rows (e.g. the from-scratch RVQ action
    tokens) train even when the rest of the LM is frozen. Grouping mirrors the trainer's
    optimizer buckets. Sets ``requires_grad`` on every parameter; returns ``True`` if
    anything was frozen (so DDP can enable find_unused_parameters). Raises if nothing
    would train.
    """
    vision_lr = getattr(training_args, "vision_lr", None)
    projector_lr = getattr(training_args, "projector_lr", None)
    if projector_lr is None:
        projector_lr = getattr(training_args, "merger_lr", None)
    llm_lr = getattr(training_args, "llm_lr", None)
    embedding_lr = getattr(training_args, "embedding_lr", None)

    # A pluggable swap installs a FRESH randomly-initialized projector; freezing
    # it (no --projector_lr) would train encoder/LM against permanently
    # scrambled vision features with no error -- refuse instead.
    if getattr(model.config, "vision_backbone", "qwen") != "qwen" and projector_lr is None:
        raise ValueError(
            "--vision_backbone != qwen requires --projector_lr: the swapped-in "
            "projector is freshly initialized, and an unset LR knob freezes it."
        )

    def trains(name: str) -> bool:
        if ".visual." in name and ".merger." in name:
            return projector_lr is not None
        if ".visual." in name:
            return vision_lr is not None
        if "embed_tokens" in name or "lm_head" in name:
            return llm_lr is not None or embedding_lr is not None
        return llm_lr is not None

    any_frozen = any_trainable = False
    for name, p in model.named_parameters():
        t = trains(name)
        p.requires_grad = t
        any_frozen |= not t
        any_trainable |= t
    if not any_trainable:
        raise ValueError(
            "No component LR set -> nothing would train. Set at least one of "
            "--vision_lr / --projector_lr / --llm_lr (optionally --embedding_lr)."
        )
    if int(os.environ.get("RANK", "0")) == 0:
        n_train = sum(p.requires_grad for p in model.parameters())
        n_tot = sum(1 for _ in model.parameters())
        print(
            f"[lr] per-component training: vision_lr={vision_lr} projector_lr={projector_lr} "
            f"llm_lr={llm_lr} embedding_lr={embedding_lr}; trainable tensors {n_train}/{n_tot}",
            flush=True,
        )
    return any_frozen


def load_data_cfg(data_config: str):
    """Load the x2robot_dataset_v2 YAML config."""
    import yaml

    with open(data_config) as f:
        return yaml.safe_load(f)


def maybe_register_rvq_action_tokenizer(cfg, processor, model):
    """Build the RVQ action tokenizer + register its tokens (for action co-train).

    When the epilogue is ``multimodal_action_qwen3_5`` and an action-tokenizer
    checkpoint is configured (``epilogue.params.action_tokenizer_checkpoint_path``),
    this:

    1. loads the external RVQ-delta codec via ``RVQActionTokenizer`` (CPU);
    2. adds its ``<rvq_group>`` / ``<rvq_r{q}_{idx}>`` special tokens to the LM
       tokenizer (``1 + num_quantizers*codebook_size`` rows, e.g. 4097);
    3. resizes the model's token embeddings to the new vocab size -- **before**
       gradient-checkpointing's ``enable_input_require_grads`` so the new rows
       are trainable (ported from wall-x ``load_wallx_processors`` +
       ``resize_token_embeddings``);
    4. returns the tokenizer instance so the epilogue can be handed it.

    The new embedding rows are initialized by HF (mean-resizing) and trained from
    scratch.  No flow head is added -- loss is the model's native cross-entropy on
    the assistant-span labels (the RVQ tokens).

    Returns ``None`` when no action tokenizer is configured (pure-VQA runs).
    """
    epi_cfg = cfg.get("dataset", {}).get("processors", {}).get("epilogue", {})
    epi_params = epi_cfg.get("params", {})
    ckpt = epi_params.get("action_tokenizer_checkpoint_path")
    if not ckpt:
        return None

    from qwenvl.data.rvq_tokenizer import RVQActionTokenizer

    action_tokenizer = RVQActionTokenizer(
        checkpoint_path=ckpt,
        config_dir=epi_params.get("action_tokenizer_config_dir"),
        device=epi_params.get("action_tokenizer_device", "cpu"),
        rvq_version=epi_params.get("rvq_version", "v3_2"),
    )

    rvq_tokens = action_tokenizer.get_special_tokens()
    n_added = processor.tokenizer.add_tokens(rvq_tokens)
    if n_added > 0:
        model.resize_token_embeddings(len(processor.tokenizer))
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"[rvq] registered {n_added} RVQ action tokens "
            f"(nq={action_tokenizer.num_quantizers}, "
            f"codebook={action_tokenizer.codebook_size}); "
            f"vocab -> {len(processor.tokenizer)}",
            flush=True,
        )
    return action_tokenizer


def maybe_setup_packing(cfg, model):
    """Enable neat-packing when ``epilogue.params.packing`` is set.

    Applies the Qwen3.5 GDN/decoder monkey-patch (so linear-attention layers
    reset at document boundaries) and returns a picklable, config-only
    ``get_rope_index`` for the epilogue to build per-document MRoPE positions.
    Returns ``None`` when packing is disabled.
    """
    epi_params = (
        cfg.get("dataset", {}).get("processors", {}).get("epilogue", {}).get("params", {})
    )
    if not epi_params.get("packing"):
        return None

    from qwenvl.data.packing import apply_qwen3_5_packing_patch, make_packed_rope_index_fn

    apply_qwen3_5_packing_patch()
    if int(os.environ.get("RANK", "0")) == 0:
        print("[packing] neat-packing enabled (GDN+FA2 cu_seqlens at doc boundaries)", flush=True)
    return make_packed_rope_index_fn(model)


def build_dataset_v2(
    cfg, processor, action_tokenizer, model_max_length: int, get_rope_index=None
):
    """Build (dataset, sampler) from a loaded x2robot_dataset_v2 config dict.

    The model's already-loaded ``processor`` is injected into the epilogue so
    tokenization/vision preprocessing are identical to the model and the
    processor is not loaded twice.  When co-training discrete actions, the
    pre-built ``action_tokenizer`` (RVQ codec) is injected too.  When sequence
    packing is enabled, the config-only ``get_rope_index`` is injected so the
    epilogue can build per-document MRoPE positions.  Distributed rank/world are
    taken from the launcher env so the sampler shards correctly even before
    ``torch.distributed`` is initialized.
    """
    from x2robot_dataset_v2.datasets.x2robot_dataset import X2RobotDataset

    ds_cfg = cfg["dataset"]

    # Sampler: pin rank/world from the launcher env (RANK/WORLD_SIZE), since the
    # sampler is built before torch.distributed.init in the Trainer.
    sampler_cfg = ds_cfg.setdefault("sampler", {})
    sampler_cfg.setdefault("num_replicas", int(os.environ.get("WORLD_SIZE", "1")))
    sampler_cfg.setdefault("rank", int(os.environ.get("RANK", "0")))

    # Length-balanced packing (this project's application-layer samplers, kept out
    # of x2robot_dataset_v2): importing registers knapsack_packed in the
    # dataset_v2 sampler registry so from_config can build it by name.
    from qwenvl.data.length_samplers import inject_length_estimator_cfg

    # Epilogue: reuse the model's processor + tokenizer (no second load).
    epi_params = ds_cfg["processors"]["epilogue"].setdefault("params", {})
    epi_params["hf_processor_instance"] = processor
    epi_params["tokenizer_instance"] = processor.tokenizer
    epi_params.setdefault("max_seq_length", model_max_length)
    if get_rope_index is not None:
        epi_params["get_rope_index"] = get_rope_index
        # The packing collate itself is also THIS project's policy: inject it
        # (module-level function, picklable) so dataset_v2 stays free of any
        # qwenvl import.
        from qwenvl.data.packing import pack_sequences

        epi_params["pack_sequences_fn"] = pack_sequences
    if action_tokenizer is not None:
        epi_params["action_tokenizer_instance"] = action_tokenizer
        # Drop the codec-loading keys: the instance is already built + injected.
        for k in (
            "action_tokenizer_checkpoint_path",
            "action_tokenizer_config_dir",
            "action_tokenizer_device",
            "rvq_version",
        ):
            epi_params.pop(k, None)

    # One resolution budget everywhere: the yaml vision processor resizes first,
    # the injected HF processor resizes again, and the length estimator plans
    # from the yaml numbers -- smart_resize is only idempotent (and the packing
    # plan only exact) when all of them share min/max_pixels. A CLI
    # --image_min/max_pixels that drifts from the yaml fails here, loudly.
    vis_params = (ds_cfg.get("processors", {}).get("vision", {}) or {}).get("params", {}) or {}
    ip = getattr(processor, "image_processor", None)
    for key in ("min_pixels", "max_pixels"):
        y, p_val = vis_params.get(key), getattr(ip, key, None)
        if y is not None and p_val is not None and int(y) != int(p_val):
            raise ValueError(
                f"vision.{key}={y} (data yaml) != image_processor.{key}={p_val} "
                "(model processor / --image_min_pixels / --image_max_pixels). "
                "The dataset vision processor, the HF processor and the length "
                "estimator must share one pixel budget."
            )

    # Derive the length-estimator config from the vision/epilogue params (no-op
    # unless sampler.type is knapsack_packed). Done here, after
    # the epilogue params are set, so processor_path / max_seq_length are visible.
    inject_length_estimator_cfg(ds_cfg)

    dataset, sampler = X2RobotDataset.from_config(cfg)
    return dataset, sampler
