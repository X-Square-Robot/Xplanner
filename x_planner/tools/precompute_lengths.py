#!/usr/bin/env python
"""Offline length precompute -- warm the length-balanced-packing cache.

Runs the SAME length estimator the ``knapsack_packed`` sampler uses, but
standalone *before* any model is loaded -- so it uses ``fork``
(no per-worker re-import / tokenizer reload) and as many workers as you give it.
Once this finishes, the per-source ``lengths.*.npy`` files are cached and training
starts instantly (the sampler just loads them).

The cache is keyed by a config hash (resolution / video params / model /
max_seq_length / estimator logic version), so use the SAME data config you train
with and it will hit. Image dimensions are always read from the actual files
(JSONL ``width``/``height`` are ignored). Frames whose length cannot be measured
(unreadable/corrupt media, unrenderable dialogue, or a non-multimodal source with
no length branch yet) are flagged ``-1`` and listed in a ``lengths.*.report.jsonl``
sidecar; the sampler drops them at train time.

Usage:
    python -m x_planner.tools.precompute_lengths --config path/to/data.yml --workers 40
    python -m x_planner.tools.precompute_lengths --config data.yml --image-tokens-mode assume_max
"""

from __future__ import annotations

import argparse
import os
from multiprocessing import cpu_count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", "-c", required=True, help="dataset YAML config path")
    ap.add_argument("--workers", "-w", type=int, default=min(cpu_count(), 32),
                    help="estimator processes (default min(cpu_count,32))")
    ap.add_argument("--cutoff", type=int, default=None,
                    help="per-row token budget for knapsack_packed")
    ap.add_argument("--cache-dir", default=None, help="override length_cache_dir")
    ap.add_argument("--token-factor", type=int, default=None,
                    help="vision-token divisor = patch_size * spatial_merge of the "
                         "backbone you TRAIN with. Qwen native = 32 (leave unset; == "
                         "image_factor). DINOv3 per-patch (patch16, no merge) = 16. "
                         "Training auto-derives this from the swapped processor, but "
                         "precompute loads no model -- so pass it here for a pluggable "
                         "backbone or the cache hash won't match training.")
    ap.add_argument("--image-tokens-mode", default=None,
                    choices=["header", "assume_max"],
                    help="'assume_max' skips image-header reads entirely (coarser)")
    args = ap.parse_args()

    import yaml

    # Registers knapsack_packed and exposes the cfg injector.
    from x_planner.data.length_samplers import inject_length_estimator_cfg
    from x2robot_dataset_v2.datasets.x2robot_dataset import X2RobotDataset

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ds = cfg["dataset"]

    sc = dict(ds.get("sampler", {}))
    sc["type"] = "knapsack_packed"
    sc.setdefault("num_replicas", 1)
    sc.setdefault("rank", 0)
    sc["length_num_workers"] = args.workers
    sc.setdefault("batch_size", 1)
    if args.cutoff is not None:
        sc["cutoff"] = args.cutoff
    if args.cache_dir is not None:
        sc["length_cache_dir"] = args.cache_dir
    if args.token_factor is not None:
        sc.setdefault("length_estimator", {})["token_factor"] = args.token_factor
    if args.image_tokens_mode is not None:
        sc.setdefault("length_estimator", {})["image_tokens_mode"] = args.image_tokens_mode
    ds["sampler"] = sc

    # Derive the estimator config from vision/epilogue params (same as training).
    inject_length_estimator_cfg(ds)

    # Length estimation only needs the SAMPLER, not the epilogue's packing (which
    # packs at training collate and requires a model-bound get_rope_index). Turn
    # packing off for this offline run -- it does NOT affect the cached lengths or
    # the config hash (which keys on processor_path / resolution / max_seq_length).
    epi_params = (ds.get("processors", {}).get("epilogue", {}) or {}).get("params")
    if isinstance(epi_params, dict) and epi_params.get("packing"):
        epi_params["packing"] = False

    print(f"[precompute] config={args.config} sampler=knapsack_packed "
          f"workers={args.workers} (cpu_count={cpu_count()})", flush=True)
    # No model loaded here -> the estimator pool uses fork (fast startup).
    _dataset, sampler = X2RobotDataset.from_config(cfg)
    lengths = getattr(sampler, "_lengths", None)
    if lengths is not None:
        import numpy as np

        cache = (
            sc.get("length_cache_dir")
            or (sc.get("length_estimator") or {}).get("cache_dir")
            or "~/.cache/x2robot_dataset_v2/lengths"
        )  # mirror _build_lengths' resolution so the printout is truthful
        n_flagged = int(np.sum(lengths < 0))
        valid = lengths[lengths >= 0]
        print(f"[precompute] DONE: {lengths.shape[0]} frames cached  "
              f"mean={float(np.mean(valid)) if valid.size else 0:.0f} "
              f"max={int(np.max(valid)) if valid.size else 0}  "
              f"flagged_unestimable={n_flagged} "
              f"({100.0 * n_flagged / max(lengths.shape[0], 1):.2f}%, dropped at "
              f"train time; see lengths.*.report.jsonl)", flush=True)
        # Truncation preview: the estimate already mirrors the epilogue's
        # _safe_truncate (length clamped to max(max_seq_length, vision)). Report the
        # two overflow modes so a too-large resolution / multi-image budget shows up
        # here rather than as silent truncation + budget-blowing rows at train time.
        max_seq = int((sc.get("length_estimator") or {}).get("max_seq_length") or 0)
        if max_seq and valid.size:
            n_at_cap = int(np.sum(valid == max_seq))  # filled to cap -> text right-truncated
            n_over = int(np.sum(valid > max_seq))       # vision span alone > cap -> oversized row
            print(f"[precompute] truncation @ max_seq_length={max_seq}: "
                  f"text-truncated (len==cap) {n_at_cap} "
                  f"({100.0 * n_at_cap / valid.size:.3f}%); "
                  f"vision-floor oversized (len>cap, blows the sampler token budget) "
                  f"{n_over} ({100.0 * n_over / valid.size:.3f}%)", flush=True)
        print(f"[precompute] cache dir: {os.path.expanduser(cache)}", flush=True)
    else:
        print("[precompute] DONE (no _lengths on sampler?)", flush=True)


if __name__ == "__main__":
    main()
