"""Length-balanced packing samplers for Qwen3.5-VL SFT (application layer).

These live in Penguin-VL (not in the generic ``x2robot_dataset_v2`` library) on
purpose: length estimation / balancing is *this project's* training concern.  They
plug into dataset_v2 only through its public extension point ``@register_sampler``
(so ``X2RobotDataset.from_config`` can build them by name) and subclass
``X2RobotSampler`` to reuse its task-balanced phase-1 sampling.

Importing this module registers ``knapsack_packed`` in dataset_v2's sampler
registry.  ``qwenvl.train.builders`` imports it before building the dataset; the
offline ``qwenvl/tools/precompute_lengths.py`` does the same.

See ``docs/qwenvl/length_balanced_packing.md``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from x2robot_dataset_v2.samplers.frame_sampler import (
    X2RobotSampler,
    register_sampler,
    _resolve_task_balance_report_path,
)
from x2robot_dataset_v2.samplers.task_balance import TaskBalanceStrategy

from qwenvl.data.length_estimator import (
    LengthEstimatorConfig,
    estimate_lengths_for_sources,
)

logger = logging.getLogger("qwenvl.data.length_samplers")

_LENGTH_AWARE_SAMPLERS = frozenset({"knapsack_packed"})


def _lengths_digest(lengths: np.ndarray) -> str:
    """Content digest of the lengths array itself.

    The resumed epoch order depends on the exact array -- through the -1 filter
    AND the sort/knapsack over values -- but the config hash is unchanged across
    a cache delete-and-rescan (which the estimator's own drop-ratio warning
    recommends). Persisting this digest lets resume fail loudly instead of
    silently training a diverged stream.
    """
    import hashlib

    return hashlib.blake2b(
        np.ascontiguousarray(lengths).tobytes(), digest_size=8
    ).hexdigest()


def _filter_unestimable(
    arr: np.ndarray, lengths: np.ndarray, *, who: str, epoch: int
) -> np.ndarray:
    """Drop tagged indices the length estimator flagged unestimable (``length < 0``).

    The estimator writes ``ESTIMATE_FAILED`` (-1) for any frame whose length could
    not be measured truthfully -- corrupt/unreadable media, an unrenderable
    dialogue, or a source type with no length branch yet (action/subtask).  Packing
    or grouping such a frame under a fabricated length silently corrupts the plan
    (an under-estimate overflows its bin and the epilogue truncates the packed row),
    so we exclude it here.  The per-frame reasons live in the estimator's report
    sidecar (``lengths.*.report.jsonl``).
    """
    if lengths is None or lengths.size == 0 or arr.size == 0:
        return arr
    keep = lengths[arr] >= 0
    n_drop = int((~keep).sum())
    if n_drop and epoch == 0:  # count is stable across epochs -> log once
        logger.warning(
            "%s: dropping %d/%d sampled frames flagged unestimable by the length "
            "estimator (see the lengths.*.report.jsonl sidecar for reasons).",
            who, n_drop, int(arr.size),
        )
    return arr[keep]


# ======================================================================
# Length-estimator config injection (from vision/epilogue params)
# ======================================================================


def inject_length_estimator_cfg(dataset_cfg: Dict[str, Any]) -> None:
    """Populate ``dataset_cfg['sampler']['length_estimator']`` from processor params.

    Length-aware samplers estimate per-frame token length up front; to keep that
    estimate aligned with the running pipeline, the resolution budget (vision
    processor), the video sampling params (vision), and the tokenizer +
    max_seq_length + per_turn_think (epilogue) are copied into the estimator
    config.  Call this on the dataset-level config *before*
    ``X2RobotDataset.from_config`` (the trainer and the precompute tool both do).
    No-op for non-length-aware samplers; user-set values win (``setdefault``).
    """
    sampler_cfg = dataset_cfg.get("sampler", {})
    if sampler_cfg.get("type") not in _LENGTH_AWARE_SAMPLERS:
        return

    processors = dataset_cfg.get("processors", {}) or {}
    vision_params = (processors.get("vision", {}) or {}).get("params", {}) or {}
    epi_cfg = processors.get("epilogue", {}) or {}
    epi_params = epi_cfg.get("params", {}) or {}

    est = dict(sampler_cfg.get("length_estimator", {}))

    def _default(key: str, value: Any) -> None:
        if value is not None and key not in est:
            est[key] = value

    processor_path = epi_params.get("processor_path")
    inst = epi_params.get("hf_processor_instance")
    if processor_path is None and inst is not None:
        processor_path = (
            getattr(inst, "name_or_path", None)
            or getattr(getattr(inst, "tokenizer", None), "name_or_path", None)
        )
    _default("processor_path", processor_path)
    _default("image_factor", vision_params.get("image_factor"))
    # token_factor = patch_size * spatial_merge of the ACTUAL image processor (the one
    # injected into the epilogue). Qwen: 16*2=32; pluggable per-patch (DINOv3): 16*1=16.
    # Only set it when it DIFFERS from image_factor (the resize factor) -- i.e. the
    # pluggable case. For Qwen (token_factor == image_factor) we leave it unset so the
    # config hash is identical whether or not a processor instance is present (training
    # injects one; offline precompute_lengths does not) -> the cache still hits.
    ip = getattr(inst, "image_processor", None) if inst is not None else None
    if ip is not None:
        token_factor = int(getattr(ip, "patch_size", 16)) * int(getattr(ip, "merge_size", 2))
        img_factor = est.get("image_factor", 32)  # effective resize factor (default 32)
        # A user-set token_factor (yaml / precompute --token-factor) that
        # contradicts the ACTUAL processor means the cache was built for a
        # different backbone -- estimating under it would be 4x off. Fail loud.
        if "token_factor" in est and int(est["token_factor"]) != token_factor:
            raise ValueError(
                f"length_estimator.token_factor={est['token_factor']} but the "
                f"injected processor is patch{getattr(ip, 'patch_size', 16)}*"
                f"merge{getattr(ip, 'merge_size', 2)} = {token_factor}. The "
                "precomputed cache belongs to a different vision backbone."
            )
        if token_factor != img_factor:
            _default("token_factor", token_factor)
    # Per-patch backbones do no temporal pairing (one block per frame). The
    # temporal and spatial divisors are independent knobs, so set the temporal
    # one explicitly instead of letting the estimator infer it -- and do it for
    # BOTH token_factor origins (processor instance above, or an explicit
    # value from yaml / precompute --token-factor).
    if est.get("token_factor") and int(est["token_factor"]) != int(est.get("image_factor", 32)):
        _default("video_temporal_patch_size", 1)
    _default("image_min_pixels", vision_params.get("min_pixels"))
    _default("image_max_pixels", vision_params.get("max_pixels"))
    _default("max_pixels_split_by_images", vision_params.get("max_pixels_split_by_images"))
    _default("video_fps", vision_params.get("video_fps"))
    _default("video_maxlen", vision_params.get("video_maxlen"))
    _default("video_max_pixels", vision_params.get("video_max_pixels"))
    _default("video_min_pixels", vision_params.get("video_min_pixels"))
    _default("max_seq_length", epi_params.get("max_seq_length"))
    _default("per_turn_think", epi_params.get("per_turn_think"))

    # The estimator's dataclass defaults (32 / 1024 / 589824) differ from the
    # vision processor's own defaults (28 / 3136 / 1003520): a yaml that omits
    # these keys would train under one budget while estimating under another --
    # silent, systematic mis-estimation. Require them explicitly.
    missing = [
        k for k in ("image_factor", "image_min_pixels", "image_max_pixels")
        if k not in est
    ]
    if missing:
        raise ValueError(
            "length-aware sampler: the estimator config is missing "
            f"{missing}. Set processors.vision.params (image_factor / "
            "min_pixels / max_pixels) explicitly in the yaml -- the estimator "
            "cannot fall back to defaults because they differ from the vision "
            "processor's."
        )
    for k in ("video_fps", "video_maxlen", "video_max_pixels", "video_min_pixels"):
        if k not in est:
            logger.warning(
                "length-aware sampler: estimator config has no %r; if any source "
                "contains video, set processors.vision.params.%s explicitly or "
                "video lengths will be estimated under the estimator's default.",
                k, k,
            )

    # The estimator applies ONE global config to every source. A per-source
    # vision override with a different resolution budget would silently
    # mis-estimate that source (bin overflow -> packed-row truncation) and is
    # invisible to the cache key, so enforce the yaml contract ("per-source
    # vision params MUST match the global vision block") here.
    _RES_KEYS = (
        "image_factor", "min_pixels", "max_pixels", "max_pixels_split_by_images",
        "video_fps", "video_maxlen", "video_max_pixels", "video_min_pixels",
    )
    for src in dataset_cfg.get("sources", []) or []:
        src_vis = (((src.get("processors") or {}).get("vision") or {}).get("params") or {})
        for k in _RES_KEYS:
            if k in src_vis and src_vis[k] != vision_params.get(k):
                raise ValueError(
                    f"length-aware sampler: source {src.get('name')!r} overrides "
                    f"vision.{k}={src_vis[k]!r} but the global vision block has "
                    f"{vision_params.get(k)!r}. The (global) length estimator "
                    "cannot see per-source overrides -- make them match the "
                    "global vision params."
                )

    sampler_cfg["length_estimator"] = est
    if sampler_cfg.get("type") == "knapsack_packed":
        sampler_cfg.setdefault("cutoff", epi_params.get("max_seq_length"))
    dataset_cfg["sampler"] = sampler_cfg


def _build_lengths(sources: list, cfg: dict):
    """Estimate (or load cached) per-frame lengths from the sampler config."""
    est_cfg = dict(cfg.get("length_estimator", {}))
    # Allow flat keys on the sampler block too (convenience / injection).
    for k in (
        "processor_path", "image_factor", "token_factor",
        "image_min_pixels", "image_max_pixels",
        "video_fps", "video_maxlen", "video_max_pixels", "video_min_pixels",
        "video_temporal_patch_size", "max_seq_length", "per_turn_think",
        "image_tokens_mode",
    ):
        if k in cfg and k not in est_cfg:
            est_cfg[k] = cfg[k]
    if "processor_path" not in est_cfg:
        raise ValueError(
            "length-aware sampler needs a processor_path to estimate lengths "
            "(set sampler.length_estimator.processor_path, or run via "
            "qwenvl.train.launcher / qwenvl.tools.precompute_lengths which inject "
            "it from the epilogue config)."
        )
    config = LengthEstimatorConfig(
        **{k: v for k, v in est_cfg.items() if k in LengthEstimatorConfig.__dataclass_fields__}
    )
    cache_dir = cfg.get("length_cache_dir") or est_cfg.get(
        "cache_dir", "~/.cache/x2robot_dataset_v2/lengths"
    )
    # Mirror X2RobotSampler.__init__'s distributed auto-detect: _build_lengths
    # runs BEFORE super().__init__, and a caller that skips rank injection
    # (direct X2RobotDataset.from_config use) must not have every rank run the
    # full metadata scan concurrently.
    rank, world = cfg.get("rank"), cfg.get("num_replicas")
    if rank is None or world is None:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank() if rank is None else rank
                world = dist.get_world_size() if world is None else world
        except Exception:
            pass
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    if world is None:
        world = int(os.environ.get("WORLD_SIZE", "1"))
    lengths = estimate_lengths_for_sources(
        sources, config, cache_dir,
        num_workers=int(cfg.get("length_num_workers", 8)),
        rank=int(rank),
        world_size=int(world),
    )
    return lengths, config.config_hash()


# ======================================================================
# KnapsackPackedSampler
# ======================================================================


def greedy_knapsack(
    items: Sequence[Tuple[int, int]], capacity: int
) -> List[List[int]]:
    """Pack ``(length, index)`` items into bins with sum(length) <= ``capacity``.

    Best-fit-decreasing, the LLaMA-Factory algorithm: each bin seeds with the
    largest remaining item, then repeatedly adds the LARGEST remaining item that
    still fits the residual capacity, closing the bin only when nothing fits.
    Compared to next-fit (walk the sorted stream, close on overflow) this both
    packs tighter AND mixes content -- a bin of long videos gets its tail filled
    with short image-QA docs instead of leaving dead space, so length-clustered
    single-modality bins mostly disappear.

    LF's reference implementation is ``list.pop(mid)`` -- O(n^2), fine for its
    per-preprocessing-batch use but not for re-binning hundreds of thousands of
    docs every epoch.  Same algorithm here via per-length buckets + bisect over
    the (<= cutoff distinct) sizes: O(n log k + k^2), k = distinct lengths.

    Ties (same length) pop in ``items`` order, so the caller's per-epoch shuffle
    still randomizes WHICH doc fills a slot.  An item longer than ``capacity``
    gets its own bin (the epilogue truncates it); nothing is dropped.
    """
    from bisect import bisect_right
    from collections import deque

    bins: List[List[int]] = []
    buckets: Dict[int, Any] = {}
    for length, idx in items:
        if length > capacity:
            bins.append([idx])  # oversize: own bin
        else:
            buckets.setdefault(int(length), deque()).append(idx)

    sizes = sorted(buckets)  # ascending unique lengths
    while sizes:
        remaining = capacity
        cur: List[int] = []
        while sizes:
            pos = bisect_right(sizes, remaining) - 1
            if pos < 0:
                break  # nothing fits the residual capacity
            length = sizes[pos]
            cur.append(buckets[length].popleft())
            remaining -= length
            if not buckets[length]:
                del buckets[length]
                sizes.pop(pos)
        bins.append(cur)
    return bins


# Stamped into the sampler state: resuming under a DIFFERENT binning algorithm
# would silently rebuild a different bin stream for the same consumed count.
_KNAPSACK_ALGO = "best_fit_decreasing_v2"


@register_sampler("knapsack_packed")
class KnapsackPackedSampler(X2RobotSampler):
    """Knapsack bin-packing batch sampler: length-balanced packing rows."""

    yields_batches = True  # DataLoader uses batch_sampler= (one bin per iteration)

    def __init__(
        self,
        sources: list,
        *,
        lengths: np.ndarray,
        cutoff: int,
        lengths_config_hash: str = "",
        pad_to_cutoff: bool = False,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("batch_size", None)  # the bin IS the batch
        super().__init__(sources, batch_size=1, **kwargs)
        lengths = np.asarray(lengths, dtype=np.int64)
        if lengths.shape[0] != self._offsets[-1]:
            raise ValueError(
                f"KnapsackPackedSampler: lengths has {lengths.shape[0]} entries "
                f"but total frames = {self._offsets[-1]}."
            )
        self._lengths = lengths
        self._cutoff = int(cutoff)
        self._lengths_config_hash = lengths_config_hash
        self._lengths_digest = _lengths_digest(lengths)
        self._pad_to_cutoff = bool(pad_to_cutoff)
        self._cached_bins: Any = None

    @classmethod
    def from_config(cls, sources: list, cfg: dict) -> "KnapsackPackedSampler":
        task_balance = None
        if "task_balance" in cfg:
            task_balance = TaskBalanceStrategy.from_config(cfg["task_balance"])

        lengths = cfg.get("lengths")
        lengths_config_hash = cfg.get("lengths_config_hash", "")
        if lengths is None:
            lengths, lengths_config_hash = _build_lengths(sources, cfg)

        cutoff = cfg.get("cutoff") or cfg.get("max_seq_length")
        if not cutoff:
            raise ValueError(
                "KnapsackPackedSampler needs a 'cutoff' (or 'max_seq_length') -- "
                "the per-row token budget."
            )

        return cls(
            sources=sources,
            lengths=lengths,
            cutoff=int(cutoff),
            lengths_config_hash=lengths_config_hash,
            pad_to_cutoff=bool(cfg.get("pad_to_cutoff", False)),
            task_balance=task_balance,
            seed=cfg.get("seed", 42),
            num_replicas=cfg.get("num_replicas"),
            rank=cfg.get("rank"),
            task_balance_report_path=_resolve_task_balance_report_path(cfg),
            sample_episode_prefix_ratio=cfg.get("sample_episode_prefix_ratio", 1.0),
        )

    def _compute_bins(self) -> List[List[int]]:
        rng = np.random.RandomState(self._seed + self.epoch)
        global_indices = self._phase1_global_sample(rng)
        if not global_indices:
            return []

        arr = np.asarray(global_indices, dtype=np.int64)
        arr = _filter_unestimable(
            arr, self._lengths, who=type(self).__name__, epoch=self.epoch
        )
        if arr.size == 0:
            return []
        rng.shuffle(arr)
        items = [(int(self._lengths[idx]), int(idx)) for idx in arr]
        bins = greedy_knapsack(items, self._cutoff)

        perm = rng.permutation(len(bins))
        bins = [bins[i] for i in perm]

        if self._world_size > 1:
            pad = (-len(bins)) % self._world_size
            if pad:
                # Tile the pad: a tiny epoch (smoke run / huge cutoff) can hold
                # fewer bins than the pad needs, and a single short slice would
                # leave the total a non-multiple of world_size -> some ranks get
                # one bin fewer (or zero) -> DDP collectives desync/NCCL hang.
                reps = -(-pad // len(bins))
                bins = bins + (bins * reps)[:pad]
        rank_bins = bins[self._rank :: self._world_size]

        if self._lengths.size:
            fills = [sum(self._lengths[i] for i in b) for b in rank_bins] or [0]
            logger.info(
                "KnapsackPackedSampler: epoch=%d rank=%d bins=%d cutoff=%d "
                "mean_fill=%.0f (%.0f%%) pad_to_cutoff=%s",
                self.epoch, self._rank, len(rank_bins), self._cutoff,
                float(np.mean(fills)), 100.0 * float(np.mean(fills)) / self._cutoff,
                self._pad_to_cutoff,
            )
        return rank_bins

    def _ensure_bins(self) -> None:
        if self._cached_bins is None:
            self._cached_bins = self._compute_bins()
            # Mirror the base sampler's length cache: keeps __len__ O(1) after
            # __iter__ frees the bins, and lets advance() clamp at the epoch end.
            self._cached_length = len(self._cached_bins)

    def __iter__(self) -> Iterator[List[int]]:
        self._ensure_bins()
        bins = self._cached_bins[self._consumed :]
        self._cached_bins = None  # free; _cached_length keeps __len__ O(1)
        return iter(bins)

    def __len__(self) -> int:
        """Full bin count for this epoch (constant; see ``X2RobotSampler.__len__``
        for why this must NOT shrink with ``consumed``).

        NOTE: re-binning under a new shuffle can change the bin count slightly
        between epochs, so HF's setup-time ``num_update_steps_per_epoch`` is
        exact for the first epoch and approximate for later ones.
        """
        if self._cached_length is None:
            self._ensure_bins()
        return self._cached_length

    def set_epoch(self, epoch: int) -> None:
        if epoch == self.epoch:
            return
        self.epoch = epoch
        self._consumed = 0
        self._cached_bins = None
        self._cached_length = None

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["cutoff"] = self._cutoff
        state["pad_to_cutoff"] = self._pad_to_cutoff
        state["lengths_config_hash"] = self._lengths_config_hash
        state["lengths_digest"] = self._lengths_digest
        state["knapsack_algo"] = _KNAPSACK_ALGO
        return state

    def load_state_dict(self, state: dict) -> None:
        ckpt_hash = state.get("lengths_config_hash", "")
        if (
            ckpt_hash
            and self._lengths_config_hash
            and ckpt_hash != self._lengths_config_hash
        ):
            raise ValueError(
                "KnapsackPackedSampler: lengths_config_hash mismatch on resume "
                f"(checkpoint={ckpt_hash!r}, current={self._lengths_config_hash!r})."
            )
        ckpt_digest = state.get("lengths_digest", "")
        if ckpt_digest and ckpt_digest != self._lengths_digest:
            raise ValueError(
                "KnapsackPackedSampler: the lengths array changed since the "
                "checkpoint (same config, different content -- e.g. the cache "
                "was rescanned and a different frame set was flagged), so the "
                "resumed bin stream would silently diverge. Restore the original "
                "length cache or start a fresh run."
            )
        ckpt_algo = state.get("knapsack_algo")
        if ckpt_algo is not None and ckpt_algo != _KNAPSACK_ALGO:
            raise ValueError(
                f"KnapsackPackedSampler: checkpoint was built with binning "
                f"algorithm {ckpt_algo!r} but this code uses {_KNAPSACK_ALGO!r}; "
                "the resumed bin stream would silently diverge. Start a fresh run."
            )
        self._cutoff = int(state.get("cutoff", self._cutoff))
        self._pad_to_cutoff = bool(state.get("pad_to_cutoff", self._pad_to_cutoff))
        super().load_state_dict(state)
        self._cached_bins = None
        self._cached_length = None


__all__ = [
    "KnapsackPackedSampler",
    "greedy_knapsack",
    "inject_length_estimator_cfg",
]
