"""Tests for length-balanced packing, CPU-only.

Covers the data-side guarantees of docs/x_planner/length_balanced_packing.md without
a GPU or model weights:

* ``KnapsackPackedSampler`` -- bins <= cutoff, DDP-equal bin counts, resume;
* ``pack_sequences``        -- non-dropping concat + the pad-to-cutoff path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from x2robot_dataset_v2.common.episode import Episode
from x2robot_dataset_v2.samplers.data_source import DataSource
from x2robot_dataset_v2.samplers.frame_index import X2RobotFrameIndex

# Application-layer samplers live in this project, not in dataset_v2.
from x_planner.data.length_samplers import (
    KnapsackPackedSampler,
    inject_length_estimator_cfg,
    greedy_knapsack,
)


def make_episode(path="/d/ep", num_frames=100, st_frame=0, ed_frame=100,
                 length=100, episode_type="x2_multimodal", task_name="others",
                 **kw):
    return Episode(path=path, num_frames=num_frames, st_frame=st_frame,
                   ed_frame=ed_frame, length=length, episode_type=episode_type,
                   task_name=task_name, cam_mapping={"cam0": "cam0"}, **kw)


def _sources(total_frames: int = 120):
    eps = [
        make_episode("/d/ep0", num_frames=total_frames // 2,
                     ed_frame=total_frames // 2, length=total_frames // 2),
        make_episode("/d/ep1", num_frames=total_frames - total_frames // 2,
                     ed_frame=total_frames - total_frames // 2,
                     length=total_frames - total_frames // 2),
    ]
    return [DataSource("mm", X2RobotFrameIndex(eps), source_type="multimodal")]


def _rand_lengths(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randint(50, 4000, size=n).astype(np.int64)


# ----------------------------------------------------------------------
# KnapsackPackedSampler
# ----------------------------------------------------------------------
class TestKnapsackPackedSampler:
    def _make(self, world, rank, cutoff=8000, total=200, seed=0, pad=False):
        srcs = _sources(total)
        lengths = _rand_lengths(total, seed=2)
        return KnapsackPackedSampler(
            srcs, lengths=lengths, cutoff=cutoff, lengths_config_hash="h",
            pad_to_cutoff=pad, seed=seed, num_replicas=world, rank=rank,
        )

    def test_yields_batches_flag(self):
        assert self._make(1, 0).yields_batches is True

    def test_bins_within_cutoff(self):
        cutoff = 8000
        s = self._make(1, 0, cutoff=cutoff)
        lengths = _rand_lengths(200, seed=2)
        for b in iter(s):
            # each bin sums to <= cutoff (single oversized items can't occur here:
            # max length 3999 < cutoff)
            assert sum(int(lengths[i]) for i in b) <= cutoff

    def test_ddp_equal_bin_counts(self):
        world = 4
        counts = [len(self._make(world, r)) for r in range(world)]
        assert len(set(counts)) == 1, counts

    def test_resume_consumed_bins(self):
        s = self._make(2, 0)
        full = [list(b) for b in iter(s)]
        s2 = self._make(2, 0)
        s2.advance(2)  # 2 bins consumed
        assert [list(b) for b in iter(s2)] == full[2:]

    def test_greedy_knapsack_tightness(self):
        rng = np.random.RandomState(0)
        lens = rng.randint(100, 2000, size=500)
        items = [(int(L), i) for i, L in enumerate(lens)]
        cutoff = 8192
        bins = greedy_knapsack(items, cutoff)
        # no bin exceeds cutoff
        assert all(sum(items[i][0] for i in b) <= cutoff for b in bins)
        # nothing lost / duplicated
        flat = [i for b in bins for i in b]
        assert sorted(flat) == list(range(len(items)))
        # reasonably tight: total/cutoff is the theoretical min bin count
        lower_bound = int(np.ceil(lens.sum() / cutoff))
        assert len(bins) <= lower_bound + 5


# ----------------------------------------------------------------------
# pack_sequences: non-dropping + B' pad-to-cutoff
# ----------------------------------------------------------------------
def _fake_rope_index(input_ids, image_grid_thw=None, video_grid_thw=None,
                     attention_mask=None):
    """Stand-in get_rope_index: 3D positions = arange per doc (no model needed)."""
    L = input_ids.shape[-1]
    pos = torch.arange(L).view(1, 1, L).expand(3, 1, L).contiguous()
    return pos, None


class TestPackSequences:
    def _docs(self, lengths):
        docs = []
        for n in lengths:
            ids = torch.arange(n, dtype=torch.long)
            docs.append({"input_ids": ids, "labels": ids.clone()})
        return docs

    def test_no_drop_concatenates_all(self):
        from x_planner.data.packing import pack_sequences
        lengths = [100, 200, 50, 4096]  # total far exceeds an old max_length
        out = pack_sequences(self._docs(lengths), _fake_rope_index, max_length=512)
        assert out["input_ids"].shape[1] == sum(lengths)  # nothing dropped
        cu = out["cu_seq_lens_q"].tolist()
        assert cu == np.cumsum([0] + lengths).tolist()
        assert out["attention_mask"] is None

    def test_pad_to_cutoff_fixed_shape(self):
        from x_planner.data.packing import pack_sequences
        lengths = [100, 200, 50]  # total 350
        cutoff = 512
        out = pack_sequences(
            self._docs(lengths), _fake_rope_index, max_length=cutoff,
            pad_to_cutoff=True, pad_token_id=7,
        )
        assert out["input_ids"].shape[1] == cutoff
        # pad span is masked and is its own final cu_seqlens segment
        assert out["cu_seq_lens_q"].tolist()[-1] == cutoff
        assert int((out["labels"] == -100).sum()) >= cutoff - sum(lengths)
        assert int(out["input_ids"][0, -1]) == 7

    def test_pad_to_cutoff_overflow_raises(self):
        from x_planner.data.packing import pack_sequences
        with pytest.raises(ValueError, match="exceeds cutoff"):
            pack_sequences(self._docs([400, 400]), _fake_rope_index,
                           max_length=512, pad_to_cutoff=True)


# ----------------------------------------------------------------------
# Regression: __len__/resume semantics + degenerate-size padding (2026-07)
# ----------------------------------------------------------------------
class TestResumeLenSemantics:
    """__len__ must report the FULL epoch: HF derives num_update_steps_per_epoch,
    max_steps and epochs_trained from len(dataloader) (and evaluates each epoch's
    steps_in_epoch BEFORE set_epoch), so a consumed-shrunken length breaks the LR
    schedule and mid-epoch resume. The fast-forward lives in __iter__ only."""

    def _knapsack(self, world=2, rank=0, total=120, cutoff=6000):
        return KnapsackPackedSampler(
            _sources(total), lengths=_rand_lengths(total, seed=1), cutoff=cutoff,
            lengths_config_hash="h1", seed=0, num_replicas=world, rank=rank,
        )

    def test_len_constant_after_advance(self):
        s = self._knapsack()
        full = len(s)
        s.advance(4)
        assert len(s) == full                    # NOT full - 4
        assert len(list(iter(s))) == full - 4    # iter does the skipping

    def test_advance_clamps_at_epoch_end(self):
        s = self._knapsack()
        full = len(s)
        s.advance(full + 3)  # partial final accumulation window overshoots
        assert s.consumed == full
        assert list(iter(s)) == []

    def test_set_epoch_resets_consumed(self):
        s = self._knapsack()
        s.advance(len(s))
        s.set_epoch(1)
        assert s.consumed == 0

    def test_lengths_digest_mismatch_raises(self):
        s = self._knapsack()
        state = s.state_dict()
        state["lengths_digest"] = "0" * 16
        s2 = self._knapsack()
        with pytest.raises(ValueError, match="lengths array changed"):
            s2.load_state_dict(state)

    def test_knapsack_len_cheap_after_iter(self):
        s = self._knapsack()
        full = len(s)
        list(iter(s))
        assert s._cached_bins is None  # freed by __iter__
        assert len(s) == full          # answered from the cached count


class TestDegenerateSizePadding:
    """Tiny epochs must still hand every rank the same non-zero work
    (a short pad slice used to leave ranks unequal -> DDP hang)."""

    def test_knapsack_tiny_dataset_equal_nonzero_bins(self):
        world = 8
        counts = []
        for r in range(world):
            s = KnapsackPackedSampler(
                _sources(6), lengths=_rand_lengths(6, seed=1), cutoff=100000,
                lengths_config_hash="h1", seed=0, num_replicas=world, rank=r,
            )
            counts.append(len(s))
        assert len(set(counts)) == 1
        assert counts[0] >= 1


class TestInjectEstimatorCfg:
    """token_factor (spatial) and video_temporal_patch_size (temporal) are
    independent divisors; injection must set BOTH for per-patch backbones,
    from either origin (processor instance or explicit --token-factor)."""

    def _cfg(self, ip=None, est=None):
        from types import SimpleNamespace
        epi = {"processor_path": "/m", "max_seq_length": 8192}
        if ip is not None:
            epi["hf_processor_instance"] = SimpleNamespace(
                image_processor=ip, name_or_path="/m", tokenizer=None
            )
        sampler = {"type": "knapsack_packed", "cutoff": 8192}
        if est:
            sampler["length_estimator"] = dict(est)
        return {
            "sampler": sampler,
            "processors": {
                "vision": {"params": {"image_factor": 32, "min_pixels": 1024,
                                      "max_pixels": 589824}},
                "epilogue": {"params": epi},
            },
            "sources": [],
        }

    def test_per_patch_instance_sets_temporal_1(self):
        from types import SimpleNamespace
        cfg = self._cfg(ip=SimpleNamespace(patch_size=16, merge_size=1))
        inject_length_estimator_cfg(cfg)
        est = cfg["sampler"]["length_estimator"]
        assert est["token_factor"] == 16
        assert est["video_temporal_patch_size"] == 1

    def test_offline_token_factor_sets_temporal_1(self):
        # precompute --token-factor path: no processor instance at all
        cfg = self._cfg(ip=None, est={"token_factor": 16})
        inject_length_estimator_cfg(cfg)
        est = cfg["sampler"]["length_estimator"]
        assert est["video_temporal_patch_size"] == 1

    def test_qwen_instance_leaves_defaults(self):
        from types import SimpleNamespace
        cfg = self._cfg(ip=SimpleNamespace(patch_size=16, merge_size=2))
        inject_length_estimator_cfg(cfg)
        est = cfg["sampler"]["length_estimator"]
        assert "token_factor" not in est               # == image_factor: unset
        assert "video_temporal_patch_size" not in est  # keep qwen hash stable

    def test_conflicting_token_factor_raises(self):
        from types import SimpleNamespace
        cfg = self._cfg(ip=SimpleNamespace(patch_size=16, merge_size=2),
                        est={"token_factor": 16})
        with pytest.raises(ValueError, match="different vision backbone"):
            inject_length_estimator_cfg(cfg)


class TestBestFitKnapsack:
    """LF-style best-fit-decreasing: bins top up residual capacity with the
    largest fitting smaller docs -> long-doc bins mix in short docs instead of
    leaving dead space (and single-modality length-cluster bins disappear)."""

    def test_long_bins_topped_up_with_short_docs(self):
        # 3 "videos" (3000) + 30 "image QAs" (100), capacity 8192.
        items = [(3000, i) for i in range(3)] + [(100, 10 + i) for i in range(30)]
        bins = greedy_knapsack(items, 8192)
        by_idx = dict((idx, ln) for ln, idx in items)
        first = bins[0]
        lens = [by_idx[i] for i in first]
        assert lens.count(3000) == 2          # 2 videos seed the bin (3 would overflow)
        assert lens.count(100) == 21          # residual 2192 topped up with QAs
        assert sum(lens) == 8100
        # coverage: nothing dropped, nothing duplicated
        flat = sorted(i for b in bins for i in b)
        assert flat == sorted(idx for _, idx in items)

    def test_oversize_gets_own_bin(self):
        bins = greedy_knapsack([(9000, 0), (100, 1)], 8192)
        assert [0] in bins
        assert [1] in bins or any(1 in b and len(b) == 1 for b in bins)
