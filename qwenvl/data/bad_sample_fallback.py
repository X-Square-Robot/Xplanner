# Copyright (c) 2026
"""Fallback-substitute collator: keep training alive when a whole packed bin is bad.

x2robot_dataset_v2's ``bad_sample_tolerance`` (``dataset.bad_sample_tolerance`` in
the YAML) drops decode-time-bad samples from a batch instead of crashing -- but a
bin whose **only** doc is bad collates to an *empty* batch and raises
``RuntimeError``, which kills the whole DDP job (one rank dies -> the others hang
on the next collective -> NCCL timeout). Under sequence packing this is a real
edge: a doc with estimated length >= cutoff packs **alone** (``greedy_knapsack``
gives it its own bin), so a single decode-time failure on one big image/video is
fatal.

This wraps ``dataset.collate_fn`` so that when a DataLoader batch is *entirely*
bad, one known-good sample is substituted and the step runs on a short, valid row
instead of raising:

* **DDP-safe** -- the step still happens, so per-rank step counts stay in lockstep
  (the substitution is transparent to the sampler's ``consumed`` bookkeeping).
* **No overflow** -- the substituted row is a single doc <= cutoff, so it can never
  push a packed row past the budget (unlike substituting into a *partially* full
  bin, which is why we only intervene on the all-bad case).
* **No-op otherwise** -- a batch with >=1 good sample passes straight through to the
  upstream collate (which already drops the bad ones and packs the survivors); and
  a passthrough if the running dataset_v2 predates bad-sample tolerance.

The good sample is found lazily (interior points, so a bad head-cluster doesn't
defeat the search) under a wall-clock budget (each probe is a full decode; a
video-heavy dataset must not stall the rank toward the NCCL timeout), cached by
INDEX, then re-fetched fresh on each use -- re-fetching avoids deep-copying a
processed sample dict (which carries heavy, non-copyable refs like the frame
decoder / config), and all-bad bins are rare enough that the extra decode is
negligible.
"""

import logging
import time

logger = logging.getLogger("qwenvl.data.bad_sample_fallback")

_MAX_WARN = 20  # cap the per-worker substitution warnings so a bad source can't spam


class BadSampleFallbackCollator:
    """Picklable collate wrapper (see module docstring for semantics).

    A module-level class rather than a closure: ``import deepspeed`` flips the
    global multiprocessing start method to ``spawn``, under which DataLoader
    workers must pickle the collate callable -- a closure dies with "Can't
    pickle local object" in any entrypoint that doesn't force ``fork`` the way
    ``qwenvl.train.launcher`` does (eval scripts, notebooks).
    """

    def __init__(
        self,
        dataset,
        bad_sample_cls,
        max_probe: int = 64,
        probe_budget_s: float = 120.0,
    ):
        self.dataset = dataset
        self.inner = dataset.collate_fn
        self.bad_sample_cls = bad_sample_cls
        self.max_probe = max_probe
        self.probe_budget_s = probe_budget_s
        self._idx = None
        self._warned = 0

    def _fetch(self, i):
        """Fetch sample ``i``; return the dict, or None if it is bad/errors."""
        try:
            s = self.dataset[i]
        except Exception:
            return None
        return None if isinstance(s, self.bad_sample_cls) else s

    def _find_fallback_idx(self):
        n = len(self.dataset)
        if n <= 0:
            return None
        probes = min(self.max_probe, n)
        t0 = time.monotonic()
        for k in range(probes):
            # Interior points only (never exactly the head): the head cluster is
            # the most likely to share the failure that triggered us.
            i = ((k + 1) * n) // (probes + 1)
            if self._fetch(i) is not None:
                return i
            if time.monotonic() - t0 > self.probe_budget_s:
                logger.warning(
                    "bad-sample fallback: probe budget (%.0fs) exhausted after "
                    "%d probes without a good sample; skipping substitution.",
                    self.probe_budget_s, k + 1,
                )
                return None
        return None

    def __call__(self, batch):
        if batch and all(isinstance(s, self.bad_sample_cls) for s in batch):
            fb = self._fetch(self._idx) if self._idx is not None else None
            if fb is None:  # first time, or the cached index went bad -> (re)find
                self._idx = self._find_fallback_idx()
                fb = self._fetch(self._idx) if self._idx is not None else None
            if fb is not None:
                if self._warned < _MAX_WARN:
                    self._warned += 1
                    logger.warning(
                        "bad-sample fallback: a full DataLoader bin was all-bad; "
                        "substituting good sample idx=%s so the step runs on one "
                        "short valid row (no overflow). See the bad_sample_tolerance "
                        "report for the failed samples.",
                        self._idx,
                    )
                batch = [fb]
            # else: no good sample found -> fall through; inner raises as before.
        return self.inner(batch)


def make_bad_sample_fallback_collator(dataset, *, max_probe: int = 64):
    """Wrap ``dataset.collate_fn`` to survive an all-bad DataLoader batch.

    Returns the wrapped collate callable, or the original ``dataset.collate_fn``
    unchanged if the installed dataset_v2 has no ``_BadSample`` sentinel (nothing
    to guard against).
    """
    try:
        from x2robot_dataset_v2.datasets.x2robot_dataset import _BadSample
    except Exception:
        return dataset.collate_fn  # older dataset_v2 -> nothing to guard against
    return BadSampleFallbackCollator(dataset, _BadSample, max_probe=max_probe)


__all__ = ["BadSampleFallbackCollator", "make_bad_sample_fallback_collator"]
