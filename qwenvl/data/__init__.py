# Copyright (c) 2026
"""Data-pipeline helpers for Qwen3.5-VL SFT (x2robot_dataset_v2 backend).

Modules (imported by path where they pull heavy deps, so ``import qwenvl.data``
stays light and does not require x2robot_dataset_v2 on ``sys.path`` yet):

* :mod:`~qwenvl.data.bad_sample_fallback` -- collate guard for all-bad packed bins;
* :mod:`~qwenvl.data.packing`             -- neat-packing model patch + get_rope_index;
* :mod:`~qwenvl.data.length_estimator`    -- metadata-only per-frame length estimate;
* :mod:`~qwenvl.data.length_samplers`     -- knapsack_packed sampler
  (importing this registers it in the dataset_v2 sampler registry).

Only the lightweight collate guard is re-exported here; the sampler / estimator /
packing modules are imported directly (lazily) by the trainer + tools so their
x2robot_dataset_v2 / transformers imports fire only after sys.path is set up.
"""

from qwenvl.data.bad_sample_fallback import make_bad_sample_fallback_collator

__all__ = [
    "make_bad_sample_fallback_collator",
]
