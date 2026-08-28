# Copyright (c) 2026
"""Training entry point + Trainer for Qwen3.5-VL full SFT.

* :mod:`qwenvl.train.launcher`  -- ``-m qwenvl.train.launcher`` CLI / ``train()``;
* :mod:`qwenvl.train.builders`  -- model + dataset_v2 assembly helpers;
* :mod:`qwenvl.train.trainer`   -- :class:`QwenVLTrainer` (decoupled LRs + sampler).

The Trainer is re-exported here for convenience; the launcher is intentionally
*not* imported at package level so ``import qwenvl.train.trainer`` stays light.
"""

from qwenvl.train.trainer import QwenVLTrainer

__all__ = ["QwenVLTrainer"]
