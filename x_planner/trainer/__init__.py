# Copyright (c) 2026
"""Training entry point + Trainer for Qwen3.5-VL full SFT.

* :mod:`x_planner.trainer.launcher`  -- ``-m x_planner.trainer.launcher`` CLI / ``train()``;
* :mod:`x_planner.trainer.builders`  -- model + dataset_v2 assembly helpers;
* :mod:`x_planner.trainer.trainer`   -- :class:`QwenVLTrainer` (decoupled LRs + sampler).

The Trainer is re-exported here for convenience; the launcher is intentionally
*not* imported at package level so ``import x_planner.trainer.trainer`` stays light.
"""

from x_planner.trainer.trainer import QwenVLTrainer

__all__ = ["QwenVLTrainer"]
