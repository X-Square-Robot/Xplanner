# Copyright (c) 2026
"""X-Planner data, training, inference, and evaluation package.

The lightweight top-level namespace exposes shared multimodal constants. Import
the task-specific subpackages directly for data construction, modeling,
training, inference, and rollout evaluation.

Layout::

    x_planner/
      constants.py   shared IGNORE_INDEX / media tags / think block
      data/          event data, context, discovery, and packing utilities
      modeling/      Qwen3.5-VL modeling extensions
      trainer/       launcher (CLI), builders (assembly), trainer
      evaluation/    rollout evaluation and review tools
      tools/         offline CLIs (precompute lengths, packing smoke test)

Run training with ``-m x_planner.trainer.launcher`` (see ``scripts/train/``).

Only the cheap, dependency-free constants are re-exported here; import the
submodules directly for the heavier (torch / transformers / x2robot_dataset_v2)
surfaces so ``import x_planner`` stays light.
"""

from x_planner.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    EMPTY_THINK_BLOCK,
    IGNORE_INDEX,
)

__all__ = [
    "IGNORE_INDEX",
    "DEFAULT_IMAGE_TOKEN",
    "DEFAULT_VIDEO_TOKEN",
    "EMPTY_THINK_BLOCK",
]
