# Copyright (c) 2026
"""Qwen3.5-VL supervised fine-tuning package.

Self-contained training code aligned to transformers ``qwen3_5``
(``Qwen3_5ForConditionalGeneration``) + qwen-vl-utils. Independent from the
legacy ``penguinvl`` model code.

Layout::

    qwenvl/
      constants.py   shared IGNORE_INDEX / media tags / think block
      data/          legacy JSONL dataset + collator (reference backend)
      model/         neat-packing patch for the hybrid GDN backbone
      sampling/      length estimator + length-balanced packing samplers
      train/         launcher (CLI), builders (assembly), trainer
      tools/         offline CLIs (precompute lengths, packing smoke test)

Run training with ``-m qwenvl.train.launcher`` (see ``scripts/train/``).

Only the cheap, dependency-free constants are re-exported here; import the
submodules directly for the heavier (torch / transformers / x2robot_dataset_v2)
surfaces so ``import qwenvl`` stays light.
"""

from qwenvl.constants import (
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
