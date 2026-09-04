# Copyright (c) 2026
"""Shared constants for the Qwen3.5-VL SFT package.

Single source of truth for the label-ignore index, the media placeholder tags,
and the empty ``<think>`` block, so the data pipeline, the packing collator, and
the length estimator can no longer drift apart.
"""

# Label id that cross-entropy ignores (HF convention).
IGNORE_INDEX = -100

# LLaVA-style media placeholders found in the raw ``conversations`` text; mapped,
# in order, to the media listed under the ``image`` / ``video`` keys.
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

# Empty per-turn reasoning block Qwen3.5 emits when ``per_turn_think`` is on; the
# length estimator charges this once per assistant turn.
EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"

__all__ = [
    "IGNORE_INDEX",
    "DEFAULT_IMAGE_TOKEN",
    "DEFAULT_VIDEO_TOKEN",
    "EMPTY_THINK_BLOCK",
]
