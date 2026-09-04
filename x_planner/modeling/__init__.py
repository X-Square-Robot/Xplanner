# Copyright (c) 2026
"""Model-side adaptations for Qwen3.5-VL SFT.

Holds the CCE fused loss and the pluggable vision backbone. The neat-packing
monkey-patch + length samplers now live in :mod:`x_planner.data` (data pipeline).
"""

from x_planner.modeling.cce import apply_cce_patch, cce_loss_from_hidden
from x_planner.modeling.vision import (
    build_pluggable_processor,
    build_vision_backbone,
    load_pluggable_qwen35,
)

__all__ = [
    "apply_cce_patch",
    "cce_loss_from_hidden",
    "build_vision_backbone",
    "build_pluggable_processor",
    "load_pluggable_qwen35",
]
