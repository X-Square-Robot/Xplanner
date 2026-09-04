# Copyright (c) 2026
"""Pluggable vision encoder for Qwen3.5-VL SFT (Qwen native default; DINOv3 wired).

Swaps ``model.model.visual`` for a foreign encoder + per-patch MLP projector while
keeping the LM, MRoPE, packing and the x2robot_dataset_v2 image path untouched. Three
modules by concern:

* :mod:`~x_planner.modeling.vision.tower`     -- encoder + projector + tower + build factory;
* :mod:`~x_planner.modeling.vision.processor` -- the HF image processor (data side);
* :mod:`~x_planner.modeling.vision.loader`    -- checkpoint stamp + reload (inference / eval).
"""

from x_planner.modeling.vision.tower import (
    BACKBONES,
    PluggableVisualTower,
    build_projector,
    build_vision_backbone,
    build_vision_backbone_from_config,
)
from x_planner.modeling.vision.processor import (
    PluggableImageProcessor,
    build_pluggable_processor,
)
from x_planner.modeling.vision.loader import (
    load_pluggable_processor,
    load_pluggable_qwen35,
    stamp_pluggable_config,
)

__all__ = [
    "BACKBONES",
    "build_vision_backbone",
    "build_vision_backbone_from_config",
    "PluggableImageProcessor",
    "build_pluggable_processor",
    "load_pluggable_processor",
    "load_pluggable_qwen35",
    "stamp_pluggable_config",
    "PluggableVisualTower",
    "build_projector",
]
