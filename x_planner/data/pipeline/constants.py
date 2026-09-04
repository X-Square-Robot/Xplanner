"""Shared V10 constants and profile contracts."""

from __future__ import annotations

SCHEMA_VERSION = "v10_continuous_v1"
SCAN_RULE_VERSION = "v10_scan_20260804_1"

LEVEL_ORDER = ("L2", "L1", "L0")
LEVEL_TO_FIELD = {"L2": "subtask", "L1": "action", "L0": "l0"}
FIELD_TO_LEVEL = {value: key for key, value in LEVEL_TO_FIELD.items()}

PROFILE_FIELDS = {
    "full": ("subtask", "action", "l0"),
    "L3L2L1": ("subtask", "action"),
    "L3L2L0": ("subtask", "l0"),
    "L3L2": ("subtask",),
    "L3L1L0": ("action", "l0"),
    "L3L1": ("action",),
    "L3L0": ("l0",),
}

PROFILE_PRECEDENCE = tuple(PROFILE_FIELDS)
PROFILE_LEVELS = {
    profile: tuple(FIELD_TO_LEVEL[field] for field in fields)
    for profile, fields in PROFILE_FIELDS.items()
}

UNIT_LEVEL = {
    profile: ("L2" if "L2" in PROFILE_LEVELS[profile] else
              "L1" if "L1" in PROFILE_LEVELS[profile] else "L0")
    for profile in PROFILE_FIELDS
}
UNIT_TYPE = {"L2": "subtask", "L1": "action", "L0": "segment"}

VIEW_PRIORITY = ("head", "left_wrist", "right_wrist", "side")
VIEW_ALIASES = {
    "face_view": "head",
    "faceImg": "head",
    "camera_head": "head",
    "head": "head",
    "left_wrist_view": "left_wrist",
    "leftImg": "left_wrist",
    "camera_left_wrist": "left_wrist",
    "left_wrist": "left_wrist",
    "right_wrist_view": "right_wrist",
    "rightImg": "right_wrist",
    "camera_right_wrist": "right_wrist",
    "right_wrist": "right_wrist",
    "side_view": "side",
    "sideImg": "side",
    "camera_side": "side",
    "side": "side",
}

DEFAULT_STRIDE = 10
DEFAULT_MAX_TEMPORAL_STEPS = 3
DEFAULT_MAX_CAMERA_VIEWS = 3
DEFAULT_MAX_VISUAL_INPUTS = 9
DEFAULT_SHORT_MEMORY_K = 1
DEFAULT_MAX_SHORT_MEMORY_K = 2
DEFAULT_VISIBLE_LONG_MEMORY_LIMIT = 8
DEFAULT_MAX_MEMORY_NOISE_PROB = 0.4
DEFAULT_DONE_THRESHOLD = 80
DEFAULT_STABLE_STEPS = 2
DEFAULT_SIMILARITY_THRESHOLD = 0.8
DEFAULT_MIN_L1_COUNT = 3
DEFAULT_MIN_L0_COUNT = 3
DEFAULT_MIN_INTERVAL_FRAMES = 5

