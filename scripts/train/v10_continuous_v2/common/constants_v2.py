"""Constants for the v2 pipeline.

Sampling-critical values are re-exported from the frozen v1 package rather than
redefined, so the two pipelines can never drift.
"""

from __future__ import annotations

from ...v10_continuous.constants import (  # noqa: F401  (re-export)
    DEFAULT_MAX_CAMERA_VIEWS,
    DEFAULT_MAX_VISUAL_INPUTS,
    DEFAULT_MIN_INTERVAL_FRAMES,
    DEFAULT_MIN_L0_COUNT,
    DEFAULT_MIN_L1_COUNT,
    DEFAULT_STRIDE,
    VIEW_ALIASES,
    VIEW_PRIORITY,
)

SCAN_RULE_VERSION_V2 = "v10_v2_scan_20260806_6"

DEFAULT_OUTPUT_ROOT = "/mnt/cpfs/zbl-cpfs-new/USERS/luhao/checkpoint/v10_continuous_v2"
ANNOTATION_ROOT = (
    "/mnt/cpfs/zbl-cpfs-new/open_data/video_caption/video_caption_for_x2_check_v2v3umi"
)

DEFAULT_NUM_SHARDS = 256
DEFAULT_BATCH_FLUSH = 200
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_EPISODE_TIMEOUT_S = 300.0
DEFAULT_ANCHOR_STRIDE = 10
DEFAULT_ANCHOR_QUANTILES = (0.25, 0.5, 0.75)

# Scan stages, used for error attribution.
STAGE_DISCOVER = "discover"
STAGE_VALIDATE = "validate"
STAGE_ADAPT = "adapt"
STAGE_CANONICALIZE = "canonicalize"
STAGE_SAMPLE = "sample"
STAGE_WRITE = "write"

# Error taxonomy (spec section 9.2).
ERROR_TYPES: tuple[str, ...] = (
    "missing_annotation",
    "annotation_parse_error",
    "missing_required_field",
    "invalid_episode_key",
    "missing_view",
    "unknown_view",
    "duplicate_view",
    "view_alias_conflict",
    "view_video_mismatch",
    "view_sync_error",
    "missing_video",
    "video_metadata_error",
    "video_decode_error",
    "empty_video",
    "invalid_timestamp",
    "unit_build_error",
    "memory_build_error",
    "annotation_conflict",
    "profile_unavailable",
    "invalid_num_frames",
    "invalid_l3",
    "invalid_split",
    "missing_l2",
    "missing_l1",
    "missing_l0",
    "level_mismatch",
    "duplicate_unit_id",
    "invalid_caption",
    "invalid_interval_type",
    "interval_out_of_bounds",
    "interval_too_short",
    "same_level_overlap",
    "invalid_l0_source",
    "invalid_parent_relation",
    "parent_without_required_child",
    "no_valid_views",
    "no_mapped_valid_views",
    "current_frame_out_of_bounds",
    "current_unit_not_found",
    "prediction_field_unavailable",
    "invalid_visual_count",
    "current_frame_not_last",
    "anchor_unit_mismatch",
    "no_valid_current_frame",
    "sampling_error",
    "sample_serialize_error",
    "write_error",
    "unknown_error",
)

# Transient/environmental failures are worth another attempt; semantic
# rejections are deterministic and must not be retried.
RETRYABLE_ERROR_TYPES: frozenset[str] = frozenset({
    "missing_video",
    "video_metadata_error",
    "video_decode_error",
    "write_error",
    "unknown_error",
})


def is_retryable(error_type: str) -> bool:
    return error_type in RETRYABLE_ERROR_TYPES
