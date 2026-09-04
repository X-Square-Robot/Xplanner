"""Structured scan errors (spec section 9.3)."""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any

from .constants import ERROR_TYPES, is_retryable


class ScanError(Exception):
    """Raised by validation/sampling stages with a classified error type."""

    def __init__(
        self,
        error_type: str,
        message: str = "",
        *,
        stage: str = "",
        view_name: str = "",
        input_paths: tuple[str, ...] = (),
    ) -> None:
        if error_type not in ERROR_TYPES:
            raise ValueError(f"unknown error_type: {error_type}")
        self.error_type = error_type
        self.message = message
        self.stage = stage
        self.view_name = view_name
        self.input_paths = tuple(input_paths)
        super().__init__(f"{error_type}: {message}" if message else error_type)


@dataclass(slots=True)
class ErrorRecord:
    run_id: str
    source_id: str
    episode_key: str
    global_episode_key: str
    stage: str
    error_type: str
    error_message: str
    view_name: str = ""
    traceback: str = ""
    input_paths: tuple[str, ...] = ()
    retryable: bool = False
    worker_id: str = ""
    shard_id: int = -1
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["input_paths"] = list(self.input_paths)
        return value


def error_record(
    exc: BaseException,
    *,
    run_id: str,
    source_id: str,
    episode_key: str,
    global_episode_key: str,
    stage: str,
    worker_id: str = "",
    shard_id: int = -1,
    input_paths: tuple[str, ...] = (),
    include_traceback: bool = True,
) -> ErrorRecord:
    """Normalize any exception into an ErrorRecord."""
    if isinstance(exc, ScanError):
        error_type = exc.error_type
        message = exc.message
        view_name = exc.view_name
        paths = exc.input_paths or input_paths
        resolved_stage = exc.stage or stage
    else:
        error_type = _classify(exc)
        message = str(exc)
        view_name = ""
        paths = input_paths
        resolved_stage = stage
    return ErrorRecord(
        run_id=run_id,
        source_id=source_id,
        episode_key=episode_key,
        global_episode_key=global_episode_key,
        stage=resolved_stage,
        error_type=error_type,
        error_message=message[:2000],
        view_name=view_name,
        traceback="".join(traceback.format_exception(exc))[:8000] if include_traceback else "",
        input_paths=tuple(paths),
        retryable=is_retryable(error_type),
        worker_id=worker_id,
        shard_id=shard_id,
    )


def _classify(exc: BaseException) -> str:
    """Map non-ScanError exceptions onto the taxonomy."""
    import json as _json

    if isinstance(exc, FileNotFoundError):
        return "missing_annotation"
    if isinstance(exc, _json.JSONDecodeError):
        return "annotation_parse_error"
    if isinstance(exc, (KeyError, AttributeError)):
        return "missing_required_field"
    if isinstance(exc, OSError):
        return "write_error"
    if type(exc).__name__ == "EpisodeValidationError":
        reason = validation_reason(exc)
        return reason if reason in ERROR_TYPES else "sampling_error"
    return "unknown_error"


def validation_reason(exc: BaseException) -> str:
    """Extract the v1 EpisodeValidationError reason, when present."""
    reason = getattr(exc, "reason", None)
    return str(reason) if reason else type(exc).__name__
