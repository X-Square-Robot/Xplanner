"""Read-only adapter for the fixed final Takeover-Q bilingual review snapshot.

This module deliberately does not consume V2 scanner catalogs.  The generic
scanner cannot resolve Takeover-Q's separate reviewed annotations and its
zero-sample result is not label truth.  Labels come only from the immutable
``new_completed/episodes.jsonl`` index and its referenced ``episodes/*.json``
files.  Task conditioning is deliberately separate from the reviewed action
labels: it is resolved from the exact episode entry in the media-side
``instruction.json``.  The configured field order is data policy, not a
string-quality heuristic, and every emitted row records the source file,
field, hash, and policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

try:
    from .schema_v5 import (
        FAILURE_TYPE_BY_SOURCE_CODE,
        V5ValidationError,
        validate_model_visible_text,
    )
except ImportError:
    from schema_v5 import (
        FAILURE_TYPE_BY_SOURCE_CODE,
        V5ValidationError,
        validate_model_visible_text,
    )


SNAPSHOT_ID = "20260816_232103"
DEFAULT_SNAPSHOT_ROOT = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/luhao/datasets/takeover_q/"
    f"{SNAPSHOT_ID}/rootfs"
)
REVIEWED_ROOT_RELATIVE = Path(
    "mnt/cpfs/zbl-cpfs-new/SHARE/yujiexiao/takeover_q_dataset/"
    "reviewed_bilingual/current/new_completed"
)
DEFAULT_REVIEWED_ROOT = DEFAULT_SNAPSHOT_ROOT / REVIEWED_ROOT_RELATIVE

ANCHOR_STRIDE_FRAMES = 10
ANCHOR_WINDOW_START_NUMERATOR = 7
ANCHOR_WINDOW_START_DENOMINATOR = 10
ANCHOR_SELECTION_POLICY = "q2_late_70_to_100_percent_stride10_v1"
HISTORY_OFFSETS_FRAMES = (-20, -10, 0)
VIEW_KEYS = (
    ("face", "head"),
    ("left", "left_wrist"),
    ("right", "right_wrist"),
)

_FAILURE_CODE_RE = re.compile(r"^\s*(\d+\.\d+)\b")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_EXPECTED_PREFIX_RE = re.compile(r"^\s*expected(?:\s+action)?\s*:\s*", re.I)
_ACTUAL_PREFIX_RE = re.compile(r"^\s*actual(?:\s+failure)?\s*:\s*", re.I)
_RECOVER_BOTH_ARMS_RE = re.compile(r"\brecover both arms\b", re.I)
_NUMBERED_LETTER_TASK_RE = re.compile(
    r"^\s*\d+\s*(?:[-_/]|\s)\s*\d+\s+letters?\s+task\s*$",
    re.I,
)
_UNDERSCORE_TASK_SLUG_RE = re.compile(
    r"^[a-z0-9]+(?:_[a-z0-9]+)+_task$",
    re.I,
)

MODEL_VISIBLE_TEXT_NORMALIZATION_VERSION = (
    "takeover_q_retract_and_exact_instruction_json_v3"
)
TAKEOVER_INSTRUCTION_FIELDS = ("instruction", "detailed_instruction")
TAKEOVER_INSTRUCTION_POLICY = (
    "exact_episode_instruction_json_instruction_then_detailed_with_placeholder_rejection_v1"
)


class TakeoverQDataError(ValueError):
    """The final reviewed snapshot violates the adapter contract."""


class UnparseableQ2Error(TakeoverQDataError):
    """Q2 cannot be deterministically split into expected and actual text."""

    def __init__(
        self,
        detail: str,
        *,
        reason: str = "q2_missing_deterministic_delimiter",
    ) -> None:
        super().__init__(detail)
        self.reason = reason


class ModelVisibleTextError(TakeoverQDataError):
    """A reviewed model-visible text cell violates the V5 text contract."""

    def __init__(self, *, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason


@dataclass(frozen=True)
class VideoMetadata:
    """Original video timing needed to map seconds to source frames."""

    fps: Fraction
    frame_count: int

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError("video fps must be positive")
        if self.frame_count <= 0:
            raise ValueError("video frame_count must be positive")


VideoProbe = Callable[[Path], VideoMetadata]


@dataclass(frozen=True)
class TaskInstructionMetadata:
    """An exact, auditable episode instruction selected from source JSON."""

    text: str
    source: str
    source_path: str
    source_field: str
    source_sha256: str
    checked_paths: tuple[str, ...]
    rejected_candidates: tuple[str, ...]
    policy: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(*parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _english(value: Any, field: str, *, reason_prefix: str | None = None) -> str:
    prefix = reason_prefix or field.replace(".", "_")
    if not isinstance(value, str) or not value.strip():
        raise ModelVisibleTextError(
            reason=f"{prefix}_missing_english_caption",
            detail=f"{field} must be non-empty English text",
        )
    result = value.strip()
    if _CJK_RE.search(result):
        raise ModelVisibleTextError(
            reason=f"{prefix}_contains_cjk",
            detail=f"{field} contains model-visible CJK text",
        )
    if not _LATIN_RE.search(result):
        raise ModelVisibleTextError(
            reason=f"{prefix}_missing_english_caption",
            detail=f"{field} must be non-empty English text",
        )
    # One reviewed task-completion case uses "recover both arms" to mean
    # retracting the manipulators. Recover is a retired V5 decision name,
    # so normalize this exact, unambiguous phrase without editing raw labels.
    def retract(match: re.Match[str]) -> str:
        replacement = "retract both arms"
        return replacement.capitalize() if match.group(0)[0].isupper() else replacement

    result = _RECOVER_BOTH_ARMS_RE.sub(retract, result)
    return result


def _model_visible_english(
    value: Any,
    field: str,
    *,
    reason_prefix: str | None = None,
) -> str:
    """Validate a reviewed cell before it can enter a model-visible field."""

    prefix = reason_prefix or field.replace(".", "_")
    result = _english(value, field, reason_prefix=prefix)
    try:
        return validate_model_visible_text(result, field)
    except V5ValidationError as exc:
        detail = str(exc)
        if "raw failure code" in detail:
            suffix = "contains_raw_failure_code"
        elif "forbidden legacy term" in detail:
            suffix = "contains_forbidden_v5_term"
        else:
            suffix = "violates_v5_text_contract"
        raise ModelVisibleTextError(
            reason=f"{prefix}_{suffix}",
            detail=detail,
        ) from exc


def _task_instruction_placeholder_reason(value: str) -> str | None:
    """Return a versioned, deterministic rejection reason for known non-tasks."""

    stripped = value.strip()
    digits = sum(char.isdigit() for char in stripped)
    letters = sum(char.isalpha() for char in stripped)
    if (
        _NUMBERED_LETTER_TASK_RE.fullmatch(stripped)
        or _UNDERSCORE_TASK_SLUG_RE.fullmatch(stripped)
    ):
        return "task_slug_or_placeholder"
    if (
        len(stripped) >= 24
        and not any(char.isspace() for char in stripped)
        and digits >= 8
        and letters >= 1
        and digits / max(1, len(stripped)) >= 0.4
    ):
        return "compact_mixed_identifier"
    return None


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TakeoverQDataError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise TakeoverQDataError(f"{field} must be finite and non-negative")
    return result


def _window(segment: Mapping[str, Any], field: str) -> dict[str, Any]:
    start = _number(segment.get("start_sec"), f"{field}.start_sec")
    end = _number(segment.get("end_sec"), f"{field}.end_sec")
    if end < start:
        raise TakeoverQDataError(f"{field} ends before it starts")
    result: dict[str, Any] = {"start_sec": start, "end_sec": end}
    if "time_locked" in segment:
        result["time_locked"] = bool(segment["time_locked"])
    return result


def _single_segment(segments: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    values = segments.get(name)
    if not isinstance(values, list) or len(values) != 1:
        raise TakeoverQDataError(f"final case must contain exactly one {name} segment")
    value = values[0]
    if not isinstance(value, Mapping):
        raise TakeoverQDataError(f"{name} segment must be an object")
    return value


def _split_q2(value: Any) -> tuple[str, str]:
    caption = _english(value, "q2q3.q2_caption", reason_prefix="q2")
    if " -> " not in caption:
        raise UnparseableQ2Error(
            "q2_caption lacks the deterministic literal ' -> ' delimiter"
        )
    expected, actual = caption.split(" -> ", 1)
    expected = _EXPECTED_PREFIX_RE.sub("", expected).strip()
    actual = _ACTUAL_PREFIX_RE.sub("", actual).strip()
    try:
        return _model_visible_english(
            expected, "q2 expected action", reason_prefix="q2"
        ), _model_visible_english(
            actual, "q2 observed failure", reason_prefix="q2"
        )
    except ModelVisibleTextError as exc:
        if exc.reason in {
            "q2_contains_forbidden_v5_term",
            "q2_contains_raw_failure_code",
            "q2_violates_v5_text_contract",
        }:
            raise
        raise UnparseableQ2Error(
            "q2_caption has an empty or non-English expected/actual side",
            reason="q2_unusable_expected_or_actual",
        ) from exc


def _failure_type(value: Any) -> tuple[str, str, str]:
    raw = _english(value, "q2q3.q3_type")
    match = _FAILURE_CODE_RE.match(raw)
    if match is None:
        raise TakeoverQDataError("q3_type is missing its offline source code")
    source_code = match.group(1)
    normalized = FAILURE_TYPE_BY_SOURCE_CODE.get(source_code)
    if normalized is None:
        raise TakeoverQDataError(f"unsupported q3 source code: {source_code}")
    return source_code, normalized, raw


def _fraction(value: Any, field: str) -> Fraction:
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise TakeoverQDataError(f"invalid {field}: {value!r}") from exc
    if result <= 0:
        raise TakeoverQDataError(f"{field} must be positive")
    return result


def probe_video(path: Path) -> VideoMetadata:
    """Probe a source video without decoding frames or mutating media."""

    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
    except FileNotFoundError as exc:
        if exc.filename != "ffprobe":
            raise TakeoverQDataError(f"ffprobe failed for {path}: {exc}") from exc
        try:
            from .video_probe_v5 import probe_video_pyav
            fps, frame_count = probe_video_pyav(path)
        except Exception as fallback_exc:
            raise TakeoverQDataError(
                f"ffprobe is unavailable and PyAV metadata probe failed for {path}: "
                f"{fallback_exc}"
            ) from fallback_exc
        return VideoMetadata(fps=fps, frame_count=frame_count)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TakeoverQDataError(f"ffprobe failed for {path}: {exc}") from exc
    try:
        payload = json.loads(completed.stdout)
        stream = payload["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
        raise TakeoverQDataError(f"ffprobe returned no video stream for {path}") from exc
    raw_rate = stream.get("avg_frame_rate")
    if raw_rate in (None, "0/0", "N/A"):
        raw_rate = stream.get("r_frame_rate")
    fps = _fraction(raw_rate, "video fps")
    raw_frames = stream.get("nb_frames")
    if raw_frames not in (None, "N/A"):
        try:
            frame_count = int(raw_frames)
        except (TypeError, ValueError) as exc:
            raise TakeoverQDataError(
                f"invalid video frame count for {path}: {raw_frames!r}"
            ) from exc
    else:
        duration = _number(stream.get("duration"), "video duration")
        frame_count = int(round(duration * float(fps)))
    return VideoMetadata(fps=fps, frame_count=frame_count)


def _frame_at_or_after(seconds: float, fps: Fraction) -> int:
    value = Fraction(str(seconds)) * fps
    return -(-value.numerator // value.denominator)


def _frame_at_or_before(seconds: float, fps: Fraction) -> int:
    value = Fraction(str(seconds)) * fps
    return value.numerator // value.denominator


def _anchor_frame_bounds(
    window: Mapping[str, Any], video: VideoMetadata
) -> tuple[int, int, int]:
    """Return inclusive Q2 bounds and the inclusive late-window start.

    The supervision point is restricted to the final 30% of the reviewed Q2
    failure window.  Integer ceiling is used so an anchor can never fall
    before the exact 70% boundary, including for fractional frame rates.
    """

    start = _frame_at_or_after(float(window["start_sec"]), video.fps)
    end = _frame_at_or_before(float(window["end_sec"]), video.fps)
    end = min(end, video.frame_count - 1)
    if start > end:
        raise TakeoverQDataError(
            f"label window contains no source frame: start={start} end={end}"
        )
    span = end - start
    offset_numerator = ANCHOR_WINDOW_START_NUMERATOR * span
    offset = -(-offset_numerator // ANCHOR_WINDOW_START_DENOMINATOR)
    late_start = start + offset
    return start, late_start, end


def _anchor_frames(window: Mapping[str, Any], video: VideoMetadata) -> tuple[int, ...]:
    _start, late_start, end = _anchor_frame_bounds(window, video)
    anchors = tuple(range(late_start, end + 1, ANCHOR_STRIDE_FRAMES))
    # A valid inclusive window always produces at least one point, but retain
    # the terminal-frame fallback as a defensive invariant if stride handling
    # is ever changed.
    return anchors or (end,)


def _context_frames(anchor: int, window_start: int) -> tuple[int, ...]:
    return tuple(
        dict.fromkeys(max(window_start, anchor + offset) for offset in HISTORY_OFFSETS_FRAMES)
    )


class TakeoverQAdapter:
    """Stream no-memory Takeover candidates from final review files.

    Model-visible label defects are recorded in :attr:`exclusions` and do not
    block later cases.  Q1 is intentionally outside the V5 training source;
    Q2/Q4/recovery defects remove the corresponding Takeover candidates.
    Structural/index/media violations, including a missing Q4 segment, remain
    fatal.  ``strict_q2`` is retained only for caller compatibility; reviewed
    text defects are always exclusions because no label may be guessed.
    """

    def __init__(
        self,
        *,
        snapshot_id: str = SNAPSHOT_ID,
        snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
        reviewed_root: Path | None = None,
        video_probe: VideoProbe = probe_video,
        instruction_fields: Sequence[str] = TAKEOVER_INSTRUCTION_FIELDS,
    ) -> None:
        if not isinstance(snapshot_id, str) or not snapshot_id.strip():
            raise ValueError("snapshot_id must be a non-empty string")
        self.snapshot_id = snapshot_id.strip()
        self.snapshot_root = Path(snapshot_root)
        self.reviewed_root = (
            Path(reviewed_root)
            if reviewed_root is not None
            else self.snapshot_root / REVIEWED_ROOT_RELATIVE
        )
        self.index_path = self.reviewed_root / "episodes.jsonl"
        self.video_probe = video_probe
        fields = tuple(dict.fromkeys(instruction_fields))
        if not fields or any(
            field not in {"detailed_instruction", "instruction"} for field in fields
        ):
            raise ValueError(
                "instruction_fields must be a non-empty ordered subset of "
                "('detailed_instruction', 'instruction')"
            )
        self.instruction_fields = fields
        self.instruction_policy = (
            TAKEOVER_INSTRUCTION_POLICY
            if fields == TAKEOVER_INSTRUCTION_FIELDS
            else "exact_episode_instruction_json_fields_"
            + "_then_".join(fields)
            + "_with_placeholder_rejection_v1"
        )
        self._video_cache: dict[Path, VideoMetadata] = {}
        self._instruction_json_cache: dict[Path, tuple[Mapping[str, Any], str]] = {}
        self._index_sha256: str | None = None
        self._exclusions: list[dict[str, Any]] = []

    @property
    def exclusions(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(value) for value in self._exclusions)

    def _exclude(
        self,
        *,
        index_row: Mapping[str, Any],
        case: Mapping[str, Any],
        episode_sha256: str,
        reason: str,
        detail: str,
        source_label_id: str | None = None,
    ) -> None:
        value: dict[str, Any] = {
            "schema_version": "v5_takeover_adapter_exclusion_v1",
            "episode_key": str(index_row["episode_key"]),
            "episode_id": str(index_row["episode_id"]),
            "case_id": str(case["case_id"]),
            "reason": reason,
            "detail": detail,
            "reviewed_episode_sha256": episode_sha256,
        }
        if source_label_id is not None:
            value["source_label_id"] = source_label_id
        self._exclusions.append(value)

    def _index_digest(self) -> str:
        if self._index_sha256 is None:
            self._index_sha256 = _sha256(self.index_path)
        return self._index_sha256

    def _reviewed_child(self, relative: Any) -> Path:
        if not isinstance(relative, str) or not relative:
            raise TakeoverQDataError("episode index has no reviewed episode path")
        value = Path(relative)
        if value.is_absolute() or ".." in value.parts:
            raise TakeoverQDataError(f"unsafe reviewed episode path: {relative!r}")
        path = self.reviewed_root / value
        if not path.is_file():
            raise TakeoverQDataError(f"reviewed episode file is missing: {path}")
        return path

    def _snapshot_video(self, raw_path: Any) -> tuple[Path, str]:
        if not isinstance(raw_path, str) or not raw_path:
            raise TakeoverQDataError("episode index contains an invalid video path")
        original = Path(raw_path)
        if not original.is_absolute() or ".." in original.parts:
            raise TakeoverQDataError(f"video path must be absolute and normalized: {raw_path!r}")
        snapshot = self.snapshot_root.joinpath(*original.parts[1:])
        if not snapshot.is_file():
            raise TakeoverQDataError(f"snapshot video is missing: {snapshot}")
        return snapshot, raw_path

    def _snapshot_absolute(self, raw_path: Any, field: str) -> Path:
        if not isinstance(raw_path, str) or not raw_path:
            raise TakeoverQDataError(f"episode index contains an invalid {field}")
        original = Path(raw_path)
        if not original.is_absolute() or ".." in original.parts:
            raise TakeoverQDataError(
                f"{field} must be absolute and normalized: {raw_path!r}"
            )
        return self.snapshot_root.joinpath(*original.parts[1:])

    def _load_instruction_json(self, path: Path) -> tuple[Mapping[str, Any], str]:
        cached = self._instruction_json_cache.get(path)
        if cached is not None:
            return cached
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TakeoverQDataError(
                f"cannot read task instruction JSON {path}: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise TakeoverQDataError(f"task instruction JSON root is not an object: {path}")
        result = (payload, _sha256(path))
        self._instruction_json_cache[path] = result
        return result

    def resolve_task_instruction(
        self, index_row: Mapping[str, Any]
    ) -> TaskInstructionMetadata:
        """Resolve only the exact episode record from corresponding source JSON."""

        episode_id = index_row.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise TakeoverQDataError("episode index has no valid episode_id")
        raw_episode = self._snapshot_absolute(
            index_row.get("raw_episode_dir"), "raw_episode_dir"
        )
        candidates = (
            (
                "media_task_instruction_json.episode",
                raw_episode.parent / "instruction.json",
            ),
            (
                "media_episode_instruction_json.episode",
                raw_episode / "instruction.json",
            ),
        )
        checked_paths: list[str] = []
        rejected: list[str] = []
        for source_prefix, path in candidates:
            checked_paths.append(str(path))
            if not path.is_file():
                rejected.append(f"missing {path}")
                continue
            payload, digest = self._load_instruction_json(path)
            record = payload.get(episode_id)
            if not isinstance(record, Mapping):
                rejected.append(f"{path} has no exact object entry for {episode_id!r}")
                continue
            for field in self.instruction_fields:
                try:
                    text = _model_visible_english(
                        record.get(field),
                        f"instruction_json[{episode_id!r}].{field}",
                        reason_prefix=f"task_instruction_json_{field}",
                    )
                except ModelVisibleTextError as exc:
                    rejected.append(f"{path}:{field}: {exc}")
                    continue
                placeholder_reason = _task_instruction_placeholder_reason(text)
                if placeholder_reason is not None:
                    rejected.append(f"{path}:{field}: {placeholder_reason}: {text!r}")
                    continue
                return TaskInstructionMetadata(
                    text=text,
                    source=f"{source_prefix}.{field}",
                    source_path=str(path),
                    source_field=field,
                    source_sha256=digest,
                    checked_paths=tuple(checked_paths),
                    rejected_candidates=tuple(rejected),
                    policy=self.instruction_policy,
                )
        raise ModelVisibleTextError(
            reason="task_instruction_json_no_usable_configured_field",
            detail=(
                f"no exact {episode_id!r} instruction using ordered fields "
                f"{self.instruction_fields!r}; checked={checked_paths!r}; "
                f"rejected={rejected!r}"
            ),
        )

    def _probe(self, path: Path) -> VideoMetadata:
        value = self._video_cache.get(path)
        if value is None:
            value = self.video_probe(path)
            if not isinstance(value, VideoMetadata):
                raise TypeError("video_probe must return VideoMetadata")
            self._video_cache[path] = value
        return value

    def _media(self, index_row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], VideoMetadata, dict[str, str]]:
        videos = index_row.get("videos")
        if not isinstance(videos, Mapping):
            raise TakeoverQDataError("episode index is missing videos")
        media: list[dict[str, Any]] = []
        raw_paths: dict[str, str] = {}
        metadata: list[VideoMetadata] = []
        for raw_view, view in VIEW_KEYS:
            snapshot, raw = self._snapshot_video(videos.get(raw_view))
            info = self._probe(snapshot)
            media.append({"view": view, "video": str(snapshot)})
            raw_paths[view] = raw
            metadata.append(info)
        rates = {value.fps for value in metadata}
        if len(rates) != 1:
            raise TakeoverQDataError(
                f"synchronized Takeover-Q views have different fps: {sorted(map(str, rates))}"
            )
        return media, VideoMetadata(
            fps=metadata[0].fps,
            frame_count=min(value.frame_count for value in metadata),
        ), raw_paths

    @staticmethod
    def _images(
        media: Sequence[Mapping[str, Any]], context_frames: Sequence[int]
    ) -> list[dict[str, Any]]:
        return [
            {"video": str(item["video"]), "frame": frame, "view": str(item["view"])}
            for frame in context_frames
            for item in media
        ]

    def _provenance(
        self,
        *,
        index_row: Mapping[str, Any],
        index_line: int,
        episode_path: Path,
        episode_sha256: str,
        case: Mapping[str, Any],
        raw_paths: Mapping[str, str],
        video: VideoMetadata,
        task_instruction: TaskInstructionMetadata,
    ) -> dict[str, Any]:
        review = case.get("review")
        if not isinstance(review, Mapping) or review.get("review_status") != "completed":
            raise TakeoverQDataError("adapter accepts completed final reviews only")
        return {
            "dataset": "takeover_q",
            "snapshot_id": self.snapshot_id,
            "reviewed_label_source": "final_reviewed_bilingual",
            "reviewed_index": str(self.index_path),
            "reviewed_index_sha256": self._index_digest(),
            "reviewed_index_line": index_line,
            "reviewed_episode": str(episode_path),
            "reviewed_episode_sha256": episode_sha256,
            "episode_key": str(index_row["episode_key"]),
            "episode_id": str(index_row["episode_id"]),
            "case_id": str(case["case_id"]),
            "review_status": "completed",
            "review_revision": review.get("review_revision"),
            "review_updated_at": review.get("review_updated_at"),
            "source_patch_sha256": review.get("source_patch_sha256"),
            "raw_video_paths": dict(raw_paths),
            "fps": float(video.fps),
            "fps_fraction": f"{video.fps.numerator}/{video.fps.denominator}",
            "frame_count_min_across_views": video.frame_count,
            "anchor_stride_frames": ANCHOR_STRIDE_FRAMES,
            "anchor_selection_policy": ANCHOR_SELECTION_POLICY,
            "anchor_window_start_fraction": (
                f"{ANCHOR_WINDOW_START_NUMERATOR}/"
                f"{ANCHOR_WINDOW_START_DENOMINATOR}"
            ),
            "generic_scanner_labels_used": False,
            "model_visible_failure_code": False,
            "memory_variant": "no_memory",
            "model_visible_text_normalization_version": (
                MODEL_VISIBLE_TEXT_NORMALIZATION_VERSION
            ),
            "task_instruction_source": task_instruction.source,
            "task_instruction_source_path": task_instruction.source_path,
            "task_instruction_source_field": task_instruction.source_field,
            "task_instruction_source_sha256": task_instruction.source_sha256,
            "task_instruction_checked_paths": list(task_instruction.checked_paths),
            "task_instruction_rejected_candidates": list(
                task_instruction.rejected_candidates
            ),
            "task_instruction_policy": task_instruction.policy,
        }

    def _record(
        self,
        *,
        decision: str,
        case_id: str,
        segment_id: str,
        anchor: int,
        window: Mapping[str, Any],
        video: VideoMetadata,
        media: Sequence[Mapping[str, Any]],
        instruction: str,
        labels: Mapping[str, Any],
        timing: Mapping[str, Any],
        provenance: Mapping[str, Any],
        label_sources: Mapping[str, str],
    ) -> dict[str, Any]:
        start_frame, late_start_frame, end_frame = _anchor_frame_bounds(window, video)
        if not late_start_frame <= anchor <= end_frame:
            raise TakeoverQDataError(
                "Takeover anchor is outside the causal late-Q2 supervision window: "
                f"anchor={anchor} late_start={late_start_frame} end={end_frame}"
            )
        context = _context_frames(anchor, start_frame)
        kind = decision.lower()
        sample_id = f"v5_{kind}_{_stable_id(self.snapshot_id, case_id, segment_id, anchor, decision, ANCHOR_SELECTION_POLICY)}"
        visible_labels = dict(labels)
        if any(_CJK_RE.search(str(value)) for value in visible_labels.values()):
            raise TakeoverQDataError("model-visible labels contain CJK text")
        return {
            "schema_version": "v5_takeover_adapter_v2",
            "sample_id": sample_id,
            "task_type": "ongoing",
            "decision_class": decision,
            "memory_variant": "no_memory",
            "conditioning": {"task_instruction": instruction},
            "images": self._images(media, context),
            "anchor_frame": anchor,
            "context_frames": list(context),
            "labels": visible_labels,
            "label_timing_sec": dict(timing),
            "supervision": {
                "status": "direct_or_reviewed_derived",
                "label_sources": dict(label_sources),
            },
            "provenance": {
                **dict(provenance),
                "anchor_frame": anchor,
                "anchor_time_sec": float(Fraction(anchor, 1) / video.fps),
                "q2_start_frame": start_frame,
                "q2_late_start_frame": late_start_frame,
                "q2_end_frame": end_frame,
                "anchor_selection_policy": ANCHOR_SELECTION_POLICY,
                "future_takeover_frames_used": False,
            },
        }

    def iter_records(
        self,
        *,
        decisions: Sequence[str] = ("Takeover",),
        strict_q2: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Yield deterministic records and retain any unguessed Q2 exclusions."""

        selected = tuple(dict.fromkeys(decisions))
        unknown = sorted(set(selected) - {"Takeover"})
        if unknown or not selected:
            raise ValueError(f"invalid Takeover-Q decisions: {selected}; unknown={unknown}")
        if not self.index_path.is_file():
            raise FileNotFoundError(f"final reviewed index is missing: {self.index_path}")
        self._exclusions = []
        with self.index_path.open(encoding="utf-8") as index_handle:
            for index_line, line in enumerate(index_handle, 1):
                if not line.strip():
                    continue
                index_row = json.loads(line)
                if not isinstance(index_row, Mapping):
                    raise TakeoverQDataError(f"index line {index_line} is not an object")
                episode_path = self._reviewed_child(index_row.get("episode"))
                with episode_path.open(encoding="utf-8") as episode_handle:
                    episode = json.load(episode_handle)
                if not isinstance(episode, Mapping):
                    raise TakeoverQDataError(f"reviewed episode is not an object: {episode_path}")
                for key in ("episode_key", "episode_id"):
                    if episode.get(key) != index_row.get(key):
                        raise TakeoverQDataError(
                            f"reviewed episode {key} disagrees with index: {episode_path}"
                        )
                cases = episode.get("cases")
                if not isinstance(cases, list) or len(cases) != index_row.get("case_count"):
                    raise TakeoverQDataError(f"case_count mismatch: {episode_path}")
                index_case_ids = index_row.get("case_ids")
                episode_case_ids = episode.get("case_ids")
                if (
                    not isinstance(index_case_ids, list)
                    or not isinstance(episode_case_ids, list)
                    or index_case_ids != episode_case_ids
                ):
                    raise TakeoverQDataError(f"case_ids disagree with index: {episode_path}")
                actual_case_ids = [
                    case.get("case_id") if isinstance(case, Mapping) else None
                    for case in cases
                ]
                if actual_case_ids != episode_case_ids or len(set(actual_case_ids)) != len(
                    actual_case_ids
                ):
                    raise TakeoverQDataError(f"case_ids disagree with cases: {episode_path}")
                media, video, raw_paths = self._media(index_row)
                episode_sha256 = _sha256(episode_path)
                try:
                    task_instruction = self.resolve_task_instruction(index_row)
                except ModelVisibleTextError as exc:
                    for case in cases:
                        if not isinstance(case, Mapping):
                            raise TakeoverQDataError("reviewed case must be an object")
                        self._exclude(
                            index_row=index_row,
                            case=case,
                            episode_sha256=episode_sha256,
                            reason=exc.reason,
                            detail=str(exc),
                        )
                    continue
                for case in cases:
                    if not isinstance(case, Mapping):
                        raise TakeoverQDataError("reviewed case must be an object")
                    case_id = case.get("case_id")
                    if not isinstance(case_id, str) or not case_id:
                        raise TakeoverQDataError("reviewed case has an invalid case_id")
                    if case.get("episode_id") != index_row.get("episode_id"):
                        raise TakeoverQDataError("reviewed case episode_id disagrees with index")
                    bilingual = case.get("bilingual")
                    if not isinstance(bilingual, Mapping):
                        raise TakeoverQDataError("reviewed case is missing bilingual labels")
                    segments = bilingual.get("segments")
                    if not isinstance(segments, Mapping):
                        raise TakeoverQDataError("reviewed case is missing segments")
                    q2q3 = _single_segment(segments, "q2q3")
                    q4 = _single_segment(segments, "q4")
                    takeover = _single_segment(segments, "takeover")
                    q2_window = _window(q2q3, "q2q3")
                    q4_window = _window(q4, "q4")
                    takeover_window = _window(takeover, "takeover")
                    if takeover_window.get("time_locked") is not True:
                        raise TakeoverQDataError("final takeover timing must be time_locked")
                    source_code, failure_type, raw_q3 = _failure_type(q2q3.get("q3_type"))
                    provenance = self._provenance(
                        index_row=index_row,
                        index_line=index_line,
                        episode_path=episode_path,
                        episode_sha256=episode_sha256,
                        case=case,
                        raw_paths=raw_paths,
                        video=video,
                        task_instruction=task_instruction,
                    )

                    if "Takeover" not in selected:
                        continue
                    try:
                        failed_action_context = _model_visible_english(
                            q4.get("caption"),
                            "q4.caption",
                            reason_prefix="q4",
                        )
                        recovery_action = _model_visible_english(
                            takeover.get("caption"),
                            "takeover.caption",
                            reason_prefix="takeover",
                        )
                        expected_action, observed_failure = _split_q2(
                            q2q3.get("q2_caption")
                        )
                    except ModelVisibleTextError as exc:
                        self._exclude(
                            index_row=index_row,
                            case=case,
                            episode_sha256=episode_sha256,
                            reason=exc.reason,
                            detail=str(exc),
                        )
                        continue
                    except UnparseableQ2Error as exc:
                        self._exclude(
                            index_row=index_row,
                            case=case,
                            episode_sha256=episode_sha256,
                            reason=exc.reason,
                            detail=str(exc),
                        )
                        continue
                    takeover_provenance = {
                        **provenance,
                        "raw_failure_source_key": source_code,
                        "raw_q3_type": raw_q3,
                    }
                    timing = {
                        "q2q3": q2_window,
                        "q4": q4_window,
                        "takeover": takeover_window,
                    }
                    for anchor in _anchor_frames(q2_window, video):
                        yield self._record(
                            decision="Takeover",
                            case_id=str(case["case_id"]),
                            segment_id=str(q2q3.get("id") or "q2q3"),
                            anchor=anchor,
                            window=q2_window,
                            video=video,
                            media=media,
                            instruction=task_instruction.text,
                            labels={
                                "execution_decision": "Takeover",
                                "expected_action": expected_action,
                                "observed_failure": observed_failure,
                                "failure_type": failure_type,
                                "failed_action_context": failed_action_context,
                                "recovery_action": recovery_action,
                            },
                            timing=timing,
                            provenance=takeover_provenance,
                            label_sources={
                                "execution_decision": "takeover case membership",
                                "expected_action": "q2q3.q2_caption expected side",
                                "observed_failure": "q2q3.q2_caption actual side",
                                "failure_type": "q2q3.q3_type normalized offline",
                                "failed_action_context": "q4.caption",
                                "recovery_action": "takeover.caption",
                            },
                        )


if len(FAILURE_TYPE_BY_SOURCE_CODE) != 15:
    raise RuntimeError("Takeover-Q failure taxonomy must contain exactly 15 classes")
if len(set(FAILURE_TYPE_BY_SOURCE_CODE.values())) != 15:
    raise RuntimeError("Takeover-Q failure labels must be unique")
if any(_FAILURE_CODE_RE.match(value) for value in FAILURE_TYPE_BY_SOURCE_CODE.values()):
    raise RuntimeError("model-visible Takeover-Q failure labels must be code-free")
