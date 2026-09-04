"""Read-only adapter for clean Action/Segment rows in the pinned V2 snapshot.

The V2 snapshot contains several legacy hierarchy combinations.  V5 is allowed
to consume only rows whose *anchor itself* is a clean Action or Segment anchor:
Action-only, Segment-only, or Action+Segment.  A nested Action/Segment value in
any rejected higher-level anchor is never projected into V5.

Rows are streamed and converted to canonical ongoing candidates.  Missing
Action or Segment labels are represented explicitly with
``label_available=False``.  ``EpisodeActionPlanCollector`` keeps only compact
per-episode label state, so callers can collect initial-plan candidates without
loading the source JSONL into memory.  A last row with no next label remains an
ongoing candidate; this adapter has no evidence that would justify an End label.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from .schema_v5 import V5ValidationError, validate_model_visible_text
except ImportError:  # pragma: no cover - direct fixture execution fallback
    from schema_v5 import V5ValidationError, validate_model_visible_text


SCHEMA_VERSION = "v5_baseline_canonical_v1"
SNAPSHOT_VERSION = "5bffa78a3d8581aea08d5496"
SNAPSHOT_CONTENT_DIGEST = (
    "6334f4beb17ba47f4a4c71f794c1f5d58c72d1a4bb5cc13a7729604cd917b83a"
)
DEFAULT_SNAPSHOT_ROOT = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/luhao/checkpoint/v10_continuous_v2/"
    "runs/fast-maxdata-z-view-20260807T014550Z/snapshots/"
    f"{SNAPSHOT_VERSION}"
)

# These strings are source-reader implementation details.  They are deliberately
# never copied into a canonical record or its provenance.
_CLEAN_SOURCE_LAYOUT: Mapping[tuple[str, str], tuple[str, str]] = {
    ("L3L1", "action"): ("action", "action"),
    ("L3L0", "segment"): ("segment", "segment"),
    ("L3L1L0", "action"): ("action", "action_and_segment"),
}
_TRAIN_SOURCE_DIRECTORIES = ("L3L1", "L3L0", "L3L1L0")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_SPACE_RE = re.compile(r"\s+")


class BaselineAdapterError(ValueError):
    """The pinned snapshot or a source row violates the adapter contract."""


def _stable_id(*parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def _english(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise BaselineAdapterError(f"{field_name} must be a string")
    result = _SPACE_RE.sub(" ", value).strip()
    if not result or not _LATIN_RE.search(result) or _CJK_RE.search(result):
        raise BaselineAdapterError(f"{field_name} must be non-empty English text")
    return result


def _integer(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BaselineAdapterError(
            f"{field_name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _progress(value: Any, field_name: str) -> int:
    result = _integer(value, field_name)
    if result > 100:
        raise BaselineAdapterError(f"{field_name} must not exceed 100")
    return result


def _unavailable_unit() -> dict[str, Any]:
    return {"label_available": False, "caption": "", "progress_percent": 0}


def _available_unit(raw: Any, field_name: str) -> dict[str, Any]:
    if raw is None:
        return _unavailable_unit()
    if not isinstance(raw, Mapping):
        raise BaselineAdapterError(f"{field_name} must be an object")
    caption = _english(raw.get("caption"), f"{field_name}.caption")
    progress = _progress(raw.get("progress_percent"), f"{field_name}.progress_percent")
    return {
        "label_available": True,
        "caption": caption,
        "progress_percent": progress,
    }


def _prediction(
    raw: Mapping[str, Any] | None,
    *,
    index: int,
    action_source_key: str | None,
    segment_source_key: str | None,
) -> dict[str, Any]:
    role = "current" if index == 1 else "next"
    if raw is None:
        return {
            "index": index,
            "role": role,
            "action": _unavailable_unit(),
            "segment": _unavailable_unit(),
        }
    if raw.get("index") != index:
        raise BaselineAdapterError(
            f"prediction {index} has source index {raw.get('index')!r}"
        )
    allowed_keys = {"index"}
    if action_source_key is not None:
        allowed_keys.add(action_source_key)
    if segment_source_key is not None:
        allowed_keys.add(segment_source_key)
    unexpected = sorted(set(raw) - allowed_keys)
    if unexpected:
        raise BaselineAdapterError(
            f"prediction {index} has fields outside its clean anchor contract: "
            f"{unexpected}"
        )
    result = {
        "index": index,
        "role": role,
        "action": _available_unit(
            raw.get(action_source_key) if action_source_key is not None else None,
            f"prediction {index} action",
        ),
        "segment": _available_unit(
            raw.get(segment_source_key) if segment_source_key is not None else None,
            f"prediction {index} segment",
        ),
    }
    if index == 2:
        for unit_name in ("action", "segment"):
            unit = result[unit_name]
            if unit["label_available"] and unit["progress_percent"] != 0:
                raise BaselineAdapterError(
                    f"next {unit_name} progress must be zero"
                )
    return result


def _images(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise BaselineAdapterError("images must be a non-empty list")
    result: list[dict[str, Any]] = []
    for offset, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise BaselineAdapterError(f"images[{offset}] must be an object")
        video = raw.get("video")
        view = raw.get("view")
        frame = raw.get("frame")
        if not isinstance(video, str) or not video:
            raise BaselineAdapterError(f"images[{offset}].video must be a path")
        if not isinstance(view, str) or not view:
            raise BaselineAdapterError(f"images[{offset}].view must be non-empty")
        frame = _integer(frame, f"images[{offset}].frame")
        result.append({"video": video, "frame": frame, "view": view})
    return result


def _history(value: Any, *, anchor_kind: str) -> dict[str, Any]:
    if not isinstance(value, list):
        raise BaselineAdapterError("long history must be a list")
    long_entries = [
        {"caption": _english(caption, f"long history[{index}]")}
        for index, caption in enumerate(value)
    ]
    if long_entries:
        # The source contract defines these entries as already completed units;
        # 100 is therefore observed completion state, not an invented estimate.
        short = {
            "label_available": True,
            "unit_kind": anchor_kind,
            "caption": long_entries[-1]["caption"],
            "progress_percent": 100,
        }
    else:
        short = {
            "label_available": False,
            "unit_kind": anchor_kind,
            "caption": "",
            "progress_percent": 0,
        }
    return {
        "long": long_entries,
        "short": short,
        "with_memory_eligible": bool(long_entries),
    }


def _validate_canonical_model_text(
    record: Mapping[str, Any],
    *,
    validator: Any = validate_model_visible_text,
) -> None:
    """Apply the exact V5 model-visible text gate before plan collection."""

    validator(str(record.get("task_instruction") or ""), "task_instruction")
    history = record.get("history_material")
    if isinstance(history, Mapping):
        for index, item in enumerate(history.get("long") or ()):
            if isinstance(item, Mapping):
                validator(
                    str(item.get("caption") or ""),
                    f"history_material.long[{index}].caption",
                )
    supervision = record.get("supervision")
    if isinstance(supervision, Mapping):
        for prediction_index, prediction in enumerate(
            supervision.get("predictions") or ()
        ):
            if not isinstance(prediction, Mapping):
                continue
            for unit_name in ("action", "segment"):
                unit = prediction.get(unit_name)
                if isinstance(unit, Mapping) and unit.get("label_available") is True:
                    validator(
                        str(unit.get("caption") or ""),
                        f"predictions[{prediction_index}].{unit_name}.caption",
                    )


def _text_exclusion_reason(error: V5ValidationError) -> str:
    message = str(error)
    if "raw failure code" in message:
        return "model_visible_raw_failure_code"
    if "forbidden legacy term" in message:
        return "model_visible_forbidden_v5_term"
    return "model_visible_invalid_text"


@dataclass(slots=True)
class AdapterStatistics:
    read_rows: int = 0
    emitted_ongoing_rows: int = 0
    excluded_rows: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "read_rows": self.read_rows,
            "emitted_ongoing_rows": self.emitted_ongoing_rows,
            "excluded_rows": dict(sorted(self.excluded_rows.items())),
        }


class BaselineAdapter:
    """Validate the pinned snapshot and stream canonical ongoing candidates."""

    def __init__(
        self,
        *,
        snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
        expected_content_digest: str | None = SNAPSHOT_CONTENT_DIGEST,
        expected_version: str | None = SNAPSHOT_VERSION,
    ) -> None:
        self.snapshot_root = Path(snapshot_root)
        self.expected_content_digest = expected_content_digest
        self.expected_version = expected_version
        self.manifest = self._load_manifest()
        self.statistics = AdapterStatistics()
        # Task, Action, Segment and history captions repeat across millions of
        # frame anchors.  Cache the pure text-gate result per unique string so
        # full scans do not rerun identical regex checks for every anchor.
        self._model_text_cache: dict[str, str | None] = {}

    def _validate_model_text_cached(self, value: str, where: str) -> str:
        if value in self._model_text_cache:
            error = self._model_text_cache[value]
            if error is not None:
                raise V5ValidationError(error)
            return value.strip()
        try:
            result = validate_model_visible_text(value, where)
        except V5ValidationError as exc:
            self._model_text_cache[value] = str(exc)
            raise
        self._model_text_cache[value] = None
        return result

    def _load_manifest(self) -> Mapping[str, Any]:
        path = self.snapshot_root / "manifest.json"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BaselineAdapterError(f"cannot read snapshot manifest {path}: {exc}") from exc
        if not isinstance(manifest, Mapping):
            raise BaselineAdapterError("snapshot manifest must be an object")
        if manifest.get("complete") is not True:
            raise BaselineAdapterError("baseline snapshot is not complete")
        if (
            self.expected_content_digest is not None
            and manifest.get("content_digest") != self.expected_content_digest
        ):
            raise BaselineAdapterError("baseline snapshot content digest is not pinned")
        if self.expected_version is not None and manifest.get("version") != self.expected_version:
            raise BaselineAdapterError("baseline snapshot version is not pinned")
        return manifest

    def _default_paths(self, split: str) -> tuple[Path, ...]:
        if split == "train":
            paths = tuple(
                self.snapshot_root / "train_profiles" / directory / "data.jsonl"
                for directory in _TRAIN_SOURCE_DIRECTORIES
            )
        elif split == "validation":
            paths = (self.snapshot_root / "validation" / "data.jsonl",)
        else:
            raise ValueError("split must be 'train' or 'validation'")
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"baseline source files are missing: {missing}")
        return paths

    def adapt_row(self, row: Any, *, source_line: int) -> dict[str, Any] | None:
        """Convert one row, returning ``None`` for a disallowed anchor.

        This method is public so full/validation streams and fixture audits can
        prove that rejected higher-level anchors are never projected.
        """

        self.statistics.read_rows += 1
        if not isinstance(row, Mapping):
            raise BaselineAdapterError(f"source line {source_line} is not an object")
        sample = row.get("v10_sample")
        if not isinstance(sample, Mapping):
            raise BaselineAdapterError(f"source line {source_line} has no sample object")
        source_shape = (sample.get("profile"), sample.get("unit_type"))
        clean_spec = _CLEAN_SOURCE_LAYOUT.get(source_shape)
        if clean_spec is None:
            self.statistics.excluded_rows["invalid_anchor_kind"] += 1
            return None
        anchor_kind, label_mode = clean_spec
        if row.get("profile") != source_shape[0] or row.get("unit_type") != source_shape[1]:
            raise BaselineAdapterError(
                f"source line {source_line} wrapper and sample metadata disagree"
            )
        if row.get("data_id") != sample.get("sample_id"):
            raise BaselineAdapterError(f"source line {source_line} sample id mismatch")
        if row.get("episode_key") != sample.get("episode_key"):
            raise BaselineAdapterError(f"source line {source_line} episode id mismatch")

        target = sample.get("target")
        if not isinstance(target, Mapping):
            raise BaselineAdapterError(f"source line {source_line} has no target")
        task = target.get("task")
        if not isinstance(task, Mapping):
            raise BaselineAdapterError(f"source line {source_line} has no task target")
        task_instruction = _english(
            sample.get("task_caption"), f"source line {source_line} task instruction"
        )
        if _SPACE_RE.sub(" ", str(task.get("caption", ""))).strip() != task_instruction:
            raise BaselineAdapterError(f"source line {source_line} task captions disagree")
        task_progress = _progress(
            task.get("progress_percent"), f"source line {source_line} task progress"
        )
        raw_predictions = target.get("predictions")
        if not isinstance(raw_predictions, list) or not 1 <= len(raw_predictions) <= 2:
            raise BaselineAdapterError(
                f"source line {source_line} must have one or two predictions"
            )
        action_key = "action" if label_mode in {"action", "action_and_segment"} else None
        segment_key = "l0" if label_mode in {"segment", "action_and_segment"} else None
        by_index: dict[int, Mapping[str, Any]] = {}
        for raw_prediction in raw_predictions:
            if not isinstance(raw_prediction, Mapping):
                raise BaselineAdapterError(
                    f"source line {source_line} prediction must be an object"
                )
            index = raw_prediction.get("index")
            if index not in {1, 2} or index in by_index:
                raise BaselineAdapterError(
                    f"source line {source_line} has invalid prediction indices"
                )
            by_index[index] = raw_prediction
        if 1 not in by_index:
            raise BaselineAdapterError(f"source line {source_line} has no current prediction")
        predictions = [
            _prediction(
                by_index.get(index),
                index=index,
                action_source_key=action_key,
                segment_source_key=segment_key,
            )
            for index in (1, 2)
        ]
        source_sample_id = str(sample.get("sample_id"))
        episode_key = str(sample.get("episode_key"))
        split = str(sample.get("split"))
        anchor_index = _integer(
            sample.get("unit_index"), f"source line {source_line} anchor sequence index"
        )
        anchor_frame = _integer(
            sample.get("current_frame"), f"source line {source_line} anchor frame"
        )
        history = _history(sample.get("long_memory"), anchor_kind=anchor_kind)
        result = {
            "schema_version": SCHEMA_VERSION,
            "record_id": f"v5_baseline_{_stable_id(source_sample_id)}",
            "source": "pinned_complete_baseline",
            "category": "ongoing",
            "canonical_episode_id": episode_key,
            "split": split,
            "task_instruction": task_instruction,
            "anchor_frame": anchor_frame,
            "images": _images(sample.get("images")),
            "history_material": history,
            "supervision": {
                "task_progress_percent": task_progress,
                "predictions": predictions,
                "execution_decision": {"label_available": False, "value": ""},
            },
            "provenance": {
                "dataset": "pinned_complete_baseline",
                "snapshot_version": str(self.manifest.get("version")),
                "snapshot_content_digest": str(self.manifest.get("content_digest")),
                "source_sample_id": source_sample_id,
                "episode_key": episode_key,
                "source_split": split,
                "source_line": source_line,
                "anchor_kind": anchor_kind,
                "anchor_sequence_index": anchor_index,
            },
        }
        try:
            _validate_canonical_model_text(
                result, validator=self._validate_model_text_cached
            )
        except V5ValidationError as exc:
            self.statistics.excluded_rows[_text_exclusion_reason(exc)] += 1
            return None
        self.statistics.emitted_ongoing_rows += 1
        return result

    def iter_ongoing(
        self,
        *,
        split: str = "train",
        source_paths: Sequence[Path] | None = None,
        max_rows: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream canonical ongoing rows from default or explicitly supplied files."""

        if max_rows is not None and max_rows < 0:
            raise ValueError("max_rows must be non-negative")
        paths = tuple(Path(path) for path in source_paths) if source_paths else self._default_paths(split)
        seen = 0
        for path in paths:
            with path.open(encoding="utf-8") as handle:
                for source_line, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    if max_rows is not None and seen >= max_rows:
                        return
                    seen += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise BaselineAdapterError(
                            f"invalid JSON at {path}:{source_line}: {exc}"
                        ) from exc
                    result = self.adapt_row(row, source_line=source_line)
                    if result is not None:
                        yield result

    def iter_initial_plans(
        self,
        *,
        split: str = "train",
        source_paths: Sequence[Path] | None = None,
        max_rows: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        collector = EpisodeActionPlanCollector(
            snapshot_version=str(self.manifest.get("version")),
            snapshot_content_digest=str(self.manifest.get("content_digest")),
        )
        collector.extend(
            self.iter_ongoing(split=split, source_paths=source_paths, max_rows=max_rows)
        )
        yield from collector.iter_records()


@dataclass(slots=True)
class _EpisodePlanState:
    canonical_episode_id: str
    split: str
    task_instruction: str
    earliest_frame: int
    earliest_images: list[dict[str, Any]]
    actions: dict[int, str] = field(default_factory=dict)
    segments: dict[int, dict[str, int]] = field(default_factory=dict)
    source_sample_ids: set[str] = field(default_factory=set)
    conflict: bool = False


def _normalized_caption(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip().casefold().rstrip(".")


class EpisodeActionPlanCollector:
    """Collect compact Action-plan state while arbitrary-order rows stream by."""

    def __init__(self, *, snapshot_version: str, snapshot_content_digest: str) -> None:
        self.snapshot_version = snapshot_version
        self.snapshot_content_digest = snapshot_content_digest
        self._episodes: dict[str, _EpisodePlanState] = {}
        self.excluded_conflicts = 0
        self.excluded_incomplete = 0

    def _put_action(self, state: _EpisodePlanState, index: int, caption: str) -> None:
        old = state.actions.get(index)
        if old is None:
            state.actions[index] = caption
        elif _normalized_caption(old) != _normalized_caption(caption):
            state.conflict = True

    def add(self, record: Mapping[str, Any]) -> None:
        if record.get("category") != "ongoing":
            raise BaselineAdapterError("plan collector accepts ongoing records only")
        provenance = record.get("provenance")
        supervision = record.get("supervision")
        if not isinstance(provenance, Mapping) or not isinstance(supervision, Mapping):
            raise BaselineAdapterError("canonical ongoing record is incomplete")
        if provenance.get("anchor_kind") != "action":
            return
        episode_id = str(record["canonical_episode_id"])
        anchor_index = _integer(
            provenance.get("anchor_sequence_index"), "anchor sequence index"
        )
        anchor_frame = _integer(record.get("anchor_frame"), "anchor frame")
        state = self._episodes.get(episode_id)
        if state is None:
            state = _EpisodePlanState(
                canonical_episode_id=episode_id,
                split=str(record["split"]),
                task_instruction=str(record["task_instruction"]),
                earliest_frame=anchor_frame,
                earliest_images=[dict(value) for value in record["images"]],
            )
            self._episodes[episode_id] = state
        elif (
            state.split != record.get("split")
            or state.task_instruction != record.get("task_instruction")
        ):
            state.conflict = True
        if anchor_frame < state.earliest_frame:
            state.earliest_frame = anchor_frame
            state.earliest_images = [dict(value) for value in record["images"]]
        state.source_sample_ids.add(str(provenance["source_sample_id"]))

        history = record.get("history_material")
        if not isinstance(history, Mapping) or not isinstance(history.get("long"), list):
            raise BaselineAdapterError("action record has no canonical long history")
        for index, item in enumerate(history["long"]):
            if not isinstance(item, Mapping):
                raise BaselineAdapterError("canonical long history item is not an object")
            self._put_action(state, index, str(item["caption"]))

        predictions = supervision.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != 2:
            raise BaselineAdapterError("canonical ongoing record needs two predictions")
        for offset, prediction in enumerate(predictions):
            if not isinstance(prediction, Mapping):
                raise BaselineAdapterError("canonical prediction is not an object")
            action_index = anchor_index + offset
            action = prediction.get("action")
            segment = prediction.get("segment")
            if isinstance(action, Mapping) and action.get("label_available") is True:
                self._put_action(state, action_index, str(action["caption"]))
            if isinstance(segment, Mapping) and segment.get("label_available") is True:
                caption = str(segment["caption"])
                existing = state.segments.setdefault(action_index, {})
                existing[caption] = min(anchor_frame, existing.get(caption, anchor_frame))

    def extend(self, records: Iterable[Mapping[str, Any]]) -> None:
        for record in records:
            self.add(record)

    def iter_records(self) -> Iterator[dict[str, Any]]:
        for episode_id in sorted(self._episodes):
            state = self._episodes[episode_id]
            if state.conflict:
                self.excluded_conflicts += 1
                continue
            indices = sorted(state.actions)
            if not indices or indices != list(range(indices[-1] + 1)):
                self.excluded_incomplete += 1
                continue
            plan: list[dict[str, Any]] = []
            for output_index, source_index in enumerate(indices, 1):
                segments = sorted(
                    state.segments.get(source_index, {}).items(),
                    key=lambda value: (value[1], value[0]),
                )
                plan.append({
                    "index": output_index,
                    "action": {
                        "label_available": True,
                        "caption": state.actions[source_index],
                    },
                    "segments": [
                        {
                            "index": segment_index,
                            "segment": {
                                "label_available": True,
                                "caption": caption,
                            },
                        }
                        for segment_index, (caption, _) in enumerate(segments, 1)
                    ],
                    "segment_labels_available": bool(segments),
                })
            yield {
                "schema_version": SCHEMA_VERSION,
                "record_id": f"v5_baseline_plan_{_stable_id(episode_id)}",
                "source": "pinned_complete_baseline",
                "category": "initial_plan",
                "canonical_episode_id": episode_id,
                "split": state.split,
                "task_instruction": state.task_instruction,
                "anchor_frame": state.earliest_frame,
                "images": state.earliest_images,
                "history_material": {
                    "long": [],
                    "short": {
                        "label_available": False,
                        "unit_kind": "action",
                        "caption": "",
                        "progress_percent": 0,
                    },
                    "with_memory_eligible": False,
                },
                "supervision": {"initial_plan": plan},
                "provenance": {
                    "dataset": "pinned_complete_baseline",
                    "snapshot_version": self.snapshot_version,
                    "snapshot_content_digest": self.snapshot_content_digest,
                    "episode_key": episode_id,
                    "source_split": state.split,
                    "source_sample_count": len(state.source_sample_ids),
                    "anchor_kind": "action",
                },
            }


__all__ = [
    "BaselineAdapter",
    "BaselineAdapterError",
    "DEFAULT_SNAPSHOT_ROOT",
    "EpisodeActionPlanCollector",
    "SCHEMA_VERSION",
    "SNAPSHOT_CONTENT_DIGEST",
    "SNAPSHOT_VERSION",
]
