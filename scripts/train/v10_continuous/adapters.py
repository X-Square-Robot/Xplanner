"""Source discovery and independent adapters for nested and flat annotations."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .captions import is_valid_english_caption, parse_interval_key, select_l3, unique_valid_caption
from .constants import VIEW_ALIASES, VIEW_PRIORITY
from .models import TemporalUnit


class AnnotationValidationError(ValueError):
    """Expected annotation content is absent or unusable for one episode."""


@dataclass(frozen=True, slots=True)
class EpisodeJob:
    source: str
    kind: str
    episode_key: str
    episode_name: str
    topic: str
    episode_dir: str
    annotation_paths: tuple[str, ...]
    hierarchy_path: str = ""
    instruction_path: str = ""
    metadata_path: str = ""
    cam_mapping: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["annotation_paths"] = list(self.annotation_paths)
        value["cam_mapping"] = [list(item) for item in self.cam_mapping]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeJob":
        return cls(
            source=str(value["source"]),
            kind=str(value["kind"]),
            episode_key=str(value["episode_key"]),
            episode_name=str(value["episode_name"]),
            topic=str(value["topic"]),
            episode_dir=str(value["episode_dir"]),
            annotation_paths=tuple(value.get("annotation_paths", ())),
            hierarchy_path=str(value.get("hierarchy_path", "")),
            instruction_path=str(value.get("instruction_path", "")),
            metadata_path=str(value.get("metadata_path", "")),
            cam_mapping=tuple(tuple(item) for item in value.get("cam_mapping", ())),
        )


@dataclass(frozen=True, slots=True)
class AdaptedEpisode:
    task_caption: str
    num_frames: int
    raw_levels: dict[str, tuple[TemporalUnit, ...]]
    video_candidates: dict[str, str]
    annotation_sources: tuple[str, ...]
    metadata: dict[str, Any]


def _l3_candidates(annotation: Mapping[str, Any]) -> dict[str, str]:
    """Keep explicit task fields for downstream conflict auditing."""
    return {
        field_name: caption
        for field_name in ("task_caption", "instruction", "detailed_instruction", "task")
        if (caption := unique_valid_caption(annotation.get(field_name)))
    }


def _rewrite_path(path: str, rewrites: list[Mapping[str, Any]]) -> str:
    for rule in rewrites:
        logical = str(rule.get("from", ""))
        physical = str(rule.get("to", ""))
        if logical and physical and (path == logical or path.startswith(logical.rstrip("/") + "/")):
            return physical.rstrip("/") + path[len(logical.rstrip("/")):]
    return path


@lru_cache(maxsize=128)
def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _annotation_keys(paths: list[str]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if not os.path.isfile(path):
            continue
        value = _load_json(os.path.abspath(path))
        if isinstance(value, Mapping):
            keys.update(str(key) for key, item in value.items() if isinstance(item, Mapping))
    return keys


def _format_instruction_paths(
    templates: list[str], *, topic_path: str, raw_topic: str, rewrites: list[Mapping[str, Any]]
) -> list[str]:
    parts = raw_topic.rstrip("/").split("/")
    robot_id = parts[-2] if len(parts) >= 2 else ""
    topic_id = parts[-1] if parts else ""
    paths = []
    for template in templates:
        rendered = str(template).format(
            topic_path=topic_path,
            robot_id=robot_id,
            topic_id=topic_id,
        )
        paths.append(os.path.abspath(_rewrite_path(rendered, rewrites)))
    return list(dict.fromkeys(paths))


def _episode_dir(topic_path: str, episode_name: str) -> str:
    for name in (episode_name, f"episode_{episode_name}"):
        candidate = os.path.join(topic_path, name)
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(os.path.join(topic_path, episode_name))


def _metadata_path(episode_dir: str, episode_name: str) -> str:
    exact = os.path.join(episode_dir, f"{episode_name}.json")
    if os.path.isfile(exact):
        return exact
    directory_name = os.path.basename(episode_dir)
    alternative = os.path.join(episode_dir, f"{directory_name}.json")
    if os.path.isfile(alternative):
        return alternative
    candidates = sorted(
        path
        for path in Path(episode_dir).glob("*.json")
        if not path.name.endswith((".meta.json", "_hierarchy.json"))
        and path.name not in {"instruction.json", "instruction_meta.json"}
    )
    return str(candidates[0]) if candidates else exact


def _discover_flat(source: Mapping[str, Any]) -> Iterator[EpisodeJob]:
    source_name = str(source["name"])
    rewrites = list(source.get("path_rewrites") or ())
    instruction_rewrites = list(source.get("instruction_path_rewrites") or rewrites)
    default_templates = list(source.get("instruction_templates") or ("{topic_path}/instruction.json",))
    for datalist in source.get("datalists") or ():
        with open(datalist, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        topics = config.get("dataset_path")
        if not isinstance(topics, list):
            raise ValueError(f"dataset_path list missing in {datalist}")
        for topic in topics:
            if not isinstance(topic, Mapping):
                continue
            raw_topic = str(topic.get("path") or "")
            if not raw_topic:
                continue
            topic_path = os.path.abspath(_rewrite_path(raw_topic, rewrites))
            templates = list(topic.get("instruction_templates") or default_templates)
            annotation_paths = _format_instruction_paths(
                templates,
                topic_path=topic_path,
                raw_topic=raw_topic,
                rewrites=instruction_rewrites,
            )
            names = _annotation_keys(annotation_paths)
            whitelist = topic.get("episode_whitelist")
            if whitelist is not None:
                names.intersection_update(str(item) for item in whitelist)
            cam_mapping = tuple(
                (str(key), str(value))
                for key, value in (topic.get("cam_mapping") or {}).items()
            )
            for episode_name in sorted(names):
                episode_dir = _episode_dir(topic_path, episode_name)
                episode_key = f"{source_name}/{raw_topic.strip('/').split('/', 1)[-1]}/{episode_name}"
                yield EpisodeJob(
                    source=source_name,
                    kind="flat",
                    episode_key=episode_key,
                    episode_name=episode_name,
                    topic=raw_topic,
                    episode_dir=episode_dir,
                    annotation_paths=tuple(annotation_paths),
                    metadata_path=_metadata_path(episode_dir, episode_name),
                    cam_mapping=cam_mapping,
                )


def _discover_collection(source: Mapping[str, Any]) -> Iterator[EpisodeJob]:
    source_name = str(source["name"])
    for root_value in source.get("roots") or ():
        root = Path(str(root_value)).resolve()
        if not root.is_dir():
            continue
        for hierarchy in root.rglob("*_hierarchy.json"):
            episode_dir = hierarchy.parent
            episode_name = hierarchy.name[: -len("_hierarchy.json")]
            metadata = episode_dir / f"{episode_name}.json"
            instruction = episode_dir / "instruction.json"
            relative = episode_dir.relative_to(root).as_posix()
            yield EpisodeJob(
                source=source_name,
                kind="collection",
                episode_key=f"{source_name}/{relative}",
                episode_name=episode_name,
                topic=str(episode_dir.parent),
                episode_dir=str(episode_dir),
                annotation_paths=tuple(str(path) for path in (hierarchy, instruction)),
                hierarchy_path=str(hierarchy),
                instruction_path=str(instruction),
                metadata_path=str(metadata),
                cam_mapping=(
                    ("faceImg", "face_view"),
                    ("leftImg", "left_wrist_view"),
                    ("rightImg", "right_wrist_view"),
                    ("sideImg", "side_view"),
                ),
            )


def discover_jobs(config: Mapping[str, Any], max_episodes: int = 0) -> Iterator[EpisodeJob]:
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources config must contain a non-empty sources list")
    iterators: list[Iterator[EpisodeJob]] = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        kind = source.get("kind")
        iterators.append(iter(
            _discover_collection(source) if kind == "collection" else _discover_flat(source)
        ))

    seen_dirs: set[str] = set()
    emitted = 0
    active = list(iterators)
    while active:
        next_active: list[Iterator[EpisodeJob]] = []
        for jobs in active:
            try:
                job = next(jobs)
            except StopIteration:
                continue
            next_active.append(jobs)
            physical = os.path.realpath(job.episode_dir)
            if physical in seen_dirs:
                continue
            seen_dirs.add(physical)
            yield job
            emitted += 1
            if max_episodes and emitted >= max_episodes:
                return
        active = next_active


_TOTAL_RE = re.compile(rb'"total"\s*:\s*(\d+)')


def read_num_frames(path: str) -> int:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as handle:
        prefix = handle.read(256 * 1024)
    match = _TOTAL_RE.search(prefix)
    if match:
        return int(match.group(1))
    value = _load_json(os.path.abspath(path))
    total = value.get("total") if isinstance(value, Mapping) else None
    if isinstance(total, bool) or not isinstance(total, int):
        raise ValueError(f"missing integer total in {path}")
    return total


def _merge_flat_annotation(paths: tuple[str, ...], episode_name: str) -> tuple[dict[str, Any], tuple[str, ...]]:
    loaded: list[tuple[str, Mapping[str, Any]]] = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        root = _load_json(os.path.abspath(path))
        record = root.get(episode_name) if isinstance(root, Mapping) else None
        if isinstance(record, Mapping):
            loaded.append((os.path.abspath(path), record))
    merged: dict[str, Any] = {}
    for _path, record in reversed(loaded):
        merged.update(record)
    return merged, tuple(path for path, _record in loaded)


def _flat_entries(value: Any) -> tuple[list[tuple[int, int, str]], bool]:
    if value is None or value == {} or value == []:
        return [], False
    entries: list[tuple[int, int, str]] = []
    structural_error = False
    if isinstance(value, Mapping):
        iterable = value.items()
        for interval_key, caption in iterable:
            interval = parse_interval_key(interval_key)
            if interval is None:
                structural_error = True
                continue
            entries.append((interval[0], interval[1], caption if isinstance(caption, str) else ""))
    elif isinstance(value, list):
        for item in value:
            if not isinstance(item, Mapping):
                structural_error = True
                continue
            start, end = item.get("start_frame"), item.get("end_frame")
            if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
                structural_error = True
                continue
            caption = item.get("caption_en", item.get("caption", ""))
            entries.append((start, end, caption if isinstance(caption, str) else ""))
    else:
        structural_error = True
    return entries, structural_error


def _flat_level(level: str, value: Any, source: str | None = None) -> tuple[TemporalUnit, ...]:
    entries, structural_error = _flat_entries(value)
    if structural_error:
        return ()
    return tuple(
        TemporalUnit(
            unit_id=f"{level}-{index}",
            level=level,
            caption=caption,
            start_frame=start,
            end_frame=end,
            source=source,
        )
        for index, (start, end, caption) in enumerate(entries)
    )


def _flat_l0(annotation: Mapping[str, Any]) -> tuple[TemporalUnit, ...]:
    human, human_structural_error = _flat_entries(annotation.get("human_segment_caption"))
    segment, segment_structural_error = _flat_entries(annotation.get("segment_caption"))
    if human_structural_error:
        human = []
    if segment_structural_error and not human:
        return ()
    human_by_interval = {(start, end): caption for start, end, caption in human}
    segment_by_interval = {(start, end): caption for start, end, caption in segment}
    intervals = sorted(set(human_by_interval) | set(segment_by_interval))
    units = []
    for index, (start, end) in enumerate(intervals):
        human_caption = human_by_interval.get((start, end), "")
        segment_caption = segment_by_interval.get((start, end), "")
        if is_valid_english_caption(human_caption):
            caption, source = human_caption, "human_segment"
        elif is_valid_english_caption(segment_caption):
            caption, source = segment_caption, "segment"
        else:
            caption, source = "", "segment"
        units.append(TemporalUnit(
            unit_id=f"L0-{index}",
            level="L0",
            caption=caption,
            start_frame=start,
            end_frame=end,
            source=source,
        ))
    return tuple(units)


def _normalize_view(value: str) -> str | None:
    if value in VIEW_ALIASES:
        return VIEW_ALIASES[value]
    stem = Path(value).stem
    return VIEW_ALIASES.get(stem)


def _video_candidates(episode_dir: str, mapping: tuple[tuple[str, str], ...]) -> dict[str, str]:
    candidates: dict[str, str] = {}
    for file_key, raw_view in mapping:
        view = _normalize_view(raw_view) or _normalize_view(file_key)
        if not view:
            continue
        filename = file_key if file_key.endswith(".mp4") else f"{file_key}.mp4"
        candidates.setdefault(view, os.path.abspath(os.path.join(episode_dir, filename)))
    if not candidates:
        for stem, raw_view in (
            ("faceImg", "head"),
            ("camera_head", "head"),
            ("camera_left_wrist", "left_wrist"),
            ("leftImg", "left_wrist"),
            ("camera_right_wrist", "right_wrist"),
            ("rightImg", "right_wrist"),
            ("sideImg", "side"),
            ("camera_side", "side"),
        ):
            path = os.path.abspath(os.path.join(episode_dir, f"{stem}.mp4"))
            if os.path.isfile(path):
                candidates.setdefault(raw_view, path)
    return {view: candidates[view] for view in VIEW_PRIORITY if view in candidates}


def _adapt_flat(job: EpisodeJob) -> AdaptedEpisode:
    if not os.path.isdir(job.episode_dir):
        raise FileNotFoundError(job.episode_dir)
    annotation, used_paths = _merge_flat_annotation(job.annotation_paths, job.episode_name)
    if not annotation:
        raise AnnotationValidationError("missing episode annotation")
    return AdaptedEpisode(
        task_caption=select_l3(annotation),
        num_frames=read_num_frames(job.metadata_path),
        raw_levels={
            "L2": _flat_level("L2", annotation.get("subtask_caption")),
            "L1": _flat_level("L1", annotation.get("action_caption")),
            "L0": _flat_l0(annotation),
        },
        video_candidates=_video_candidates(job.episode_dir, job.cam_mapping),
        annotation_sources=used_paths + (os.path.abspath(job.metadata_path),),
        metadata={"adapter": "flat", "l3_candidates": _l3_candidates(annotation)},
    )


def _collection_instruction(job: EpisodeJob) -> tuple[dict[str, Any], tuple[str, ...]]:
    if not os.path.isfile(job.instruction_path):
        return {}, ()
    root = _load_json(os.path.abspath(job.instruction_path))
    if not isinstance(root, Mapping):
        return {}, (os.path.abspath(job.instruction_path),)
    record = root.get(job.episode_name)
    if not isinstance(record, Mapping) and len(root) == 1:
        only = next(iter(root.values()))
        record = only if isinstance(only, Mapping) else None
    return dict(record or {}), (os.path.abspath(job.instruction_path),)


def _adapt_collection(job: EpisodeJob) -> AdaptedEpisode:
    if not os.path.isdir(job.episode_dir):
        raise FileNotFoundError(job.episode_dir)
    hierarchy = _load_json(os.path.abspath(job.hierarchy_path))
    if not isinstance(hierarchy, Mapping):
        raise ValueError("hierarchy root is not an object")
    instruction, instruction_sources = _collection_instruction(job)
    l2_units: list[TemporalUnit] = []
    l1_units: list[TemporalUnit] = []
    l0_units: list[TemporalUnit] = []
    subtasks = hierarchy.get("subtasks")
    if not isinstance(subtasks, list):
        subtasks = []
    for l2_index, subtask in enumerate(subtasks):
        if not isinstance(subtask, Mapping):
            continue
        l2_units.append(TemporalUnit(
            unit_id=f"L2-{l2_index}",
            level="L2",
            caption=subtask.get("caption_en") if isinstance(subtask.get("caption_en"), str) else "",
            start_frame=subtask.get("start_frame"),
            end_frame=subtask.get("end_frame"),
        ))
        actions = subtask.get("actions")
        if not isinstance(actions, list):
            actions = []
        for action in actions:
            if not isinstance(action, Mapping):
                continue
            l1_index = len(l1_units)
            l1_units.append(TemporalUnit(
                unit_id=f"L1-{l1_index}",
                level="L1",
                caption=action.get("caption_en") if isinstance(action.get("caption_en"), str) else "",
                start_frame=action.get("start_frame"),
                end_frame=action.get("end_frame"),
            ))
            segments = action.get("segment_details")
            if not isinstance(segments, list):
                segments = []
            for segment in segments:
                if not isinstance(segment, Mapping):
                    continue
                l0_index = len(l0_units)
                l0_units.append(TemporalUnit(
                    unit_id=f"L0-{l0_index}",
                    level="L0",
                    caption=segment.get("caption_en") if isinstance(segment.get("caption_en"), str) else "",
                    start_frame=segment.get("start_frame"),
                    end_frame=segment.get("end_frame"),
                    source="segment",
                ))
    return AdaptedEpisode(
        task_caption=select_l3(instruction),
        num_frames=read_num_frames(job.metadata_path),
        raw_levels={"L2": tuple(l2_units), "L1": tuple(l1_units), "L0": tuple(l0_units)},
        video_candidates=_video_candidates(job.episode_dir, job.cam_mapping),
        annotation_sources=(os.path.abspath(job.hierarchy_path),) + instruction_sources + (os.path.abspath(job.metadata_path),),
        metadata={
            "adapter": "collection",
            "l3_candidates": _l3_candidates(instruction),
            "external_validation": hierarchy.get("metadata", {}).get("validation")
            if isinstance(hierarchy.get("metadata"), Mapping)
            else None,
        },
    )


def adapt_job(job: EpisodeJob) -> AdaptedEpisode:
    if job.kind == "collection":
        return _adapt_collection(job)
    if job.kind == "flat":
        return _adapt_flat(job)
    raise ValueError(f"unsupported adapter kind: {job.kind}")
