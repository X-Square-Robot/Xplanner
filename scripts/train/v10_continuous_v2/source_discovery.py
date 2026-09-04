"""Root-complete V2 source discovery with stable, source-scoped identities."""

from __future__ import annotations

import json
import os
import glob
import re
import subprocess
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from ..v10_continuous.adapters import EpisodeJob
from .common.hashing import global_episode_key


DEFAULT_CAMERA_MAPPING = (
    ("faceImg", "face_view"),
    ("leftImg", "left_wrist_view"),
    ("rightImg", "right_wrist_view"),
    ("sideImg", "side_view"),
    ("move1Img", "face_view"),
    ("move2Img", "side_view"),
)


@dataclass(frozen=True, slots=True)
class DiscoveredEpisode:
    source_id: str
    episode_key: str
    dataset_name: str
    root: str
    job: EpisodeJob
    discovery_warnings: tuple[str, ...] = field(default_factory=tuple)
    resolved_media_realpath: str = field(default="", repr=False, compare=False)

    @property
    def global_episode_key(self) -> str:
        return global_episode_key(self.source_id, self.episode_key)

    @property
    def media_realpath(self) -> str:
        return self.resolved_media_realpath or os.path.realpath(self.job.episode_dir)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "episode_key": self.episode_key,
            "global_episode_key": self.global_episode_key,
            "dataset_name": self.dataset_name,
            "root": self.root,
            "media_realpath": self.media_realpath,
            "job": self.job.to_dict(),
            "discovery_warnings": list(self.discovery_warnings),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DiscoveredEpisode":
        return cls(
            source_id=str(value["source_id"]),
            episode_key=str(value["episode_key"]),
            dataset_name=str(value.get("dataset_name") or ""),
            root=str(value["root"]),
            job=EpisodeJob.from_dict(value["job"]),
            discovery_warnings=tuple(value.get("discovery_warnings") or ()),
            resolved_media_realpath=str(value.get("media_realpath") or ""),
        )


@dataclass(frozen=True, slots=True)
class AnnotationEntry:
    paths: tuple[str, ...]
    cam_mapping: tuple[tuple[str, str], ...]


def _metadata_path(episode_dir: Path, episode_name: str) -> Path:
    exact = episode_dir / f"{episode_name}.json"
    if exact.is_file():
        return exact
    candidates = sorted(
        path for path in episode_dir.glob("*.json")
        if not path.name.endswith((".meta.json", "_hierarchy.json"))
        and path.name not in {"instruction.json", "instruction_meta.json"}
    )
    return candidates[0] if candidates else exact


def _metadata_path_from_names(
    episode_dir: Path, episode_name: str, filenames: list[str]
) -> Path:
    """Resolve metadata from an existing directory listing without re-scanning."""
    exact_name = f"{episode_name}.json"
    if exact_name in filenames:
        return episode_dir / exact_name
    candidates = sorted(
        name for name in filenames
        if name.endswith(".json")
        and not name.endswith((".meta.json", "_hierarchy.json"))
        and name not in {"instruction.json", "instruction_meta.json"}
    )
    return episode_dir / (candidates[0] if candidates else exact_name)


def _load_annotation_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return set()
    if not isinstance(value, Mapping):
        return set()
    return {str(key) for key, record in value.items() if isinstance(record, Mapping)}


def _rewrite_path(path: str, rewrites: list[Mapping[str, Any]]) -> str:
    for rule in rewrites:
        old = str(rule.get("from") or "").rstrip("/")
        new = str(rule.get("to") or "").rstrip("/")
        if old and new and (path == old or path.startswith(old + "/")):
            return new + path[len(old):]
    return path


def _datalist_index(source: Mapping[str, Any]) -> dict[str, AnnotationEntry]:
    """Map real topic directories to annotation files without loading huge JSON."""
    index: dict[str, AnnotationEntry] = {}
    for datalist_value in source.get("datalists") or ():
        datalist = Path(str(datalist_value))
        if not datalist.is_file():
            continue
        try:
            config = yaml.safe_load(datalist.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        for topic in config.get("dataset_path") or ():
            if not isinstance(topic, Mapping):
                continue
            raw_topic = str(topic.get("path") or "")
            if not raw_topic:
                continue
            rewrites = list(topic.get("path_rewrites") or source.get("path_rewrites") or ())
            topic_path = Path(_rewrite_path(raw_topic, rewrites)).resolve()
            templates = list(
                topic.get("instruction_templates")
                or source.get("instruction_templates")
                or ("{topic_path}/instruction.json",)
            )
            relative_parent = ""
            for root_value in source.get("roots") or ():
                try:
                    relative_parent = topic_path.relative_to(Path(str(root_value)).resolve()).as_posix()
                    break
                except ValueError:
                    continue
            parts = relative_parent.split("/") if relative_parent else topic_path.parts
            paths = tuple(dict.fromkeys(os.path.abspath(str(template).format(
                topic_path=str(topic_path),
                root_relative_parent=relative_parent,
                dataset_name=parts[0] if parts else "",
                topic_id=parts[-1] if parts else "",
            )) for template in templates))
            mapping = tuple(
                (str(key), str(value))
                for key, value in (topic.get("cam_mapping") or {}).items()
            ) or DEFAULT_CAMERA_MAPPING
            index[os.path.realpath(topic_path)] = AnnotationEntry(paths, mapping)
    return index


def _hierarchy_jobs(source: Mapping[str, Any]) -> Iterator[DiscoveredEpisode]:
    source_id = str(source["source_id"])
    for root_value in source.get("roots") or ():
        root = Path(str(root_value)).resolve()
        if not root.is_dir():
            continue
        process = subprocess.Popen(
            ["find", str(root), "-type", "f", "-name", "*_hierarchy.json", "-print"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                hierarchy = Path(line.rstrip("\n"))
                episode_dir = hierarchy.parent
                episode_name = hierarchy.name[:-len("_hierarchy.json")]
                relative = episode_dir.relative_to(root).as_posix()
                dataset_name = relative.split("/", 1)[0]
                metadata = _metadata_path(episode_dir, episode_name)
                instruction = episode_dir / "instruction.json"
                if not instruction.is_file():
                    parent_instruction = episode_dir.parent / "instruction.json"
                    if parent_instruction.is_file():
                        instruction = parent_instruction
                job = EpisodeJob(
                    source=source_id,
                    kind="collection",
                    episode_key=relative,
                    episode_name=episode_name,
                    topic=str(episode_dir.parent),
                    episode_dir=str(episode_dir),
                    annotation_paths=(str(hierarchy), str(instruction)),
                    hierarchy_path=str(hierarchy),
                    instruction_path=str(instruction),
                    metadata_path=str(metadata),
                    cam_mapping=DEFAULT_CAMERA_MAPPING,
                )
                yield DiscoveredEpisode(source_id, relative, dataset_name, str(root), job)
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if process.poll() is None:
                process.terminate()
            process.wait()


def _flat_root_jobs(source: Mapping[str, Any]) -> Iterator[DiscoveredEpisode]:
    source_id = str(source["source_id"])
    datalist_index = _datalist_index(source)
    for root_value in source.get("roots") or ():
        root = Path(str(root_value)).resolve()
        if not root.is_dir():
            continue
        for directory, child_dirs, filenames in os.walk(root):
            child_dirs.sort()
            filenames.sort()
            if not any(name.endswith(".mp4") for name in filenames):
                continue
            episode_dir = Path(directory)
            episode_name = episode_dir.name
            metadata = _metadata_path(episode_dir, episode_name)
            if not metadata.is_file():
                continue
            relative = episode_dir.relative_to(root).as_posix()
            dataset_name = relative.split("/", 1)[0]
            indexed = datalist_index.get(os.path.realpath(episode_dir.parent))
            warnings: list[str] = []
            if indexed is not None:
                annotation_paths = indexed.paths
                cam_mapping = indexed.cam_mapping
            else:
                parent_relative = episode_dir.parent.relative_to(root).as_posix()
                parts = parent_relative.split("/")
                annotation_paths = tuple(dict.fromkeys(os.path.abspath(str(template).format(
                    topic_path=str(episode_dir.parent),
                    root_relative_parent=parent_relative,
                    dataset_name=parts[0] if parts else dataset_name,
                    topic_id=parts[-1] if parts else "",
                )) for template in source.get("instruction_templates") or ("{topic_path}/instruction.json",)))
                cam_mapping = DEFAULT_CAMERA_MAPPING
                warnings.append("not_in_datalist")
            job = EpisodeJob(
                source=source_id,
                kind="flat",
                episode_key=relative,
                episode_name=episode_name,
                topic=str(episode_dir.parent),
                episode_dir=str(episode_dir),
                annotation_paths=annotation_paths,
                metadata_path=str(metadata),
                cam_mapping=cam_mapping,
            )
            yield DiscoveredEpisode(
                source_id, relative, dataset_name, str(root), job, tuple(warnings)
            )


def _annotation_mirror_jobs(
    config: Mapping[str, Any], *, source_filter: set[str] | None
) -> Iterator[DiscoveredEpisode]:
    """Discover annotation-owned Episodes and mirror their relative media path."""
    annotation_root = Path(str(config.get("annotation_root") or "")).resolve()
    if not annotation_root.is_dir():
        raise FileNotFoundError(f"annotation_root is missing: {annotation_root}")
    sources = [
        source for source in config.get("sources") or ()
        if isinstance(source, Mapping) and source.get("kind") == "annotation_mirror"
    ]
    compiled = [
        (source, re.compile(str(source.get("dir_pattern") or ".*")))
        for source in sources
    ]
    max_depth = int(config.get("topic_max_depth", 4))
    l3_roots = [Path(str(value)).resolve() for value in config.get("l3_roots") or ()]
    instructions = sorted(annotation_root.rglob("instruction.json"))
    seen_realpaths: set[str] = set()
    for instruction in instructions:
        relative_topic = instruction.parent.relative_to(annotation_root)
        if not relative_topic.parts or len(relative_topic.parts) > max_depth + 1:
            continue
        top = relative_topic.parts[0]
        owner = next((source for source, pattern in compiled if pattern.fullmatch(top)), None)
        if owner is None:
            continue
        source_id = str(owner["source_id"])
        if source_filter and source_id not in source_filter:
            continue
        episode_names = sorted(_load_annotation_keys(instruction))
        if not episode_names:
            continue
        media_topic: Path | None = None
        media_root_used: Path | None = None
        for root_pattern in owner.get("media_roots") or ():
            expanded = sorted(glob.glob(str(root_pattern))) or [str(root_pattern)]
            for root_value in expanded:
                root = Path(root_value).resolve()
                candidate = root / relative_topic
                if candidate.is_dir():
                    media_topic, media_root_used = candidate, root
                    break
            if media_topic is not None:
                break
        if media_topic is None or media_root_used is None:
            continue
        annotation_paths = [str(instruction)]
        for l3_root in l3_roots:
            candidate = l3_root / relative_topic / "instruction.json"
            if candidate.is_file():
                annotation_paths.insert(0, str(candidate))
                break
        mapping = tuple(
            (str(key), str(value)) for key, value in (owner.get("cam_mapping") or {}).items()
        ) or DEFAULT_CAMERA_MAPPING
        for episode_name in episode_names:
            episode_dir = media_topic / episode_name
            if not episode_dir.is_dir():
                alternate = media_topic / f"episode_{episode_name}"
                episode_dir = alternate if alternate.is_dir() else episode_dir
            if not episode_dir.is_dir():
                continue
            physical = os.path.realpath(episode_dir)
            if physical in seen_realpaths:
                continue
            seen_realpaths.add(physical)
            episode_relative = (relative_topic / episode_dir.name).as_posix()
            job = EpisodeJob(
                source=source_id,
                kind="flat",
                episode_key=episode_relative,
                episode_name=episode_name,
                topic=str(media_topic),
                episode_dir=str(episode_dir.resolve()),
                annotation_paths=tuple(dict.fromkeys(annotation_paths)),
                metadata_path=str(_metadata_path(episode_dir, episode_name)),
                cam_mapping=mapping,
            )
            yield DiscoveredEpisode(
                source_id=source_id,
                episode_key=episode_relative,
                dataset_name=top,
                root=str(media_root_used),
                job=job,
            )


def _iter_media_dirs(root: Path) -> Iterator[tuple[Path, list[str]]]:
    """Stream ``(directory, filenames)`` for dirs holding mp4/json, one pass.

    ``find`` emits a directory's entries contiguously, so grouping needs only a
    single-directory buffer instead of a full in-memory index.
    """
    process = subprocess.Popen(
        [
            "find", str(root), "-type", "f",
            "(", "-name", "*.mp4", "-o", "-name", "*.json", ")",
            "-printf", "%h\t%f\n",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    current: str | None = None
    names: list[str] = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            directory, _, name = line.rstrip("\n").partition("\t")
            if not directory or not name:
                continue
            if directory != current:
                if current is not None and names:
                    yield Path(current), names
                current, names = directory, []
            names.append(name)
        if current is not None and names:
            yield Path(current), names
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()


def _iter_fixed_depth_dirs(root: Path, depth: int) -> Iterator[tuple[Path, list[str]]]:
    """Enumerate data directories by relative depth, regardless of media layout."""
    process = subprocess.Popen(["find", str(root), "-mindepth", str(depth), "-maxdepth", str(depth), "-type", "d", "-print0"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert process.stdout is not None
    buffer = b""
    try:
        while True:
            chunk = process.stdout.read(1024 * 1024)
            if not chunk: break
            buffer += chunk
            parts = buffer.split(b"\0")
            buffer = parts.pop()
            for raw in parts:
                directory = Path(raw.decode(errors="replace"))
                try: names = [entry.name for entry in os.scandir(directory) if entry.is_file()]
                except OSError: names = []
                yield directory, names
        if buffer:
            directory = Path(buffer.decode(errors="replace"))
            try: names = [entry.name for entry in os.scandir(directory) if entry.is_file()]
            except OSError: names = []
            yield directory, names
    finally:
        process.stdout.close()
        if process.poll() is None: process.terminate()
        process.wait()


def _media_sweep_jobs(
    config: Mapping[str, Any], source: Mapping[str, Any]
) -> Iterator[DiscoveredEpisode]:
    """Discover media Episodes, or every fixed-depth directory when configured.

    Annotation-driven discovery only sees Episodes keyed in the annotation root,
    which silently skips whole trees (e.g. zhengwei/10000 has ~21k Episodes with
    video and zero ``*_hierarchy.json``). This sweep is media-driven, so nothing
    with video can be missed; annotations are then resolved by mirroring the
    relative path into the annotation and L3 roots, falling back to
    Episode-local ``instruction.json`` / ``*_hierarchy.json``.  Fixed-depth
    discovery intentionally includes directories without either signal so the
    inventory remains a complete directory audit.
    """
    source_id = str(source["source_id"])
    annotation_root_value = str(config.get("annotation_root") or "")
    annotation_root = Path(annotation_root_value).resolve() if annotation_root_value else None
    l3_roots = [Path(str(value)).resolve() for value in config.get("l3_roots") or ()]
    strip = int(source.get("annotation_relative_strip", 0))
    mapping = tuple(
        (str(key), str(value)) for key, value in (source.get("cam_mapping") or {}).items()
    ) or DEFAULT_CAMERA_MAPPING
    topic_annotation_cache: dict[str, tuple[str, ...]] = {}

    for root_pattern in source.get("roots") or ():
        for root_value in sorted(glob.glob(str(root_pattern))) or [str(root_pattern)]:
            root = Path(root_value).resolve()
            if not root.is_dir():
                continue
            depth = source.get("episode_depth")
            iterator = _iter_fixed_depth_dirs(root, int(depth)) if depth is not None else _iter_media_dirs(root)
            for episode_dir, filenames in iterator:
                videos = [name for name in filenames if name.endswith(".mp4")]
                if depth is None and not videos:
                    continue
                episode_name = episode_dir.name
                try:
                    relative = episode_dir.relative_to(root)
                except ValueError:
                    continue
                parts = relative.parts
                if len(parts) <= strip:
                    continue
                # Relative path as the annotation roots see it. For the
                # collection root this drops the <batch>/<rd> prefix, which is
                # also what makes the 44 multi-batch XRRD ids collapse to one
                # stable episode_key.
                annotation_relative = Path(*parts[strip:])
                dataset_name = annotation_relative.parts[0]
                topic_relative = annotation_relative.parent

                warnings: list[str] = ["media_sweep"]
                jsons = {name for name in filenames if name.endswith(".json")}
                hierarchy_name = f"{episode_name}_hierarchy.json"
                annotation_paths: list[str] = []

                # Episode-local hierarchy is the richest signal when present.
                kind = "flat"
                hierarchy_path = ""
                instruction_path = ""
                if hierarchy_name in jsons:
                    kind = "collection"
                    hierarchy_path = str(episode_dir / hierarchy_name)
                    annotation_paths.append(hierarchy_path)
                    if "instruction.json" in jsons:
                        instruction_path = str(episode_dir / "instruction.json")
                    elif (episode_dir.parent / "instruction.json").is_file():
                        instruction_path = str(episode_dir.parent / "instruction.json")
                    if instruction_path:
                        annotation_paths.append(instruction_path)
                else:
                    # Topic-level annotation keyed by episode name, mirrored
                    # into the L3 root first so task captions win.
                    topic_key = topic_relative.as_posix()
                    topic_paths = topic_annotation_cache.get(topic_key)
                    if topic_paths is None:
                        resolved_topic_paths: list[str] = []
                        for l3_root in l3_roots:
                            candidate = l3_root / topic_relative / "instruction.json"
                            if candidate.is_file():
                                resolved_topic_paths.append(str(candidate))
                                break
                        if annotation_root is not None:
                            candidate = annotation_root / topic_relative / "instruction.json"
                            if candidate.is_file():
                                resolved_topic_paths.append(str(candidate))
                        topic_paths = tuple(resolved_topic_paths)
                        topic_annotation_cache[topic_key] = topic_paths
                    annotation_paths.extend(topic_paths)
                    if "instruction.json" in jsons:
                        annotation_paths.append(str(episode_dir / "instruction.json"))
                    if not annotation_paths:
                        warnings.append("no_annotation_found")

                metadata = _metadata_path_from_names(episode_dir, episode_name, filenames)
                if metadata.name not in filenames:
                    warnings.append("no_metadata_json")

                job = EpisodeJob(
                    source=source_id,
                    kind=kind,
                    episode_key=annotation_relative.as_posix(),
                    episode_name=episode_name,
                    topic=str(episode_dir.parent),
                    episode_dir=str(episode_dir),
                    annotation_paths=tuple(dict.fromkeys(annotation_paths)),
                    hierarchy_path=hierarchy_path,
                    instruction_path=instruction_path,
                    metadata_path=str(metadata),
                    cam_mapping=mapping,
                )
                yield DiscoveredEpisode(
                    source_id=source_id,
                    episode_key=annotation_relative.as_posix(),
                    dataset_name=dataset_name,
                    root=str(root),
                    job=job,
                    discovery_warnings=tuple(warnings),
                )


def discover_source(
    source: Mapping[str, Any], config: Mapping[str, Any] | None = None
) -> Iterator[DiscoveredEpisode]:
    kind = str(source.get("kind") or "")
    if kind == "hierarchy_root":
        yield from _hierarchy_jobs(source)
    elif kind == "flat_root":
        yield from _flat_root_jobs(source)
    elif kind == "media_sweep":
        yield from _media_sweep_jobs(config or {}, source)
    else:
        raise ValueError(f"unsupported V2 source kind: {kind}")


def discover_episodes(
    config: Mapping[str, Any], *, source_filter: set[str] | None = None, max_episodes: int = 0
) -> Iterator[DiscoveredEpisode]:
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources.yml must contain a non-empty sources list")

    active: list[Iterator[DiscoveredEpisode]] = []
    # annotation_mirror is config-level (it fans out over all such sources at
    # once), so it contributes a single iterator.
    if any(
        isinstance(source, Mapping)
        and source.get("kind") == "annotation_mirror"
        and (
            not source_filter
            or str(source.get("source_id") or "") in source_filter
        )
        for source in sources
    ):
        active.append(iter(_annotation_mirror_jobs(config, source_filter=source_filter)))
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        if source.get("kind") == "annotation_mirror":
            continue
        source_id = str(source.get("source_id") or "")
        if not source_id or (source_filter and source_id not in source_filter):
            continue
        active.append(iter(discover_source(source, config)))

    # Physical identity, not (source_id, path): the same Episode reachable from
    # two roots must be emitted once so shard assignment and dedup stay stable.
    seen_realpaths: set[str] = set()
    emitted = 0
    try:
        while active:
            next_active: list[Iterator[DiscoveredEpisode]] = []
            for iterator in active:
                try:
                    item = next(iterator)
                except StopIteration:
                    continue
                next_active.append(iterator)
                physical = os.path.realpath(item.job.episode_dir)
                if physical in seen_realpaths:
                    continue
                seen_realpaths.add(physical)
                yield replace(item, resolved_media_realpath=physical)
                emitted += 1
                if max_episodes and emitted >= max_episodes:
                    return
            active = next_active
    finally:
        for iterator in active:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
