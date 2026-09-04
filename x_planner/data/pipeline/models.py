"""Serializable data contracts shared by scan, train, validation and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TemporalUnit:
    unit_id: str
    level: str
    caption: str
    start_frame: int
    end_frame: int
    source: str | None = None
    parent_id: str | None = None

    @property
    def duration(self) -> int:
        return self.end_frame - self.start_frame

    def contains_frame(self, frame: int) -> bool:
        return self.start_frame <= frame < self.end_frame

    def contains_unit(self, child: "TemporalUnit") -> bool:
        return (
            self.start_frame <= child.start_frame
            and child.end_frame <= self.end_frame
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TemporalUnit":
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ImageRef:
    view: str
    video: str
    frame: int
    relative_frame: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ImageRef":
        return cls(**value)


@dataclass(frozen=True, slots=True)
class CanonicalEpisode:
    source: str
    episode_key: str
    episode_name: str
    split: str
    num_frames: int
    task_caption: str
    profile: str
    unit_type: str
    levels: dict[str, tuple[TemporalUnit, ...]]
    videos: dict[str, str]
    annotation_sources: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "episode_key": self.episode_key,
            "episode_name": self.episode_name,
            "split": self.split,
            "num_frames": self.num_frames,
            "task_caption": self.task_caption,
            "profile": self.profile,
            "unit_type": self.unit_type,
            "levels": {
                level: [unit.to_dict() for unit in units]
                for level, units in self.levels.items()
            },
            "videos": dict(self.videos),
            "annotation_sources": list(self.annotation_sources),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CanonicalEpisode":
        levels = {
            level: tuple(TemporalUnit.from_dict(unit) for unit in units)
            for level, units in value["levels"].items()
        }
        return cls(
            source=value["source"],
            episode_key=value["episode_key"],
            episode_name=value["episode_name"],
            split=value["split"],
            num_frames=value["num_frames"],
            task_caption=value["task_caption"],
            profile=value["profile"],
            unit_type=value["unit_type"],
            levels=levels,
            videos=dict(value["videos"]),
            annotation_sources=tuple(value.get("annotation_sources", ())),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class V10Sample:
    sample_id: str
    episode_key: str
    split: str
    profile: str
    unit_type: str
    unit_index: int
    current_frame: int
    task_caption: str
    long_memory: tuple[str, ...]
    images: tuple[ImageRef, ...]
    target: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "episode_key": self.episode_key,
            "split": self.split,
            "profile": self.profile,
            "unit_type": self.unit_type,
            "unit_index": self.unit_index,
            "current_frame": self.current_frame,
            "task_caption": self.task_caption,
            "long_memory": list(self.long_memory),
            "images": [image.to_dict() for image in self.images],
            "target": self.target,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "V10Sample":
        return cls(
            sample_id=value["sample_id"],
            episode_key=value["episode_key"],
            split=value["split"],
            profile=value["profile"],
            unit_type=value["unit_type"],
            unit_index=value["unit_index"],
            current_frame=value["current_frame"],
            task_caption=value["task_caption"],
            long_memory=tuple(value.get("long_memory", ())),
            images=tuple(ImageRef.from_dict(item) for item in value["images"]),
            target=dict(value["target"]),
        )


@dataclass(frozen=True, slots=True)
class ScanFailure:
    source: str
    episode_key: str
    reason: str
    detail: str = ""
    topic: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

