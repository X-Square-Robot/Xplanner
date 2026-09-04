"""Deterministic indexed-JSONL leaf I/O for V5 samples."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .schema_v5 import validate_sample


LEAF_SCHEMA_VERSION = "v10_action_segment_v5_leaf_v2"
_SAFE_PART_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=False
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_part(value: str, field: str) -> str:
    if not isinstance(value, str) or not _SAFE_PART_RE.fullmatch(value):
        raise ValueError(f"unsafe {field} path component: {value!r}")
    return value


@dataclass(frozen=True, order=True, slots=True)
class LeafKey:
    source: str
    memory_variant: str
    category: str
    output_profile: str
    task: str
    split: str

    def __post_init__(self) -> None:
        for field, value in asdict(self).items():
            _safe_part(value, field)

    def relative_path(self) -> Path:
        return Path(
            self.source,
            self.memory_variant,
            self.category,
            self.output_profile,
            self.task,
            self.split,
        )

    @classmethod
    def from_sample(cls, sample: Mapping[str, Any]) -> "LeafKey":
        provenance = sample.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("V5 sample provenance must be an object")
        return cls(
            source=str(sample["source"]),
            memory_variant=str(sample["memory_variant"]),
            category=str(sample["category"]),
            output_profile=str(sample["output_profile_id"]),
            task=str(provenance.get("task_name") or ""),
            split=str(provenance.get("split") or ""),
        )


class IndexedLeafWriter:
    """Write one immutable V5 leaf and its byte-offset index."""

    def __init__(self, root: Path | str, leaf: LeafKey) -> None:
        self.root = Path(root)
        self.leaf = leaf
        self.root.mkdir(parents=True, exist_ok=False)
        self._data: BinaryIO = (self.root / "data.jsonl").open("wb")
        self._index: BinaryIO = (self.root / "data.index").open("wb")
        self._sample_ids: set[str] = set()
        self._episodes: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._canonical_sources: set[str] = set()
        self._memory_pair_eligible: bool | None = None
        self._samples = 0
        self._closed = False

    def write(self, sample: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("cannot write a closed indexed leaf")
        validated = validate_sample(sample)
        observed_leaf = LeafKey.from_sample(validated)
        if observed_leaf != self.leaf:
            raise ValueError(
                f"sample leaf mismatch: expected={self.leaf}, observed={observed_leaf}"
            )
        sample_id = str(validated["sample_id"])
        if not sample_id or sample_id in self._sample_ids:
            raise ValueError(f"duplicate or empty sample_id: {sample_id!r}")
        provenance = validated["provenance"]
        self._canonical_sources.add(
            str(provenance.get("canonical_source") or validated["source"])
        )
        memory_pair_eligible = provenance.get("memory_pair_eligible")
        if not isinstance(memory_pair_eligible, bool):
            raise ValueError(
                "sample provenance.memory_pair_eligible must be boolean"
            )
        if self._memory_pair_eligible is None:
            self._memory_pair_eligible = memory_pair_eligible
        elif self._memory_pair_eligible != memory_pair_eligible:
            raise ValueError(
                "memory_pair_eligible must be identical within an indexed leaf"
            )
        episode_key = str(provenance.get("episode_key") or "")
        if not episode_key:
            raise ValueError("sample provenance is missing episode_key")

        outer = {
            "data_id": sample_id,
            "v5_sample": validated,
            "image": validated["images"],
        }
        offset = self._data.tell()
        self._index.write(struct.pack("<Q", offset))
        self._data.write(_json_bytes(outer) + b"\n")

        episode = self._episodes.get(episode_key)
        if episode is None:
            episode = {
                "episode_key": episode_key,
                "source": self.leaf.source,
                "task": self.leaf.task,
                "split": self.leaf.split,
                "first_sample_index": self._samples,
                "last_sample_index": self._samples,
                "num_samples": 0,
            }
            self._episodes[episode_key] = episode
        episode["last_sample_index"] = self._samples
        episode["num_samples"] += 1
        self._sample_ids.add(sample_id)
        self._samples += 1

    def close(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("indexed leaf is already closed")
        if self._samples <= 0:
            raise ValueError("indexed leaf cannot be empty")
        for handle in (self._data, self._index):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()

        episodes_path = self.root / "episodes.jsonl"
        with episodes_path.open("wb") as handle:
            for episode in self._episodes.values():
                handle.write(_json_bytes(episode) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

        if (self.root / "data.index").stat().st_size != self._samples * 8:
            raise RuntimeError("data.index size does not match the sample count")
        manifest = {
            "schema_version": LEAF_SCHEMA_VERSION,
            "complete": True,
            "source": self.leaf.source,
            "category": self.leaf.category,
            "memory_variant": self.leaf.memory_variant,
            "output_profile_id": self.leaf.output_profile,
            "task_name": self.leaf.task,
            "split": self.leaf.split,
            "jsonl_file": "data.jsonl",
            "index_file": "data.index",
            "episodes_file": "episodes.jsonl",
            "num_samples": self._samples,
            "num_episodes": len(self._episodes),
            "canonical_raw_sources": sorted(self._canonical_sources),
            "memory_pair_eligible": self._memory_pair_eligible,
            "leaf": asdict(self.leaf),
            "data_sha256": _sha256(self.root / "data.jsonl"),
            "index_sha256": _sha256(self.root / "data.index"),
            "episodes_sha256": _sha256(episodes_path),
        }
        manifest_path = self.root / "manifest.json"
        with manifest_path.open("wb") as handle:
            handle.write(_json_bytes(manifest) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(self.root)
        self._closed = True
        return manifest

    def abort(self) -> None:
        for handle in (self._data, self._index):
            if not handle.closed:
                handle.close()
        self._closed = True

    def __enter__(self) -> "IndexedLeafWriter":
        return self

    def __exit__(self, exc_type, _exc, _traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def write_indexed_leaf(
    root: Path | str,
    leaf: LeafKey,
    samples: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    writer = IndexedLeafWriter(root, leaf)
    try:
        for sample in samples:
            writer.write(sample)
        return writer.close()
    except BaseException:
        writer.abort()
        raise


def read_indexed_item(root: Path | str, index: int) -> dict[str, Any]:
    """Random-access one outer row using the little-endian uint64 offset index."""

    root = Path(root)
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise IndexError(index)
    index_path = root / "data.index"
    count = index_path.stat().st_size // 8
    if index >= count:
        raise IndexError(index)
    with index_path.open("rb") as offsets:
        offsets.seek(index * 8)
        payload = offsets.read(8)
    if len(payload) != 8:
        raise EOFError(f"truncated offset at index {index}")
    offset = struct.unpack("<Q", payload)[0]
    with (root / "data.jsonl").open("rb") as data:
        data.seek(offset)
        line = data.readline()
    if not line:
        raise EOFError(f"missing data row at index {index}")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError(f"indexed row {index} is not an object")
    return value


__all__ = [
    "IndexedLeafWriter",
    "LEAF_SCHEMA_VERSION",
    "LeafKey",
    "read_indexed_item",
    "write_indexed_leaf",
]
