"""Atomic physical bucket writer and exact prompt-contract exporter for V5.3."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, TextIO

from .prompt_v5 import prompt_renderer_digest_v53, render_user
from .schema_v5 import dumps_assistant, validate_sample


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AtomicJsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        # Keep sealed JSONL/manifest evidence controller-readable even when
        # the DLC worker UID differs from the workspace UID.
        os.fchmod(descriptor, 0o644)
        self.temporary = Path(temporary)
        self.handle: TextIO = os.fdopen(descriptor, "w", encoding="utf-8")
        self.count = 0
        self.closed = False

    def write(self, value: Mapping[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("writer is closed")
        self.handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.count += 1

    def close(self, *, publish: bool = True) -> None:
        if self.closed:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        if publish:
            os.replace(self.temporary, self.path)
        else:
            try:
                self.temporary.unlink()
            except FileNotFoundError:
                pass
        self.closed = True


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    writer = AtomicJsonlWriter(path)
    try:
        writer.write(value)
        writer.close()
    except BaseException:
        writer.close(publish=False)
        raise


def contract_row(sample: Mapping[str, Any], *, source_record: Any = None) -> dict[str, Any]:
    return {
        "sample_id": sample["sample_id"],
        "training_bucket": sample["training_bucket"],
        "category": sample["category"],
        "context_variant": sample["context_variant"],
        "output_profile_id": sample["output_profile_id"],
        "split": sample["split"],
        "images": sample["images"],
        "source_record": source_record,
        "input_json": {
            "task_instruction": sample["task_instruction"],
            "images": sample["images"],
            "prompt_context": sample["prompt_context"],
            "output_spec": sample["output_spec"],
        },
        "prompt": render_user(dict(sample)),
        "output_json": sample["target"],
        "output_text": dumps_assistant(
            sample["target"], sample["category"], sample["output_spec"]
        ),
        "provenance": sample["provenance"],
    }


class BucketWriter:
    """Write train/test files into independent physical bucket directories."""

    def __init__(self, output_root: Path, *, contract_examples_per_key: int = 1) -> None:
        self.output_root = output_root
        self.contract_examples_per_key = max(0, contract_examples_per_key)
        self.data_writers: dict[tuple[str, str], AtomicJsonlWriter] = {}
        self.contract_writers: dict[str, AtomicJsonlWriter] = {}
        self.contract_counts: Counter[tuple[str, str, str, str, str]] = Counter()
        self.sample_counts: Counter[tuple[str, str]] = Counter()
        self.category_counts: Counter[tuple[str, str]] = Counter()
        self.context_counts: Counter[tuple[str, str]] = Counter()
        self.profile_counts: Counter[tuple[str, str]] = Counter()
        self.closed = False

    def _data(self, bucket: str, split: str) -> AtomicJsonlWriter:
        key = (bucket, split)
        if key not in self.data_writers:
            self.data_writers[key] = AtomicJsonlWriter(
                self.output_root / "buckets" / bucket / f"{split}.jsonl"
            )
        return self.data_writers[key]

    def _contracts(self, bucket: str) -> AtomicJsonlWriter:
        if bucket not in self.contract_writers:
            self.contract_writers[bucket] = AtomicJsonlWriter(
                self.output_root / "buckets" / bucket / "contracts.jsonl"
            )
        return self.contract_writers[bucket]

    def write(self, sample: Mapping[str, Any], *, source_record: Any = None) -> None:
        value = validate_sample(sample)
        bucket = str(value["training_bucket"])
        split = str(value["split"])
        self._data(bucket, split).write({
            "data_id": value["sample_id"],
            "v5_sample": value,
            "image": value["images"],
        })
        self.sample_counts[(bucket, split)] += 1
        self.category_counts[(bucket, str(value["category"]))] += 1
        self.context_counts[(bucket, str(value["context_variant"]))] += 1
        self.profile_counts[(bucket, str(value["output_profile_id"]))] += 1
        contract_key = (
            bucket,
            split,
            str(value["category"]),
            str(value["context_variant"]),
            str(value["output_profile_id"]),
        )
        if self.contract_counts[contract_key] < self.contract_examples_per_key:
            self._contracts(bucket).write(contract_row(value, source_record=source_record))
            self.contract_counts[contract_key] += 1

    def close(self, *, provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("BucketWriter already closed")
        try:
            for writer in self.data_writers.values():
                writer.close()
            for writer in self.contract_writers.values():
                writer.close()
        except BaseException:
            for writer in (*self.data_writers.values(), *self.contract_writers.values()):
                writer.close(publish=False)
            raise
        buckets = sorted({bucket for bucket, _split in self.sample_counts})
        bucket_manifests: dict[str, Any] = {}
        for bucket in buckets:
            root = self.output_root / "buckets" / bucket
            files: dict[str, Any] = {}
            for split in ("train", "test"):
                path = root / f"{split}.jsonl"
                if path.is_file():
                    files[path.name] = {
                        "records": self.sample_counts[(bucket, split)],
                        "sha256": file_sha256(path),
                    }
            contracts = root / "contracts.jsonl"
            if contracts.is_file():
                files[contracts.name] = {
                    "records": sum(
                        count for key, count in self.contract_counts.items() if key[0] == bucket
                    ),
                    "sha256": file_sha256(contracts),
                }
            manifest = {
                "schema_version": "v10_action_segment_v5_3_bucket_v1",
                "complete": True,
                "training_bucket": bucket,
                "records": sum(
                    count for (name, _split), count in self.sample_counts.items() if name == bucket
                ),
                "split_counts": {
                    split: self.sample_counts[(bucket, split)]
                    for split in ("train", "test") if self.sample_counts[(bucket, split)]
                },
                "category_counts": {
                    category: count
                    for (name, category), count in sorted(self.category_counts.items())
                    if name == bucket
                },
                "context_variant_counts": {
                    variant: count
                    for (name, variant), count in sorted(self.context_counts.items())
                    if name == bucket
                },
                "output_profile_counts": {
                    profile: count
                    for (name, profile), count in sorted(self.profile_counts.items())
                    if name == bucket
                },
                "prompt_renderer_sha256": prompt_renderer_digest_v53(),
                "files": files,
                "provenance": dict(provenance or {}),
            }
            atomic_json(root / "manifest.json", manifest)
            bucket_manifests[bucket] = manifest
        root_manifest = {
            "schema_version": "v10_action_segment_v5_3_artifact_v1",
            "complete": True,
            "total_records": sum(self.sample_counts.values()),
            "buckets": {
                bucket: {
                    "records": bucket_manifests[bucket]["records"],
                    "manifest": f"buckets/{bucket}/manifest.json",
                }
                for bucket in buckets
            },
            "prompt_renderer_sha256": prompt_renderer_digest_v53(),
            "provenance": dict(provenance or {}),
        }
        atomic_json(self.output_root / "manifest.json", root_manifest)
        self.closed = True
        return root_manifest

    def abort(self) -> None:
        for writer in (*self.data_writers.values(), *self.contract_writers.values()):
            writer.close(publish=False)
        self.closed = True


__all__ = [
    "AtomicJsonlWriter",
    "BucketWriter",
    "atomic_json",
    "contract_row",
    "file_sha256",
]
