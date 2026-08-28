"""Compose immutable V5 source builds and verify incremental equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..v10_continuous_v2.common.atomic import atomic_write
from .indexed_io_v5 import LEAF_SCHEMA_VERSION
from .holdout_v5 import (
    Benchmark3Holdout,
    DEFAULT_BENCHMARK3_MANIFEST,
    DEFAULT_BENCHMARK3_SHA256,
    audit_artifact,
)
from .prompt_v5 import prompt_renderer_digest
from .schema_v5 import FAILURE_TYPE_BY_SOURCE_CODE, SNAPSHOT_SCHEMA_VERSION


EXPECTED_SOURCES = ("baseline", "robodojo", "takeover_q")
SOURCE_MANIFEST_SCHEMA = "v10_action_segment_v5_materialization_v3"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _validate_takeover_exclusion_report(
    path: Path,
    *,
    require_complete: bool,
) -> dict[str, Any]:
    report = _load_object(path)
    if report.get("schema_version") != "v5_takeover_q_exclusion_report_v1":
        raise ValueError("invalid Takeover-Q exclusion report schema")
    exclusions = report.get("exclusions")
    if not isinstance(exclusions, list) or any(
        not isinstance(value, Mapping) for value in exclusions
    ):
        raise ValueError("Takeover-Q exclusion report has invalid exclusions")
    if report.get("num_exclusions") != len(exclusions):
        raise ValueError("Takeover-Q exclusion report count mismatch")
    expected_counts = dict(sorted(Counter(
        str(value.get("reason") or "unknown") for value in exclusions
    ).items()))
    if report.get("exclusion_reason_counts") != expected_counts:
        raise ValueError("Takeover-Q exclusion reason counts mismatch")
    if require_complete and report.get("scan_complete") is not True:
        raise ValueError("formal Takeover-Q exclusion report is not scan-complete")
    return report


def _copy_or_link(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def _source_inventory(source_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = source_root / "manifest.json"
    manifest = _load_object(manifest_path)
    if (
        manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA
        or manifest.get("complete") is not True
    ):
        raise ValueError(f"incomplete V5 source materialization: {source_root}")
    source = str(manifest.get("source") or "")
    if source not in EXPECTED_SOURCES:
        raise ValueError(f"unknown V5 materialized source: {source!r}")
    split_metadata = manifest.get("robodojo_official_split")
    if split_metadata is not None:
        if source != "robodojo" or not isinstance(split_metadata, Mapping):
            raise ValueError(
                f"invalid RoboDojo official split metadata: {source_root}"
            )
        relative = Path(str(split_metadata.get("copied_relative_path") or ""))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(
                f"unsafe RoboDojo official split copy path: {relative}"
            )
        copied_split = source_root / relative
        if (
            not copied_split.is_file()
            or _sha256(copied_split) != split_metadata.get("sha256")
        ):
            raise ValueError(
                f"RoboDojo official split copy/digest mismatch: {copied_split}"
            )
    leaves: list[dict[str, Any]] = []
    for leaf_manifest_path in sorted(source_root.rglob("manifest.json")):
        if leaf_manifest_path == manifest_path:
            continue
        leaf = _load_object(leaf_manifest_path)
        if leaf.get("schema_version") != LEAF_SCHEMA_VERSION:
            continue
        if leaf.get("complete") is not True or leaf.get("source") != source:
            raise ValueError(f"invalid V5 leaf manifest: {leaf_manifest_path}")
        relative = leaf_manifest_path.parent.relative_to(source_root)
        if not relative.parts or relative.parts[0] != source:
            raise ValueError(f"V5 source folder mismatch: {leaf_manifest_path}")
        leaves.append({
            "relative_path": relative.as_posix(),
            "manifest_path": str(leaf_manifest_path),
            "source": source,
            "category": str(leaf["category"]),
            "memory_variant": str(leaf["memory_variant"]),
            "output_profile_id": str(leaf["output_profile_id"]),
            "memory_pair_eligible": bool(leaf["memory_pair_eligible"]),
            "task_name": str(leaf["task_name"]),
            "split": str(leaf["split"]),
            "num_samples": int(leaf["num_samples"]),
            "num_episodes": int(leaf["num_episodes"]),
            "data_sha256": str(leaf["data_sha256"]),
            "index_sha256": str(leaf["index_sha256"]),
            "episodes_sha256": str(leaf["episodes_sha256"]),
        })
    if not leaves or sum(item["num_samples"] for item in leaves) != int(manifest["num_samples"]):
        raise ValueError(f"V5 source leaf totals disagree: {source_root}")
    return manifest, leaves


def _implementation_digests() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    paths = {
        name: root / name
        for name in (
            "schema_v5.py",
            "prompt_v5.py",
            "memory_v5.py",
            "loss_mask_v5.py",
            "baseline_adapter.py",
            "baseline_materialize_v5.py",
            "robodojo_adapter.py",
            "takeover_adapter.py",
            "materialize_v5.py",
            "indexed_io_v5.py",
            "mix_v5.py",
            "dataset_v5.py",
            "epilogue_v5.py",
            "train_v5.py",
            "data_smoke_v5.py",
            "smoke_report_v5.py",
            "validate_v5.py",
            "incremental_cache_v5.py",
            "snapshot_v5.py",
            "holdout_v5.py",
            "infer_v5.py",
        )
    }
    paths["run_v10_continuous_v5.sh"] = root.parent / "run_v10_continuous_v5.sh"
    return {name: _sha256(path) for name, path in paths.items()}


def _publish_current(current_path: Path, manifest: Mapping[str, Any], output: Path) -> None:
    value = {
        "schema_version": "v10_action_segment_v5_current_v1",
        "snapshot_id": manifest["snapshot_id"],
        "snapshot_path": str(output.resolve()),
        "content_digest": manifest["content_digest"],
        "input_cache_key": manifest["input_cache_key"],
    }
    current_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_write(str(current_path)) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def compose_snapshot(
    source_roots: Sequence[Path | str],
    output: Path | str,
    *,
    snapshot_id: str,
    current_path: Path | str | None = None,
    expected_sources: Sequence[str] = EXPECTED_SOURCES,
    benchmark3_holdout: Benchmark3Holdout | None = None,
) -> dict[str, Any]:
    """Atomically compose exactly three already-immutable source builds."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"V5 snapshot already exists: {output}")
    by_source: dict[str, tuple[Path, dict[str, Any], list[dict[str, Any]]]] = {}
    for raw_root in source_roots:
        root = Path(raw_root).resolve()
        manifest, leaves = _source_inventory(root)
        source = str(manifest["source"])
        if source in by_source:
            raise ValueError(f"duplicate V5 source build: {source}")
        by_source[source] = (root, manifest, leaves)
    expected = tuple(expected_sources)
    if not expected or len(expected) != len(set(expected)) or set(expected) - set(EXPECTED_SOURCES):
        raise ValueError(f"invalid V5 expected source set: {expected}")
    if set(by_source) != set(expected):
        raise ValueError(
            f"V5 snapshot requires {expected}, got {sorted(by_source)}"
        )

    holdout_source_audits: dict[str, Any] = {}
    if benchmark3_holdout is not None:
        for source in expected:
            source_root = by_source[source][0]
            holdout_source_audits[source] = audit_artifact(
                source_root,
                benchmark3_holdout,
                fail_on_match=True,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    staging.mkdir()
    final_leaves: list[dict[str, Any]] = []
    source_summaries: dict[str, Any] = {}
    try:
        for source in expected:
            source_root, source_manifest, leaves = by_source[source]
            source_summaries[source] = {
                "materialization_manifest_sha256": _sha256(source_root / "manifest.json"),
                "partial": bool(source_manifest.get("partial")),
                "limit": source_manifest.get("limit"),
                "num_samples": int(source_manifest["num_samples"]),
                "num_leaves": int(source_manifest["num_leaves"]),
                "canonical_raw_sources": source_manifest.get("canonical_raw_sources", []),
                "selector": source_manifest.get("selector"),
            }
            plan_report = source_root / "plan_cache_report.json"
            if plan_report.is_file():
                source_summaries[source]["plan_cache_report_sha256"] = (
                    _sha256(plan_report)
                )
            if source == "takeover_q":
                if not plan_report.is_file() and not source_manifest.get("partial"):
                    raise ValueError(
                        "formal Takeover-Q source lacks plan_cache_report.json"
                    )
                if plan_report.is_file():
                    report = _validate_takeover_exclusion_report(
                        plan_report,
                        require_complete=not bool(source_manifest.get("partial")),
                    )
                    source_summaries[source]["num_exclusions"] = report[
                        "num_exclusions"
                    ]
            metadata_root = staging / "metadata" / "sources" / source
            metadata_root.mkdir(parents=True)
            shutil.copy2(source_root / "manifest.json", metadata_root / "manifest.json")
            split_metadata = source_manifest.get("robodojo_official_split")
            if isinstance(split_metadata, Mapping):
                source_split_path = source_root / str(
                    split_metadata["copied_relative_path"]
                )
                snapshot_split_relative = Path(
                    "metadata",
                    "sources",
                    source,
                    "robodojo_official_split.json",
                )
                shutil.copy2(
                    source_split_path,
                    staging / snapshot_split_relative,
                )
                snapshot_split_metadata = dict(split_metadata)
                snapshot_split_metadata["snapshot_relative_path"] = (
                    snapshot_split_relative.as_posix()
                )
                source_summaries[source]["robodojo_official_split"] = (
                    snapshot_split_metadata
                )
            for optional in ("plan_cache_report.json",):
                if (source_root / optional).is_file():
                    shutil.copy2(source_root / optional, metadata_root / optional)
            if (source_root / "review_fixtures").is_dir():
                shutil.copytree(
                    source_root / "review_fixtures",
                    metadata_root / "review_fixtures",
                )
            for leaf in leaves:
                relative = Path(leaf["relative_path"])
                source_leaf = source_root / relative
                destination = staging / "data" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source_leaf, destination, copy_function=_copy_or_link)
                copied = dict(leaf)
                copied.pop("manifest_path", None)
                copied["relative_path"] = (Path("data") / relative).as_posix()
                final_leaves.append(copied)

        final_leaves.sort(key=lambda item: item["relative_path"])
        implementation = _implementation_digests()
        taxonomy_digest = _digest(FAILURE_TYPE_BY_SOURCE_CODE)
        content_closure = [
            {
                key: leaf[key]
                for key in (
                    "relative_path",
                    "source",
                    "category",
                    "memory_variant",
                    "output_profile_id",
                    "memory_pair_eligible",
                    "task_name",
                    "split",
                    "num_samples",
                    "data_sha256",
                    "index_sha256",
                    "episodes_sha256",
                )
            }
            for leaf in final_leaves
        ]
        cache_closure = {
            "source_summaries": source_summaries,
            "leaf_content_closure": content_closure,
            "implementation_digests": implementation,
            "prompt_renderer_sha256": prompt_renderer_digest(),
            "failure_taxonomy_sha256": taxonomy_digest,
        }
        if benchmark3_holdout is not None:
            cache_closure["benchmark3_holdout"] = benchmark3_holdout.metadata()
        manifest = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "complete": True,
            "snapshot_id": snapshot_id,
            "build_mode": "full_compose",
            "expected_sources": list(expected),
            "partial": (
                any(value["partial"] for value in source_summaries.values())
                or set(expected) != set(EXPECTED_SOURCES)
            ),
            "formal_source_set_complete": set(expected) == set(EXPECTED_SOURCES),
            "missing_formal_sources": sorted(set(EXPECTED_SOURCES) - set(expected)),
            "input_cache_key": _digest(cache_closure),
            "content_digest": _digest(content_closure),
            "prompt_renderer_sha256": prompt_renderer_digest(),
            "failure_taxonomy_sha256": taxonomy_digest,
            "implementation_digests": implementation,
            "num_samples": sum(leaf["num_samples"] for leaf in final_leaves),
            "num_leaves": len(final_leaves),
            "sources": source_summaries,
            "leaves": final_leaves,
        }
        if benchmark3_holdout is not None:
            holdout_report = {
                "schema_version": "v5_benchmark3_snapshot_compose_audit_v1",
                "passed": True,
                "holdout": benchmark3_holdout.metadata(),
                "source_audits": holdout_source_audits,
                "checked_samples": sum(
                    int(value["checked_samples"])
                    for value in holdout_source_audits.values()
                ),
                "overlap_samples": 0,
            }
            holdout_relative = Path("metadata", "benchmark3_holdout_report.json")
            _write_json(staging / holdout_relative, holdout_report)
            manifest["benchmark3_holdout"] = {
                **benchmark3_holdout.metadata(),
                "policy": "compose_requires_zero_source_overlap",
                "checked_samples": holdout_report["checked_samples"],
                "overlap_samples": 0,
                "report_relative_path": holdout_relative.as_posix(),
                "report_sha256": _sha256(staging / holdout_relative),
            }
        ledger = [
            {
                "schema_version": "v10_action_segment_v5_completed_leaf_v1",
                "relative_path": leaf["relative_path"],
                "source": leaf["source"],
                "num_samples": leaf["num_samples"],
                "data_sha256": leaf["data_sha256"],
                "episodes_sha256": leaf["episodes_sha256"],
            }
            for leaf in final_leaves
        ]
        ledger_path = staging / "completed_ledger.jsonl"
        with ledger_path.open("wb") as handle:
            for value in ledger:
                handle.write(_json_bytes(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        manifest["completed_ledger_sha256"] = _sha256(ledger_path)
        _write_json(staging / "manifest.json", manifest)
        _fsync_directory(staging)
        os.replace(staging, output)
        _fsync_directory(output.parent)
        if current_path is not None:
            _publish_current(Path(current_path), manifest, output)
        return {**manifest, "snapshot_path": str(output.resolve())}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def clone_incremental_snapshot(
    base: Path | str,
    output: Path | str,
    *,
    snapshot_id: str,
    current_path: Path | str | None = None,
) -> dict[str, Any]:
    """Publish a no-delta incremental cache without mutating the base snapshot."""
    base = Path(base).resolve()
    output = Path(output)
    base_manifest = _load_object(base / "manifest.json")
    if (
        base_manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION
        or base_manifest.get("complete") is not True
    ):
        raise ValueError("incremental base is not a complete V5 snapshot")
    if output.exists():
        raise FileExistsError(f"incremental snapshot already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        shutil.copytree(base, staging, copy_function=_copy_or_link)
        manifest = dict(base_manifest)
        manifest.update({
            "snapshot_id": snapshot_id,
            "build_mode": "incremental_no_delta",
            "base_snapshot": str(base),
            "base_snapshot_manifest_sha256": _sha256(base / "manifest.json"),
        })
        _write_json(staging / "manifest.json", manifest)
        _fsync_directory(staging)
        os.replace(staging, output)
        _fsync_directory(output.parent)
        comparison = compare_snapshots(base, output)
        if comparison["equivalent"] is not True:
            raise RuntimeError("incremental clone is not content-equivalent to its base")
        if current_path is not None:
            _publish_current(Path(current_path), manifest, output)
        return {**manifest, "snapshot_path": str(output.resolve()), "equivalence": comparison}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def compare_snapshots(left: Path | str, right: Path | str) -> dict[str, Any]:
    left_path = Path(left).resolve()
    right_path = Path(right).resolve()
    left_manifest = _load_object(left_path / "manifest.json")
    right_manifest = _load_object(right_path / "manifest.json")
    fields = (
        "content_digest",
        "input_cache_key",
        "num_samples",
        "num_leaves",
        "completed_ledger_sha256",
    )
    equal_fields = {
        field: left_manifest.get(field) == right_manifest.get(field) for field in fields
    }
    left_leaves = {
        value["relative_path"]: (
            value["num_samples"],
            value["data_sha256"],
            value["index_sha256"],
            value["episodes_sha256"],
        )
        for value in left_manifest.get("leaves", [])
    }
    right_leaves = {
        value["relative_path"]: (
            value["num_samples"],
            value["data_sha256"],
            value["index_sha256"],
            value["episodes_sha256"],
        )
        for value in right_manifest.get("leaves", [])
    }
    leaves_equal = left_leaves == right_leaves
    return {
        "schema_version": "v10_action_segment_v5_equivalence_v1",
        "left": str(left_path),
        "right": str(right_path),
        "equal_fields": equal_fields,
        "leaf_hash_multiset_equal": leaves_equal,
        "equivalent": all(equal_fields.values()) and leaves_equal,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compose = subparsers.add_parser("compose")
    compose.add_argument("--source-root", type=Path, action="append", required=True)
    compose.add_argument("--output", type=Path, required=True)
    compose.add_argument("--snapshot-id", required=True)
    compose.add_argument("--current", type=Path)
    compose.add_argument(
        "--benchmark3-manifest",
        type=Path,
        default=DEFAULT_BENCHMARK3_MANIFEST,
    )
    compose.add_argument(
        "--benchmark3-expected-sha256",
        default=DEFAULT_BENCHMARK3_SHA256,
    )
    compose.add_argument(
        "--expected-source",
        action="append",
        choices=EXPECTED_SOURCES,
        help="Repeat to compose an intentional source subset; default is all sources.",
    )
    clone = subparsers.add_parser("incremental-clone")
    clone.add_argument("--base", type=Path, required=True)
    clone.add_argument("--output", type=Path, required=True)
    clone.add_argument("--snapshot-id", required=True)
    clone.add_argument("--current", type=Path)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "compose":
        benchmark3_holdout = Benchmark3Holdout.load(
            args.benchmark3_manifest,
            expected_sha256=args.benchmark3_expected_sha256,
        )
        result = compose_snapshot(
            args.source_root,
            args.output,
            snapshot_id=args.snapshot_id,
            current_path=args.current,
            expected_sources=(args.expected_source or EXPECTED_SOURCES),
            benchmark3_holdout=benchmark3_holdout,
        )
    elif args.command == "incremental-clone":
        result = clone_incremental_snapshot(
            args.base,
            args.output,
            snapshot_id=args.snapshot_id,
            current_path=args.current,
        )
    else:
        result = compare_snapshots(args.left, args.right)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("equivalent", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_SOURCES",
    "clone_incremental_snapshot",
    "compare_snapshots",
    "compose_snapshot",
]
