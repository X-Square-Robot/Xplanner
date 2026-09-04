"""Build per-folder and global training manifests from a merged V2 catalog."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

from .artifact_ledger import register_artifacts
from .common.atomic import read_json, write_json


def _slug(value: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "dataset"
    return f"{base[:80]}-{hashlib.sha256(value.encode()).hexdigest()[:8]}"


def build_manifests(
    run_root: Path,
    *,
    run_id: str,
    ledger_path: str | None = None,
) -> dict[str, Any]:
    current = read_json(str(run_root / "current_merge.json"))
    if not isinstance(current, dict):
        raise FileNotFoundError("current_merge.json is missing; run merge first")
    merge_root = Path(str(current["root"]))
    catalog = merge_root / "catalog.jsonl"
    manifest_id = str(current["merge_id"])
    final = run_root / "manifests" / manifest_id
    if final.is_dir():
        result = read_json(str(final / "manifest_statistics.json"), {})
        return {**result, "root": str(final)}
    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{manifest_id}-", dir=final.parent))
    by_folder = temporary / "by_folder"
    by_folder.mkdir()
    handles: dict[str, Any] = {}
    dataset_paths: dict[str, Path] = {}
    dataset_counts: Counter[str] = Counter()
    profile_counts: dict[str, Counter] = defaultdict(Counter)
    unit_counts: dict[str, Counter] = defaultdict(Counter)
    view_counts: dict[str, Counter] = defaultdict(Counter)
    total = 0
    try:
        with catalog.open(encoding="utf-8") as catalog_handle:
            for line in catalog_handle:
                row = json.loads(line)
                dataset = str(row.get("dataset_name") or "unknown")
                path = dataset_paths.setdefault(dataset, by_folder / f"{_slug(dataset)}.jsonl")
                handle = handles.get(dataset)
                if handle is None:
                    handle = path.open("a", encoding="utf-8")
                    handles[dataset] = handle
                # The merged catalog is already canonical JSONL. Preserve its
                # exact row bytes instead of serializing every large sample a
                # second time merely to partition it by dataset.
                handle.write(line if line.endswith("\n") else line + "\n")
                dataset_counts[dataset] += 1
                profile_counts[dataset][str(row.get("profile") or "unknown")] += 1
                unit_counts[dataset][str(row.get("unit_level") or "unknown")] += 1
                view_counts[dataset]["+".join(row.get("views") or ())] += 1
                total += 1
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        handles.clear()
        try:
            os.link(catalog, temporary / "all.jsonl")
        except OSError:
            shutil.copy2(catalog, temporary / "all.jsonl")
        datasets = []
        for dataset in sorted(dataset_paths):
            datasets.append({
                "name": dataset,
                "manifest": f"by_folder/{dataset_paths[dataset].name}",
                "sample_count": dataset_counts[dataset],
                "weight": 1.0,
                "enabled": True,
            })
        write_json(str(temporary / "datasets.json"), {"datasets": datasets})
        mixture_path = temporary / "mixture.yml"
        with mixture_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump({"version": 1, "datasets": datasets}, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        statistics = {
            "run_id": run_id,
            "manifest_id": manifest_id,
            "merged_catalog": str(catalog),
            "sample_count": total,
            "dataset_count": len(datasets),
            "by_dataset": {
                dataset: {
                    "samples": dataset_counts[dataset],
                    "profiles": dict(profile_counts[dataset]),
                    "unit_levels": dict(unit_counts[dataset]),
                    "view_combinations": dict(view_counts[dataset]),
                    "global_share": dataset_counts[dataset] / total if total else 0.0,
                }
                for dataset in sorted(dataset_counts)
            },
        }
        write_json(str(temporary / "manifest_statistics.json"), statistics)
        os.replace(temporary, final)
        write_json(str(run_root / "current_manifest.json"), {**statistics, "root": str(final)})
        artifacts = [
            final / "all.jsonl", final / "datasets.json", final / "mixture.yml",
            final / "manifest_statistics.json", run_root / "current_manifest.json",
            *(final / "by_folder" / path.name for path in dataset_paths.values()),
        ]
        register_artifacts(
            ledger_path,
            artifacts,
            purpose="V2 per-folder and global training manifests",
            source_id=",".join(sorted(current.get("by_source") or {})),
            run_id=run_id,
        )
        return {**statistics, "root": str(final)}
    except BaseException:
        for handle in handles.values():
            handle.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
