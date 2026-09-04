"""Independent row, leaf, memory-pair, and source validation for V5 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from .indexed_io import LEAF_SCHEMA_VERSION, LeafKey
from .holdout import (
    EvaluationHoldout,
    DEFAULT_EVALUATION_MANIFEST,
    DEFAULT_EVALUATION_SHA256,
    HoldoutFilter,
    HoldoutViolationError,
)
from .prompt import normalized_execution_instruction, prompt_renderer_digest, render_user
from .robodojo_adapter import (
    DEFAULT_OFFICIAL_SPLIT,
    OFFICIAL_SPLITS,
    ROBODOJO_TASKS,
    load_official_split_assignments,
)
from .schema import FAILURE_TYPES, validate_sample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(sample: dict[str, Any]) -> str:
    provenance = sample["provenance"]
    closure = {
        "base_sample_id": sample["base_sample_id"],
        "source": sample["source"],
        "category": sample["category"],
        "output_spec": sample["output_spec"],
        "output_profile_id": sample["output_profile_id"],
        "task_instruction": sample["task_instruction"],
        "images": sample["images"],
        "target": sample["target"],
        "supervision": sample["supervision"],
        "episode_key": provenance.get("episode_key"),
        "task_name": provenance.get("task_name"),
        "split": provenance.get("split"),
    }
    payload = json.dumps(
        closure, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _manifest_paths(root: Path) -> list[Path]:
    values: list[Path] = []
    for path in sorted(root.rglob("manifest.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("schema_version") == LEAF_SCHEMA_VERSION:
            values.append(path)
    if not values:
        raise ValueError(f"no V5 leaf manifests below {root}")
    return values


def _official_robodojo_contract(
    split_path: Path | str,
) -> tuple[dict[str, str], dict[str, set[str]]]:
    assignments = load_official_split_assignments(split_path)
    tasks_by_split = {split: set() for split in OFFICIAL_SPLITS}
    for episode_key, split in assignments.items():
        task_name = episode_key.rsplit("/", 2)[-2]
        tasks_by_split[split].add(task_name)
    observed_tasks = set().union(*tasks_by_split.values())
    if observed_tasks != set(ROBODOJO_TASKS):
        raise ValueError(
            "formal RoboDojo official split inventory is not exactly 35 tasks"
        )
    if any(not tasks_by_split[split] for split in OFFICIAL_SPLITS):
        raise ValueError("formal RoboDojo official split has an empty split group")
    return assignments, tasks_by_split


def _validate_snapshot_split_digest(
    root: Path,
    root_manifest: dict[str, Any],
    current_split_path: Path | str,
) -> tuple[str, Path]:
    sources = root_manifest.get("sources")
    robodojo = sources.get("robodojo") if isinstance(sources, dict) else None
    metadata = (
        robodojo.get("robodojo_official_split")
        if isinstance(robodojo, dict)
        else None
    )
    if not isinstance(metadata, dict):
        raise ValueError(
            "formal V5 snapshot lacks RoboDojo official split provenance"
        )
    expected_digest = metadata.get("sha256")
    if (
        not isinstance(expected_digest, str)
        or len(expected_digest) != 64
        or any(character not in "0123456789abcdef" for character in expected_digest)
    ):
        raise ValueError("formal V5 snapshot has an invalid official split digest")
    snapshot_relative = Path(
        str(metadata.get("snapshot_relative_path") or "")
    )
    if (
        snapshot_relative.is_absolute()
        or ".." in snapshot_relative.parts
        or not snapshot_relative.parts
    ):
        raise ValueError("formal V5 snapshot has an unsafe official split copy path")
    snapshot_copy = root / snapshot_relative
    if not snapshot_copy.is_file() or _sha256(snapshot_copy) != expected_digest:
        raise ValueError(
            "formal V5 snapshot official split copy/digest mismatch"
        )
    current_split = Path(current_split_path).resolve()
    if _sha256(current_split) != expected_digest:
        raise ValueError(
            "current RoboDojo official split digest disagrees with the snapshot"
        )
    return expected_digest, snapshot_copy


def _validate_snapshot_takeover_report(
    root: Path,
    root_manifest: dict[str, Any],
    *,
    require_complete: bool,
) -> dict[str, Any] | None:
    sources = root_manifest.get("sources")
    takeover = sources.get("takeover_q") if isinstance(sources, dict) else None
    digest = (
        takeover.get("plan_cache_report_sha256")
        if isinstance(takeover, dict)
        else None
    )
    if digest is None and not require_complete:
        return None
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("formal Takeover-Q snapshot lacks a valid report digest")
    path = root / "metadata" / "sources" / "takeover_q" / "plan_cache_report.json"
    if not path.is_file() or _sha256(path) != digest:
        raise ValueError("Takeover-Q exclusion report copy/digest mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != "v5_takeover_q_exclusion_report_v1"
    ):
        raise ValueError("invalid Takeover-Q exclusion report schema")
    exclusions = report.get("exclusions")
    if not isinstance(exclusions, list) or any(
        not isinstance(value, dict) for value in exclusions
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


def validate_dataset(
    root: Path | str,
    *,
    expect_composed: bool,
    require_full: bool,
    official_split_path: Path | str = DEFAULT_OFFICIAL_SPLIT,
    evaluation_holdout: EvaluationHoldout | None = None,
    require_evaluation_holdout_fence: bool = False,
) -> dict[str, Any]:
    root = Path(root).resolve()
    root_manifest_path = root / "manifest.json"
    root_manifest = (
        json.loads(root_manifest_path.read_text(encoding="utf-8"))
        if root_manifest_path.is_file()
        else {}
    )
    if require_full and root_manifest.get("partial") is not False:
        raise ValueError("formal validation requires a non-partial root manifest")
    root_holdout = root_manifest.get("evaluation_holdout")
    if require_evaluation_holdout_fence and not isinstance(root_holdout, dict):
        raise ValueError("V5 root manifest lacks the required Evaluation holdout fence")
    if evaluation_holdout is not None and isinstance(root_holdout, dict):
        if root_holdout.get("manifest_sha256") != evaluation_holdout.manifest_sha256:
            raise ValueError("V5 root Evaluation holdout digest differs from the active holdout")
        relative = Path(str(root_holdout.get("report_relative_path") or ""))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("V5 root has an unsafe Evaluation holdout report path")
        report_path = root / relative
        if (
            not report_path.is_file()
            or _sha256(report_path) != root_holdout.get("report_sha256")
        ):
            raise ValueError("V5 root Evaluation holdout report copy/digest mismatch")
    official_assignments: dict[str, str] = {}
    official_tasks_by_split: dict[str, set[str]] = {
        split: set() for split in OFFICIAL_SPLITS
    }
    official_split_digest: str | None = None
    frozen_official_split: Path | None = None
    sources_metadata = root_manifest.get("sources")
    robodojo_metadata = (
        sources_metadata.get("robodojo")
        if isinstance(sources_metadata, dict)
        else None
    )
    has_official_split_metadata = isinstance(robodojo_metadata, dict) and isinstance(
        robodojo_metadata.get("robodojo_official_split"), dict
    )
    if require_full or has_official_split_metadata:
        official_split_digest, frozen_official_split = (
            _validate_snapshot_split_digest(
                root,
                root_manifest,
                official_split_path,
            )
        )
        official_assignments, official_tasks_by_split = (
            _official_robodojo_contract(frozen_official_split)
        )
    takeover_exclusion_report = _validate_snapshot_takeover_report(
        root,
        root_manifest,
        require_complete=require_full,
    )

    with tempfile.TemporaryDirectory(prefix="v5_validate_") as temporary:
        database = sqlite3.connect(str(Path(temporary) / "audit.sqlite3"))
        database.execute("PRAGMA journal_mode=WAL")
        database.execute("PRAGMA synchronous=NORMAL")
        database.execute("CREATE TABLE ids (sample_id TEXT PRIMARY KEY)")
        database.execute(
            "CREATE TABLE pairs ("
            "base_id TEXT NOT NULL, variant TEXT NOT NULL, fingerprint TEXT NOT NULL, "
            "PRIMARY KEY(base_id, variant))"
        )
        counts: Counter[str] = Counter()
        failure_counts: Counter[str] = Counter()
        source_set: set[str] = set()
        robodojo_tasks: set[str] = set()
        leaf_reports: list[dict[str, Any]] = []
        holdout_filter = (
            HoldoutFilter(evaluation_holdout)
            if evaluation_holdout is not None
            else None
        )

        for manifest_path in _manifest_paths(root):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            leaf_root = manifest_path.parent
            if manifest.get("complete") is not True:
                raise ValueError(f"incomplete V5 leaf: {leaf_root}")
            for filename, digest_key in (
                ("data.jsonl", "data_sha256"),
                ("data.index", "index_sha256"),
                ("episodes.jsonl", "episodes_sha256"),
            ):
                path = leaf_root / filename
                if not path.is_file() or _sha256(path) != manifest[digest_key]:
                    raise ValueError(f"V5 leaf hash mismatch: {path}")
            num_samples = int(manifest["num_samples"])
            if (leaf_root / "data.index").stat().st_size != num_samples * 8:
                raise ValueError(f"V5 index size mismatch: {leaf_root}")
            relative_parts = leaf_root.relative_to(root).parts
            expected_parts = (
                str(manifest["source"]),
                str(manifest["memory_variant"]),
                str(manifest["category"]),
                str(manifest["output_profile_id"]),
                str(manifest["task_name"]),
                str(manifest["split"]),
            )
            if tuple(relative_parts[-6:]) != expected_parts:
                raise ValueError(f"V5 folder/manifest mismatch: {leaf_root}")
            eligible = manifest.get("memory_pair_eligible")
            if not isinstance(eligible, bool):
                raise ValueError(f"V5 leaf lacks memory eligibility: {leaf_root}")

            observed = 0
            with (leaf_root / "data.jsonl").open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    outer = json.loads(line)
                    if not isinstance(outer, dict) or tuple(outer) != (
                        "data_id", "v5_sample", "image"
                    ):
                        raise ValueError(f"invalid indexed outer row: {leaf_root}:{line_number}")
                    sample = validate_sample(outer["v5_sample"])
                    if holdout_filter is not None and not holdout_filter.keep(sample):
                        raise HoldoutViolationError(
                            "Evaluation holdout sample found in V5 artifact: "
                            f"{sample['sample_id']}"
                        )
                    if outer["data_id"] != sample["sample_id"] or outer["image"] != sample["images"]:
                        raise ValueError(f"outer/sample mismatch: {leaf_root}:{line_number}")
                    if LeafKey.from_sample(sample) != LeafKey(
                        source=expected_parts[0],
                        memory_variant=expected_parts[1],
                        category=expected_parts[2],
                        output_profile=expected_parts[3],
                        task=expected_parts[4],
                        split=expected_parts[5],
                    ):
                        raise ValueError(f"row/leaf mismatch: {leaf_root}:{line_number}")
                    try:
                        database.execute(
                            "INSERT INTO ids(sample_id) VALUES (?)", (sample["sample_id"],)
                        )
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f"duplicate global sample_id: {sample['sample_id']}") from exc

                    source = str(sample["source"])
                    category = str(sample["category"])
                    memory = str(sample["memory_variant"])
                    provenance = sample["provenance"]
                    source_set.add(source)
                    if source == "takeover_q" and category != "takeover":
                        raise ValueError("Takeover-Q may contain Takeover rows only")
                    counts[f"source:{source}"] += 1
                    counts[f"category:{category}"] += 1
                    counts[f"memory:{memory}"] += 1
                    counts[f"leaf:{source}|{memory}|{category}"] += 1
                    if provenance.get("memory_pair_eligible") is not eligible:
                        raise ValueError(f"row/leaf memory eligibility mismatch: {leaf_root}")
                    if memory == "with_memory" and not eligible:
                        raise ValueError("ineligible sample cannot contain memory")
                    if eligible:
                        database.execute(
                            "INSERT INTO pairs(base_id,variant,fingerprint) VALUES (?,?,?)",
                            (sample["base_sample_id"], memory, _fingerprint(sample)),
                        )

                    prompt = render_user(sample)
                    if category != "initial_plan":
                        normalized_execution_instruction(prompt)
                    if category == "takeover":
                        if memory != "no_memory" or sample["prompt_context"]:
                            raise ValueError("Takeover sample contains memory")
                        detail = sample["target"]["decision_detail"]
                        failure = detail["failure_analysis"]
                        failure_counts[failure["failure_type"]] += 1
                        label_sources = provenance.get("label_sources")
                        if not isinstance(label_sources, dict) or "q4" not in str(
                            label_sources.get("failed_action_context", "")
                        ).casefold():
                            raise ValueError("Takeover sample lacks Q4 label provenance")
                        if provenance.get("future_takeover_frames_used") is not False:
                            raise ValueError("Takeover sample used future takeover frames")
                    if source == "robodojo":
                        task_name = str(provenance["task_name"])
                        robodojo_tasks.add(task_name)
                        if official_assignments:
                            episode_key = str(provenance.get("episode_key") or "")
                            assigned_split = official_assignments.get(episode_key)
                            if (
                                provenance.get("split") != "train"
                                or expected_parts[5] != "train"
                                or assigned_split != "train"
                            ):
                                raise ValueError(
                                    "formal RoboDojo train leaf contains a "
                                    f"task/trajectory holdout: {episode_key}"
                                )
                            episode_task = episode_key.rsplit("/", 2)[-2]
                            if episode_task != task_name:
                                raise ValueError(
                                    "RoboDojo episode/task provenance mismatch: "
                                    f"{episode_key} != {task_name}"
                                )
                    observed += 1
            if observed != num_samples:
                raise ValueError(f"V5 leaf count mismatch: {leaf_root}")
            leaf_reports.append({
                "path": str(leaf_root),
                "source": expected_parts[0],
                "category": expected_parts[2],
                "output_profile_id": expected_parts[3],
                "memory_variant": expected_parts[1],
                "memory_pair_eligible": eligible,
                "num_samples": observed,
            })
            database.commit()

        bad_pairs = database.execute(
            "SELECT base_id, COUNT(*), COUNT(DISTINCT fingerprint), "
            "SUM(variant='no_memory'), SUM(variant='with_memory') "
            "FROM pairs GROUP BY base_id "
            "HAVING COUNT(*) != 2 OR COUNT(DISTINCT fingerprint) != 1 "
            "OR SUM(variant='no_memory') != 1 OR SUM(variant='with_memory') != 1 "
            "LIMIT 20"
        ).fetchall()
        if bad_pairs:
            raise ValueError(f"invalid with/no memory pairs: {bad_pairs[:3]}")
        pair_count = int(database.execute(
            "SELECT COUNT(DISTINCT base_id) FROM pairs"
        ).fetchone()[0])
        sample_count = int(database.execute("SELECT COUNT(*) FROM ids").fetchone()[0])
        database.close()

    if expect_composed:
        formal_sources = {"baseline", "robodojo", "takeover_q"}
        declared_sources = (
            set(root_manifest.get("expected_sources") or ())
            if root_manifest
            else formal_sources
        )
        if source_set != declared_sources:
            raise ValueError(
                "composed snapshot source set differs from its declaration: "
                f"observed={sorted(source_set)}, declared={sorted(declared_sources)}"
            )
        if require_full and source_set != formal_sources:
            raise ValueError(
                f"formal composed snapshot source set is incomplete: {sorted(source_set)}"
            )
        if not source_set or source_set - formal_sources:
            raise ValueError(f"invalid composed snapshot source set: {sorted(source_set)}")
    if require_full:
        if set(failure_counts) != set(FAILURE_TYPES):
            raise ValueError("formal V5 snapshot does not cover all 15 failure types")
        expected_train_tasks = official_tasks_by_split["train"]
        if robodojo_tasks != expected_train_tasks:
            raise ValueError(
                "formal RoboDojo train task inventory does not match the "
                f"official train split: observed={len(robodojo_tasks)}, "
                f"expected={len(expected_train_tasks)}"
            )
    if root_manifest and int(root_manifest.get("num_samples", sample_count)) != sample_count:
        raise ValueError("root manifest sample total disagrees with validated leaves")
    result = {
        "schema_version": "v10_action_segment_v5_validation_v1",
        "passed": True,
        "root": str(root),
        "formal_full": require_full,
        "sample_count": sample_count,
        "leaf_count": len(leaf_reports),
        "memory_pair_count": pair_count,
        "sources": sorted(source_set),
        "counts": dict(sorted(counts.items())),
        "failure_type_counts": {
            value: failure_counts[value] for value in FAILURE_TYPES
        },
        "takeover_q_exclusion_count": (
            takeover_exclusion_report["num_exclusions"]
            if takeover_exclusion_report is not None
            else None
        ),
        "takeover_q_exclusion_reason_counts": (
            takeover_exclusion_report["exclusion_reason_counts"]
            if takeover_exclusion_report is not None
            else None
        ),
        "robodojo_task_count": len(robodojo_tasks),
        "robodojo_official_task_count": len(
            set().union(*official_tasks_by_split.values())
        ),
        "robodojo_official_train_task_count": len(
            official_tasks_by_split["train"]
        ),
        "robodojo_official_split_sha256": official_split_digest,
        "prompt_renderer_sha256": prompt_renderer_digest(),
        "leaf_reports": leaf_reports,
    }
    if holdout_filter is not None:
        result["evaluation_holdout"] = {
            **holdout_filter.report(),
            "root_manifest_fence_present": isinstance(root_holdout, dict),
            "root_manifest_fence_required": require_evaluation_holdout_fence,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-composed", action="store_true")
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--require-evaluation_holdout-fence", action="store_true")
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        default=DEFAULT_EVALUATION_MANIFEST,
    )
    parser.add_argument(
        "--evaluation-expected-sha256",
        default=DEFAULT_EVALUATION_SHA256,
    )
    parser.add_argument(
        "--robodojo-official-split",
        type=Path,
        default=DEFAULT_OFFICIAL_SPLIT,
    )
    args = parser.parse_args()
    evaluation_holdout = EvaluationHoldout.load(
        args.evaluation_manifest,
        expected_sha256=args.evaluation_expected_sha256,
    )
    result = validate_dataset(
        args.root,
        expect_composed=args.expect_composed,
        require_full=args.require_full,
        official_split_path=args.robodojo_official_split,
        evaluation_holdout=evaluation_holdout,
        require_evaluation_holdout_fence=args.require_evaluation_holdout_fence,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["validate_dataset"]
