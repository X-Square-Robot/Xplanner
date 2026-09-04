"""Export deterministic, production-rendered V5 examples for human review."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .dataset import dialogues_from_sample
from .schema import FAILURE_TYPES, SNAPSHOT_SCHEMA_VERSION, validate_sample


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _iter_rows(root: Path) -> Iterable[dict[str, Any]]:
    data_root = root / "data" if (root / "data").is_dir() else root
    for path in sorted(data_root.rglob("data.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                sample = validate_sample(row["v5_sample"])
                yield {
                    "leaf": str(path.parent.relative_to(root)),
                    "line_number": line_number,
                    "row": row,
                    "sample": sample,
                }


def _render(item: dict[str, Any]) -> dict[str, Any]:
    sample = item["sample"]
    dialogues = dialogues_from_sample(sample)
    return {
        "sample_id": sample["sample_id"],
        "source": sample["source"],
        "category": sample["category"],
        "memory_variant": sample["memory_variant"],
        "output_spec": sample["output_spec"],
        "output_profile_id": sample["output_profile_id"],
        "leaf": item["leaf"],
        "model_input": dialogues[0]["text"],
        "model_output": dialogues[1]["text"],
        "loss_mask_char_spans": dialogues[1]["loss_mask_char_spans"],
        "use_for_training": True,
    }


def export(
    snapshot: Path,
    output: Path,
    *,
    takeover_source: Path | None = None,
) -> dict[str, Any]:
    snapshot_manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if snapshot_manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("input is not a V5 snapshot")

    items = list(_iter_rows(snapshot))
    by_pair: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    initial: dict[str, dict[str, Any]] = {}
    initial_profiles: dict[str, dict[str, Any]] = {}
    ongoing_profiles: dict[str, dict[str, Any]] = {}
    segment_only: dict[str, Any] | None = None
    failures: dict[str, dict[str, Any]] = {}
    for item in items:
        sample = item["sample"]
        source = sample["source"]
        category = sample["category"]
        variant = sample["memory_variant"]
        if category == "initial_plan" and source not in initial:
            initial[source] = item
        if category == "initial_plan":
            initial_profiles.setdefault(sample["output_profile_id"], item)
        if category == "ongoing":
            ongoing_profiles.setdefault(sample["output_profile_id"], item)
        if category == "ongoing" and sample["provenance"].get("memory_pair_eligible"):
            by_pair[(source, sample["base_sample_id"])][variant] = item
        if (
            category == "ongoing"
            and source == "baseline"
            and sample["provenance"].get("clean_label_mode") == "segment"
            and segment_only is None
        ):
            segment_only = item
        if category == "takeover":
            failure = sample["target"]["decision_detail"]["failure_analysis"]["failure_type"]
            failures.setdefault(failure, item)

    if takeover_source is not None:
        takeover_manifest = json.loads(
            (takeover_source / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            takeover_manifest.get("source") != "takeover_q"
            or takeover_manifest.get("complete") is not True
            or takeover_manifest.get("partial") is not False
        ):
            raise ValueError(
                "takeover_source must be a complete, non-partial Takeover-Q source"
            )
        failures = {}
        for item in _iter_rows(takeover_source):
            sample = item["sample"]
            if sample["source"] != "takeover_q":
                continue
            if sample["category"] == "takeover":
                failure = sample["target"]["decision_detail"]["failure_analysis"]["failure_type"]
                failures.setdefault(failure, item)
            if set(failures) == set(FAILURE_TYPES):
                break

    pair_examples: list[dict[str, Any]] = []
    for source in ("baseline", "robodojo"):
        candidates = [
            variants
            for (pair_source, _base_id), variants in sorted(by_pair.items())
            if pair_source == source and set(variants) == {"no_memory", "with_memory"}
        ]
        if not candidates:
            raise ValueError(f"no complete memory pair for {source}")
        pair_examples.extend([candidates[0]["no_memory"], candidates[0]["with_memory"]])

    if set(failures) != set(FAILURE_TYPES):
        missing = sorted(set(FAILURE_TYPES) - set(failures))
        raise ValueError(f"smoke snapshot lacks failure examples: {missing}")
    if set(initial) != {"baseline", "robodojo"}:
        raise ValueError("smoke snapshot lacks Baseline/RoboDojo initial-plan examples")
    if segment_only is None:
        raise ValueError("smoke snapshot lacks a Segment-only example")

    selected = (
        [initial["baseline"], initial["robodojo"]]
        + pair_examples
        + [segment_only]
        + [failures[value] for value in FAILURE_TYPES]
    )
    raw_rows = [item["row"] for item in selected]
    rendered = [_render(item) for item in selected]
    _write_jsonl(output / "actual_train" / "train_rows.jsonl", raw_rows)
    _write_json(output / "actual_train" / "rendered_input_output.json", rendered)
    _write_json(
        output / "actual_train" / "initial_plan.json",
        [_render(initial["baseline"]), _render(initial["robodojo"])],
    )
    _write_json(output / "actual_train" / "ongoing_memory_pairs.json", [_render(v) for v in pair_examples])
    _write_json(output / "actual_train" / "ongoing_segment_only.json", [_render(segment_only)])
    _write_json(
        output / "actual_train" / "initial_plan_output_profiles.json",
        [_render(initial_profiles[key]) for key in sorted(initial_profiles)],
    )
    _write_json(
        output / "actual_train" / "ongoing_output_profiles.json",
        [_render(ongoing_profiles[key]) for key in sorted(ongoing_profiles)],
    )
    _write_json(
        output / "actual_train" / "takeover_all_failure_types.json",
        [_render(failures[value]) for value in FAILURE_TYPES],
    )

    end_source = (
        snapshot
        / "metadata/sources/robodojo/review_fixtures/end_candidates.jsonl"
    )
    end_row = json.loads(next(line for line in end_source.read_text(encoding="utf-8").splitlines() if line.strip()))
    _write_json(
        output / "review_only" / "end_candidate.json",
        {
            "use_for_training": False,
            "reason": end_row["reason"],
            "record": end_row,
        },
    )

    manifest = {
        "schema_version": "v5_confirmation_sample_pack_v3",
        "snapshot": str(snapshot.resolve()),
        "snapshot_manifest_sha256": _sha256(snapshot / "manifest.json"),
        "takeover_source": (
            str(takeover_source.resolve())
            if takeover_source
            else str(snapshot.resolve())
        ),
        "takeover_source_manifest_sha256": (
            _sha256(takeover_source / "manifest.json")
            if takeover_source
            else _sha256(snapshot / "manifest.json")
        ),
        "num_actual_train_rows": len(raw_rows),
        "num_failure_types": len(failures),
        "category_index": "CATEGORY_INDEX.json",
        "categories": {
            "initial_plan": 2,
            "ongoing_memory_pair_rows": len(pair_examples),
            "ongoing_segment_only": 1,
            "initial_plan_output_profiles": len(initial_profiles),
            "ongoing_output_profiles": len(ongoing_profiles),
            "takeover": len(failures),
            "end_review_only": 1,
        },
        "training_rule": "Only files under actual_train are trainable. End remains review-only until independent physical completion evidence exists.",
    }
    category_index = {
        "schema_version": "v5_confirmation_category_index_v3",
        "note": "Files are deterministic review exports from production rows; loaders must consume the immutable snapshot, not glob this directory.",
        "categories": {
            "initial_plan": {
                "examples": "actual_train/initial_plan.json",
                "profile_examples": "actual_train/initial_plan_output_profiles.json",
            },
            "ongoing_memory_pairs": {
                "examples": "actual_train/ongoing_memory_pairs.json",
            },
            "ongoing_output_profiles": {
                "examples": "actual_train/ongoing_output_profiles.json",
                "meaning": "Prediction 1 and Prediction 2 independently request only the Action/Segment units backed by labels.",
            },
            "ongoing_segment_only": {
                "examples": "actual_train/ongoing_segment_only.json",
            },
            "takeover": {
                "examples": "actual_train/takeover_all_failure_types.json",
                "meaning": "All fifteen failure types; exact assistant top-level keys are execution_decision and decision_detail.",
            },
            "end": {
                "examples": "review_only/end_candidate.json",
                "use_for_training": False,
            },
        },
        "forbidden_absent_file": "actual_train/takeover_q_continue.json",
    }
    readme = """# V5.1 production-format confirmation\n\nAll examples are rendered from immutable snapshot rows with the production V5.1 renderer. `actual_train/train_rows.jsonl` contains the deduplicated 22-row confirmation selection: two initial plans, two complete memory pairs, one Segment-only ongoing row, and all fifteen Takeover failure types. Additional `*_output_profiles.json` files expose every prompt/output capability profile found in the smoke snapshot for review; they can repeat rows from the main selection.\n\nTakeover-Q Q1/Continue is intentionally absent. There is no `actual_train/takeover_q_continue.json`. Takeover outputs contain only `execution_decision` and `decision_detail`. End remains review-only because the available RoboDojo endpoint lacks independent physical-completion evidence. Training loaders must read the immutable snapshot manifests rather than recursively glob this review directory.\n"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    _write_json(output / "CATEGORY_INDEX.json", category_index)
    _write_json(output / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--takeover-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export(
        args.snapshot,
        args.output,
        takeover_source=args.takeover_source,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
