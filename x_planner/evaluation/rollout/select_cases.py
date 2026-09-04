"""Select deterministic validation Episodes for causal Memory V4 evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from x2robot_dataset_v2.readers.multimodal_jsonl_reader import (
    load_indexed_jsonl_item,
)


SEED = 20260813
RICHNESS = {
    "full": 7,
    "L3L2L1": 6,
    "L3L2L0": 5,
    "L3L2": 4,
    "L3L1L0": 3,
    "L3L1": 2,
    "L3L0": 1,
}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object at {path}:{line_number}")
            yield value


def _score(*values: Any) -> str:
    payload = "\0".join(str(value) for value in (SEED, *values))
    return hashlib.sha256(payload.encode()).hexdigest()


def _source(global_episode_key: str) -> str:
    source, separator, _ = global_episode_key.partition(":")
    if not separator or not source:
        raise ValueError(f"invalid global_episode_key: {global_episode_key!r}")
    return source


@dataclass(frozen=True)
class EpisodeCandidate:
    global_episode_key: str
    source_id: str
    profile: str
    initial_plan_row: int
    continuous_rows: tuple[int, ...]
    start_row: int
    process_row: int
    terminal_row: int
    selection_score: str


def _candidate(
    group: list[dict[str, Any]],
    *,
    initial_plan_row: int,
    terminal_rows: set[int],
) -> EpisodeCandidate | None:
    key = str(group[0]["global_episode_key"])
    profiles = {str(row["profile"]) for row in group}
    if len(profiles) != 1:
        raise ValueError(f"cross-profile Episode group: {key}")
    indices = tuple(int(row["row_index"]) for row in group)
    terminal = [index for index in indices if index in terminal_rows]
    nonterminal = [index for index in indices if index not in terminal_rows]
    if not terminal or len(nonterminal) < 2 or indices[0] in terminal_rows:
        return None
    process = nonterminal[len(nonterminal) // 2]
    return EpisodeCandidate(
        global_episode_key=key,
        source_id=_source(key),
        profile=profiles.pop(),
        initial_plan_row=initial_plan_row,
        continuous_rows=indices,
        start_row=indices[0],
        process_row=process,
        terminal_row=terminal[-1],
        selection_score=_score(key),
    )


def _retain(
    values: dict[tuple[str, str], list[EpisodeCandidate]],
    candidate: EpisodeCandidate,
    *,
    limit: int,
) -> None:
    cell = values.setdefault((candidate.source_id, candidate.profile), [])
    cell.append(candidate)
    cell.sort(key=lambda item: (len(item.continuous_rows), item.selection_score))
    del cell[limit:]


def _load_sample(root: Path, task: str, row_index: int) -> dict[str, Any]:
    dataset = root / "datasets" / task / "validation"
    row = load_indexed_jsonl_item(str(dataset), row_index)
    sample = row.get("v4_sample")
    if not isinstance(sample, dict):
        raise ValueError(f"{task} row {row_index} has no v4_sample")
    return sample


def _enrich(root: Path, candidate: EpisodeCandidate) -> dict[str, Any]:
    samples = [
        _load_sample(root, "continuous", index)
        for index in candidate.continuous_rows
    ]
    unit_indices = sorted({int(sample["unit_index"]) for sample in samples})
    anchor_gaps = [
        {
            "previous": int(previous["anchor_frame"]),
            "current": int(current["anchor_frame"]),
            "current_forced_terminal": bool(current.get("forced_terminal_anchor", False)),
        }
        for previous, current in zip(samples, samples[1:])
        if int(current["anchor_frame"]) - int(previous["anchor_frame"]) != 20
        and not bool(current.get("forced_terminal_anchor", False))
    ]
    terminal_sample = next(
        sample
        for sample in reversed(samples)
        if int(sample["anchor_frame"])
        == int(_load_sample(root, "continuous", candidate.terminal_row)["anchor_frame"])
    )
    initial = _load_sample(root, "initial_plan", candidate.initial_plan_row)
    if initial["global_episode_key"] != candidate.global_episode_key:
        raise ValueError("Initial Plan/continuous Episode mismatch")
    return {
        "episode_id": hashlib.sha256(candidate.global_episode_key.encode()).hexdigest()[:16],
        "global_episode_key": candidate.global_episode_key,
        "source_id": candidate.source_id,
        "profile": candidate.profile,
        "task_instruction": initial["task_instruction"],
        "initial_plan_row": candidate.initial_plan_row,
        "continuous_rows": list(candidate.continuous_rows),
        "stage_rows": {
            "initial": candidate.initial_plan_row,
            "start": candidate.start_row,
            "process": candidate.process_row,
            "terminal": candidate.terminal_row,
        },
        "anchor_count": len(candidate.continuous_rows),
        "unit_indices": unit_indices,
        "distinct_unit_count": len(unit_indices),
        "anchor_stride_contract_valid": not anchor_gaps,
        "invalid_anchor_gaps": anchor_gaps,
        "terminal_task_progress_percent": int(
            terminal_sample["target"]["task_progress_percent"]
        ),
        "selection_score": candidate.selection_score,
    }


def _review_set(episodes: list[dict[str, Any]]) -> list[str]:
    selected: list[str] = []
    for source in ("collection", "open_action", "zhengwei"):
        source_items = [
            item
            for item in episodes
            if item["source_id"] == source and item["anchor_stride_contract_valid"]
        ]
        if not source_items:
            raise ValueError(f"no validation Episodes selected for {source}")
        transition = min(
            source_items,
            key=lambda item: (
                0
                if item["distinct_unit_count"] >= 2 and item["anchor_count"] <= 50
                else 1,
                -item["distinct_unit_count"],
                -RICHNESS[item["profile"]],
                item["anchor_count"],
                item["selection_score"],
            ),
        )
        remaining = [item for item in source_items if item is not transition]
        terminal = min(
            remaining or source_items,
            key=lambda item: (
                0 if item["terminal_task_progress_percent"] >= 90 else 1,
                -RICHNESS[item["profile"]],
                item["anchor_count"],
                item["selection_score"],
            ),
        )
        selected.extend([transition["global_episode_key"], terminal["global_episode_key"]])
    if len(set(selected)) != 6:
        raise ValueError("review set must contain six distinct Episodes")
    return selected


def select(snapshot: Path, *, shortlist: int = 6) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    initial_path = snapshot / "lists" / "initial_plan_val.list"
    continuous_path = snapshot / "lists" / "continuous_val.list"
    terminal_path = snapshot / "lists" / "terminal_val.list"
    initial_by_episode = {
        str(row["global_episode_key"]): int(row["row_index"])
        for row in _rows(initial_path)
    }
    terminal_rows = {int(row["row_index"]) for row in _rows(terminal_path)}

    cells: dict[tuple[str, str], list[EpisodeCandidate]] = {}
    active_key: str | None = None
    active_group: list[dict[str, Any]] = []

    def finish() -> None:
        nonlocal active_key, active_group
        if active_key is None:
            return
        initial_row = initial_by_episode.get(active_key)
        if initial_row is not None:
            candidate = _candidate(
                active_group,
                initial_plan_row=initial_row,
                terminal_rows=terminal_rows,
            )
            if candidate is not None:
                _retain(cells, candidate, limit=shortlist)
        active_key = None
        active_group = []

    for row in _rows(continuous_path):
        key = str(row["global_episode_key"])
        if active_key is None:
            active_key = key
        elif key != active_key:
            finish()
            active_key = key
        active_group.append(row)
    finish()

    enriched_cells: list[dict[str, Any]] = []
    all_enriched: list[dict[str, Any]] = []
    for source_profile, candidates in sorted(cells.items()):
        enriched = [_enrich(snapshot, candidate) for candidate in candidates]
        enriched.sort(
            key=lambda item: (
                item["anchor_count"],
                item["selection_score"],
            )
        )
        valid = [item for item in enriched if item["anchor_stride_contract_valid"]]
        if not valid:
            raise ValueError(f"no stride-valid validation Episode for {source_profile}")
        chosen = valid[0]
        all_enriched.extend(enriched)
        enriched_cells.append({
            "source_id": source_profile[0],
            "profile": source_profile[1],
            "episode": chosen,
            "stages": ["initial", "start", "process", "terminal"],
        })
    if len(enriched_cells) != 17:
        raise ValueError(f"expected 17 validation source/profile cells, got {len(enriched_cells)}")
    episodes = [cell["episode"] for cell in enriched_cells]
    review_keys = _review_set(all_enriched)
    review = [
        next(item for item in all_enriched if item["global_episode_key"] == key)
        for key in review_keys
    ]
    return {
        "schema_version": "memory_v4_causal_eval_manifest_v1",
        "seed": SEED,
        "snapshot": str(snapshot),
        "split": "validation",
        "selection_rule": (
            "Shortest stride-valid Episode with Initial Plan, a nonterminal start/process "
            "row, and a final-unit row; only explicit forced-terminal gaps are allowed; "
            "SHA-256 tie break."
        ),
        "source_profile_cells": enriched_cells,
        "cell_count": len(enriched_cells),
        "stage_count": len(enriched_cells) * 4,
        "probe_episodes": episodes,
        "review_episodes": review,
        "review_episode_count": len(review),
        "terminal_label_semantics": "last_same_scale_unit_not_physical_task_completion",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shortlist", type=int, default=6)
    args = parser.parse_args()
    if args.shortlist < 2:
        raise ValueError("shortlist must be at least two")
    value = select(args.snapshot, shortlist=args.shortlist)
    _atomic_json(args.output.resolve(), value)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "cell_count": value["cell_count"],
        "stage_count": value["stage_count"],
        "review_episode_count": value["review_episode_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
