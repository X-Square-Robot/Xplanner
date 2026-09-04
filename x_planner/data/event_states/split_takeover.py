"""Freeze an episode-disjoint 80/20 split for new_completed Takeover-Q."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from .holdout import EvaluationHoldout, DEFAULT_EVALUATION_MANIFEST, DEFAULT_EVALUATION_SHA256


DEFAULT_ROOT = Path(
    "/data/takeover_q_dataset/"
    "reviewed_bilingual/current/new_completed"
)
SPLIT_VERSION = "v5_3_takeover_episode_split_v1"
FAILURE_CODES = (
    "1.1", "1.2", "1.3", "1.4", "1.5", "2.1", "2.2", "3.1",
    "4.1", "4.2", "4.3", "5.1", "6.1", "7.1", "8.1",
)
_FAILURE_CODE = re.compile(r"(?<!\d)([1-8]\.[1-5])(?!\d)")
_TASK_PREFIX = re.compile(r"^\d{8}-(?:day|night)-\d+-", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_fraction(*parts: object) -> float:
    payload = "\0".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _task_key(episode_id: str) -> str:
    base = episode_id.split("@", 1)[0]
    return _TASK_PREFIX.sub("", base) or "unknown"


def _case_bin(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 7:
        return "4-7"
    return "8+"


def _failure_codes(episode: Mapping[str, Any]) -> tuple[str, ...]:
    result: set[str] = set()
    cases = episode.get("cases")
    if not isinstance(cases, list):
        return ()
    for case in cases:
        bilingual = case.get("bilingual") if isinstance(case, Mapping) else None
        segments = bilingual.get("segments") if isinstance(bilingual, Mapping) else None
        q2q3 = segments.get("q2q3") if isinstance(segments, Mapping) else None
        if not isinstance(q2q3, list):
            continue
        for item in q2q3:
            if not isinstance(item, Mapping):
                continue
            match = _FAILURE_CODE.search(str(item.get("q3_type") or ""))
            if match and match.group(1) in FAILURE_CODES:
                result.add(match.group(1))
    return tuple(sorted(result))


def load_episode_inventory(
    root: Path,
    *,
    holdout: EvaluationHoldout,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index_path = root / "episodes.jsonl"
    inventory: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    with index_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            episode_relative = Path(str(row.get("episode") or ""))
            if episode_relative.is_absolute() or ".." in episode_relative.parts:
                raise ValueError(f"unsafe episode path on line {line_number}")
            episode_path = root / episode_relative
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            codes = _failure_codes(episode)
            episode_key = str(row.get("episode_key") or "")
            episode_id = str(row.get("episode_id") or "")
            videos = row.get("videos") if isinstance(row.get("videos"), Mapping) else {}
            pseudo_sample = {
                "images": list(videos.values()),
                "provenance": {
                    "episode_key": episode_key,
                    "episode_id": episode_id,
                    "episode_path": str(row.get("raw_episode_dir") or ""),
                    "raw_video_paths": dict(videos),
                },
            }
            matches = holdout.match_sample(pseudo_sample)
            if matches or not codes:
                excluded.append({
                    "episode_key": episode_key,
                    "reason": "evaluation_holdout_overlap" if matches else "no_supported_failure_type",
                    "matches": matches,
                })
                continue
            inventory.append({
                "episode_key": episode_key,
                "episode_id": episode_id,
                "episode": episode_relative.as_posix(),
                "case_count": int(row.get("case_count") or 0),
                "case_ids": list(row.get("case_ids") or ()),
                "failure_codes": list(codes),
                "task_key": _task_key(episode_id),
                "case_count_bin": _case_bin(int(row.get("case_count") or 0)),
                "raw_episode_dir": str(row.get("raw_episode_dir") or ""),
                "videos": dict(videos),
                "source_index_line": line_number,
            })
    inventory.sort(key=lambda item: item["episode_key"])
    return inventory, excluded


def stratified_split(
    inventory: Sequence[Mapping[str, Any]],
    *,
    test_fraction: float = 0.20,
    seed: int = 827,
) -> tuple[set[str], dict[str, Any]]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    if not inventory:
        raise ValueError("Takeover inventory is empty")
    target_size = max(1, round(len(inventory) * test_fraction))
    class_totals = Counter(code for item in inventory for code in item["failure_codes"])
    missing = sorted(set(FAILURE_CODES) - set(class_totals))
    scarce = sorted(code for code, count in class_totals.items() if count < 2)
    if missing or scarce:
        raise ValueError(
            f"failure types cannot cover both train/test; missing={missing}, scarce={scarce}"
        )
    class_targets = {
        code: min(count - 1, max(1, round(count * test_fraction)))
        for code, count in class_totals.items()
    }
    task_totals = Counter(str(item["task_key"]) for item in inventory)
    task_targets = {
        task: round(count * test_fraction)
        for task, count in task_totals.items()
        if count >= 5
    }
    bin_totals = Counter(str(item["case_count_bin"]) for item in inventory)
    bin_targets = {name: round(count * test_fraction) for name, count in bin_totals.items()}
    total_cases = sum(int(item["case_count"]) for item in inventory)
    target_cases = round(total_cases * test_fraction)

    selected: set[str] = set()
    class_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    bin_counts: Counter[str] = Counter()
    selected_cases = 0
    by_key = {str(item["episode_key"]): item for item in inventory}

    while len(selected) < target_size:
        best_key: str | None = None
        best_score: tuple[float, float] | None = None
        for item in inventory:
            key = str(item["episode_key"])
            if key in selected:
                continue
            codes = tuple(str(code) for code in item["failure_codes"])
            if any(class_counts[code] + 1 >= class_totals[code] for code in codes):
                continue
            class_gain = sum(
                max(0, class_targets[code] - class_counts[code]) / class_totals[code]
                for code in codes
            )
            task = str(item["task_key"])
            task_gain = (
                max(0, task_targets.get(task, 0) - task_counts[task]) / task_totals[task]
                if task in task_targets else 0
            )
            case_bin = str(item["case_count_bin"])
            bin_gain = max(0, bin_targets[case_bin] - bin_counts[case_bin]) / bin_totals[case_bin]
            case_gain = max(0, target_cases - selected_cases) / max(total_cases, 1)
            score = (
                class_gain * 100 + task_gain * 8 + bin_gain * 5 + case_gain,
                _stable_fraction(seed, key),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_key = key
        if best_key is None:
            raise ValueError("cannot complete Takeover test split without emptying a train class")
        item = by_key[best_key]
        selected.add(best_key)
        class_counts.update(item["failure_codes"])
        task_counts[str(item["task_key"])] += 1
        bin_counts[str(item["case_count_bin"])] += 1
        selected_cases += int(item["case_count"])

    test_missing = sorted(code for code in FAILURE_CODES if class_counts[code] == 0)
    train_missing = sorted(
        code for code in FAILURE_CODES if class_counts[code] == class_totals[code]
    )
    if test_missing or train_missing:
        raise ValueError(
            f"Takeover split lost failure coverage; test={test_missing}, train={train_missing}"
        )
    report = {
        "target_test_fraction": test_fraction,
        "seed": seed,
        "eligible_episodes": len(inventory),
        "train_episodes": len(inventory) - len(selected),
        "test_episodes": len(selected),
        "observed_test_fraction": len(selected) / len(inventory),
        "eligible_cases": total_cases,
        "test_cases": selected_cases,
        "observed_test_case_fraction": selected_cases / max(total_cases, 1),
        "failure_type_episodes": {
            code: {
                "all": class_totals[code],
                "train": class_totals[code] - class_counts[code],
                "test": class_counts[code],
                "test_fraction": class_counts[code] / class_totals[code],
            }
            for code in FAILURE_CODES
        },
        "case_count_bins": {
            name: {"all": bin_totals[name], "test": bin_counts[name]}
            for name in sorted(bin_totals)
        },
        "train_test_episode_overlap": 0,
        "all_failure_types_in_both_splits": True,
    }
    return selected, report


def materialize_split(
    *,
    root: Path,
    output_root: Path,
    holdout: EvaluationHoldout,
    test_fraction: float = 0.20,
    seed: int = 827,
) -> dict[str, Any]:
    inventory, excluded = load_episode_inventory(root, holdout=holdout)
    test_keys, distribution = stratified_split(
        inventory, test_fraction=test_fraction, seed=seed
    )
    rows = [
        {"schema_version": SPLIT_VERSION, **dict(item), "split": (
            "test" if item["episode_key"] in test_keys else "train"
        )}
        for item in inventory
    ]
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    _atomic_jsonl(output_root / "episodes.jsonl", rows)
    _atomic_jsonl(output_root / "train_episodes.jsonl", train)
    _atomic_jsonl(output_root / "test_episodes.jsonl", test)
    _atomic_jsonl(output_root / "excluded_episodes.jsonl", excluded)
    index_path = root / "episodes.jsonl"
    report = {
        "schema_version": SPLIT_VERSION,
        "complete": True,
        "source_root": str(root.resolve()),
        "source_index": str(index_path.resolve()),
        "source_index_sha256": _sha256(index_path),
        "legacy_gold_considered": False,
        "evaluation_holdout": holdout.metadata(),
        "evaluation_holdout_excluded": sum(item["reason"] == "evaluation_holdout_overlap" for item in excluded),
        "excluded_episodes": len(excluded),
        "distribution": distribution,
        "files": {},
    }
    for name in ("episodes.jsonl", "train_episodes.jsonl", "test_episodes.jsonl", "excluded_episodes.jsonl"):
        report["files"][name] = {"sha256": _sha256(output_root / name)}
    _atomic_jsonl(output_root / "split_report.json", [report])
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=827)
    parser.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_EVALUATION_MANIFEST)
    parser.add_argument("--evaluation-sha256", default=DEFAULT_EVALUATION_SHA256)
    args = parser.parse_args(argv)
    holdout = EvaluationHoldout.load(args.evaluation_holdout, expected_sha256=args.evaluation_holdout_sha256)
    report = materialize_split(
        root=args.root,
        output_root=args.output_root,
        holdout=holdout,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FAILURE_CODES", "load_episode_inventory", "materialize_split", "stratified_split"]
