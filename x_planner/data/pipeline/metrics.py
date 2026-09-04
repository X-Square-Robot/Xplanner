"""Strict JSON parsing and transparent teacher/rollout scoring."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .captions import normalize_caption
from .constants import PROFILE_FIELDS
from .schema import TargetValidationError, loads_target


@dataclass(frozen=True, slots=True)
class TargetScore:
    score: float
    exact_match: bool
    valid_json: bool
    matched_components: int
    total_components: int
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _caption_equal(left: Any, right: Any) -> bool:
    return isinstance(left, str) and isinstance(right, str) and (
        normalize_caption(left) == normalize_caption(right)
    )


def score_target_text(text: str, target: dict[str, Any], profile: str) -> TargetScore:
    """Score every non-constant target leaf equally after strict wire validation."""

    try:
        predicted = loads_target(text.strip(), profile)
    except (TargetValidationError, TypeError, AttributeError) as exc:
        return TargetScore(0.0, False, False, 0, 1, str(exc)[:500])

    comparisons: list[bool] = [
        _caption_equal(predicted["task"]["caption"], target["task"]["caption"]),
        predicted["task"]["progress_percent"] == target["task"]["progress_percent"],
        len(predicted["predictions"]) == len(target["predictions"]),
    ]
    fields = PROFILE_FIELDS[profile]
    for position in range(max(len(predicted["predictions"]), len(target["predictions"]))):
        if position >= len(predicted["predictions"]) or position >= len(target["predictions"]):
            comparisons.extend(
                False
                for field in fields
                for _leaf in range(3 if field == "l0" else 2)
            )
            continue
        prediction = predicted["predictions"][position]
        truth = target["predictions"][position]
        for field in fields:
            comparisons.append(_caption_equal(prediction[field]["caption"], truth[field]["caption"]))
            comparisons.append(
                prediction[field]["progress_percent"] == truth[field]["progress_percent"]
            )
            if field == "l0":
                comparisons.append(prediction[field]["source"] == truth[field]["source"])
    matched = sum(comparisons)
    total = len(comparisons)
    return TargetScore(
        score=matched / total if total else 0.0,
        exact_match=all(comparisons),
        valid_json=True,
        matched_components=matched,
        total_components=total,
    )


class ScoreAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.score_sum = 0.0
        self.exact = 0
        self.valid = 0
        self.errors: Counter[str] = Counter()

    def add(self, result: TargetScore) -> None:
        self.count += 1
        self.score_sum += result.score
        self.exact += int(result.exact_match)
        self.valid += int(result.valid_json)
        if result.error:
            self.errors[result.error] += 1

    def report(self, *, name: str) -> dict[str, Any]:
        denominator = max(1, self.count)
        return {
            name: self.score_sum / denominator,
            "samples": self.count,
            "exact_match_rate": self.exact / denominator,
            "valid_json_rate": self.valid / denominator,
            "top_errors": dict(self.errors.most_common(10)),
        }


def write_jsonl(path, rows: Iterable[dict[str, Any]]) -> None:
    from pathlib import Path

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(destination)
