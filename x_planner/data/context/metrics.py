"""Terminal guardrail metrics for Memory V3 evaluation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .common import write_json
from .schema import is_terminal_prediction


def terminal_metrics(
    rows: Iterable[tuple[Mapping[str, Any], Mapping[str, Any], str]]
) -> dict[str, Any]:
    true_positive = false_positive = false_negative = true_negative = 0
    for prediction, target, profile in rows:
        predicted_terminal = is_terminal_prediction(prediction, profile)
        actual_terminal = is_terminal_prediction(target, profile)
        if predicted_terminal and actual_terminal:
            true_positive += 1
        elif predicted_terminal:
            false_positive += 1
        elif actual_terminal:
            false_negative += 1
        else:
            true_negative += 1
    terminal_total = true_positive + false_negative
    nonterminal_total = false_positive + true_negative
    return {
        "terminal_recall": true_positive / terminal_total if terminal_total else 0.0,
        "premature_terminal_rate": false_positive / nonterminal_total if nonterminal_total else 0.0,
        "missed_terminal_rate": false_negative / terminal_total if terminal_total else 0.0,
        "counts": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        },
    }


def evaluate_jsonl(path: Path) -> dict[str, Any]:
    counters = {"evaluated_rows": 0, "invalid_prediction_json": 0}

    def rows():
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                profile = str(record["profile"])
                target = record["target"]
                if isinstance(target, str):
                    target = json.loads(target)
                if not isinstance(target, Mapping):
                    raise TypeError(f"line {line_number}: target must be an object")
                prediction = record.get("prediction", record.get("assistant_json"))
                try:
                    if isinstance(prediction, str):
                        prediction = json.loads(prediction)
                    if not isinstance(prediction, Mapping):
                        raise TypeError("prediction must be an object")
                except (json.JSONDecodeError, TypeError):
                    counters["invalid_prediction_json"] += 1
                    prediction = {}
                counters["evaluated_rows"] += 1
                yield prediction, target, profile

    result = terminal_metrics(rows())
    return {
        "schema_version": "memory_v3_terminal_metrics_v1",
        **result,
        **counters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_jsonl(args.predictions)
    write_json(str(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
