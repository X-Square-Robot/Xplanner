"""Export decision-level distillation samples from local Harbor trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def export_trial(trial_dir: Path) -> list[dict[str, Any]]:
    config = _read_json(trial_dir / "config.json")
    result = _read_json(trial_dir / "result.json") if (trial_dir / "result.json").exists() else {}
    failure_path = trial_dir / "failure" / "report.json"
    failure = _read_json(failure_path) if failure_path.exists() else {}
    actions = {int(x["step"]): x for x in _read_jsonl(trial_dir / "executor" / "actions.jsonl") if "step" in x}
    samples = []
    for decision in _read_jsonl(trial_dir / "planner" / "decisions.jsonl"):
        if "step" not in decision:
            continue
        step = int(decision["step"])
        action = actions.get(step)
        samples.append({
            "schema_version": "xplanner_harbor_distill_v1",
            "sample_id": f"{config.get('trial_id', trial_dir.name)}_step_{step:06d}",
            "source_trial": config.get("trial_id", trial_dir.name),
            "step": step,
            "observation": decision.get("observation") or decision.get("observation_path"),
            "history": decision.get("history", []),
            "current_event": decision.get("event_state") or decision.get("current_event"),
            "teacher_decision": decision.get("decision"),
            "teacher_plan": decision.get("plan") or decision.get("raw_output"),
            "action": action.get("actual_action") if action else None,
            "outcome": "success" if result.get("success") else "failure",
            "reward": result.get("reward"),
            "failure_category": failure.get("category"),
            "correction": failure.get("correction"),
        })
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trials = sorted(args.trials_root.glob("**/config.json"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as out:
        for config_path in trials:
            trial_dir = config_path.parent
            if not (trial_dir / "manifest.json").exists():
                continue
            for sample in export_trial(trial_dir):
                out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                count += 1
    print(f"exported {count} samples to {args.output}")


if __name__ == "__main__":
    main()
