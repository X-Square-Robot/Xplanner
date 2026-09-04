"""Probe all 68 valid source/profile/stage cells under the trained V4 contract.

This is deliberately an Oracle-context compatibility probe: continuous prompts use
their stored GT memories and profile scale.  Its metrics must never be presented as
causal rollout quality.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from x2robot_dataset_v2.readers.multimodal_jsonl_reader import load_indexed_jsonl_item

from .infer_tasks import Generator, _extract_json
from .prompt import render_user
from .schema import dumps_assistant, loads_assistant


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--initial-max-new-tokens", type=int, default=2048)
    parser.add_argument("--continuous-max-new-tokens", type=int, default=512)
    parser.add_argument("--stop-after-root-json", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cells = manifest["source_profile_cells"]
    if len(cells) * 4 != 68:
        raise ValueError(f"expected 68 cells, got {len(cells) * 4}")
    snapshot = args.snapshot.resolve()
    initial_root = snapshot / "datasets" / "initial_plan" / "validation"
    continuous_root = snapshot / "datasets" / "continuous" / "validation"
    generator = Generator(args.checkpoint.resolve(), device=args.device)
    records: list[dict[str, Any]] = []
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for cell in cells:
        episode = cell["episode"]
        for stage in cell["stages"]:
            task = "initial_plan" if stage == "initial" else "continuous"
            root = initial_root if task == "initial_plan" else continuous_root
            row_index = int(episode["stage_rows"][stage])
            row = load_indexed_jsonl_item(str(root), row_index)
            sample = row["v4_sample"]
            rendered_prompt = render_user(sample)
            stored_prompt = row["text"][0]["text"]
            if stored_prompt != rendered_prompt:
                raise ValueError(
                    f"stored/rendered prompt mismatch for {sample['sample_key']}"
                )
            budget = (
                args.initial_max_new_tokens
                if task == "initial_plan"
                else args.continuous_max_new_tokens
            )
            raw, generation = generator.generate(
                row,
                root,
                max_new_tokens=budget,
                stop_after_root_json=args.stop_after_root_json,
            )
            prediction = None
            error = None
            try:
                prediction = loads_assistant(
                    _extract_json(raw),
                    str(sample["profile"]),
                    str(sample["task_type"]),
                    instruction=str(sample["task_instruction"]),
                    is_terminal_window=bool(sample.get("is_terminal_window", False)),
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            record = {
                "source_id": cell["source_id"],
                "profile": cell["profile"],
                "stage": stage,
                "task": task,
                "row_index": row_index,
                "sample_key": sample["sample_key"],
                "task_instruction": sample["task_instruction"],
                "input_contract": "trained_profile_scale_with_gt_memory_oracle_diagnostic",
                "prompt": rendered_prompt,
                "prediction_raw": raw,
                "prediction": prediction,
                "prediction_schema_valid": prediction is not None,
                "prediction_schema_error": error,
                "gt": sample["target"],
                "gt_text": dumps_assistant(
                    sample["target"],
                    str(sample["profile"]),
                    str(sample["task_type"]),
                    instruction=str(sample["task_instruction"]),
                    is_terminal_window=bool(sample.get("is_terminal_window", False)),
                ),
                **generation,
            }
            records.append(record)
            cell_dir = output / str(cell["source_id"]) / str(cell["profile"])
            _atomic(cell_dir / f"{stage}.json", record)
            print(json.dumps({
                "source": cell["source_id"],
                "profile": cell["profile"],
                "stage": stage,
                "schema_valid": prediction is not None,
            }, sort_keys=True), flush=True)
    by_stage = Counter(record["stage"] for record in records if record["prediction_schema_valid"])
    by_source = Counter(record["source_id"] for record in records if record["prediction_schema_valid"])
    summary = {
        "schema_version": "memory_v4_contract_68_probe_v1",
        "started_at": started,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": generator.checkpoint,
        "snapshot": str(snapshot),
        "manifest": str(args.manifest.resolve()),
        "evaluation_kind": "oracle_context_trained_contract_compatibility_only",
        "causal_rollout_metric": False,
        "stop_after_root_json": args.stop_after_root_json,
        "sample_count": len(records),
        "schema_valid_count": sum(record["prediction_schema_valid"] for record in records),
        "schema_valid_by_stage": dict(sorted(by_stage.items())),
        "schema_valid_by_source": dict(sorted(by_source.items())),
        "records": [{
            key: record[key]
            for key in (
                "source_id", "profile", "stage", "row_index", "sample_key",
                "prediction_schema_valid", "prediction_schema_error", "input_tokens",
                "output_tokens", "image_count", "generation_seconds", "peak_allocated_mib",
            )
        } for record in records],
    }
    _atomic(output / "summary.json", summary)
    print(json.dumps({
        "sample_count": len(records),
        "schema_valid_count": summary["schema_valid_count"],
        "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
