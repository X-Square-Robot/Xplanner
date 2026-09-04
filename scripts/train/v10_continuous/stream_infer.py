#!/usr/bin/env python3
"""Stream V10 samples through the serving MemoryBank and emit strict JSON results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .memory import MemoryBank
from .schema import loads_target
from .validation_runtime import ModelGenerator, _observation, iter_snapshot_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--processor-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    generator = ModelGenerator(
        args.model_path.resolve(),
        args.processor_path.resolve(),
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    bank = MemoryBank()
    active_episode = None
    samples = iter_snapshot_samples(args.snapshot.resolve(), args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for sample in samples:
            if args.limit > 0 and count >= args.limit:
                break
            if sample.episode_key != active_episode:
                bank.reset(sample.episode_key)
                active_episode = sample.episode_key
            input_memory = tuple(bank.long_memory)
            output = generator.generate(sample, input_memory)
            update = None
            try:
                target = loads_target(output, sample.profile)
                state = bank.step(
                    sample.episode_key,
                    _observation(target, sample.unit_type),
                )
                update = {
                    "committed": state.committed,
                    "reason": state.reason,
                    "long_memory": list(state.long_memory),
                }
            except Exception as exc:
                update = {"error": f"{type(exc).__name__}: {exc}"[:500]}
            row = {
                "sample_id": sample.sample_id,
                "episode_key": sample.episode_key,
                "current_frame": sample.current_frame,
                "input_long_memory": list(input_memory),
                "assistant_json": output,
                "memory_update": update,
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            count += 1
    os.replace(temporary, args.output)
    print(json.dumps({"samples": count, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
