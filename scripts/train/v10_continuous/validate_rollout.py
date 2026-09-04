#!/usr/bin/env python3
"""Resumable V10 rollout validation using the serving MemoryBank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .validation_runtime import ModelGenerator, OracleGenerator, run_validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--processor-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--oracle", action="store_true", help="Exercise MemoryBank/resume without loading a model.")
    args = parser.parse_args()
    if args.oracle:
        generator = OracleGenerator()
    else:
        if args.model_path is None or args.processor_path is None:
            parser.error("--model-path and --processor-path are required without --oracle")
        generator = ModelGenerator(
            args.model_path.resolve(),
            args.processor_path.resolve(),
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
    result = run_validation(
        mode="rollout",
        snapshot=args.snapshot.resolve(),
        split=args.split,
        output_path=args.output.resolve(),
        generator=generator,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
