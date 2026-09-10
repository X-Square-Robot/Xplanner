#!/usr/bin/env python3
"""Check the optional data backend required by X-Planner event-state entry points."""

from __future__ import annotations

import importlib
import sys


REQUIRED_MODULES = (
    "x2robot_dataset_v2.datasets.x2robot_dataset",
    "x2robot_dataset_v2.processors.epilogue.qwen3_5_epilogue",
    "x2robot_dataset_v2.processors.text.base",
    "x2robot_dataset_v2.processors.text.multimodal_jsonl_qwen3_5_text_processor",
    "x2robot_dataset_v2.processors.vision.base",
    "x2robot_dataset_v2.processors.vision.multimodal_jsonl_vision_processor",
    "x2robot_dataset_v2.readers.multimodal_jsonl_reader",
    "x2robot_dataset_v2.samplers.frame_sampler",
    "x2robot_dataset_v2.samplers.task_balance",
    "x2robot_dataset_v2.utils.multimodal_schema",
    "x2robot_dataset_v2.utils.multimodal_utils",
)


def main() -> int:
    missing: list[tuple[str, str]] = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - error text is the interface
            missing.append((name, f"{type(exc).__name__}: {exc}"))

    if missing:
        print("Missing X-Planner data-backend modules:", file=sys.stderr)
        for name, reason in missing:
            print(f"  - {name} ({reason})", file=sys.stderr)
        return 1

    print("X-Planner data-backend contract: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
