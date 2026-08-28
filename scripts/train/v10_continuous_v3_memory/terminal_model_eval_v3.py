"""Small deterministic terminal/nonterminal generation evaluation for Memory V3."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from x2robot_dataset_v2.readers.multimodal_jsonl_reader import (
    get_indexed_jsonl_sample_count,
    load_indexed_jsonl_item,
)
from x2robot_dataset_v2.utils.multimodal_utils import process_dialogue

from ..v10_continuous.runtime import apply_b30z_compatibility
from ..v10_continuous.validation_runtime import render_generation_prompt
from .common_v3 import write_json
from .dataset_v3 import MemoryV3VisionProcessor
from .metrics_v3 import evaluate_jsonl


PROFILES = ("full", "L3L2L1", "L3L2L0", "L3L2", "L3L1L0", "L3L1", "L3L0")


def _stratified_rows(root: Path, *, terminal: bool, limit: int) -> list[dict[str, Any]]:
    count = get_indexed_jsonl_sample_count(str(root))
    if count <= 0:
        raise ValueError(f"empty indexed dataset: {root}")
    selected: list[dict[str, Any]] = []
    per_profile: dict[str, int] = defaultdict(int)
    target_per_profile = max(1, limit // len(PROFILES))
    candidate_count = min(count, max(4096, limit * 256))
    indices = sorted({min(count - 1, index * count // candidate_count)
                      for index in range(candidate_count)})
    for index in indices:
        row = load_indexed_jsonl_item(str(root), index)
        if bool(row.get("is_terminal_window")) != terminal:
            continue
        profile = str(row.get("profile"))
        if profile not in PROFILES:
            continue
        if per_profile[profile] >= target_per_profile and len(selected) < len(PROFILES):
            continue
        row["_indexed_sample"] = index
        selected.append(row)
        per_profile[profile] += 1
        if len(selected) >= limit and all(per_profile[name] for name in PROFILES):
            break
    if len(selected) < limit:
        raise RuntimeError(
            f"only selected {len(selected)}/{limit} rows from {root}; profiles={dict(per_profile)}"
        )
    return selected[:limit]


class Generator:
    def __init__(self, checkpoint: Path, *, max_new_tokens: int, device: str) -> None:
        import torch
        from transformers import AutoProcessor
        from qwenvl.train.builders import load_model
        from x2robot_dataset_v2.processors.epilogue.qwen3_5_epilogue import (
            build_qwen_messages,
        )

        apply_b30z_compatibility()
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(str(checkpoint))
        self.model = load_model(str(checkpoint), torch.bfloat16, "sdpa")
        self.model.to(device)
        self.model.eval()
        self.model.config.use_cache = True
        if hasattr(self.model.config, "text_config"):
            self.model.config.text_config.use_cache = True
        self.vision = MemoryV3VisionProcessor(
            image_factor=32,
            min_pixels=1024,
            max_pixels=589824,
            pixel_cap=589824,
            target_long_edge=640,
            resize_policy_id="auto_near_640_no_upscale_v1",
            decoder_backend="av",
        )
        self.build_qwen_messages = build_qwen_messages
        self.max_new_tokens = int(max_new_tokens)
        self.device = device

    def generate(self, row: dict[str, Any]) -> str:
        user_turns = [turn for turn in row["text"] if turn.get("role") == "user"]
        if len(user_turns) != 1:
            raise ValueError(f"expected one user turn: {row.get('sample_key')}")
        refs = row["image"]
        images = self.vision.load_image_refs(refs, refs[0]["video"])
        dialogues = process_dialogue(
            [{"role": "user", "text": str(user_turns[0]["text"])}],
            seed=random.Random(zlib.crc32(str(row["sample_key"]).encode())).getrandbits(64),
            num_images=len(images),
        )
        messages, used = self.build_qwen_messages(dialogues, images)
        rendered = render_generation_prompt(self.processor, messages)
        inputs = self.processor(
            text=[rendered], images=images[:used] if used else None,
            padding=False, return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        input_length = int(inputs["input_ids"].shape[1])
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        return self.processor.tokenizer.decode(
            generated[0, input_length:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--terminal-limit", type=int, default=7)
    parser.add_argument("--nonterminal-limit", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"
    if prediction_path.exists():
        raise FileExistsError(prediction_path)
    terminal_root = args.snapshot / "datasets" / "terminal" / "validation"
    continuous_root = args.snapshot / "datasets" / "continuous" / "validation"
    rows = [
        *_stratified_rows(terminal_root, terminal=True, limit=args.terminal_limit),
        *_stratified_rows(continuous_root, terminal=False, limit=args.nonterminal_limit),
    ]
    generator = Generator(
        args.checkpoint.resolve(), max_new_tokens=args.max_new_tokens, device=args.device
    )
    with prediction_path.open("x", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            started = time.monotonic()
            output = generator.generate(row)
            target = row["v3_sample"]["target"]
            record = {
                "row_index": row_index,
                "sample_key": row["sample_key"],
                "indexed_sample": row["_indexed_sample"],
                "profile": row["profile"],
                "actual_terminal": bool(row["is_terminal_window"]),
                "assistant_json": output,
                "target": target,
                "generation_seconds": time.monotonic() - started,
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            print(json.dumps({
                "row": row_index + 1, "total": len(rows), "profile": row["profile"],
                "actual_terminal": bool(row["is_terminal_window"]),
                "generation_seconds": record["generation_seconds"],
            }, sort_keys=True), flush=True)
    report = evaluate_jsonl(prediction_path)
    report.update({
        "evaluation_kind": "stratified_generation_smoke_not_statistical_benchmark",
        "checkpoint": str(args.checkpoint.resolve()),
        "snapshot": str(args.snapshot.resolve()),
        "predictions": str(prediction_path.resolve()),
        "terminal_examples": args.terminal_limit,
        "nonterminal_examples": args.nonterminal_limit,
        "max_new_tokens": args.max_new_tokens,
    })
    write_json(str(args.output_dir / "terminal_metrics.json"), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
