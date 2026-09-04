"""Decode, tokenize and collate a real Memory V3 batch with JSON-only loss."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from . import dataset as _dataset  # noqa: F401
from .common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=6)
    args = parser.parse_args()

    from x_planner.trainer.builders import build_dataset_v2

    config = copy.deepcopy(yaml.safe_load(args.data_config.read_text(encoding="utf-8")))
    config["dataset"]["sampler"]["batch_size"] = args.batch_size
    config["dataset"]["processors"]["epilogue"]["params"]["packing"] = False
    processor = AutoProcessor.from_pretrained(str(args.model_path.resolve()))
    processor.image_processor.min_pixels = 1024
    processor.image_processor.max_pixels = 589824
    dataset, sampler = build_dataset_v2(
        config,
        processor,
        action_tokenizer=None,
        model_max_length=4096,
        get_rope_index=None,
    )
    loader = DataLoader(
        dataset,
        batch_size=getattr(sampler, "batch_size", args.batch_size),
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )
    batch = next(iter(loader))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    if input_ids.shape != attention_mask.shape or input_ids.shape != labels.shape:
        raise ValueError("input/attention/label shape mismatch")
    if input_ids.shape[0] != args.batch_size:
        raise ValueError(f"expected batch {args.batch_size}, got {input_ids.shape[0]}")
    supervised = labels != -100
    if not bool(supervised.any()) or bool((labels[attention_mask == 0] != -100).any()):
        raise ValueError("invalid JSON-only supervision mask")
    decoded = []
    for row in labels:
        values = row[row != -100].detach().cpu().tolist()
        text = processor.tokenizer.decode(values, skip_special_tokens=True).strip()
        # Token boundaries may retain template punctuation, but a complete JSON
        # object must exist and consume the supervised semantic payload.
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError(f"supervised labels contain no JSON object: {text[:200]!r}")
        json.loads(text[start:end + 1])
        decoded.append(text[start:end + 1])
    result = {
        "schema_version": "memory_v3_data_smoke_v1",
        "passed": True,
        "batch_size": args.batch_size,
        "input_shape": list(input_ids.shape),
        "supervised_tokens": int(supervised.sum()),
        "masked_tokens": int((~supervised).sum()),
        "pixel_values_shape": list(batch["pixel_values"].shape),
        "image_grid_thw_shape": list(batch["image_grid_thw"].shape),
        "finite_tensors": all(
            bool(torch.isfinite(value).all())
            for value in batch.values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        ),
        "json_rows_parsed": len(decoded),
        "first_supervised_json": decoded[0],
    }
    if not result["finite_tensors"]:
        raise FloatingPointError("non-finite tensor in Memory V3 batch")
    write_json(str(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
