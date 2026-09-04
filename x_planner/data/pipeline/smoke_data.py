#!/usr/bin/env python3
"""Load, decode, prompt, tokenize and collate one real V10 batch."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoProcessor

# Registration happens on import and is intentionally shared with training.
from . import runtime as _runtime  # noqa: F401
from .snapshot import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from x_planner.trainer.builders import build_dataset_v2

    config = yaml.safe_load(args.data_config.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    dataset_config = config["dataset"]
    dataset_config["sampler"] = {"type": "default", "batch_size": 1, "seed": 42}
    dataset_config["processors"]["epilogue"]["params"]["packing"] = False
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
        batch_size=getattr(sampler, "batch_size", 1),
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        num_workers=0,
    )
    batch = next(iter(loader))
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    if input_ids.shape != attention_mask.shape or input_ids.shape != labels.shape:
        raise ValueError(
            f"shape mismatch: {input_ids.shape}, {attention_mask.shape}, {labels.shape}"
        )
    supervised = int((labels != -100).sum())
    if supervised <= 0:
        raise ValueError("no Assistant JSON supervision tokens")
    if bool((labels[attention_mask == 0] != -100).any()):
        raise ValueError("padding participates in loss")
    supervised_ids = labels[0][labels[0] != -100].detach().cpu().tolist()
    supervised_text = processor.tokenizer.decode(
        supervised_ids, skip_special_tokens=False
    )
    special_tokens_in_labels = [
        token for token in processor.tokenizer.all_special_tokens
        if token and token in supervised_text
    ]
    result = {
        "passed": True,
        "input_shape": list(input_ids.shape),
        "attention_shape": list(attention_mask.shape),
        "label_shape": list(labels.shape),
        "supervised_tokens": supervised,
        "masked_tokens": int((labels == -100).sum()),
        "shifted_supervised_tokens": int((labels[:, 1:] != -100).sum()),
        "supervised_first_token_id": int(supervised_ids[0]),
        "supervised_last_token_id": int(supervised_ids[-1]),
        "supervised_text": supervised_text,
        "special_tokens_in_labels": special_tokens_in_labels,
        "pixel_values_shape": (
            list(batch["pixel_values"].shape) if "pixel_values" in batch else None
        ),
        "image_grid_thw_shape": (
            list(batch["image_grid_thw"].shape) if "image_grid_thw" in batch else None
        ),
        "finite_tensors": all(
            bool(torch.isfinite(value).all())
            for value in batch.values()
            if isinstance(value, torch.Tensor) and value.is_floating_point()
        ),
    }
    if not result["finite_tensors"]:
        raise FloatingPointError("non-finite tensor in collated V10 batch")
    atomic_write_json(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
