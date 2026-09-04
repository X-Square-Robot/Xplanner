"""Decode, tokenize, collate, and audit a real V5 batch."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
import yaml

from . import dataset_v5 as _dataset_v5  # noqa: F401
from .loss_mask_v5 import contains_mask_marker
from .holdout_v5 import (
    Benchmark3Holdout,
    DEFAULT_BENCHMARK3_MANIFEST,
    DEFAULT_BENCHMARK3_SHA256,
)
from ..v10_continuous_v3_memory.common_v3 import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    holdout = Benchmark3Holdout.load(
        DEFAULT_BENCHMARK3_MANIFEST,
        expected_sha256=DEFAULT_BENCHMARK3_SHA256,
    )
    prepare_path = args.data_config.resolve().parent / "prepare_summary.json"
    if not prepare_path.is_file():
        raise FileNotFoundError(
            f"V5 data smoke requires a fenced prepare summary: {prepare_path}"
        )
    prepare_summary = json.loads(prepare_path.read_text(encoding="utf-8"))
    prepared_holdout = prepare_summary.get("benchmark3_holdout")
    if (
        not isinstance(prepared_holdout, dict)
        or prepared_holdout.get("manifest_sha256") != holdout.manifest_sha256
        or prepared_holdout.get("training_overlap_samples") != 0
    ):
        raise ValueError("V5 data smoke refuses an unfenced data config")
    from qwenvl.train.builders import build_dataset_v2

    config = copy.deepcopy(yaml.safe_load(args.data_config.read_text(encoding="utf-8")))
    config["dataset"]["sampler"]["batch_size"] = args.batch_size
    epilogue = config["dataset"]["processors"]["epilogue"]["params"]
    epilogue["packing"] = False
    max_length = int(epilogue["max_seq_length"])
    processor = AutoProcessor.from_pretrained(str(args.model_path.resolve()))
    checkpoint_tokenizer_size = len(processor.tokenizer)
    processor.image_processor.min_pixels = 1024
    processor.image_processor.max_pixels = 589824
    dataset, sampler = build_dataset_v2(
        config,
        processor,
        action_tokenizer=None,
        model_max_length=max_length,
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
        raise ValueError("V5 input/attention/label shape mismatch")
    if input_ids.shape[0] != args.batch_size:
        raise ValueError("V5 data smoke batch size mismatch")
    # Mask delimiter AddedTokens deliberately live above the checkpoint's
    # original vocabulary. They must all be removed by the V5 epilogue before
    # a batch reaches the model.
    if int(input_ids.max().item()) >= checkpoint_tokenizer_size:
        raise ValueError("V5 removable marker token ID leaked into model input")
    supervised = labels != -100
    if not bool(supervised.any()):
        raise ValueError("V5 batch has no supervised assistant tokens")
    if bool((labels[attention_mask == 0] != -100).any()):
        raise ValueError("V5 padding tokens are supervised")
    decoded_inputs = [
        processor.tokenizer.decode(row.detach().cpu().tolist(), skip_special_tokens=False)
        for row in input_ids
    ]
    if any(contains_mask_marker(value) or "V5MASK_" in value for value in decoded_inputs):
        raise ValueError("V5 removable mask marker leaked into model input")
    finite = all(
        bool(torch.isfinite(value).all())
        for value in batch.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    )
    if not finite:
        raise FloatingPointError("non-finite tensor in V5 data batch")
    result = {
        "schema_version": "v10_action_segment_v5_data_smoke_v1",
        "passed": True,
        "batch_size": args.batch_size,
        "max_length": max_length,
        "input_shape": list(input_ids.shape),
        "supervised_tokens": int(supervised.sum().item()),
        "masked_tokens": int((labels == -100).sum().item()),
        "marker_leaks": 0,
        "max_input_token_id": int(input_ids.max().item()),
        "checkpoint_tokenizer_size": checkpoint_tokenizer_size,
        "pixel_values_shape": list(batch["pixel_values"].shape),
        "image_grid_thw_shape": list(batch["image_grid_thw"].shape),
        "finite_tensors": finite,
        "benchmark3_manifest_sha256": holdout.manifest_sha256,
        "benchmark3_overlap_samples": 0,
    }
    write_json(str(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
