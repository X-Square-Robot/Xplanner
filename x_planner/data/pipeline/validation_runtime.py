"""Shared model generation and resumable teacher/rollout validation runtime."""

from __future__ import annotations

import json
import os
import random
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from .constants import UNIT_TYPE
from .memory import MemoryBank, MemoryCodec, UnitObservation
from .metrics import ScoreAccumulator, score_target_text
from .models import V10Sample
from .prompt import render_user_text
from .schema import dumps_target, loads_target
from .snapshot import atomic_write_json


UNIT_FIELD = {"subtask": "subtask", "action": "action", "segment": "l0"}


def render_generation_prompt(processor: Any, messages: list[dict[str, Any]]) -> str:
    """Render the same empty-think prefix that precedes JSON during V10 SFT.

    The V10 epilogue inserts ``<think>\n\n</think>\n\n`` immediately before
    the Assistant JSON and masks that prefix from loss.  Qwen3.5's default
    generation template instead opens an unclosed ``<think>`` block, which
    makes inference emit free-form reasoning before the JSON.  Disabling
    thinking here preserves the trained prefix while keeping the generated
    continuation JSON-only.  ``ModelGenerator`` is shared by validation and
    streaming inference, so both paths use this exact contract.
    """

    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def iter_snapshot_samples(snapshot: Path, split: str) -> Iterator[V10Sample]:
    data_path = snapshot / split / "data.jsonl"
    with data_path.open(encoding="utf-8") as handle:
        samples = [
            V10Sample.from_dict(json.loads(line)["v10_sample"])
            for line in handle if line.strip()
        ]
    yield from sorted(
        samples,
        key=lambda sample: (sample.episode_key, sample.unit_index, sample.current_frame),
    )


def _observation(target: dict[str, Any], unit_type: str) -> UnitObservation:
    field = UNIT_FIELD[unit_type]
    predictions = target["predictions"]
    current = predictions[0][field]
    following = predictions[1][field]["caption"] if len(predictions) == 2 else None
    return UnitObservation(
        caption=current["caption"],
        progress_percent=current["progress_percent"],
        next_caption=following,
    )


class OracleGenerator:
    """Deterministic structural runner used to test validation and resume logic."""

    def generate(self, sample: V10Sample, long_memory: tuple[str, ...]) -> str:
        return dumps_target(sample.target, sample.profile)


class ModelGenerator:
    def __init__(
        self,
        model_path: Path,
        processor_path: Path,
        *,
        max_new_tokens: int = 512,
        device: str = "cuda",
        memory_codec: MemoryCodec | None = None,
        # Match the stable V10 training/inference path on B30Z.  FlashAttention
        # has produced NaNs and illegal-memory-access failures for multi-image
        # sequences on this stack.
        attn_implementation: str = "sdpa",
    ) -> None:
        import torch
        from transformers import AutoProcessor

        from x_planner.data.video_frames import VideoFrameVisionProcessor
        from x2robot_dataset_v2.processors.epilogue.qwen3_5_epilogue import (
            build_qwen_messages,
        )
        from x2robot_dataset_v2.utils.multimodal_utils import process_dialogue
        from x_planner.trainer.builders import load_model

        from .runtime import apply_b30z_compatibility

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for model validation")
        apply_b30z_compatibility()
        self.torch = torch
        self.build_qwen_messages = build_qwen_messages
        self.process_dialogue = process_dialogue
        self.processor = AutoProcessor.from_pretrained(str(processor_path))
        self.model = load_model(str(model_path), torch.bfloat16, attn_implementation)
        self.model.to(device)
        self.model.eval()
        self.model.config.use_cache = True
        if hasattr(self.model.config, "text_config"):
            self.model.config.text_config.use_cache = True
        self.vision = VideoFrameVisionProcessor(
            image_factor=32,
            min_pixels=1024,
            max_pixels=589824,
            max_pixels_split_by_images=True,
            decoder_backend="av",
        )
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.memory_codec = memory_codec or MemoryCodec()

    def generate(self, sample: V10Sample, long_memory: tuple[str, ...]) -> str:
        user_text = render_user_text(
            sample,
            codec=self.memory_codec,
            long_memory=long_memory,
        )
        images = self.vision.load_image_refs(
            [
                {"video": image.video, "frame": image.frame, "view": image.view}
                for image in sample.images
            ],
            sample.images[0].video,
        )
        dialogues = self.process_dialogue(
            [{"role": "user", "text": user_text}],
            seed=random.Random(zlib.crc32(sample.sample_id.encode())).getrandbits(64),
            num_images=len(images),
        )
        messages, used = self.build_qwen_messages(dialogues, images)
        rendered = render_generation_prompt(self.processor, messages)
        inputs = self.processor(
            text=[rendered],
            images=images[:used] if used else None,
            padding=False,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        input_length = int(inputs["input_ids"].shape[1])
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        output_ids = generated[0, input_length:]
        return self.processor.tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()


def _load_partial(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    return rows, {str(row["sample_id"]): row for row in rows}


def run_validation(
    *,
    mode: str,
    snapshot: Path,
    split: str,
    output_path: Path,
    generator: Any,
    limit: int = 0,
) -> dict[str, Any]:
    if mode not in {"teacher_forced", "rollout"}:
        raise ValueError(f"unknown validation mode: {mode}")
    samples = list(iter_snapshot_samples(snapshot, split))
    if limit > 0:
        samples = samples[:limit]
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    if output_path.is_file() and not partial_path.exists():
        partial_path = output_path
    prior_rows, prior_by_id = _load_partial(partial_path)
    accumulator = ScoreAccumulator()
    for row in prior_rows:
        from .metrics import TargetScore
        accumulator.add(TargetScore(**row["score"]))

    bank = MemoryBank()
    vocabulary: dict[str, tuple[str, ...]] = defaultdict(tuple)
    per_episode: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        field = UNIT_FIELD[sample.unit_type]
        caption = sample.target["predictions"][0][field]["caption"]
        if caption not in per_episode[sample.episode_key]:
            per_episode[sample.episode_key].append(caption)
    vocabulary = {key: tuple(value) for key, value in per_episode.items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    active_episode: str | None = None
    with partial_path.open("a", encoding="utf-8") as handle:
        for sample in samples:
            if sample.episode_key != active_episode:
                bank.reset(
                    sample.episode_key,
                    canonical_captions=vocabulary[sample.episode_key],
                )
                active_episode = sample.episode_key
            existing = prior_by_id.get(sample.sample_id)
            if existing is not None:
                if mode == "rollout" and existing.get("valid_json"):
                    parsed = loads_target(existing["output"], sample.profile)
                    bank.step(sample.episode_key, _observation(parsed, sample.unit_type))
                continue

            input_memory = (
                tuple(sample.long_memory)
                if mode == "teacher_forced"
                else tuple(bank.long_memory)
            )
            output = generator.generate(sample, input_memory)
            score = score_target_text(output, sample.target, sample.profile)
            update = None
            if mode == "rollout" and score.valid_json:
                parsed = loads_target(output, sample.profile)
                update_value = bank.step(
                    sample.episode_key,
                    _observation(parsed, sample.unit_type),
                )
                update = {
                    "committed": update_value.committed,
                    "transitioned": update_value.transitioned,
                    "reason": update_value.reason,
                    "long_memory": list(update_value.long_memory),
                }
            row = {
                "sample_id": sample.sample_id,
                "episode_key": sample.episode_key,
                "profile": sample.profile,
                "unit_type": sample.unit_type,
                "current_frame": sample.current_frame,
                "input_long_memory": list(input_memory),
                "output": output,
                "valid_json": score.valid_json,
                "score": score.to_dict(),
                "memory_update": update,
            }
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            accumulator.add(score)

    if partial_path != output_path:
        os.replace(partial_path, output_path)
    metric_name = "teacher_forced_score" if mode == "teacher_forced" else "rollout_score"
    result = {
        "mode": mode,
        "snapshot": str(snapshot.resolve()),
        "split": split,
        "output": str(output_path.resolve()),
        **accumulator.report(name=metric_name),
    }
    atomic_write_json(output_path.with_suffix(".metrics.json"), result)
    return result
