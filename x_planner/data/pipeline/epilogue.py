"""V10-only Qwen3.5 epilogue with an exact JSON-only causal-LM mask."""

from __future__ import annotations

import json
from typing import Any

import torch

from x2robot_dataset_v2.processors.epilogue.base import register_epilogue
from x2robot_dataset_v2.processors.epilogue.qwen3_5_epilogue import (
    Qwen3_5MultimodalEpilogueProcessor,
)


def _last_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle:
        return -1
    for start in range(len(haystack) - len(needle), -1, -1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return -1


@register_epilogue("v10_multimodal_qwen3_5")
class V10Qwen35JsonOnlyEpilogue(Qwen3_5MultimodalEpilogueProcessor):
    """Keep chat-template tokens in context while supervising only target JSON."""

    def _encode_one(
        self,
        dialogues: list[dict[str, Any]],
        images: list[Any],
        videos: list[Any] | None = None,
        video_metas: list[dict[str, Any]] | None = None,
    ):
        encoded = list(super()._encode_one(dialogues, images, videos, video_metas))
        input_ids: torch.Tensor = encoded[0]
        assistant_turns = [turn for turn in dialogues if turn.get("role") == "assistant"]
        if len(assistant_turns) != 1:
            raise ValueError(
                f"V10 requires exactly one Assistant JSON turn, got {len(assistant_turns)}"
            )
        target_text = assistant_turns[0].get("text")
        if not isinstance(target_text, str):
            raise ValueError("V10 Assistant target must be a JSON string")
        parsed = json.loads(target_text)
        if not isinstance(parsed, dict):
            raise ValueError("V10 Assistant target must decode to a JSON object")
        target_ids = self.tokenizer.encode(target_text, add_special_tokens=False)
        start = _last_subsequence(input_ids.tolist(), target_ids)
        if start < 0:
            raise ValueError(
                "V10 JSON token sequence was truncated or not found in rendered dialogue"
            )
        labels = torch.full_like(input_ids, -100)
        labels[start : start + len(target_ids)] = input_ids[start : start + len(target_ids)]
        supervised_text = self.tokenizer.decode(
            labels[labels != -100].tolist(), skip_special_tokens=False
        )
        if supervised_text != target_text:
            raise ValueError(
                "V10 JSON-only loss mask round-trip mismatch: "
                f"expected={target_text!r} decoded={supervised_text!r}"
            )
        if any(token_id in set(target_ids) for token_id in self.tokenizer.all_special_ids):
            raise ValueError("V10 JSON target unexpectedly contains a special token")
        encoded[1] = labels
        return tuple(encoded)

