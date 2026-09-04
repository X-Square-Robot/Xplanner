"""Qwen3.5 epilogue that enforces V5 field-level loss masks."""

from __future__ import annotations

import copy
from typing import Any, Optional

import torch

from x2robot_dataset_v2.processors.epilogue.base import register_epilogue
from x2robot_dataset_v2.processors.epilogue.qwen3_5_epilogue import (
    Qwen3_5MultimodalEpilogueProcessor,
    _warn_rate_limited,
    build_qwen_messages,
    inject_per_turn_think,
    make_assistant_mask,
    substitute_video_blocks,
)

from .loss_mask import (
    add_mask_markers,
    apply_token_mask_and_remove_markers,
    ensure_mask_marker_tokens,
)


@register_epilogue("v10_action_segment_v5_qwen3_5")
class V5Qwen3_5EpilogueProcessor(Qwen3_5MultimodalEpilogueProcessor):
    """Base Qwen epilogue plus removable marker-bounded JSON value masks."""

    def _encode_one(
        self,
        dialogues: list[dict[str, Any]],
        images: list[Any],
        videos: Optional[list[Any]] = None,
        video_metas: Optional[list[dict[str, Any]]] = None,
    ):
        dialogues = copy.deepcopy(dialogues)
        all_markers = []
        for turn in dialogues:
            spans = turn.pop("loss_mask_char_spans", None)
            if spans is None:
                continue
            if turn.get("role") != "assistant":
                raise ValueError("V5 loss_mask_char_spans is valid only on assistant turns")
            text, markers = add_mask_markers(turn.get("text", "") or "", spans)
            turn["text"] = text
            all_markers.extend(markers)

        pvv = vgrid = None
        has_video = bool(videos) and any(bool(value) for value in videos)
        if has_video:
            blocks, pvv_list, grid_list = self._build_video_blocks(
                videos or [], video_metas or []
            )
            dialogues, used_videos = substitute_video_blocks(dialogues, blocks)
            if used_videos < len(blocks):
                _warn_rate_limited(
                    "v5_unused_videos",
                    "sample decoded %d video(s) but consumed %d tags",
                    len(blocks),
                    used_videos,
                )
            if used_videos:
                pvv = torch.cat(pvv_list[:used_videos], dim=0)
                vgrid = torch.cat(grid_list[:used_videos], dim=0)

        assistant_supervision: Optional[list[bool]] = None
        assistant_turns = [turn for turn in dialogues if turn.get("role") == "assistant"]
        if any("supervise" in turn for turn in assistant_turns):
            if any(not isinstance(turn.get("supervise"), bool) for turn in assistant_turns):
                raise ValueError("every V5 assistant turn must define boolean supervise")
            assistant_supervision = [bool(turn["supervise"]) for turn in assistant_turns]

        messages, used_images = build_qwen_messages(dialogues, images)
        rendered = self.hf_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if self.per_turn_think:
            rendered = inject_per_turn_think(rendered)
        # Registration must happen before hf_processor tokenizes ``rendered``;
        # registering only during post-processing would leave context-dependent
        # ordinary subword pieces in input_ids and make delimiters unfindable.
        if all_markers:
            ensure_mask_marker_tokens(self.tokenizer)
        inputs = self.hf_processor(
            text=[rendered],
            images=images[:used_images] if used_images else None,
            padding=False,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"][0]
        labels = make_assistant_mask(
            input_ids,
            self.assistant_header_ids,
            self.im_end_id,
            supervised_assistant_turns=assistant_supervision,
        )
        if all_markers:
            input_ids, labels, stats = apply_token_mask_and_remove_markers(
                input_ids, labels, self.tokenizer, all_markers
            )
            if stats["masked_value_tokens"] <= 0:
                raise ValueError("V5 field mask produced zero masked value tokens")
        input_ids, labels = self._safe_truncate(input_ids, labels)
        return (
            input_ids,
            labels,
            inputs.get("pixel_values"),
            inputs.get("image_grid_thw"),
            pvv,
            vgrid,
        )


__all__ = ["V5Qwen3_5EpilogueProcessor"]
