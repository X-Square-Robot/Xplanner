"""Character-span markers and token masking for partially supervised V5 JSON."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

import torch


IGNORE_INDEX = -100
MAX_MASK_PATHS = 64
_MARKER_RE = re.compile(r"<\|V5MASK_[0-9]{2}_(?:START|END)\|>")
_MARKER_TOKEN_POOL = tuple(
    f"<|V5MASK_{index:02d}_{boundary}|>"
    for index in range(MAX_MASK_PATHS)
    for boundary in ("START", "END")
)


@dataclass(frozen=True)
class MaskMarker:
    path: str
    start_text: str
    end_text: str


def add_mask_markers(
    text: str,
    spans: Iterable[tuple[int, int, str]],
) -> tuple[str, list[MaskMarker]]:
    """Wrap non-overlapping character spans in unique removable markers."""
    normalized = sorted((int(start), int(end), str(path)) for start, end, path in spans)
    if len(normalized) > MAX_MASK_PATHS:
        raise ValueError(
            f"V5 loss mask supports at most {MAX_MASK_PATHS} paths per assistant turn"
        )
    previous_end = 0
    for start, end, _path in normalized:
        if start < previous_end or start < 0 or end <= start or end > len(text):
            raise ValueError("V5 loss-mask character spans are invalid or overlapping")
        previous_end = end
    marked = text
    markers: list[MaskMarker] = []
    for index, (start, end, path) in reversed(list(enumerate(normalized))):
        # These fixed delimiters are registered as AddedToken special tokens
        # immediately before encoding. A fixed bounded pool avoids unbounded
        # tokenizer growth while guaranteeing a one-token delimiter that does
        # not change with the preceding JSON character.
        start_text = f"<|V5MASK_{index:02d}_START|>"
        end_text = f"<|V5MASK_{index:02d}_END|>"
        if start_text in text or end_text in text:
            raise ValueError("V5 loss-mask marker collision")
        marked = marked[:start] + start_text + marked[start:end] + end_text + marked[end:]
        markers.append(MaskMarker(path=path, start_text=start_text, end_text=end_text))
    markers.reverse()
    return marked, markers


def _find_unique_subsequence(values: list[int], needle: list[int], name: str) -> tuple[int, int]:
    if not needle:
        raise ValueError(f"empty token sequence for {name}")
    matches = [
        index
        for index in range(0, len(values) - len(needle) + 1)
        if values[index : index + len(needle)] == needle
    ]
    if len(matches) != 1:
        raise ValueError(f"{name} marker token sequence matched {len(matches)} times")
    return matches[0], matches[0] + len(needle)


def ensure_mask_marker_tokens(tokenizer: Any) -> None:
    """Register a bounded marker pool as context-independent AddedTokens.

    Marker IDs can sit above the model vocabulary because every marker token
    is removed before the batch reaches the model. Registering the entire
    fixed pool once also keeps tokenizer state bounded and deterministic.
    """
    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(_MARKER_TOKEN_POOL)}
    )
    invalid = [
        token
        for token in _MARKER_TOKEN_POOL
        if len(tokenizer.encode(token, add_special_tokens=False)) != 1
    ]
    if invalid:
        raise ValueError(
            "V5 loss-mask delimiters were not registered as single tokens: "
            f"{invalid[:3]}"
        )


def apply_token_mask_and_remove_markers(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer: Any,
    markers: Iterable[MaskMarker],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Mask marker-bounded value tokens and remove marker tokens from the input."""
    ensure_mask_marker_tokens(tokenizer)
    ids = input_ids.tolist()
    remove = torch.zeros_like(input_ids, dtype=torch.bool)
    masked = torch.zeros_like(input_ids, dtype=torch.bool)
    paths = 0
    for marker in markers:
        start_ids = tokenizer.encode(marker.start_text, add_special_tokens=False)
        end_ids = tokenizer.encode(marker.end_text, add_special_tokens=False)
        start_begin, start_end = _find_unique_subsequence(ids, start_ids, marker.path + ":start")
        end_begin, end_end = _find_unique_subsequence(ids, end_ids, marker.path + ":end")
        if end_begin < start_end:
            raise ValueError(f"V5 mask marker order is invalid for {marker.path}")
        remove[start_begin:start_end] = True
        remove[end_begin:end_end] = True
        masked[start_end:end_begin] = True
        paths += 1
    out_labels = labels.clone()
    out_labels[masked] = IGNORE_INDEX
    keep = ~remove
    return (
        input_ids[keep],
        out_labels[keep],
        {
            "mask_paths": paths,
            "masked_value_tokens": int(masked.sum().item()),
            "removed_marker_tokens": int(remove.sum().item()),
        },
    )


def contains_mask_marker(text: str) -> bool:
    return bool(_MARKER_RE.search(text))


__all__ = [
    "IGNORE_INDEX",
    "MAX_MASK_PATHS",
    "MaskMarker",
    "add_mask_markers",
    "apply_token_mask_and_remove_markers",
    "contains_mask_marker",
    "ensure_mask_marker_tokens",
]
