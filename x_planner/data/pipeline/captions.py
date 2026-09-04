"""Caption selection, validation and canonical normalization."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_SPACE_RE = re.compile(r"\s+")
_TERMINAL_PUNCT_RE = re.compile(r"[\s\.,;:!?。！？，；：…]+$")
_INVALID = {"", "none", "n/a", "na", "null", "nan", "unknown"}


def clean_caption(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = _SPACE_RE.sub(" ", value).strip()
    return "" if text.casefold() in _INVALID else text


def is_valid_english_caption(value: Any) -> bool:
    text = clean_caption(value)
    return bool(text and _LATIN_RE.search(text) and not _CJK_RE.search(text))


def normalize_caption(value: Any) -> str:
    text = clean_caption(value).casefold()
    text = _SPACE_RE.sub(" ", text).strip()
    return _TERMINAL_PUNCT_RE.sub("", text)


def unique_valid_caption(value: Any) -> str:
    """Return one unambiguous valid caption from a scalar or mapping.

    Mapping-valued fields occur in some annotations.  Accepting an arbitrary
    dict value would make task labels dependent on insertion order, so a
    mapping is usable only when all valid values collapse to one normalized
    caption.
    """

    if isinstance(value, str):
        return clean_caption(value) if is_valid_english_caption(value) else ""
    if not isinstance(value, Mapping):
        return ""
    by_normalized: dict[str, str] = {}
    for item in value.values():
        if is_valid_english_caption(item):
            cleaned = clean_caption(item)
            by_normalized.setdefault(normalize_caption(cleaned), cleaned)
    return next(iter(by_normalized.values())) if len(by_normalized) == 1 else ""


def select_l3(annotation: Mapping[str, Any]) -> str:
    for field_name in ("task_caption", "instruction", "detailed_instruction"):
        caption = unique_valid_caption(annotation.get(field_name))
        if caption:
            return caption
    return ""


def parse_interval_key(value: Any) -> tuple[int, int] | None:
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        return None
    if len(parts) != 2:
        return None
    try:
        start, end = int(parts[0]), int(parts[1])
    except (TypeError, ValueError):
        return None
    return start, end

