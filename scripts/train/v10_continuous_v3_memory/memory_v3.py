"""State-bearing Short Memory plus the proven completed-unit MemoryBank."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..v10_continuous.memory import MemoryBank, MemoryCodec, UnitObservation
from .schema_v3 import TERMINAL_CAPTION, active_field, validate_short_memory


@dataclass(frozen=True, slots=True)
class ShortMemoryState:
    caption: str
    progress_percent: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "caption": self.caption,
            "progress_percent": self.progress_percent,
        }


class MemoryCodecV3:
    def __init__(self, *, visible_long_memory_limit: int = 8) -> None:
        self.long_codec = MemoryCodec(
            short_memory_k=1,
            visible_long_memory_limit=visible_long_memory_limit,
        )

    def parse_short(self, value: Any) -> tuple[ShortMemoryState, ...]:
        return tuple(ShortMemoryState(**item) for item in validate_short_memory(value))

    def render_long(self, value: Iterable[str]) -> str:
        return self.long_codec.render_long(value)

    def render_short(self, value: Any) -> str:
        states = self.parse_short(value)
        if not states:
            return "[none]"
        return json.dumps(
            [state.to_dict() for state in states],
            ensure_ascii=False,
            separators=(",", ":"),
        )


@dataclass(slots=True)
class StreamingMemoryV3:
    """Inference state: prompt memory is read before the new model output is stored."""

    bank: MemoryBank = field(default_factory=MemoryBank)
    episode_id: str | None = None
    short_memory: tuple[ShortMemoryState, ...] = ()

    def reset(self, episode_id: str, *, canonical_captions: Iterable[str] = ()) -> None:
        self.episode_id = episode_id
        self.short_memory = ()
        self.bank.reset(episode_id, canonical_captions=canonical_captions)

    def prompt_memory(self, episode_id: str) -> tuple[tuple[str, ...], tuple[ShortMemoryState, ...]]:
        if self.episode_id != episode_id:
            self.reset(episode_id)
        return tuple(self.bank.long_memory), self.short_memory

    def observe_model_target(
        self,
        episode_id: str,
        target: Mapping[str, Any],
        profile: str,
    ) -> None:
        if self.episode_id != episode_id:
            self.reset(episode_id)
        predictions = target.get("predictions")
        if not isinstance(predictions, list) or not predictions:
            raise ValueError("model target has no Prediction 1")
        field_name = active_field(profile)
        first = predictions[0][field_name]
        next_caption = None
        if len(predictions) > 1:
            candidate = predictions[1][field_name].get("caption")
            if candidate != TERMINAL_CAPTION:
                next_caption = candidate
        state = ShortMemoryState(
            caption=str(first["caption"]),
            progress_percent=int(first["progress_percent"]),
        )
        self.bank.step(
            episode_id,
            UnitObservation(state.caption, state.progress_percent, next_caption),
        )
        self.short_memory = (state,)

