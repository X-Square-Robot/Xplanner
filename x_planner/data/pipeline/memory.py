"""Ground-truth/noisy memory handling and the shared streaming MemoryBank."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Iterable

from .captions import normalize_caption
from .constants import (
    DEFAULT_DONE_THRESHOLD,
    DEFAULT_MAX_MEMORY_NOISE_PROB,
    DEFAULT_SHORT_MEMORY_K,
    DEFAULT_STABLE_STEPS,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_VISIBLE_LONG_MEMORY_LIMIT,
)


DEFAULT_PARAPHRASE_RULES = (
    ("pick up", "lift"),
    ("picks up", "lifts"),
    ("place", "put"),
    ("places", "puts"),
    ("move toward", "approach"),
    ("moves toward", "approaches"),
    ("grasp", "grip"),
    ("grasps", "grips"),
)


class MemoryCodec:
    def __init__(
        self,
        *,
        short_memory_k: int = DEFAULT_SHORT_MEMORY_K,
        visible_long_memory_limit: int = DEFAULT_VISIBLE_LONG_MEMORY_LIMIT,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        aliases: dict[str, str] | None = None,
    ) -> None:
        if short_memory_k not in {1, 2}:
            raise ValueError("short_memory_k must be 1 or 2")
        if visible_long_memory_limit <= 0:
            raise ValueError("visible_long_memory_limit must be positive")
        self.short_memory_k = short_memory_k
        self.visible_long_memory_limit = visible_long_memory_limit
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [0, 1]")
        self.similarity_threshold = similarity_threshold
        self.aliases = {
            normalize_caption(key): normalize_caption(value)
            for key, value in (aliases or {}).items()
        }

    def normalize(self, caption: str) -> str:
        return normalize_caption(caption)

    def canonical(self, caption: str) -> str:
        normalized = self.normalize(caption)
        return self.aliases.get(normalized, normalized)

    @staticmethod
    def _embedding(caption: str, dimensions: int = 512) -> tuple[float, ...]:
        """Deterministic token/character n-gram embedding used without a service.

        This keeps rollout and serving byte-for-byte reproducible. Deployments may
        populate ``aliases`` from a stronger offline sentence-embedding map; exact
        normalized matching always remains the first decision path.
        """

        words = caption.split()
        features = list(words)
        features.extend(" ".join(words[index:index + 2]) for index in range(len(words) - 1))
        compact = " ".join(words)
        features.extend(compact[index:index + 3] for index in range(max(0, len(compact) - 2)))
        vector = [0.0] * dimensions
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "big")
            index = raw % dimensions
            vector[index] += -1.0 if raw & (1 << 63) else 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return tuple(vector)
        return tuple(value / norm for value in vector)

    def similarity(self, left: str, right: str) -> float:
        left_value = self.canonical(left)
        right_value = self.canonical(right)
        if not left_value or not right_value:
            return 0.0
        if left_value == right_value:
            return 1.0
        left_embedding = self._embedding(left_value)
        right_embedding = self._embedding(right_value)
        return max(0.0, min(1.0, sum(
            left_item * right_item
            for left_item, right_item in zip(left_embedding, right_embedding)
        )))

    def same(self, left: str, right: str) -> bool:
        return bool(
            left and right
            and (
                self.canonical(left) == self.canonical(right)
                or self.similarity(left, right) >= self.similarity_threshold
            )
        )

    def closest(self, caption: str, candidates: Iterable[str]) -> str | None:
        normalized = self.normalize(caption)
        if not normalized:
            return None
        values = tuple(self.normalize(value) for value in candidates if self.normalize(value))
        exact = next((value for value in values if self.canonical(value) == self.canonical(normalized)), None)
        if exact is not None:
            return exact
        scored = [(self.similarity(normalized, value), value) for value in values]
        score, value = max(scored, default=(0.0, None))
        return value if score >= self.similarity_threshold else None

    def short_from_long(self, long_memory: Iterable[str]) -> tuple[str, ...]:
        normalized = tuple(self.normalize(item) for item in long_memory if self.normalize(item))
        return normalized[-self.short_memory_k:]

    @staticmethod
    def _lines(items: Iterable[str]) -> str:
        values = tuple(items)
        if not values:
            return "[none]"
        return "\n".join(f"{index}. {caption}" for index, caption in enumerate(values, 1))

    def render_long(self, long_memory: Iterable[str]) -> str:
        values = tuple(self.normalize(item) for item in long_memory if self.normalize(item))
        visible = values[-self.visible_long_memory_limit:]
        prefix = "[earlier steps omitted]\n" if len(values) > len(visible) else ""
        return prefix + self._lines(visible)

    def render_short(self, short_memory: Iterable[str]) -> str:
        return self._lines(tuple(short_memory))


@dataclass(frozen=True, slots=True)
class AugmentedMemory:
    long_memory: tuple[str, ...]
    short_memory: tuple[str, ...]
    operation: str
    applied: bool
    probability: float


class MemoryAugmentor:
    _OPERATIONS = ("paraphrase", "drop", "replace", "duplicate", "empty")
    _WEIGHTS = (0.6, 0.1, 0.1, 0.1, 0.1)

    def __init__(
        self,
        codec: MemoryCodec | None = None,
        *,
        seed: int = 42,
        max_probability: float = DEFAULT_MAX_MEMORY_NOISE_PROB,
        ramp_steps: int = 60,
        paraphrase_rules: Iterable[tuple[str, str]] = DEFAULT_PARAPHRASE_RULES,
    ) -> None:
        self.codec = codec or MemoryCodec()
        self.seed = seed
        self.max_probability = max_probability
        self.ramp_steps = ramp_steps
        self.paraphrase_rules = tuple(
            (normalize_caption(left), normalize_caption(right))
            for left, right in paraphrase_rules
        )

    def probability(self, global_step: int) -> float:
        if global_step <= 0:
            return 0.0
        if self.ramp_steps <= 0:
            return self.max_probability
        return min(self.max_probability, self.max_probability * global_step / self.ramp_steps)

    def _rng(self, sample_id: str, global_step: int) -> random.Random:
        digest = hashlib.sha256(
            f"{self.seed}\0{global_step}\0{sample_id}".encode()
        ).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def _paraphrase(self, caption: str, rng: random.Random) -> str | None:
        candidates: list[str] = []
        for left, right in self.paraphrase_rules:
            if left in caption:
                candidates.append(caption.replace(left, right, 1))
        return rng.choice(candidates) if candidates else None

    def augment(
        self, long_memory: Iterable[str], *, sample_id: str, global_step: int
    ) -> AugmentedMemory:
        values = [self.codec.normalize(item) for item in long_memory if self.codec.normalize(item)]
        probability = self.probability(global_step)
        if not values or probability <= 0:
            result = tuple(values)
            return AugmentedMemory(
                result, self.codec.short_from_long(result), "gt", False, probability
            )
        rng = self._rng(sample_id, global_step)
        if rng.random() >= probability:
            result = tuple(values)
            return AugmentedMemory(
                result, self.codec.short_from_long(result), "gt", False, probability
            )

        operation = rng.choices(self._OPERATIONS, weights=self._WEIGHTS, k=1)[0]
        if operation == "paraphrase":
            indices = list(range(len(values)))
            rng.shuffle(indices)
            replacement = None
            for index in indices:
                replacement = self._paraphrase(values[index], rng)
                if replacement:
                    values[index] = replacement
                    break
            if replacement is None:
                operation = rng.choice(("drop", "duplicate", "empty"))

        if operation == "drop":
            del values[rng.randrange(len(values))]
        elif operation == "replace":
            index = rng.randrange(len(values))
            candidates = [item for pos, item in enumerate(values) if pos != index and item != values[index]]
            if not candidates:
                result = tuple(values)
                return AugmentedMemory(
                    result, self.codec.short_from_long(result), "replace_skipped", False, probability
                )
            values[index] = rng.choice(candidates)
        elif operation == "duplicate":
            index = rng.randrange(len(values))
            values.insert(index + 1, values[index])
        elif operation == "empty":
            values.clear()
        elif operation != "paraphrase":
            raise AssertionError(f"unhandled operation: {operation}")

        result = tuple(values)
        return AugmentedMemory(
            result,
            self.codec.short_from_long(result),
            operation,
            True,
            probability,
        )


@dataclass(frozen=True, slots=True)
class UnitObservation:
    caption: str
    progress_percent: int
    next_caption: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryBankUpdate:
    episode_id: str
    committed: str | None
    transitioned: bool
    active_caption: str
    active_stable_steps: int
    long_memory: tuple[str, ...]
    reason: str


@dataclass(slots=True)
class _ActiveState:
    caption: str
    canonical: str
    stable_steps: int
    max_progress: int
    next_caption: str | None


@dataclass(slots=True)
class MemoryBank:
    codec: MemoryCodec = field(default_factory=MemoryCodec)
    done_threshold: int = DEFAULT_DONE_THRESHOLD
    stable_steps: int = DEFAULT_STABLE_STEPS
    episode_id: str | None = None
    long_memory: list[str] = field(default_factory=list)
    active: _ActiveState | None = None
    candidate: _ActiveState | None = None
    canonical_vocabulary: dict[str, str] = field(default_factory=dict)

    def reset(
        self,
        episode_id: str,
        *,
        canonical_captions: Iterable[str] = (),
    ) -> None:
        self.episode_id = episode_id
        self.long_memory.clear()
        self.active = None
        self.candidate = None
        self.canonical_vocabulary = {
            self.codec.canonical(caption): self.codec.normalize(caption)
            for caption in canonical_captions
            if self.codec.normalize(caption)
        }

    def _canonicalize_observation(self, observation: UnitObservation) -> UnitObservation:
        vocabulary = tuple(self.canonical_vocabulary.values())
        canonical = self.codec.canonical(observation.caption)
        caption = self.canonical_vocabulary.get(canonical)
        if caption is None:
            caption = self.codec.closest(observation.caption, vocabulary)
        caption = caption or self.codec.normalize(observation.caption)
        next_caption = (
            self.canonical_vocabulary.get(self.codec.canonical(observation.next_caption))
            or self.codec.closest(observation.next_caption, vocabulary)
            or self.codec.normalize(observation.next_caption)
            if observation.next_caption
            else None
        )
        return UnitObservation(caption, observation.progress_percent, next_caption)

    def _update(self, reason: str, *, committed: str | None = None, transitioned: bool = False) -> MemoryBankUpdate:
        assert self.episode_id is not None and self.active is not None
        return MemoryBankUpdate(
            episode_id=self.episode_id,
            committed=committed,
            transitioned=transitioned,
            active_caption=self.active.caption,
            active_stable_steps=self.active.stable_steps,
            long_memory=tuple(self.long_memory),
            reason=reason,
        )

    def step(self, episode_id: str, observation: UnitObservation) -> MemoryBankUpdate:
        if self.episode_id != episode_id:
            self.reset(episode_id)
        if not 0 <= observation.progress_percent <= 100:
            raise ValueError("progress_percent must be in [0, 100]")
        observation = self._canonicalize_observation(observation)
        if not observation.caption:
            raise ValueError("observation caption is empty")
        canonical = self.codec.canonical(observation.caption)

        if self.active is None:
            self.active = _ActiveState(
                observation.caption, canonical, 1, observation.progress_percent, observation.next_caption
            )
            return self._update("initialized")

        if canonical == self.active.canonical:
            self.active.stable_steps += 1
            self.active.max_progress = max(self.active.max_progress, observation.progress_percent)
            if observation.next_caption:
                self.active.next_caption = observation.next_caption
            self.candidate = None
            return self._update("active_stable")

        if self.candidate is None or self.candidate.canonical != canonical:
            self.candidate = _ActiveState(
                observation.caption, canonical, 1, observation.progress_percent, observation.next_caption
            )
            return self._update("transition_candidate")

        self.candidate.stable_steps += 1
        self.candidate.max_progress = max(
            self.candidate.max_progress, observation.progress_percent
        )
        if observation.next_caption:
            self.candidate.next_caption = observation.next_caption
        if self.candidate.stable_steps < self.stable_steps:
            return self._update("transition_candidate")

        previous = self.active
        transition_matches_prediction = bool(
            previous.next_caption
            and self.codec.same(self.candidate.caption, previous.next_caption)
        )
        completed = (
            previous.max_progress >= self.done_threshold
            or transition_matches_prediction
        )
        committed: str | None = None
        reason = "transition_without_commit"
        if previous.stable_steps >= self.stable_steps and completed:
            if not self.long_memory or not self.codec.same(self.long_memory[-1], previous.caption):
                committed = previous.caption
                self.long_memory.append(committed)
                reason = "transition_committed"
            else:
                reason = "transition_duplicate_suppressed"

        self.active = self.candidate
        self.candidate = None
        return self._update(reason, committed=committed, transitioned=True)
