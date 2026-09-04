"""Deterministic clean/noisy context construction for V5.3.

The module never mutates source labels.  Corruption metadata stays offline in
``provenance.context_noise`` and every emitted noisy context is byte-different
from its clean parent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from typing import Any


def stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _different_progress(value: int, seed: int) -> int:
    candidate = seed % 101
    return candidate if candidate != value else (candidate + 1) % 101


def _progress_noise(context: dict[str, Any], seed: int) -> dict[str, Any] | None:
    short = context.get("short_memory")
    if not isinstance(short, Mapping):
        return None
    changed: list[dict[str, Any]] = []
    old_task = int(short["task_progress_percent"])
    new_task = _different_progress(old_task, seed)
    short["task_progress_percent"] = new_task
    changed.append({"path": "/short_memory/task_progress_percent", "from": old_task, "to": new_task})
    for offset, (unit, value) in enumerate(short["prediction1"].items(), 1):
        old = int(value["progress_percent"])
        new = _different_progress(old, seed >> (offset * 7))
        value["progress_percent"] = new
        changed.append({
            "path": f"/short_memory/prediction1/{unit}/progress_percent",
            "from": old,
            "to": new,
        })
    return {"kind": "progress_0_100", "changes": changed}


def _window_noise(context: dict[str, Any], seed: int) -> dict[str, Any] | None:
    history = context.get("long_memory")
    if not isinstance(history, list) or len(history) < 2:
        return None
    direction = "left" if seed % 2 == 0 else "right"
    if direction == "left":
        shifted = history[:-1]
    else:
        shifted = history[1:]
    for index, item in enumerate(shifted, 1):
        item["index"] = index
    context["long_memory"] = shifted
    return {
        "kind": "causal_window_shift",
        "direction": direction,
        "offset_items": 1,
        "future_context_used": False,
    }


def _eligible_donors(
    donors: Sequence[Mapping[str, Any]],
    *,
    episode_key: str,
    output_profile_id: str,
) -> list[Mapping[str, Any]]:
    result = []
    for donor in donors:
        if donor.get("split") != "train":
            continue
        if str(donor.get("episode_key") or "") == episode_key:
            continue
        if str(donor.get("output_profile_id") or "") != output_profile_id:
            continue
        context = donor.get("prompt_context")
        if not isinstance(context, Mapping) or not isinstance(
            context.get("short_memory"), Mapping
        ):
            continue
        result.append(donor)
    return result


def _donor_noise(
    context: dict[str, Any],
    seed: int,
    donors: Sequence[Mapping[str, Any]],
    *,
    episode_key: str,
    output_profile_id: str,
) -> dict[str, Any] | None:
    eligible = _eligible_donors(
        donors,
        episode_key=episode_key,
        output_profile_id=output_profile_id,
    )
    if not eligible:
        return None
    donor = eligible[seed % len(eligible)]
    context["short_memory"] = copy.deepcopy(
        donor["prompt_context"]["short_memory"]
    )
    return {
        "kind": "cross_train_episode_short_memory",
        "donor_sample_id": str(donor.get("sample_id") or ""),
        "donor_episode_key": str(donor.get("episode_key") or ""),
        "donor_split": "train",
    }


def _reindex_plan(plan: list[dict[str, Any]]) -> None:
    for index, item in enumerate(plan, 1):
        item["index"] = index


def _initial_plan_noise(context: dict[str, Any], seed: int) -> dict[str, Any]:
    plan = context["initial_plan_memory"]
    mode = seed % 3
    if len(plan) == 1:
        plan.append(copy.deepcopy(plan[0]))
        _reindex_plan(plan)
        return {"kind": "initial_plan_duplicate_single", "source_index": 1}
    if mode == 0:
        index = seed % len(plan)
        removed = plan.pop(index)
        _reindex_plan(plan)
        return {
            "kind": "initial_plan_drop",
            "removed_index": index + 1,
            "removed_caption": removed["action"]["caption"],
        }
    if mode == 1:
        left = seed % (len(plan) - 1)
        plan[left], plan[left + 1] = plan[left + 1], plan[left]
        _reindex_plan(plan)
        return {
            "kind": "initial_plan_adjacent_swap",
            "left_index": left + 1,
            "right_index": left + 2,
        }
    target = seed % len(plan)
    source = (target + 1 + ((seed >> 8) % (len(plan) - 1))) % len(plan)
    plan[target]["action"] = copy.deepcopy(plan[source]["action"])
    return {
        "kind": "initial_plan_step_replace",
        "target_index": target + 1,
        "source_index": source + 1,
    }


def noisy_context(
    clean_context: Mapping[str, Any],
    *,
    context_variant: str,
    sample_id: str,
    episode_key: str,
    output_profile_id: str,
    donors: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a deterministic corrupted context and its offline audit record."""

    if context_variant not in {
        "with_memory_no_initial_noisy",
        "with_memory_with_initial_noisy",
    }:
        raise ValueError("noisy_context requires a noisy context_variant")
    context = copy.deepcopy(dict(clean_context))
    before = json.dumps(context, ensure_ascii=False, sort_keys=True)
    seed = stable_seed(sample_id, context_variant, "context-noise-v1")
    operations: list[dict[str, Any]] = []
    preferred = seed % 3
    candidates = (
        lambda: _progress_noise(context, seed),
        lambda: _window_noise(context, seed),
        lambda: _donor_noise(
            context,
            seed,
            donors,
            episode_key=episode_key,
            output_profile_id=output_profile_id,
        ),
    )
    for offset in range(3):
        operation = candidates[(preferred + offset) % 3]()
        if operation is not None:
            operations.append(operation)
            break
    if context_variant == "with_memory_with_initial_noisy":
        if not isinstance(context.get("initial_plan_memory"), list) or not context[
            "initial_plan_memory"
        ]:
            raise ValueError("with-initial noisy context has no initial plan")
        operations.append(_initial_plan_noise(context, seed >> 11))
    after = json.dumps(context, ensure_ascii=False, sort_keys=True)
    if before == after or not operations:
        raise ValueError("noisy context did not change its clean parent")
    return context, {
        "changed": True,
        "seed": seed,
        "operations": operations,
        "clean_context_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "noisy_context_sha256": hashlib.sha256(after.encode()).hexdigest(),
    }


def clean_context_variants(
    *,
    initial_plan: Sequence[Mapping[str, Any]] | None,
    long_memory: Sequence[Mapping[str, Any]] | None,
    short_memory: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Build only grounded clean variants; missing memory is never synthesized."""

    result: dict[str, dict[str, Any]] = {"no_memory_no_initial": {}}
    if long_memory is None:
        return result
    memory = {
        "long_memory": copy.deepcopy(list(long_memory)),
        "short_memory": copy.deepcopy(short_memory),
    }
    result["with_memory_no_initial"] = memory
    if initial_plan:
        result["with_memory_with_initial"] = {
            "initial_plan_memory": copy.deepcopy(list(initial_plan)),
            "long_memory": copy.deepcopy(list(long_memory)),
            "short_memory": copy.deepcopy(short_memory),
        }
    return result


__all__ = ["clean_context_variants", "noisy_context", "stable_seed"]
