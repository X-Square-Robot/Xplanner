"""Exact V5 training exposure plan with bounded virtual replay.

The legacy static-count sampler never samples past the number of rows exposed
by one task.  V5 therefore represents replay as multiple virtual tasks that
point at the same immutable indexed leaf.  Each virtual task requests at most
the leaf's physical row count, so the existing sampler cannot silently cap the
approved exposure plan.

The formal plan covers every physical training row at least once.  A smoke plan
covers every non-empty leaf at least once.  Both use exact effective exposure
ratios:

* baseline / robodojo / takeover_q = 70 / 15 / 15;
* every takeover_q exposure is one of the 15 reviewed Takeover failure classes;
* every eligible ongoing/end with_memory/no_memory pair is exposed 50 / 50.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any


SOURCE_WEIGHTS: Mapping[str, int] = {
    "baseline": 70,
    "robodojo": 15,
    "takeover_q": 15,
}
TAKEOVER_CATEGORY_WEIGHTS: Mapping[str, int] = {
    "takeover": 100,
}
EXACT_RATIO_QUANTUM = 200


class V5MixError(ValueError):
    """The leaf inventory cannot satisfy the approved V5 exposure contract."""


@dataclass(frozen=True, slots=True)
class VirtualTask:
    source: str
    path: str
    task_name: str
    count: int
    physical_leaf_task: str
    replica_index: int


def _ceil_multiple(value: int, quantum: int) -> int:
    if value <= 0 or quantum <= 0:
        raise ValueError("value and quantum must be positive")
    return ((value + quantum - 1) // quantum) * quantum


def _leaf_identity(leaf: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(leaf["source"]),
        str(leaf["category"]),
        str(leaf["memory_variant"]),
        str(leaf["output_profile_id"]),
        str(leaf["task_name"]),
        str(leaf["path"]),
    )


def _validate_leaves(
    raw_leaves: Sequence[Mapping[str, Any]],
    source_weights: Mapping[str, int],
) -> list[dict[str, Any]]:
    if not raw_leaves:
        raise V5MixError("the V5 leaf inventory is empty")
    leaves: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for raw in raw_leaves:
        leaf = dict(raw)
        source = str(leaf.get("source", ""))
        if source not in source_weights:
            raise V5MixError(f"unknown V5 source: {source!r}")
        category = str(leaf.get("category", ""))
        memory = str(leaf.get("memory_variant", ""))
        if category not in {"initial_plan", "ongoing", "end", "takeover"}:
            raise V5MixError(f"unknown V5 category: {category!r}")
        if memory not in {"no_memory", "with_memory"}:
            raise V5MixError(f"unknown V5 memory variant: {memory!r}")
        if category in {"initial_plan", "takeover"} and memory != "no_memory":
            raise V5MixError(f"{category} cannot have a with_memory leaf")
        eligible = leaf.get("memory_pair_eligible")
        if not isinstance(eligible, bool):
            raise V5MixError("every V5 leaf must declare memory_pair_eligible")
        if category in {"initial_plan", "takeover"} and eligible:
            raise V5MixError(f"{category} cannot be memory-pair eligible")
        if memory == "with_memory" and not eligible:
            raise V5MixError("a with_memory leaf must be memory-pair eligible")
        count = leaf.get("num_samples")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise V5MixError("every V5 leaf must contain at least one sample")
        identity = _leaf_identity(leaf)
        if identity in seen:
            raise V5MixError(f"duplicate V5 leaf inventory entry: {identity}")
        seen.add(identity)
        leaves.append(leaf)

    present_sources = {str(leaf["source"]) for leaf in leaves}
    if present_sources != set(source_weights):
        raise V5MixError(
            "training inventory must contain exactly its requested sources; "
            f"got {sorted(present_sources)}"
        )
    takeover_categories = {
        str(leaf["category"])
        for leaf in leaves
        if leaf["source"] == "takeover_q"
    }
    if "takeover_q" in source_weights and takeover_categories != set(TAKEOVER_CATEGORY_WEIGHTS):
        raise V5MixError(
            "takeover_q must contain Takeover leaves only"
        )

    by_pair: dict[tuple[str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for leaf in leaves:
        if (
            leaf["source"] == "takeover_q"
            or leaf["category"] not in {"ongoing", "end"}
            or leaf["memory_pair_eligible"] is not True
        ):
            continue
        key = (
            str(leaf["source"]),
            str(leaf["category"]),
            str(leaf["output_profile_id"]),
            str(leaf["task_name"]),
        )
        memory = str(leaf["memory_variant"])
        if memory in by_pair[key]:
            raise V5MixError(f"duplicate memory leaf for pair {key}")
        by_pair[key][memory] = leaf
    for key, variants in by_pair.items():
        if set(variants) != {"no_memory", "with_memory"}:
            raise V5MixError(f"eligible memory pair is incomplete: {key}")
        if variants["no_memory"]["num_samples"] != variants["with_memory"]["num_samples"]:
            raise V5MixError(f"eligible memory pair counts differ: {key}")
    return leaves


def _minimum_total(
    leaves: Sequence[Mapping[str, Any]],
    *,
    full_coverage: bool,
    source_weights: Mapping[str, int],
) -> int:
    def minimum(leaf: Mapping[str, Any]) -> int:
        return int(leaf["num_samples"]) if full_coverage else 1

    source_minimum = defaultdict(int)
    for leaf in leaves:
        value = minimum(leaf)
        source_minimum[str(leaf["source"])] += value

    bounds = [
        math.ceil(source_minimum[source] * 100 / weight)
        for source, weight in source_weights.items()
    ]
    return _ceil_multiple(max(bounds), EXACT_RATIO_QUANTUM)


def _weighted_allocation(
    leaves: Sequence[Mapping[str, Any]],
    target: int,
    *,
    full_coverage: bool,
) -> dict[tuple[str, str, str, str, str, str], int]:
    """Allocate a target to unpaired leaves while preserving every minimum."""
    minima = [int(leaf["num_samples"]) if full_coverage else 1 for leaf in leaves]
    if sum(minima) > target:
        raise V5MixError("exposure target is smaller than its coverage minimum")
    weights = [int(leaf["num_samples"]) for leaf in leaves]
    allocation = list(minima)
    remaining = target - sum(allocation)
    if remaining:
        denominator = sum(weights)
        exact = [remaining * weight / denominator for weight in weights]
        for index, value in enumerate(exact):
            allocation[index] += int(value)
        remainder = target - sum(allocation)
        order = sorted(
            range(len(leaves)),
            key=lambda index: (exact[index] - int(exact[index]), _leaf_identity(leaves[index])),
            reverse=True,
        )
        for index in order[:remainder]:
            allocation[index] += 1
    return {
        _leaf_identity(leaf): count for leaf, count in zip(leaves, allocation, strict=True)
    }


def _paired_allocation(
    leaves: Sequence[Mapping[str, Any]],
    target: int,
    *,
    full_coverage: bool,
) -> dict[tuple[str, str, str, str, str, str], int]:
    """Allocate normal-source exposure and keep each memory pair exactly equal."""
    paired: dict[tuple[str, str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    singles: list[Mapping[str, Any]] = []
    for leaf in leaves:
        if (
            leaf["category"] in {"ongoing", "end"}
            and leaf["memory_pair_eligible"] is True
        ):
            key = (
                str(leaf["source"]),
                str(leaf["category"]),
                str(leaf["output_profile_id"]),
                str(leaf["task_name"]),
            )
            paired[key][str(leaf["memory_variant"])] = leaf
        else:
            singles.append(leaf)
    for key, variants in paired.items():
        if set(variants) != {"no_memory", "with_memory"}:
            raise V5MixError(f"incomplete memory pair during allocation: {key}")

    allocation: dict[tuple[str, str, str, str, str, str], int] = {}
    pair_units: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for key in sorted(paired):
        variants = paired[key]
        left = variants["no_memory"]
        right = variants["with_memory"]
        minimum = int(left["num_samples"]) if full_coverage else 1
        allocation[_leaf_identity(left)] = minimum
        allocation[_leaf_identity(right)] = minimum
        pair_units.append((left, right))
    for leaf in singles:
        allocation[_leaf_identity(leaf)] = (
            int(leaf["num_samples"]) if full_coverage else 1
        )

    remaining = target - sum(allocation.values())
    if remaining < 0:
        raise V5MixError("normal-source target is below its coverage minimum")
    # Odd exposure can only be assigned to a single no-memory plan leaf.
    if remaining % 2:
        if not singles:
            raise V5MixError("odd normal-source remainder has no unpaired plan leaf")
        chosen = max(singles, key=lambda leaf: (int(leaf["num_samples"]), _leaf_identity(leaf)))
        allocation[_leaf_identity(chosen)] += 1
        remaining -= 1
    if remaining == 0:
        return allocation

    # Allocate in two-exposure units.  A single leaf receives two at a time;
    # a memory pair receives one on each side.  This preserves pair equality.
    units: list[tuple[int, tuple[Mapping[str, Any], ...]]] = []
    for left, right in pair_units:
        units.append((int(left["num_samples"]), (left, right)))
    for leaf in singles:
        units.append((int(leaf["num_samples"]), (leaf, leaf)))
    unit_target = remaining // 2
    denominator = sum(weight for weight, _ in units)
    exact = [unit_target * weight / denominator for weight, _ in units]
    assigned = [int(value) for value in exact]
    remainder = unit_target - sum(assigned)
    order = sorted(
        range(len(units)),
        key=lambda index: (
            exact[index] - int(exact[index]),
            tuple(_leaf_identity(leaf) for leaf in units[index][1]),
        ),
        reverse=True,
    )
    for index in order[:remainder]:
        assigned[index] += 1
    for count, (_weight, pair) in zip(assigned, units, strict=True):
        for leaf in pair:
            allocation[_leaf_identity(leaf)] += count
    if sum(allocation.values()) != target:
        raise RuntimeError("paired exposure allocation did not conserve its target")
    return allocation


def plan_exposure(
    raw_leaves: Sequence[Mapping[str, Any]],
    *,
    full_coverage: bool,
    requested_total: int | None = None,
    source_weights: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return physical-leaf counts and cap-safe virtual replay tasks."""
    weights = dict(SOURCE_WEIGHTS if source_weights is None else source_weights)
    if not weights or set(weights) - set(SOURCE_WEIGHTS):
        raise V5MixError(f"unsupported source weights: {weights}")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in weights.values()):
        raise V5MixError("source weights must be positive integer percentages")
    if sum(weights.values()) != 100:
        raise V5MixError("source weights must sum to 100")
    leaves = _validate_leaves(raw_leaves, weights)
    minimum_total = _minimum_total(
        leaves,
        full_coverage=full_coverage,
        source_weights=weights,
    )
    if requested_total is not None:
        if isinstance(requested_total, bool) or not isinstance(requested_total, int):
            raise V5MixError("requested_total must be an integer")
        if requested_total <= 0:
            raise V5MixError("requested_total must be positive")
        total = _ceil_multiple(max(requested_total, minimum_total), EXACT_RATIO_QUANTUM)
    else:
        total = minimum_total

    source_targets = {
        source: total * weight // 100 for source, weight in weights.items()
    }
    leaf_counts: dict[tuple[str, str, str, str, str, str], int] = {}
    for source in sorted(set(weights) - {"takeover_q"}):
        source_leaves = [leaf for leaf in leaves if leaf["source"] == source]
        leaf_counts.update(_paired_allocation(
            source_leaves,
            source_targets[source],
            full_coverage=full_coverage,
        ))

    takeover_targets: dict[str, int] = {}
    if "takeover_q" in weights:
        takeover_source_target = source_targets["takeover_q"]
        takeover_targets = {
            category: takeover_source_target * weight // 100
            for category, weight in TAKEOVER_CATEGORY_WEIGHTS.items()
        }
        for category, target in takeover_targets.items():
            category_leaves = [
                leaf
                for leaf in leaves
                if leaf["source"] == "takeover_q" and leaf["category"] == category
            ]
            leaf_counts.update(_weighted_allocation(
                category_leaves,
                target,
                full_coverage=full_coverage,
            ))

    virtual_tasks: list[VirtualTask] = []
    for leaf in sorted(leaves, key=_leaf_identity):
        physical = int(leaf["num_samples"])
        remaining = leaf_counts[_leaf_identity(leaf)]
        replica = 0
        base_name = str(leaf["sampler_task_name"])
        while remaining:
            replica += 1
            count = min(remaining, physical)
            virtual_tasks.append(VirtualTask(
                source=str(leaf["source"]),
                path=str(leaf["path"]),
                task_name=f"{base_name}__replay_{replica:05d}",
                count=count,
                physical_leaf_task=base_name,
                replica_index=replica,
            ))
            remaining -= count

    counts_per_task = {task.task_name: task.count for task in virtual_tasks}
    if sum(counts_per_task.values()) != total:
        raise RuntimeError("virtual replay tasks do not conserve total exposure")
    if any(
        task.count > next(
            int(leaf["num_samples"])
            for leaf in leaves
            if str(leaf["path"]) == task.path
        )
        for task in virtual_tasks
    ):
        raise RuntimeError("a virtual task exceeds its physical leaf capacity")

    memory_counts = defaultdict(int)
    eligible_memory_counts = defaultdict(int)
    category_counts = defaultdict(int)
    for leaf in leaves:
        count = leaf_counts[_leaf_identity(leaf)]
        memory_counts[(str(leaf["source"]), str(leaf["category"]), str(leaf["memory_variant"]))] += count
        if leaf["memory_pair_eligible"] is True:
            eligible_memory_counts[(
                str(leaf["source"]),
                str(leaf["category"]),
                str(leaf["memory_variant"]),
            )] += count
        category_counts[(str(leaf["source"]), str(leaf["category"]))] += count
    return {
        "schema_version": "v10_action_segment_v5_exposure_v1",
        "full_coverage": full_coverage,
        "source_weights": weights,
        "ratio_quantum": EXACT_RATIO_QUANTUM,
        "total_exposures": total,
        "source_exposures": source_targets,
        "takeover_q_category_exposures": takeover_targets,
        "memory_exposures": {
            "|".join(key): value for key, value in sorted(memory_counts.items())
        },
        "eligible_memory_exposures": {
            "|".join(key): value
            for key, value in sorted(eligible_memory_counts.items())
        },
        "category_exposures": {
            "|".join(key): value for key, value in sorted(category_counts.items())
        },
        "counts_per_task": counts_per_task,
        "virtual_tasks": [asdict(task) for task in virtual_tasks],
    }


__all__ = [
    "EXACT_RATIO_QUANTUM",
    "SOURCE_WEIGHTS",
    "TAKEOVER_CATEGORY_WEIGHTS",
    "V5MixError",
    "VirtualTask",
    "plan_exposure",
]
