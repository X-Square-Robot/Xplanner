from __future__ import annotations

import unittest

from x_planner.data.event_states.exposure_plan import V5MixError, plan_exposure


def _leaf(
    source: str,
    category: str,
    memory: str,
    task: str,
    count: int,
) -> dict[str, object]:
    profile = "p1-action__p2-action__plan-action"
    name = f"v5__{source}__{memory}__{category}__{profile}__{task}"
    return {
        "source": source,
        "category": category,
        "memory_variant": memory,
        "output_profile_id": profile,
        "task_name": task,
        "path": f"/fixture/{source}/{memory}/{category}/{task}",
        "num_samples": count,
        "memory_pair_eligible": category in {"ongoing", "end"}
        and memory in {"no_memory", "with_memory"}
        and source != "takeover_q",
        "sampler_task_name": name,
    }


class MixV5Test(unittest.TestCase):
    def inventory(self) -> list[dict[str, object]]:
        return [
            _leaf("baseline", "initial_plan", "no_memory", "plan", 7),
            _leaf("baseline", "ongoing", "no_memory", "normal", 11),
            _leaf("baseline", "ongoing", "with_memory", "normal", 11),
            {
                **_leaf("baseline", "ongoing", "no_memory", "segment_only", 19),
                "memory_pair_eligible": False,
            },
            _leaf("robodojo", "initial_plan", "no_memory", "plan", 5),
            _leaf("robodojo", "ongoing", "no_memory", "normal", 13),
            _leaf("robodojo", "ongoing", "with_memory", "normal", 13),
            _leaf("takeover_q", "takeover", "no_memory", "failure", 29),
        ]

    def test_exact_nested_ratios_and_memory_pairs(self) -> None:
        plan = plan_exposure(self.inventory(), full_coverage=True)
        total = plan["total_exposures"]
        self.assertEqual(total % 200, 0)
        self.assertEqual(
            plan["source_exposures"],
            {
                "baseline": total * 70 // 100,
                "robodojo": total * 15 // 100,
                "takeover_q": total * 15 // 100,
            },
        )
        takeover_total = plan["source_exposures"]["takeover_q"]
        self.assertEqual(
            plan["takeover_q_category_exposures"],
            {
                "takeover": takeover_total,
            },
        )
        memory = plan["eligible_memory_exposures"]
        self.assertEqual(
            memory["baseline|ongoing|no_memory"],
            memory["baseline|ongoing|with_memory"],
        )
        self.assertEqual(
            memory["robodojo|ongoing|no_memory"],
            memory["robodojo|ongoing|with_memory"],
        )

    def test_virtual_replay_never_exceeds_physical_leaf_capacity(self) -> None:
        inventory = self.inventory()
        by_path = {str(leaf["path"]): int(leaf["num_samples"]) for leaf in inventory}
        plan = plan_exposure(inventory, full_coverage=True, requested_total=1000)
        self.assertTrue(any(task["replica_index"] > 1 for task in plan["virtual_tasks"]))
        for task in plan["virtual_tasks"]:
            self.assertLessEqual(task["count"], by_path[task["path"]])
        self.assertEqual(sum(plan["counts_per_task"].values()), 1000)

    def test_smoke_covers_each_leaf_without_claiming_full_coverage(self) -> None:
        plan = plan_exposure(
            self.inventory(), full_coverage=False, requested_total=200
        )
        self.assertFalse(plan["full_coverage"])
        physical = {task["physical_leaf_task"] for task in plan["virtual_tasks"]}
        expected = {str(leaf["sampler_task_name"]) for leaf in self.inventory()}
        self.assertEqual(physical, expected)

    def test_rejects_missing_memory_pair(self) -> None:
        inventory = self.inventory()
        inventory = [
            leaf
            for leaf in inventory
            if not (
                leaf["source"] == "baseline"
                and leaf["memory_variant"] == "with_memory"
            )
        ]
        with self.assertRaises(V5MixError):
            plan_exposure(inventory, full_coverage=False)

    def test_robodojo_takeover_specialization_is_exactly_balanced(self) -> None:
        inventory = [
            leaf for leaf in self.inventory()
            if leaf["source"] in {"robodojo", "takeover_q"}
        ]
        plan = plan_exposure(
            inventory,
            full_coverage=True,
            source_weights={"robodojo": 50, "takeover_q": 50},
        )
        total = plan["total_exposures"]
        self.assertEqual(
            plan["source_exposures"],
            {"robodojo": total // 2, "takeover_q": total // 2},
        )
        memory = plan["eligible_memory_exposures"]
        self.assertEqual(
            memory["robodojo|ongoing|no_memory"],
            memory["robodojo|ongoing|with_memory"],
        )


if __name__ == "__main__":
    unittest.main()
