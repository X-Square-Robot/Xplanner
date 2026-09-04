from __future__ import annotations

import unittest

from .mix_v53 import allocate_exposures, allocate_weighted_exposures


class MixV53Test(unittest.TestCase):
    def test_robodojo_is_exact_twenty_percent(self) -> None:
        counts = {
            "initial_plan": 10,
            "ongoing": 100,
            "end": 10,
            "robodojo": 500,
            "takeover": 1000,
            "replan_self": 50,
            "replan_open": 50,
        }
        result = allocate_exposures(counts, total=10_003)
        self.assertEqual(sum(result.values()), 10_003)
        self.assertEqual(result["robodojo"], round(10_003 * 0.20))
        self.assertGreater(result["replan_self"], 0)
        self.assertGreater(result["replan_open"], 0)

    def test_baseline_only_weighted_profile(self) -> None:
        result = allocate_weighted_exposures(
            {"initial_plan": 2, "ongoing": 136, "end": 28},
            total=10_003,
            weights={"initial_plan": 25, "ongoing": 50, "end": 25},
        )
        self.assertEqual(sum(result.values()), 10_003)
        self.assertEqual(set(result), {"initial_plan", "ongoing", "end"})
        self.assertNotIn("robodojo", result)
        self.assertEqual(result["ongoing"], 5_001)

    def test_weighted_profile_rejects_unknown_bucket(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown buckets"):
            allocate_weighted_exposures(
                {"initial_plan": 1, "not_a_bucket": 1},
                total=10,
                weights={"initial_plan": 1, "not_a_bucket": 1},
            )


if __name__ == "__main__":
    unittest.main()
