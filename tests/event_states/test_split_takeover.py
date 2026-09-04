from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from x_planner.data.event_states.holdout import EvaluationHoldout
from x_planner.data.event_states.split_takeover import FAILURE_CODES, materialize_split


class TakeoverSplitV53Test(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, EvaluationHoldout]:
        source = root / "new_completed"
        episodes = source / "episodes"
        episodes.mkdir(parents=True)
        index_rows = []
        # Ten distinct episodes per class make exact class coverage and a
        # 20% episode split simultaneously feasible.
        for code_index, code in enumerate(FAILURE_CODES):
            for copy_index in range(10):
                number = code_index * 10 + copy_index
                episode_key = f"episode_{number:04d}"
                episode_id = f"20260827-day-1-task_{code_index}@takeover@{copy_index}"
                relative = f"episodes/{episode_key}.json"
                payload = {
                    "episode_key": episode_key,
                    "episode_id": episode_id,
                    "cases": [{
                        "bilingual": {
                            "segments": {"q2q3": [{"q3_type": f"{code} fixture"}]}
                        }
                    }],
                }
                (source / relative).write_text(json.dumps(payload), encoding="utf-8")
                index_rows.append({
                    "episode_key": episode_key,
                    "episode_id": episode_id,
                    "episode": relative,
                    "case_count": 1 + number % 9,
                    "case_ids": [f"case_{number}"],
                    "raw_episode_dir": f"/fixture/raw/{episode_key}",
                    "videos": {"face": f"/fixture/raw/{episode_key}/faceImg.mp4"},
                })
        (source / "episodes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in index_rows), encoding="utf-8"
        )
        benchmark = root / "evaluation_holdout.jsonl"
        benchmark.write_text(
            json.dumps({"uid": "held-out", "episode_key": "held-out"}) + "\n",
            encoding="utf-8",
        )
        return source, EvaluationHoldout.load(benchmark, expected_sha256=None)

    def test_episode_split_is_exact_disjoint_and_covers_all_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, holdout = self._fixture(root)
            report = materialize_split(
                root=source,
                output_root=root / "split",
                holdout=holdout,
                test_fraction=0.20,
                seed=827,
            )
            distribution = report["distribution"]
            self.assertEqual(distribution["eligible_episodes"], 150)
            self.assertEqual(distribution["test_episodes"], 30)
            self.assertEqual(distribution["train_episodes"], 120)
            self.assertEqual(distribution["train_test_episode_overlap"], 0)
            self.assertTrue(distribution["all_failure_types_in_both_splits"])
            self.assertFalse(report["legacy_gold_considered"])
            for code in FAILURE_CODES:
                counts = distribution["failure_type_episodes"][code]
                self.assertGreater(counts["train"], 0)
                self.assertGreater(counts["test"], 0)

            train = {
                json.loads(line)["episode_key"]
                for line in (root / "split/train_episodes.jsonl").read_text().splitlines()
            }
            test = {
                json.loads(line)["episode_key"]
                for line in (root / "split/test_episodes.jsonl").read_text().splitlines()
            }
            self.assertFalse(train & test)


if __name__ == "__main__":
    unittest.main()
