from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .infer_v5 import _indexed_leaf_candidates, extract_json, parse_prediction


class InferV5ParsingTest(unittest.TestCase):
    def test_code_fenced_initial_plan_is_strictly_validated(self) -> None:
        target = {
            "initial_plan": [{
                "index": 1,
                "action": {"caption": "Move to the green block"},
            }]
        }
        sample = {
            "category": "initial_plan",
            "output_spec": {
                "prediction1_units": [],
                "prediction2_units": [],
                "plan_units": ["action"],
            },
        }
        text = "```json\n" + json.dumps(target) + "\n```"
        self.assertEqual(extract_json(text), json.dumps(target))
        parsed, error = parse_prediction(text, sample)
        self.assertIsNone(error)
        self.assertEqual(parsed, target)

    def test_non_json_generation_is_reported_not_raised(self) -> None:
        sample = {
            "category": "initial_plan",
            "output_spec": {
                "prediction1_units": [],
                "prediction2_units": [],
                "plan_units": ["action"],
            },
        }
        parsed, error = parse_prediction("not json", sample)
        self.assertIsNone(parsed)
        self.assertIn("JSONDecodeError", str(error))

    def test_indexed_leaf_candidates_use_safe_sorted_root_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary)
            (snapshot / "manifest.json").write_text(json.dumps({
                "schema_version": "v10_action_segment_v5_snapshot_v3",
                "complete": True,
                "leaves": [
                    {
                        "relative_path": "data/takeover_q/no_memory/takeover/z/train",
                        "source": "takeover_q",
                        "category": "takeover",
                        "memory_variant": "no_memory",
                    },
                    {
                        "relative_path": "data/robodojo/no_memory/initial_plan/a/train",
                        "source": "robodojo",
                        "category": "initial_plan",
                        "memory_variant": "no_memory",
                    },
                ],
            }))
            candidates = _indexed_leaf_candidates(snapshot)
            self.assertIsNotNone(candidates)
            assert candidates is not None
            self.assertEqual(
                [path.relative_to(snapshot).as_posix() for _, path in candidates],
                [
                    "data/robodojo/no_memory/initial_plan/a/train",
                    "data/takeover_q/no_memory/takeover/z/train",
                ],
            )

    def test_indexed_leaf_candidates_reject_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary)
            (snapshot / "manifest.json").write_text(json.dumps({
                "schema_version": "v10_action_segment_v5_snapshot_v3",
                "complete": True,
                "leaves": [{
                    "relative_path": "data/../outside",
                    "source": "takeover_q",
                    "category": "takeover",
                    "memory_variant": "no_memory",
                }],
            }))
            with self.assertRaisesRegex(
                ValueError, "unsafe V5 root-manifest leaf path"
            ):
                _indexed_leaf_candidates(snapshot)


if __name__ == "__main__":
    unittest.main()
