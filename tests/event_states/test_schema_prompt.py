from __future__ import annotations

import copy
import json
import re
import unittest

from x_planner.data.event_states.prompt import (
    execution_instruction,
    normalized_execution_instruction,
    render_user,
)
from x_planner.data.event_states.schema import (
    FAILURE_TYPES,
    SCHEMA_VERSION,
    V5ValidationError,
    dumps_with_mask_spans,
    output_profile_id,
    parse_json_pointer,
    validate_sample,
    validate_model_visible_text,
    validate_target,
)


def _plan() -> list[dict]:
    return [
        {
            "index": 1,
            "action": {
                "caption": "Approach and grasp the green block",
                "segments": [
                    {
                        "index": 1,
                        "segment": {"caption": "Align the gripper with the block"},
                    }
                ],
            },
        }
    ]


def _predictions() -> list[dict]:
    return [
        {
            "index": 1,
            "role": "current",
            "action": {
                "available": True,
                "caption": "Approach and grasp the green block",
                "progress_percent": 60,
            },
            "segment": {
                "available": True,
                "caption": "Align the gripper with the block",
                "progress_percent": 40,
            },
        },
        {
            "index": 2,
            "role": "next",
            "action": {
                "available": True,
                "caption": "Place the green block in the tray",
                "progress_percent": 0,
            },
        },
    ]


def _spec() -> dict[str, list[str]]:
    return {
        "prediction1_units": ["action", "segment"],
        "prediction2_units": ["action"],
        "plan_units": ["action", "segment"],
    }


def _target(decision: str, failure_type: str | None = None) -> dict:
    detail = None
    if decision == "Replan":
        detail = {
            "reason": "The original placement route is obstructed",
            "updated_plan": _plan(),
        }
    elif decision == "Takeover":
        detail = {
            "failure_analysis": {
                "failed_action_context": "Approach and grasp the green block",
                "expected_action": "Secure the green block in the gripper",
                "observed_failure": "The gripper closes without securing the block",
                "failure_type": failure_type or FAILURE_TYPES[0],
            },
            "recovery_plan": _plan(),
        }
    elif decision == "End":
        detail = {"outcome": "completed"}
    if decision == "Takeover":
        return {
            "execution_decision": decision,
            "decision_detail": detail,
        }
    return {
        "task_progress_percent": 42,
        "predictions": _predictions(),
        "execution_decision": decision,
        "decision_detail": detail,
    }


def _memory_context() -> dict:
    return {
        "initial_plan_memory": _plan(),
        "long_memory": [
            {
                "index": 1,
                "action": "Approach the green block",
                "segment": "Align the gripper",
            }
        ],
        "short_memory": {
            "task_progress_percent": 35,
            "prediction1": {
                "action": {
                    "available": True,
                    "caption": "Approach the green block",
                    "progress_percent": 70,
                },
                "segment": {
                    "available": True,
                    "caption": "Align the gripper",
                    "progress_percent": 50,
                },
            },
        },
    }


def _sample(
    decision: str,
    *,
    memory_variant: str = "no_memory",
    prompt_context: dict | None = None,
    images: list | None = None,
) -> dict:
    category = {
        "Continue": "ongoing",
        "Replan": "ongoing",
        "Takeover": "takeover",
        "End": "end",
    }[decision]
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"sample-{decision.casefold()}-{memory_variant}",
        "base_sample_id": f"base-{decision.casefold()}",
        "source": "unit_test",
        "category": category,
        "memory_variant": memory_variant,
        "output_spec": _spec(),
        "output_profile_id": output_profile_id(_spec()),
        "task_instruction": "Place the green block in the tray",
        "images": images or ["/data/head.jpg", "/data/left.jpg"],
        "prompt_context": {} if prompt_context is None else prompt_context,
        "target": _target(decision),
        "supervision": {"loss_mask_paths": []},
        "provenance": {},
    }


class SchemaPromptV5Test(unittest.TestCase):
    def test_measurement_decimals_are_not_failure_codes(self) -> None:
        captions = (
            "Rotate the robot body 1.5 circles clockwise",
            "Lift the 1.5 kg plate above the basket",
            "No motion occurs during the 4.2-second observation window",
            "Move forward 2.1 meters",
            "Increase the gripper opening by 1.5%",
            "Retract upward to recover the robotic arm",
        )
        for caption in captions:
            with self.subTest(caption=caption):
                self.assertEqual(
                    validate_model_visible_text(caption), caption
                )

    def test_raw_failure_codes_remain_forbidden(self) -> None:
        captions = (
            "4.2 Manipulation failure after grasping",
            "Failure code: 2.1 means the wrong object was grasped",
        )
        for caption in captions:
            with self.subTest(caption=caption):
                with self.assertRaisesRegex(V5ValidationError, "raw failure code"):
                    validate_model_visible_text(caption)

    def test_four_execution_decisions_validate(self) -> None:
        for decision in ("Continue", "Replan", "Takeover", "End"):
            with self.subTest(decision=decision):
                sample = _sample(decision)
                validated = validate_sample(sample)
                self.assertEqual(
                    validated["target"]["execution_decision"], decision
                )

    def test_removed_recover_decision_is_rejected(self) -> None:
        sample = _sample("Continue")
        sample["target"]["execution_decision"] = "Recover"
        with self.assertRaisesRegex(V5ValidationError, "invalid execution_decision"):
            validate_sample(sample)

    def test_all_execution_decisions_share_exact_prompt_suffix(self) -> None:
        prompts = []
        for decision in ("Continue", "Replan", "Takeover", "End"):
            sample = _sample(decision)
            validate_sample(sample)
            prompt = render_user(sample)
            self.assertEqual(
                normalized_execution_instruction(prompt),
                execution_instruction(sample["output_spec"]),
            )
            prompts.append(prompt)
        self.assertEqual(len(set(prompts)), 1)

    def test_takeover_is_no_memory_and_context_free(self) -> None:
        sample = _sample("Takeover")
        self.assertEqual(validate_sample(sample)["prompt_context"], {})

        leaked_context = copy.deepcopy(sample)
        leaked_context["prompt_context"] = _memory_context()
        with self.assertRaisesRegex(
            V5ValidationError, "no_memory prompt_context must be empty"
        ):
            validate_sample(leaked_context)

        with_memory = copy.deepcopy(sample)
        with_memory["memory_variant"] = "with_memory"
        with_memory["prompt_context"] = _memory_context()
        with self.assertRaisesRegex(V5ValidationError, "takeover must use no_memory"):
            validate_sample(with_memory)

    def test_validate_sample_checks_with_and_without_memory_context(self) -> None:
        no_memory = _sample("Continue")
        validate_sample(no_memory)

        with_memory = _sample(
            "Continue",
            memory_variant="with_memory",
            prompt_context=_memory_context(),
        )
        validate_sample(with_memory)

        end_with_memory = _sample(
            "End",
            memory_variant="with_memory",
            prompt_context=_memory_context(),
        )
        validate_sample(end_with_memory)

        missing_context = copy.deepcopy(with_memory)
        missing_context["prompt_context"] = {}
        with self.assertRaisesRegex(
            V5ValidationError, "with_memory context must use"
        ):
            validate_sample(missing_context)

        unexpected_context = copy.deepcopy(no_memory)
        unexpected_context["prompt_context"] = _memory_context()
        with self.assertRaisesRegex(
            V5ValidationError, "no_memory prompt_context must be empty"
        ):
            validate_sample(unexpected_context)

    def test_images_accept_strings_and_video_or_path_objects(self) -> None:
        sample = _sample(
            "Continue",
            images=[
                "/data/head.jpg",
                {"video": "/data/left.mp4", "frame": 20, "view": "left_wrist"},
                {"path": "/data/right.jpg", "view": "right_wrist"},
            ],
        )
        validate_sample(sample)

        for invalid in (" ", {}, {"video": ""}, {"frame": 20}, 7):
            with self.subTest(invalid=invalid):
                broken = copy.deepcopy(sample)
                broken["images"] = [invalid]
                with self.assertRaisesRegex(V5ValidationError, r"images\[0\]"):
                    validate_sample(broken)

    def test_fifteen_failure_types_validate_without_source_codes(self) -> None:
        self.assertEqual(len(FAILURE_TYPES), 15)
        self.assertEqual(len(set(FAILURE_TYPES)), 15)
        for failure_type in FAILURE_TYPES:
            with self.subTest(failure_type=failure_type):
                self.assertRegex(failure_type, r"[A-Za-z]")
                self.assertIsNone(re.search(r"(?<!\d)\d+\.\d+(?!\d)", failure_type))
                target = validate_target(
                    _target("Takeover", failure_type), "takeover", _spec()
                )
                self.assertEqual(
                    target["decision_detail"]["failure_analysis"]["failure_type"],
                    failure_type,
                )
        with self.assertRaisesRegex(V5ValidationError, "15-class taxonomy"):
            validate_target(
                _target("Takeover", "An unsupported failure type"), "takeover", _spec()
            )

    def test_json_pointer_spans_cover_exact_serialized_values(self) -> None:
        target = _target("Continue")
        paths = (
            "/task_progress_percent",
            "/predictions/0/segment/caption",
            "/predictions/1/action",
        )
        text, spans = dumps_with_mask_spans(target, "ongoing", _spec(), paths)
        self.assertEqual(json.loads(text), validate_target(target, "ongoing", _spec()))
        by_path = {path: text[start:end] for start, end, path in spans}
        self.assertEqual(set(by_path), set(paths))
        self.assertEqual(by_path["/task_progress_percent"], "42")
        self.assertEqual(
            json.loads(by_path["/predictions/0/segment/caption"]),
            "Align the gripper with the block",
        )
        self.assertEqual(
            json.loads(by_path["/predictions/1/action"]),
            {
                "available": True,
                "caption": "Place the green block in the tray",
                "progress_percent": 0,
            },
        )
        self.assertEqual(parse_json_pointer("/a~1b/~0c"), ("a/b", "~c"))

        with self.assertRaisesRegex(V5ValidationError, "do not exist"):
            dumps_with_mask_spans(target, "ongoing", _spec(), ("/missing",))
        with self.assertRaisesRegex(V5ValidationError, "overlap by ancestry"):
            dumps_with_mask_spans(
                target,
                "ongoing",
                _spec(),
                ("/predictions/0", "/predictions/0/action"),
            )

    def test_prompt_and_target_follow_each_prediction_unit_profile(self) -> None:
        profiles = (
            (["action"], ["action"], "Action only", "Segment"),
            (["segment"], ["segment"], "Segment only", "Action"),
            (["action", "segment"], ["action"], "Action and Segment", None),
        )
        for prediction1_units, prediction2_units, expected, forbidden in profiles:
            with self.subTest(prediction1_units=prediction1_units):
                sample = _sample("Continue")
                spec = {
                    "prediction1_units": prediction1_units,
                    "prediction2_units": prediction2_units,
                    "plan_units": ["action"],
                }
                sample["output_spec"] = spec
                sample["output_profile_id"] = output_profile_id(spec)
                source = _predictions()
                predictions = []
                for index, units in enumerate((prediction1_units, prediction2_units)):
                    raw = source[index]
                    selected_units = {}
                    for unit in units:
                        value = copy.deepcopy(raw.get(unit, source[0][unit]))
                        if index == 1:
                            value["progress_percent"] = 0
                        selected_units[unit] = value
                    predictions.append({
                        "index": index + 1,
                        "role": "current" if index == 0 else "next",
                        **selected_units,
                    })
                sample["target"]["predictions"] = predictions
                validate_sample(sample)
                suffix = normalized_execution_instruction(render_user(sample))
                self.assertIn(f"Prediction 1 {expected}", suffix)
                if forbidden and prediction1_units == prediction2_units:
                    self.assertNotIn(f"Prediction 1 {forbidden}", suffix)
                self.assertEqual(tuple(predictions[0]), ("index", "role", *prediction1_units))

    def test_takeover_target_has_exactly_two_top_level_keys(self) -> None:
        sample = _sample("Takeover")
        validated = validate_sample(sample)
        self.assertEqual(
            tuple(validated["target"]),
            ("execution_decision", "decision_detail"),
        )
        self.assertNotIn("predictions", validated["target"])
        self.assertNotIn("task_progress_percent", validated["target"])


if __name__ == "__main__":
    unittest.main()
