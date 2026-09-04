from __future__ import annotations

import json
import re
import tempfile
import unittest
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from typing import Any

from x_planner.data.event_states.takeover_adapter import (
    ANCHOR_SELECTION_POLICY,
    ANCHOR_STRIDE_FRAMES,
    ANCHOR_WINDOW_START_DENOMINATOR,
    ANCHOR_WINDOW_START_NUMERATOR,
    DEFAULT_REVIEWED_ROOT,
    DEFAULT_SNAPSHOT_ROOT,
    FAILURE_TYPE_BY_SOURCE_CODE,
    MODEL_VISIBLE_TEXT_NORMALIZATION_VERSION,
    SNAPSHOT_ID,
    TAKEOVER_INSTRUCTION_FIELDS,
    TAKEOVER_INSTRUCTION_POLICY,
    TakeoverQAdapter,
    TakeoverQDataError,
    UnparseableQ2Error,
    VideoMetadata,
    _anchor_frame_bounds,
    _anchor_frames,
)


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    snapshot_root = root / "rootfs"
    reviewed_root = root / "reviewed" / "new_completed"
    episode_dir = reviewed_root / "episodes"
    episode_dir.mkdir(parents=True)

    raw_dir = Path("/source/demo_episode")
    videos = {
        "face": str(raw_dir / "faceImg.mp4"),
        "left": str(raw_dir / "leftImg.mp4"),
        "right": str(raw_dir / "rightImg.mp4"),
    }
    for raw_path in videos.values():
        snapshot_path = snapshot_root.joinpath(*Path(raw_path).parts[1:])
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(b"test-video-placeholder")

    instruction_path = snapshot_root / "source" / "instruction.json"
    instruction_path.parent.mkdir(parents=True, exist_ok=True)
    instruction_path.write_text(
        json.dumps({
            "episode@TAKE_OVER_MODE@test": {
                "detailed_instruction": (
                    "Move to the green block and place it in the tray"
                ),
                "instruction": "Place the green block in the tray",
            }
        }),
        encoding="utf-8",
    )

    case = {
        "schema_version": "takeover_q_bilingual_review_v1",
        "case_id": "case_test_001",
        "episode_id": "episode@TAKE_OVER_MODE@test",
        "review": {
            "review_status": "completed",
            "reviewer_id": "offline-reviewer",
            "review_revision": 7,
            "review_updated_at": "2026-08-12T00:00:00+00:00",
            "source_patch_sha256": "a" * 64,
        },
        "bilingual": {
            "instruction_zh": "离线中文不得进入模型字段",
            "instruction": "Place the green block in the tray",
            "detailed_instruction_zh": "离线计划",
            "detailed_instruction": "Move to the block and place it in the tray",
            "segments": {
                "q1": [
                    {
                        "id": "q1-1",
                        "start_sec": 0.1,
                        "end_sec": 1.1,
                        "caption_zh": "正常移动",
                        "caption": "The arm moves toward the green block",
                    }
                ],
                "q2q3": [
                    {
                        "id": "q2q3",
                        "start_sec": 1.05,
                        "end_sec": 2.25,
                        "q2_caption_zh": "应接近 -> 实际静止",
                        "q2_caption": (
                            "Expected action: Move close to the green block "
                            "-> Actual failure: The arm remains stationary"
                        ),
                        "q3_type_zh": "1.1 离线失败类别",
                        "q3_type": "1.1 Planning failure -- the arm remains stationary",
                    }
                ],
                "q4": [
                    {
                        "id": "q4",
                        "start_sec": 0.6,
                        "end_sec": 2.25,
                        "caption_zh": "接近绿色木块",
                        "caption": "Approach and align with the green block",
                    }
                ],
                "takeover": [
                    {
                        "id": "takeover",
                        "start_sec": 2.25,
                        "end_sec": 3.0,
                        "time_locked": True,
                        "caption_zh": "人工重新接近并抓取",
                        "caption": "Move closer, realign the gripper, and grasp the green block",
                    }
                ],
            },
        },
    }
    episode = {
        "schema_version": "takeover_q_bilingual_episode_v1",
        "episode_id": "episode@TAKE_OVER_MODE@test",
        "episode_key": "episode_test",
        "case_count": 1,
        "case_ids": ["case_test_001"],
        "review": {"review_status": "completed"},
        "cases": [case],
    }
    episode_path = episode_dir / "episode_test.json"
    episode_path.write_text(json.dumps(episode, ensure_ascii=False), encoding="utf-8")
    index_row = {
        "episode_id": episode["episode_id"],
        "episode_key": episode["episode_key"],
        "episode": "episodes/episode_test.json",
        "case_count": 1,
        "case_ids": ["case_test_001"],
        "raw_episode_dir": str(raw_dir),
        "videos": videos,
    }
    (reviewed_root / "episodes.jsonl").write_text(
        json.dumps(index_row) + "\n", encoding="utf-8"
    )
    return snapshot_root, reviewed_root, episode_path


def _adapter(
    snapshot_root: Path,
    reviewed_root: Path,
    probe_calls: list[Path] | None = None,
) -> TakeoverQAdapter:
    def probe(path: Path) -> VideoMetadata:
        if probe_calls is not None:
            probe_calls.append(path)
        return VideoMetadata(fps=Fraction(20, 1), frame_count=100)

    return TakeoverQAdapter(
        snapshot_root=snapshot_root,
        reviewed_root=reviewed_root,
        video_probe=probe,
    )


def _write_cases(episode_path: Path, cases: list[dict[str, Any]]) -> None:
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    episode["cases"] = cases
    episode["case_count"] = len(cases)
    episode["case_ids"] = [case["case_id"] for case in cases]
    episode_path.write_text(json.dumps(episode, ensure_ascii=False), encoding="utf-8")
    index_path = episode_path.parent.parent / "episodes.jsonl"
    index_row = json.loads(index_path.read_text(encoding="utf-8"))
    index_row["case_count"] = len(cases)
    index_row["case_ids"] = episode["case_ids"]
    index_path.write_text(json.dumps(index_row) + "\n", encoding="utf-8")


class TakeoverQAdapterTest(unittest.TestCase):
    def test_fixed_defaults_name_only_the_final_reviewed_snapshot(self) -> None:
        self.assertEqual(SNAPSHOT_ID, "reviewed-release")
        self.assertEqual(DEFAULT_SNAPSHOT_ROOT, Path("/path/to/takeover"))
        self.assertEqual(DEFAULT_REVIEWED_ROOT.name, "reviewed")
        self.assertNotIn("cpfs", str(DEFAULT_REVIEWED_ROOT).lower())

    def test_taxonomy_has_fifteen_unique_code_free_english_labels(self) -> None:
        self.assertEqual(len(FAILURE_TYPE_BY_SOURCE_CODE), 15)
        self.assertEqual(len(set(FAILURE_TYPE_BY_SOURCE_CODE.values())), 15)
        self.assertEqual(
            set(FAILURE_TYPE_BY_SOURCE_CODE),
            {
                "1.1", "1.2", "1.3", "1.4", "1.5", "2.1", "2.2", "3.1",
                "4.1", "4.2", "4.3", "5.1", "6.1", "7.1", "8.1",
            },
        )
        for label in FAILURE_TYPE_BY_SOURCE_CODE.values():
            self.assertIsNone(re.match(r"^\s*\d+\.\d+\b", label))
            self.assertIsNone(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", label))

    def test_builds_takeover_only_in_late_q2_every_ten_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot_root, reviewed_root, _ = _fixture(Path(temporary))
            probe_calls: list[Path] = []
            adapter = _adapter(snapshot_root, reviewed_root, probe_calls)
            records = list(adapter.iter_records())

        self.assertEqual(ANCHOR_STRIDE_FRAMES, 10)
        self.assertEqual(len(probe_calls), 3)
        self.assertEqual(adapter.exclusions, ())
        takeovers = [row for row in records if row["decision_class"] == "Takeover"]
        self.assertEqual({row["decision_class"] for row in records}, {"Takeover"})
        self.assertEqual([row["anchor_frame"] for row in takeovers], [38])
        takeover = takeovers[-1]
        self.assertEqual(takeover["context_frames"], [21, 28, 38])
        self.assertEqual(len(takeover["images"]), 9)
        self.assertEqual(
            [item["view"] for item in takeover["images"][-3:]],
            ["head", "left_wrist", "right_wrist"],
        )
        self.assertEqual(takeover["labels"], {
            "execution_decision": "Takeover",
            "expected_action": "Move close to the green block",
            "observed_failure": "The arm remains stationary",
            "failure_type": "Arm remains stationary instead of approaching the object",
            "failed_action_context": "Approach and align with the green block",
            "recovery_action": (
                "Move closer, realign the gripper, and grasp the green block"
            ),
        })
        self.assertEqual(
            takeover["supervision"]["label_sources"]["failed_action_context"],
            "q4.caption",
        )
        self.assertEqual(
            takeover["supervision"]["label_sources"]["recovery_action"],
            "takeover.caption",
        )
        self.assertEqual(takeover["label_timing_sec"], {
            "q2q3": {"start_sec": 1.05, "end_sec": 2.25},
            "q4": {"start_sec": 0.6, "end_sec": 2.25},
            "takeover": {"start_sec": 2.25, "end_sec": 3.0, "time_locked": True},
        })

        model_visible: dict[str, Any] = {
            "conditioning": takeover["conditioning"],
            "labels": takeover["labels"],
        }
        visible_text = json.dumps(model_visible, ensure_ascii=False)
        self.assertNotIn("1.1", visible_text)
        self.assertIsNone(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", visible_text))
        self.assertEqual(takeover["memory_variant"], "no_memory")
        self.assertEqual(set(takeover["conditioning"]), {"task_instruction"})
        for forbidden in ("plan_memory", "execution_memory", "short_memory", "long_memory"):
            self.assertNotIn(forbidden, visible_text)
        self.assertEqual(takeover["provenance"]["raw_failure_source_key"], "1.1")
        self.assertTrue(takeover["provenance"]["raw_q3_type"].startswith("1.1 "))
        self.assertFalse(takeover["provenance"]["generic_scanner_labels_used"])
        self.assertEqual(
            takeover["provenance"]["reviewed_label_source"],
            "final_reviewed_bilingual",
        )
        self.assertFalse(takeover["provenance"]["future_takeover_frames_used"])
        self.assertEqual(
            takeover["conditioning"]["task_instruction"],
            "Place the green block in the tray",
        )
        self.assertEqual(
            takeover["provenance"]["task_instruction_source"],
            "media_task_instruction_json.episode.instruction",
        )
        self.assertEqual(
            takeover["provenance"]["task_instruction_source_field"],
            "instruction",
        )
        self.assertEqual(
            takeover["provenance"]["task_instruction_policy"],
            TAKEOVER_INSTRUCTION_POLICY,
        )
        self.assertRegex(
            takeover["provenance"]["task_instruction_source_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertEqual(
            takeover["provenance"]["anchor_selection_policy"],
            ANCHOR_SELECTION_POLICY,
        )
        self.assertEqual(takeover["provenance"]["q2_start_frame"], 21)
        self.assertEqual(takeover["provenance"]["q2_late_start_frame"], 38)
        self.assertEqual(takeover["provenance"]["q2_end_frame"], 45)
        self.assertLess(
            max(item["frame"] for item in takeover["images"]),
            45,  # takeover starts at source frame 45
        )

    def test_late_anchor_boundary_is_exact_for_fractional_fps(self) -> None:
        video = VideoMetadata(fps=Fraction(30000, 1001), frame_count=500)
        window = {"start_sec": 1.001, "end_sec": 2.469}
        start, late_start, end = _anchor_frame_bounds(window, video)
        self.assertEqual(
            (ANCHOR_WINDOW_START_NUMERATOR, ANCHOR_WINDOW_START_DENOMINATOR),
            (7, 10),
        )
        self.assertGreaterEqual(
            Fraction(late_start - start, max(1, end - start)), Fraction(7, 10)
        )
        anchors = _anchor_frames(window, video)
        self.assertEqual(anchors[0], late_start)
        self.assertTrue(all(late_start <= value <= end for value in anchors))
        self.assertTrue(
            all(right - left == 10 for left, right in zip(anchors, anchors[1:]))
        )

    def test_single_frame_q2_window_keeps_terminal_anchor(self) -> None:
        video = VideoMetadata(fps=Fraction(20, 1), frame_count=100)
        window = {"start_sec": 1.0, "end_sec": 1.0}
        self.assertEqual(_anchor_frames(window, video), (20,))

    def test_ids_and_outputs_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot_root, reviewed_root, _ = _fixture(Path(temporary))
            first = list(_adapter(snapshot_root, reviewed_root).iter_records())
            second = list(_adapter(snapshot_root, reviewed_root).iter_records())
        self.assertEqual(first, second)
        self.assertEqual(len({row["sample_id"] for row in first}), len(first))

    def test_retired_recover_decision_word_is_normalized_in_arm_retraction_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot_root, reviewed_root, episode_path = _fixture(Path(temporary))
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            segments = episode["cases"][0]["bilingual"]["segments"]
            segments["q2q3"][0]["q2_caption"] = (
                "Expected action: Recover both arms -> "
                "Actual failure: The robot did not recover both arms"
            )
            segments["q4"][0]["caption"] = (
                "After completing the task, recover both arms"
            )
            episode_path.write_text(json.dumps(episode), encoding="utf-8")

            records = list(
                _adapter(snapshot_root, reviewed_root).iter_records(
                    decisions=("Takeover",)
                )
            )

        self.assertTrue(records)
        labels = records[0]["labels"]
        self.assertEqual(labels["expected_action"], "Retract both arms")
        self.assertEqual(
            labels["observed_failure"], "The robot did not retract both arms"
        )
        self.assertEqual(
            labels["failed_action_context"],
            "After completing the task, retract both arms",
        )
        self.assertNotRegex(json.dumps(labels), r"(?i)\brecover\b")
        self.assertEqual(
            records[0]["provenance"]["model_visible_text_normalization_version"],
            MODEL_VISIBLE_TEXT_NORMALIZATION_VERSION,
        )

    def test_q4_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot_root, reviewed_root, episode_path = _fixture(Path(temporary))
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            episode["cases"][0]["bilingual"]["segments"]["q4"] = []
            episode_path.write_text(json.dumps(episode), encoding="utf-8")
            with self.assertRaisesRegex(TakeoverQDataError, "exactly one q4"):
                list(_adapter(snapshot_root, reviewed_root).iter_records())

    def test_unparseable_q2_is_always_explicitly_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot_root, reviewed_root, episode_path = _fixture(Path(temporary))
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            episode["cases"][0]["bilingual"]["segments"]["q2q3"][0][
                "q2_caption"
            ] = "The arm should approach the block. Actually it stays still."
            episode_path.write_text(json.dumps(episode), encoding="utf-8")

            adapter = _adapter(snapshot_root, reviewed_root)
            records = list(adapter.iter_records())
            self.assertEqual(
                {row["decision_class"] for row in records}, set()
            )
            self.assertEqual(len(adapter.exclusions), 1)
            self.assertEqual(
                adapter.exclusions[0]["reason"],
                "q2_missing_deterministic_delimiter",
            )

            # Compatibility mode no longer turns reviewed text quality into a
            # fatal scan error; structural/index/media errors remain fatal.
            strict = _adapter(snapshot_root, reviewed_root)
            strict_records = list(strict.iter_records(strict_q2=True))
            self.assertEqual(
                {row["decision_class"] for row in strict_records}, set()
            )
            self.assertEqual(
                strict.exclusions[0]["reason"],
                "q2_missing_deterministic_delimiter",
            )

    def test_q1_is_not_read_or_reported_by_takeover_only_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot_root, reviewed_root, episode_path = _fixture(Path(temporary))
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            episode["cases"][0]["bilingual"]["segments"]["q1"][0]["caption"] = ""
            episode_path.write_text(json.dumps(episode), encoding="utf-8")

            adapter = _adapter(snapshot_root, reviewed_root)
            records = list(adapter.iter_records())

        self.assertEqual({row["decision_class"] for row in records}, {"Takeover"})
        self.assertEqual(adapter.exclusions, ())

    def test_q2_cjk_is_an_explicit_takeover_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot_root, reviewed_root, episode_path = _fixture(Path(temporary))
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            episode["cases"][0]["bilingual"]["segments"]["q2q3"][0][
                "q2_caption"
            ] = "Expected action: Approach the block -> Actual failure: 机械臂静止"
            episode_path.write_text(json.dumps(episode, ensure_ascii=False), encoding="utf-8")

            adapter = _adapter(snapshot_root, reviewed_root)
            records = list(adapter.iter_records())

        self.assertEqual({row["decision_class"] for row in records}, set())
        self.assertEqual(adapter.exclusions[0]["reason"], "q2_contains_cjk")

    def test_dirty_subtask_and_numbered_q2_labels_are_explicitly_excluded(self) -> None:
        mutations = (
            (
                "Expected action: Complete the current subtask -> "
                "Actual failure: The arm remains stationary",
                "q2_contains_forbidden_v5_term",
            ),
            (
                "Expected action: Retract the arm -> "
                "Actual failure: Failure code 4.2 indicates that the arm remains still",
                "q2_contains_raw_failure_code",
            ),
        )
        for q2_caption, expected_reason in mutations:
            with self.subTest(reason=expected_reason), tempfile.TemporaryDirectory() as temporary:
                snapshot_root, reviewed_root, episode_path = _fixture(Path(temporary))
                episode = json.loads(episode_path.read_text(encoding="utf-8"))
                episode["cases"][0]["bilingual"]["segments"]["q2q3"][0][
                    "q2_caption"
                ] = q2_caption
                episode_path.write_text(json.dumps(episode), encoding="utf-8")

                adapter = _adapter(snapshot_root, reviewed_root)
                records = list(adapter.iter_records())

            self.assertEqual(
                {row["decision_class"] for row in records}, set()
            )
            self.assertEqual(adapter.exclusions[0]["reason"], expected_reason)

    def test_other_model_visible_text_defects_are_nonfatal_case_exclusions(self) -> None:
        mutations = (
            ("q4", "", "q4_missing_english_caption", set()),
            ("q4", "错误动作", "q4_contains_cjk", set()),
            (
                "q4",
                "Complete the current subtask",
                "q4_contains_forbidden_v5_term",
                set(),
            ),
            ("takeover", "", "takeover_missing_english_caption", set()),
            ("takeover", "人工恢复", "takeover_contains_cjk", set()),
        )
        for field, value, expected_reason, expected_decisions in mutations:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as temporary:
                snapshot_root, reviewed_root, episode_path = _fixture(Path(temporary))
                episode = json.loads(episode_path.read_text(encoding="utf-8"))
                episode["cases"][0]["bilingual"]["segments"][field][0][
                    "caption"
                ] = value
                episode_path.write_text(
                    json.dumps(episode, ensure_ascii=False), encoding="utf-8"
                )

                adapter = _adapter(snapshot_root, reviewed_root)
                records = list(adapter.iter_records())

                self.assertEqual(
                    {row["decision_class"] for row in records}, expected_decisions
                )
                self.assertEqual(len(adapter.exclusions), 1)
                self.assertEqual(adapter.exclusions[0]["reason"], expected_reason)

    def test_corresponding_json_detailed_instruction_overrides_reviewed_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot_root, reviewed_root, episode_path = _fixture(Path(temporary))
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            bilingual = episode["cases"][0]["bilingual"]
            bilingual["instruction"] = "4 5 letters task"
            bilingual["detailed_instruction"] = "Stale copied review text"
            episode_path.write_text(
                json.dumps(episode, ensure_ascii=False), encoding="utf-8"
            )

            instruction_path = snapshot_root / "source" / "instruction.json"
            payload = json.loads(instruction_path.read_text(encoding="utf-8"))
            record = payload["episode@TAKE_OVER_MODE@test"]
            record["instruction"] = "4 5 letters task"
            record["detailed_instruction"] = (
                "Pick up the requested letter and place it on the wooden board"
            )
            instruction_path.write_text(json.dumps(payload), encoding="utf-8")

            adapter = _adapter(snapshot_root, reviewed_root)
            records = list(adapter.iter_records())

        self.assertTrue(records)
        self.assertEqual(adapter.exclusions, ())
        self.assertEqual(
            {row["conditioning"]["task_instruction"] for row in records},
            {"Pick up the requested letter and place it on the wooden board"},
        )
        self.assertEqual(
            {row["provenance"]["task_instruction_source"] for row in records},
            {"media_task_instruction_json.episode.detailed_instruction"},
        )
        self.assertTrue(all(
            "task_slug_or_placeholder" in " ".join(
                row["provenance"]["task_instruction_rejected_candidates"]
            )
            for row in records
        ))

    def test_configured_json_field_order_falls_back_to_detailed_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot_root, reviewed_root, _ = _fixture(Path(temporary))
            instruction_path = snapshot_root / "source" / "instruction.json"
            payload = json.loads(instruction_path.read_text(encoding="utf-8"))
            record = payload["episode@TAKE_OVER_MODE@test"]
            record["instruction"] = ""
            record["detailed_instruction"] = "Place the green block in the tray"
            instruction_path.write_text(json.dumps(payload), encoding="utf-8")

            adapter = _adapter(snapshot_root, reviewed_root)
            records = list(adapter.iter_records())

        self.assertTrue(records)
        self.assertEqual(adapter.exclusions, ())
        self.assertEqual(
            {row["conditioning"]["task_instruction"] for row in records},
            {"Place the green block in the tray"},
        )
        self.assertEqual(
            {row["provenance"]["task_instruction_source"] for row in records},
            {"media_task_instruction_json.episode.detailed_instruction"},
        )

    def test_missing_exact_json_entry_never_falls_back_to_reviewed_or_path_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot_root, reviewed_root, _ = _fixture(Path(temporary))
            instruction_path = snapshot_root / "source" / "instruction.json"
            instruction_path.write_text(
                json.dumps({"another_episode": {"instruction": "Wrong episode"}}),
                encoding="utf-8",
            )

            adapter = _adapter(snapshot_root, reviewed_root)
            records = list(adapter.iter_records())

        self.assertEqual(records, [])
        self.assertEqual(len(adapter.exclusions), 1)
        self.assertEqual(
            adapter.exclusions[0]["reason"],
            "task_instruction_json_no_usable_configured_field",
        )

    def test_bad_case_does_not_block_a_later_good_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot_root, reviewed_root, episode_path = _fixture(Path(temporary))
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            bad = episode["cases"][0]
            good = deepcopy(bad)
            bad["bilingual"]["segments"]["q4"][0]["caption"] = "坏指令"
            good["case_id"] = "case_test_002"
            _write_cases(episode_path, [bad, good])

            adapter = _adapter(snapshot_root, reviewed_root)
            records = list(adapter.iter_records())

        self.assertEqual({row["provenance"]["case_id"] for row in records}, {"case_test_002"})
        self.assertEqual(
            {row["decision_class"] for row in records}, {"Takeover"}
        )
        self.assertEqual(adapter.exclusions[0]["case_id"], "case_test_001")
        self.assertEqual(adapter.exclusions[0]["reason"], "q4_contains_cjk")

    def test_instruction_field_configuration_is_explicit_and_validated(self) -> None:
        self.assertEqual(
            TAKEOVER_INSTRUCTION_FIELDS,
            ("instruction", "detailed_instruction"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            snapshot_root, reviewed_root, _ = _fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "instruction_fields"):
                TakeoverQAdapter(
                    snapshot_root=snapshot_root,
                    reviewed_root=reviewed_root,
                    instruction_fields=("task",),
                )


if __name__ == "__main__":
    unittest.main()
