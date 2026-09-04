from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.train.v10_continuous.schema import TargetValidationError
from scripts.train.v10_continuous_v4_instruction.convert_snapshot_v4 import (
    transform_sample,
    v4_sample_key,
)
from scripts.train.v10_continuous_v4_instruction.prompt_v4 import (
    render_continuous_user,
    render_initial_plan_user,
)
from scripts.train.v10_continuous_v4_instruction.schema_v4 import (
    dumps_assistant,
    validate_continuous_target,
    validate_initial_plan,
)


def _prediction(index: int, caption: str, progress: int) -> dict:
    return {
        "index": index,
        "l0": {
            "level": "L0",
            "source": "segment",
            "caption": caption,
            "progress_percent": progress,
        },
    }


def _continuous_sample(*, terminal: bool = False, source: str = "collection") -> dict:
    instruction = "Place the cup in the box"
    second = "the task is complete" if terminal else "Move the hand away from the box"
    return {
        "schema_version": "memory_v4",
        "task_type": "continuous",
        "sample_key": "memory-v4-example",
        "sample_id": "memory-v4-example",
        "global_episode_key": "source:episode",
        "episode_key": "episode",
        "source_id": source,
        "split": "train",
        "profile": "L3L0",
        "unit_type": "segment",
        "is_terminal_window": terminal,
        "task_instruction": instruction,
        "long_memory": [],
        "short_memory": [],
        "images": [{"view": "head", "video": "/data/video.mp4", "frame": 40, "relative_frame": 0}],
        "target": {
            "task_progress_percent": 50,
            "predictions": [_prediction(1, "Move the cup toward the box", 50), _prediction(2, second, 0)],
        },
    }


def test_continuous_prompt_conditions_on_exact_l3_and_assistant_omits_it() -> None:
    sample = _continuous_sample()
    user = render_continuous_user(sample)
    assistant = dumps_assistant(
        sample["target"],
        sample["profile"],
        sample["task_type"],
        instruction=sample["task_instruction"],
    )
    assert '[task_instruction][level=L3]\n"Place the cup in the box"\n[/task_instruction]' in user
    assert "Hierarchy (coarse to fine): Task (L3) > Subtask (L2) > Action (L1) > Segment (L0)." in user
    assert "Prediction 2 progress_percent must be 0" not in user
    assert "infer, rewrite, repeat, or output the L3 instruction" not in user
    assert "Do not output task, L3, task_caption" not in user
    assert "at the Segment (L0) scale" in user
    assert "Output levels: Segment (L0)" in user
    assert "[long_memory]\n[none]\n[/long_memory]" in user
    assert "[short_memory]\n[none]\n[/short_memory]" in user
    assert "seconds" not in user
    assert tuple(json.loads(assistant)) == ("task_progress_percent", "predictions")
    assert '"task"' not in assistant
    assert "Place the cup in the box" not in assistant


def test_only_zhengwei_prompt_has_source_rate() -> None:
    assert "20 Hz" not in render_continuous_user(_continuous_sample())
    assert "Source frame rate: 20 Hz." in render_continuous_user(
        _continuous_sample(source="zhengwei")
    )


def test_terminal_caption_and_progress_contract() -> None:
    sample = _continuous_sample(terminal=True)
    validate_continuous_target(
        sample["target"],
        sample["profile"],
        instruction=sample["task_instruction"],
        is_terminal_window=True,
    )
    sample["target"]["predictions"][1]["l0"]["progress_percent"] = 100
    with pytest.raises(TargetValidationError):
        validate_continuous_target(
            sample["target"],
            sample["profile"],
            instruction=sample["task_instruction"],
            is_terminal_window=True,
        )


def test_initial_plan_prompt_and_target_have_no_memory_or_l3_output() -> None:
    sample = _continuous_sample()
    sample.update({
        "task_type": "initial_plan",
        "target": {
            "initial_plan": [{
                "index": 1,
                "l0": {"level": "L0", "source": "segment", "caption": "Move the cup toward the box"},
            }]
        },
    })
    user = render_initial_plan_user(sample)
    validated = validate_initial_plan(
        sample["target"], sample["profile"], instruction=sample["task_instruction"]
    )
    assert tuple(validated) == ("initial_plan",)
    assert "[long_memory]" not in user and "[short_memory]" not in user
    assert json.dumps(sample["task_instruction"]) in user
    assert "Plan levels: Segment (L0)" in user
    assert "Hierarchy (coarse to fine): Task (L3) > Subtask (L2) > Action (L1) > Segment (L0)." in user


@pytest.mark.parametrize(
    ("profile", "unit_type", "labels"),
    [
        ("full", "subtask", "Subtask (L2), Action (L1), Segment (L0)"),
        ("L3L2L1", "subtask", "Subtask (L2), Action (L1)"),
        ("L3L2L0", "subtask", "Subtask (L2), Segment (L0)"),
        ("L3L2", "subtask", "Subtask (L2)"),
        ("L3L1L0", "action", "Action (L1), Segment (L0)"),
        ("L3L1", "action", "Action (L1)"),
        ("L3L0", "segment", "Segment (L0)"),
    ],
)
def test_every_profile_uses_named_levels_at_one_scale(
    profile: str, unit_type: str, labels: str
) -> None:
    sample = _continuous_sample()
    sample["profile"] = profile
    sample["unit_type"] = unit_type
    prompt = render_continuous_user(sample)
    scale = labels.split(",", 1)[0]
    assert f"at the {scale} scale" in prompt
    assert f"Output levels: {labels}." in prompt
    assert "Prediction unit: subtask" not in prompt
    assert "Prediction unit: action" not in prompt
    assert "Prediction unit: segment" not in prompt


def test_transform_v3_moves_l3_from_output_to_input_and_preserves_semantics() -> None:
    sample = _continuous_sample()
    old_key = "memory-v3-example"
    v3 = dict(sample)
    v3.update({"schema_version": "memory_v3", "sample_key": old_key, "sample_id": old_key})
    v3.pop("task_instruction")
    v3["task_caption"] = sample["task_instruction"]
    v3["target"] = {
        "task": {"level": "L3", "caption": sample["task_instruction"], "progress_percent": 50},
        "predictions": sample["target"]["predictions"],
    }
    result = transform_sample(v3)
    assert result["sample_key"] == v4_sample_key(old_key)
    assert result["task_instruction"] == sample["task_instruction"]
    assert "task_caption" not in result
    assert tuple(result["target"]) == ("task_progress_percent", "predictions")
    assert result["target"]["predictions"] == v3["target"]["predictions"]
    assert result["lineage"]["source_sample_key"] == old_key


def test_rejects_extra_l3_assistant_structure_and_non_integer_progress() -> None:
    sample = _continuous_sample()
    bad = {"task": {"caption": sample["task_instruction"]}, **sample["target"]}
    with pytest.raises(TargetValidationError):
        validate_continuous_target(
            bad, sample["profile"], instruction=sample["task_instruction"], is_terminal_window=False
        )
    sample["target"]["task_progress_percent"] = 1.5
    with pytest.raises(TargetValidationError):
        validate_continuous_target(
            sample["target"], sample["profile"], instruction=sample["task_instruction"], is_terminal_window=False
        )


def test_instruction_is_json_escaped_in_prompt() -> None:
    sample = _continuous_sample()
    sample["task_instruction"] = 'Place the "red" cup\nin the box'
    prompt = render_continuous_user(sample)
    assert json.dumps(sample["task_instruction"], ensure_ascii=False) in prompt
    assert sample["task_instruction"] not in prompt


def test_smoke_batch_size_is_wired_through_data_train_and_report() -> None:
    entry = Path(__file__).parents[2] / "run_v10_continuous_v4_instruction.sh"
    source = entry.read_text(encoding="utf-8")
    assert '--batch-size "${PER_DEVICE_BATCH_SIZE}"' in source
    assert '--per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}"' in source
    assert source.count('--batch-size "${PER_DEVICE_BATCH_SIZE}"') == 2
    assert "data_smoke_batch6.json" not in source
