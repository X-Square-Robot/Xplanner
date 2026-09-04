"""Read-only RoboDojo planner-data adapter for V5 canonical supervision.

This module intentionally reads only the ``robotwin30_x2/arx_x5`` media and
English caption trees.  It never reads robot state/action trajectories.  The
adapter keeps the official split as the episode-level authority, excludes the
known quarantined episode, and emits Action-only candidates.  Segment labels
are explicitly unavailable so the downstream renderer can mask them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Standalone import is retained for direct adapter usage.
    from .task_instruction import TaskInstructionError, select_record_instruction
except ImportError:  # pragma: no cover - exercised by standalone consumers
    from task_instruction import TaskInstructionError, select_record_instruction


SCHEMA_VERSION = "v5_robodojo_canonical_v1"
SOURCE_NAME = "robotwin30_x2/arx_x5"

DEFAULT_MEDIA_ROOT = Path(os.environ.get(
    "XPLANNER_ROBODOJO_MEDIA_ROOT", "/path/to/robodojo/media"
))
DEFAULT_LABEL_ROOT = Path(os.environ.get(
    "XPLANNER_ROBODOJO_LABEL_ROOT", "/path/to/robodojo/labels"
))
DEFAULT_OFFICIAL_SPLIT = Path(os.environ.get(
    "XPLANNER_ROBODOJO_SPLIT", "/path/to/robodojo/official_split.json",
))

ROBODOJO_TASKS = (
    "arrange_largest_number",
    "build_tower",
    "classify_objects",
    "cover_blocks",
    "deposit_coin",
    "dlc",
    "fasten_screws",
    "fill_egg_holder",
    "fill_pen_holder",
    "fold_clothes",
    "hang_mugs",
    "imitate_sorting_sequence",
    "insert_key",
    "insert_tubes",
    "make_kong",
    "make_toast",
    "match_and_pick_from_conveyor",
    "organize_table",
    "pack_objects_into_box",
    "play_Xylophone",
    "play_stacking_toy",
    "play_tic_tac_toe",
    "plug_in_charger",
    "pour_balls_into_vase",
    "pour_liquid_into_cup",
    "press_by_number",
    "push_T",
    "put_bottles_into_dustbin",
    "sort_nesting_dolls_by_size",
    "stack_blocks",
    "stack_bowls",
    "store_laptop_and_headphones",
    "swap_T",
    "swap_blocks",
    "sweep_blocks",
)

VIDEO_FILES = {
    "face_view": "faceImg.mp4",
    "left_wrist_view": "leftImg.mp4",
    "right_wrist_view": "rightImg.mp4",
}
OFFICIAL_SPLITS = ("train", "holdout_traj", "holdout_task")

_TASK_RE = re.compile(r"^[A-Za-z0-9_]+$")
_TRAJECTORY_RE = re.compile(r"^trajectory_(0|[1-9][0-9]*)$")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_SPACE_RE = re.compile(r"\s+")
_BLOCKED_DATA_TREE_MARKERS = ("open_action_dataset", "robodojo_ee_v2")


class RobodojoAdapterError(ValueError):
    """Raised when source metadata violates the canonical adapter contract."""


@dataclass(frozen=True, slots=True)
class ActionLabel:
    start_frame: int
    end_frame: int
    caption: str


@dataclass(frozen=True, slots=True)
class RobodojoEpisode:
    canonical_episode_id: str
    task_name: str
    trajectory_name: str
    split: str
    task_instruction: str
    total_frames: int
    videos: tuple[tuple[str, str], ...]
    actions: tuple[ActionLabel, ...]
    media_instruction_file: str
    action_annotation_file: str
    task_instruction_source: str = "media_instruction_json.episode.instruction"


@dataclass(frozen=True, slots=True)
class ScanIssue:
    canonical_episode_id: str
    reason: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "canonical_episode_id": self.canonical_episode_id,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class RobodojoScanResult:
    episodes: tuple[RobodojoEpisode, ...]
    issues: tuple[ScanIssue, ...]
    excluded_by_split: int
    excluded_by_quarantine: int

    def summary(self) -> dict[str, Any]:
        split_counts = Counter(episode.split for episode in self.episodes)
        task_counts = Counter(episode.task_name for episode in self.episodes)
        issue_counts = Counter(issue.reason for issue in self.issues)
        return {
            "schema_version": SCHEMA_VERSION,
            "source": SOURCE_NAME,
            "accepted_episodes": len(self.episodes),
            "accepted_tasks": len(task_counts),
            "episodes_by_split": dict(sorted(split_counts.items())),
            "episodes_by_task": dict(sorted(task_counts.items())),
            "issues": len(self.issues),
            "issues_by_reason": dict(sorted(issue_counts.items())),
            "excluded_by_split": self.excluded_by_split,
            "excluded_by_quarantine": self.excluded_by_quarantine,
        }


def _clean_english(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = _SPACE_RE.sub(" ", value).strip()
    if not text or not _LATIN_RE.search(text) or _CJK_RE.search(text):
        return ""
    return text


def _validate_task_name(task_name: str) -> None:
    if not _TASK_RE.fullmatch(task_name):
        raise RobodojoAdapterError(f"invalid task name: {task_name!r}")


def _trajectory_number(trajectory_name: str) -> int:
    match = _TRAJECTORY_RE.fullmatch(trajectory_name)
    if match is None:
        raise RobodojoAdapterError(
            f"invalid trajectory name: {trajectory_name!r}"
        )
    return int(match.group(1))


def canonical_episode_id(task_name: str, trajectory_name: str) -> str:
    """Return the stable episode identity shared by all derived V5 views."""

    _validate_task_name(task_name)
    _trajectory_number(trajectory_name)
    return f"{SOURCE_NAME}/{task_name}/{trajectory_name}"


QUARANTINED_EPISODES = frozenset(
    {canonical_episode_id("make_toast", "trajectory_72")}
)


def _load_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RobodojoAdapterError(f"cannot read JSON mapping {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RobodojoAdapterError(f"expected JSON object: {path}")
    return value


def _assert_planner_tree(path: Path) -> None:
    lowered = path.resolve().as_posix().casefold()
    if any(marker in lowered for marker in _BLOCKED_DATA_TREE_MARKERS):
        raise RobodojoAdapterError(
            f"robot state/action dataset is not a valid planner-data root: {path}"
        )


def load_official_split_assignments(
    split_path: Path | str,
    *,
    tasks: Sequence[str] = ROBODOJO_TASKS,
    expected_episodes_per_task: int | None = 100,
) -> dict[str, str]:
    """Load and validate the episode-level official split.

    The returned mapping is keyed by canonical episode id.  Overlap is rejected
    rather than resolved by precedence.
    """

    expected_tasks = tuple(tasks)
    if not expected_tasks or len(set(expected_tasks)) != len(expected_tasks):
        raise RobodojoAdapterError("tasks must be a non-empty unique sequence")
    for task_name in expected_tasks:
        _validate_task_name(task_name)

    root = _load_mapping(Path(split_path))
    missing_groups = sorted(set(OFFICIAL_SPLITS) - set(root))
    unexpected_groups = sorted(set(root) - set(OFFICIAL_SPLITS) - {"meta"})
    if missing_groups or unexpected_groups:
        raise RobodojoAdapterError(
            "official split groups do not match the contract: "
            f"missing={missing_groups}, unexpected={unexpected_groups}"
        )

    expected_task_set = set(expected_tasks)
    ids_by_task: dict[str, set[int]] = {task: set() for task in expected_tasks}
    assignments: dict[str, str] = {}
    observed_tasks: set[str] = set()

    for split_name in OFFICIAL_SPLITS:
        task_map = root[split_name]
        if not isinstance(task_map, Mapping):
            raise RobodojoAdapterError(
                f"official split group {split_name!r} must be an object"
            )
        for task_name, trajectory_ids in task_map.items():
            task_name = str(task_name)
            if task_name not in expected_task_set:
                raise RobodojoAdapterError(
                    f"unexpected task {task_name!r} in official split"
                )
            if not isinstance(trajectory_ids, list):
                raise RobodojoAdapterError(
                    f"trajectory ids must be a list for {split_name}/{task_name}"
                )
            observed_tasks.add(task_name)
            for trajectory_id in trajectory_ids:
                if (
                    isinstance(trajectory_id, bool)
                    or not isinstance(trajectory_id, int)
                    or trajectory_id < 0
                ):
                    raise RobodojoAdapterError(
                        f"invalid trajectory id for {split_name}/{task_name}: "
                        f"{trajectory_id!r}"
                    )
                trajectory_name = f"trajectory_{trajectory_id}"
                episode_id = canonical_episode_id(task_name, trajectory_name)
                previous = assignments.get(episode_id)
                if previous is not None:
                    raise RobodojoAdapterError(
                        f"split overlap for {episode_id}: {previous}, {split_name}"
                    )
                assignments[episode_id] = split_name
                ids_by_task[task_name].add(trajectory_id)

    missing_tasks = sorted(expected_task_set - observed_tasks)
    if missing_tasks:
        raise RobodojoAdapterError(
            f"tasks absent from official split: {', '.join(missing_tasks)}"
        )
    if expected_episodes_per_task is not None:
        expected_ids = set(range(expected_episodes_per_task))
        for task_name, observed_ids in ids_by_task.items():
            if observed_ids != expected_ids:
                missing = sorted(expected_ids - observed_ids)
                extra = sorted(observed_ids - expected_ids)
                raise RobodojoAdapterError(
                    f"official split coverage mismatch for {task_name}: "
                    f"missing={missing}, extra={extra}"
                )
    return assignments


def _parse_actions(
    value: Any, *, total_frames: int, episode_id: str
) -> tuple[ActionLabel, ...]:
    if not isinstance(value, Mapping) or not value:
        raise RobodojoAdapterError(f"missing action captions for {episode_id}")

    parsed: list[ActionLabel] = []
    for interval, raw_caption in value.items():
        parts = str(interval).replace(",", " ").split()
        if len(parts) != 2:
            raise RobodojoAdapterError(
                f"invalid action interval {interval!r} for {episode_id}"
            )
        try:
            start_frame, end_frame = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise RobodojoAdapterError(
                f"invalid action interval {interval!r} for {episode_id}"
            ) from exc
        caption = _clean_english(raw_caption)
        if not caption:
            raise RobodojoAdapterError(
                f"action caption is not valid English for {episode_id}: {interval!r}"
            )
        if not 0 <= start_frame < end_frame <= total_frames:
            raise RobodojoAdapterError(
                f"action interval outside [0, {total_frames}] for {episode_id}: "
                f"{start_frame} {end_frame}"
            )
        parsed.append(ActionLabel(start_frame, end_frame, caption))

    parsed.sort(key=lambda item: (item.start_frame, item.end_frame))
    for previous, current in zip(parsed, parsed[1:]):
        if current.start_frame < previous.end_frame:
            raise RobodojoAdapterError(
                f"overlapping action intervals for {episode_id}: "
                f"{previous.start_frame} {previous.end_frame}, "
                f"{current.start_frame} {current.end_frame}"
            )
    return tuple(parsed)


def _episode_videos(episode_dir: Path, episode_id: str) -> tuple[tuple[str, str], ...]:
    videos: list[tuple[str, str]] = []
    missing: list[str] = []
    for view_name, file_name in VIDEO_FILES.items():
        path = episode_dir / file_name
        if not path.is_file():
            missing.append(file_name)
        else:
            videos.append((view_name, str(path.resolve())))
    if missing:
        raise RobodojoAdapterError(
            f"missing synchronized videos for {episode_id}: {', '.join(missing)}"
        )
    return tuple(videos)


def scan_robodojo(
    media_root: Path | str = DEFAULT_MEDIA_ROOT,
    label_root: Path | str = DEFAULT_LABEL_ROOT,
    official_split_path: Path | str = DEFAULT_OFFICIAL_SPLIT,
    *,
    tasks: Sequence[str] = ROBODOJO_TASKS,
    include_splits: Sequence[str] = ("train",),
    allow_holdouts: bool = False,
    expected_episodes_per_task: int | None = 100,
) -> RobodojoScanResult:
    """Scan planner labels and videos without touching robot action data.

    Training-safe behavior is the default: only ``train`` is included.  A
    caller must both request a holdout split and set ``allow_holdouts=True`` to
    read it, preventing accidental training-set leakage.
    """

    media_root = Path(media_root)
    label_root = Path(label_root)
    _assert_planner_tree(media_root)
    _assert_planner_tree(label_root)

    requested_splits = tuple(dict.fromkeys(include_splits))
    unknown_splits = sorted(set(requested_splits) - set(OFFICIAL_SPLITS))
    if not requested_splits or unknown_splits:
        raise RobodojoAdapterError(
            f"invalid include_splits={requested_splits}; unknown={unknown_splits}"
        )
    if not allow_holdouts and any(name != "train" for name in requested_splits):
        raise PermissionError(
            "holdout access requires allow_holdouts=True; do not use holdouts for training"
        )

    task_names = tuple(tasks)
    assignments = load_official_split_assignments(
        official_split_path,
        tasks=task_names,
        expected_episodes_per_task=expected_episodes_per_task,
    )
    assignments_by_task: dict[str, list[tuple[str, str]]] = {
        task_name: [] for task_name in task_names
    }
    for episode_id, split_name in assignments.items():
        task_name, trajectory_name = episode_id.rsplit("/", 2)[-2:]
        assignments_by_task[task_name].append((trajectory_name, split_name))

    episodes: list[RobodojoEpisode] = []
    issues: list[ScanIssue] = []
    excluded_by_split = 0
    excluded_by_quarantine = 0

    for task_name in task_names:
        media_instruction_file = media_root / task_name / "instruction.json"
        action_annotation_file = label_root / task_name / "instruction.json"
        media_annotations = _load_mapping(media_instruction_file)
        action_annotations = _load_mapping(action_annotation_file)

        task_assignments = sorted(
            assignments_by_task[task_name], key=lambda item: _trajectory_number(item[0])
        )
        for trajectory_name, split_name in task_assignments:
            episode_id = canonical_episode_id(task_name, trajectory_name)
            if episode_id in QUARANTINED_EPISODES:
                excluded_by_quarantine += 1
                continue
            if split_name not in requested_splits:
                excluded_by_split += 1
                continue

            try:
                media_record = media_annotations.get(trajectory_name)
                action_record = action_annotations.get(trajectory_name)
                if not isinstance(media_record, Mapping):
                    raise RobodojoAdapterError(
                        f"missing media instruction for {episode_id}"
                    )
                if not isinstance(action_record, Mapping):
                    raise RobodojoAdapterError(
                        f"missing action annotation for {episode_id}"
                    )
                try:
                    (
                        task_instruction,
                        task_instruction_source,
                        _,
                        _,
                    ) = select_record_instruction(
                        media_record,
                        source_prefix="media_instruction_json.episode",
                        source_path=media_instruction_file,
                    )
                except TaskInstructionError as exc:
                    raise RobodojoAdapterError(
                        f"missing_task_instruction for {episode_id}: {exc}"
                    ) from exc
                total_frames = media_record.get("total")
                if (
                    isinstance(total_frames, bool)
                    or not isinstance(total_frames, int)
                    or total_frames <= 0
                ):
                    raise RobodojoAdapterError(
                        f"invalid total frame count for {episode_id}: {total_frames!r}"
                    )
                actions = _parse_actions(
                    action_record.get("action_caption"),
                    total_frames=total_frames,
                    episode_id=episode_id,
                )
                videos = _episode_videos(
                    media_root / task_name / trajectory_name, episode_id
                )
            except RobodojoAdapterError as exc:
                issues.append(
                    ScanIssue(
                        canonical_episode_id=episode_id,
                        reason=(
                            "missing_task_instruction"
                            if "missing_task_instruction" in str(exc)
                            else "episode_validation_failed"
                        ),
                        detail=str(exc),
                    )
                )
                continue

            episodes.append(
                RobodojoEpisode(
                    canonical_episode_id=episode_id,
                    task_name=task_name,
                    trajectory_name=trajectory_name,
                    split=split_name,
                    task_instruction=task_instruction,
                    total_frames=total_frames,
                    videos=videos,
                    actions=actions,
                    media_instruction_file=str(media_instruction_file.resolve()),
                    action_annotation_file=str(action_annotation_file.resolve()),
                    task_instruction_source=task_instruction_source,
                )
            )

    return RobodojoScanResult(
        episodes=tuple(episodes),
        issues=tuple(issues),
        excluded_by_split=excluded_by_split,
        excluded_by_quarantine=excluded_by_quarantine,
    )


def _action_supervision(action: ActionLabel | None) -> dict[str, Any]:
    if action is None:
        return {
            "label_available": False,
            "caption": "",
            "start_frame": None,
            "end_frame": None,
        }
    return {
        "label_available": True,
        "caption": action.caption,
        "start_frame": action.start_frame,
        "end_frame": action.end_frame,
    }


def _segment_supervision() -> dict[str, Any]:
    return {"label_available": False, "caption": ""}


def _prediction(index: int, role: str, action: ActionLabel | None) -> dict[str, Any]:
    return {
        "index": index,
        "role": role,
        "action": _action_supervision(action),
        "segment": _segment_supervision(),
    }


def _base_record(
    episode: RobodojoEpisode, *, category: str, record_suffix: str
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": f"{episode.canonical_episode_id}::{record_suffix}",
        "source": SOURCE_NAME,
        "category": category,
        "canonical_episode_id": episode.canonical_episode_id,
        "task_name": episode.task_name,
        "trajectory_name": episode.trajectory_name,
        "split": episode.split,
        "task_instruction": episode.task_instruction,
        "total_frames": episode.total_frames,
        "videos": dict(episode.videos),
        "provenance": {
            "media_instruction_file": episode.media_instruction_file,
            "action_annotation_file": episode.action_annotation_file,
            "task_instruction_source": episode.task_instruction_source,
            "task_instruction_source_path": episode.media_instruction_file,
            "task_instruction_policy": "explicit_task_caption_or_instruction_only_v1",
        },
    }


def build_canonical_records(episode: RobodojoEpisode) -> tuple[dict[str, Any], ...]:
    """Create initial, two-step ongoing, and conservative end candidates.

    Only Action supervision is activated.  The end record is deliberately a
    candidate: the permitted inputs do not independently prove physical task
    completion, so its execution-decision label remains unavailable.
    """

    if not episode.actions:
        raise RobodojoAdapterError(
            f"episode has no actions: {episode.canonical_episode_id}"
        )

    records: list[dict[str, Any]] = []
    initial = _base_record(
        episode, category="initial_plan", record_suffix="initial_plan"
    )
    initial["supervision"] = {
        "initial_plan": [
            {
                "index": index,
                "action": _action_supervision(action),
                "segment": _segment_supervision(),
            }
            for index, action in enumerate(episode.actions, start=1)
        ]
    }
    records.append(initial)

    for offset, current in enumerate(episode.actions):
        following = (
            episode.actions[offset + 1]
            if offset + 1 < len(episode.actions)
            else None
        )
        ongoing = _base_record(
            episode,
            category="ongoing",
            record_suffix=f"ongoing_action_{offset + 1:03d}",
        )
        ongoing["anchor_interval"] = {
            "start_frame": current.start_frame,
            "end_frame": current.end_frame,
        }
        ongoing["supervision"] = {
            "predictions": [
                _prediction(1, "current", current),
                _prediction(2, "next", following),
            ],
            "execution_decision": {"label_available": False, "value": ""},
        }
        records.append(ongoing)

    final_action = episode.actions[-1]
    end = _base_record(episode, category="end", record_suffix="end_candidate")
    end["anchor_frame"] = min(
        episode.total_frames - 1, final_action.end_frame - 1
    )
    end["supervision"] = {
        "predictions": [
            _prediction(1, "current", final_action),
            _prediction(2, "next", None),
        ],
        "execution_decision": {"label_available": False, "value": ""},
        "end_outcome": {"label_available": False, "value": ""},
    }
    records.append(end)
    return tuple(records)


def iter_canonical_records(scan: RobodojoScanResult) -> Iterator[dict[str, Any]]:
    for episode in scan.episodes:
        yield from build_canonical_records(episode)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--official-split", type=Path, default=DEFAULT_OFFICIAL_SPLIT)
    parser.add_argument(
        "--include-split",
        action="append",
        choices=OFFICIAL_SPLITS,
        dest="include_splits",
        help="Defaults to train. Holdouts also require --allow-holdouts.",
    )
    parser.add_argument("--allow-holdouts", action="store_true")
    parser.add_argument(
        "--fail-on-issues",
        action="store_true",
        help="Return a non-zero status when any episode is rejected.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    scan = scan_robodojo(
        media_root=args.media_root,
        label_root=args.label_root,
        official_split_path=args.official_split,
        include_splits=tuple(args.include_splits or ("train",)),
        allow_holdouts=args.allow_holdouts,
    )
    summary = scan.summary()
    summary["canonical_records"] = sum(
        len(build_canonical_records(episode)) for episode in scan.episodes
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if args.fail_on_issues and scan.issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
