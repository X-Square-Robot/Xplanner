#!/usr/bin/env python3
"""Deterministically freeze 20 pure Benchmark3 action-progress episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


USER_ROOT = Path(os.environ.get("XPLANNER_DATA_ROOT", "/path/to/xplanner-data"))
DEFAULT_BUNDLE = Path(os.environ.get(
    "XPLANNER_BENCHMARK3_BUNDLE", "/path/to/benchmark3/bundle"
))
BENCHMARK3 = Path(os.environ.get(
    "XPLANNER_BENCHMARK3_MANIFEST", "/path/to/benchmark3/manifest.jsonl"
))
LABEL_ROOT = Path(os.environ.get(
    "XPLANNER_BENCHMARK3_LABEL_ROOT", "/path/to/benchmark3/labels"
))
SELECTION_SEED = "v53-progress-parity-v1:"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(uid: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}{uid}".encode("utf-8")).hexdigest()


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_intervals(
    annotation: Any,
    field: str,
    *,
    total_frames: int | None,
    required: bool,
) -> list[dict[str, Any]]:
    if not isinstance(annotation, dict):
        raise TypeError("annotation is not an object")
    raw_values = annotation.get(field)
    if not isinstance(raw_values, dict):
        raw_values = {}
    result: list[dict[str, Any]] = []
    for raw_range, raw_caption in raw_values.items():
        match = re.fullmatch(r"\s*(\d+)\s+(\d+)\s*", str(raw_range))
        caption = str(raw_caption or "").strip()
        if match is None or not caption:
            raise ValueError(f"invalid {field} entry: {raw_range!r}")
        start, end = map(int, match.groups())
        if not 0 <= start < end:
            raise ValueError(f"invalid {field} interval [{start}, {end})")
        if total_frames is not None and end > total_frames:
            raise ValueError(
                f"{field} interval [{start}, {end}) exceeds {total_frames}"
            )
        result.append({
            "start_frame": start,
            "end_frame": end,
            "caption": caption,
        })
    result.sort(
        key=lambda item: (
            item["start_frame"],
            item["end_frame"],
            item["caption"],
        )
    )
    if required and not result:
        raise ValueError(f"missing {field}")
    return result


def video_metadata(videos: dict[str, Path]) -> tuple[int, float]:
    import av

    metadata: list[tuple[int, float]] = []
    for role, path in videos.items():
        with av.open(str(path), mode="r") as container:
            streams = list(container.streams.video)
            if len(streams) != 1:
                raise ValueError(f"{role} has {len(streams)} video streams")
            stream = streams[0]
            if stream.average_rate is None:
                raise ValueError(f"{role} has no average_rate")
            frame_count = sum(1 for _ in container.decode(stream))
            if frame_count <= 1:
                raise ValueError(f"{role} decoded only {frame_count} frames")
            metadata.append((frame_count, float(stream.average_rate)))
    if len(set(metadata)) != 1:
        raise ValueError(f"video views are not synchronized: {metadata}")
    return metadata[0]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def light_candidate(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("source_group") != "x2_numeric":
        raise ValueError("source_group")
    if row.get("path_status") != "available":
        raise ValueError("path_status")
    if int(row.get("camera_count") or 0) != 3:
        raise ValueError("camera_count")
    uid = str(row["uid"])
    dataset, body, episode_key = uid.split("/", 2)
    episode_path = Path(str(row.get("existing_episode_path") or ""))
    if not episode_path.is_dir():
        raise FileNotFoundError(episode_path)
    videos = {
        "head": episode_path / "faceImg.mp4",
        "left_wrist": episode_path / "leftImg.mp4",
        "right_wrist": episode_path / "rightImg.mp4",
    }
    if any(not path.is_file() for path in videos.values()):
        raise FileNotFoundError("one or more camera files are missing")
    label_path = LABEL_ROOT / dataset / body / "instruction.json"
    task_path = episode_path / "instruction.json"
    if not label_path.is_file() or not task_path.is_file():
        raise FileNotFoundError("label or task instruction JSON is missing")
    label_map = load_json(label_path)
    annotation = label_map.get(episode_key) if isinstance(label_map, dict) else None
    actions = parse_intervals(
        annotation,
        "action_caption",
        total_frames=None,
        required=True,
    )
    if len(actions) < 4:
        raise ValueError("action_count_lt_4")
    segments = parse_intervals(
        annotation,
        "human_segment_caption",
        total_frames=None,
        required=False,
    )
    task_map = load_json(task_path)
    task_row = task_map.get(episode_key) if isinstance(task_map, dict) else None
    if not isinstance(task_row, dict):
        raise ValueError("task instruction row is missing")
    task_instruction = str(task_row.get("instruction") or "").strip()
    if not task_instruction:
        raise ValueError("task instruction is empty")
    duration = float(row.get("episode_duration_s") or 0.0)
    if duration <= 0:
        raise ValueError("duration is not positive")
    return {
        "uid": uid,
        "dataset": dataset,
        "body": body,
        "episode_key": episode_key,
        "episode_path": episode_path,
        "videos": videos,
        "label_path": label_path,
        "task_path": task_path,
        "task_row": task_row,
        "task_instruction": task_instruction,
        "actions_unbounded": actions,
        "segments_unbounded": segments,
        "duration_s": duration,
        "manifest_row": row,
    }


def duration_strata(
    candidates: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    ordered = sorted(candidates, key=lambda item: (item["duration_s"], item["uid"]))
    strata: list[list[dict[str, Any]]] = [[] for _ in range(4)]
    for rank, candidate in enumerate(ordered):
        stratum = min(3, (rank * 4) // len(ordered))
        candidate["duration_rank"] = rank
        candidate["duration_stratum"] = stratum + 1
        strata[stratum].append(candidate)
    if any(len(stratum) < 5 for stratum in strata):
        raise ValueError([len(stratum) for stratum in strata])
    return strata


def freeze_candidate(
    candidate: dict[str, Any],
    *,
    index: int,
    bundle: Path,
    benchmark_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    total_frames, fps = video_metadata(candidate["videos"])
    label_map = load_json(candidate["label_path"])
    annotation = label_map[candidate["episode_key"]]
    actions = parse_intervals(
        annotation,
        "action_caption",
        total_frames=total_frames,
        required=True,
    )
    segments = parse_intervals(
        annotation,
        "human_segment_caption",
        total_frames=total_frames,
        required=False,
    )
    if len(actions) < 4:
        raise ValueError("decoded candidate has fewer than four Actions")
    task_source = "media_task_instruction_json.episode.instruction"
    slug = re.sub(
        r"[^a-z0-9]+", "-", candidate["body"].lower()
    ).strip("-")[:34]
    name = f"ep{index:02d}_q{candidate['duration_stratum']}_{candidate['dataset']}_{slug}"
    spec_path = bundle / "episode_specs" / f"{name}.json"
    label_sha256 = sha256_file(candidate["label_path"])
    task_sha256 = sha256_file(candidate["task_path"])
    spec = {
        "schema_version": "v5_3_frozen_episode_spec_v1",
        "benchmark3_uid": candidate["uid"],
        "episode_key": candidate["uid"],
        "source": "baseline_v2v3umi",
        "source_group": "zhengwei",
        "task_instruction": candidate["task_instruction"],
        "total_frames": total_frames,
        "fps": fps,
        "profiles": ["action_only"],
        "videos": {
            role: str(path) for role, path in candidate["videos"].items()
        },
        "actions": actions,
        "segments": segments,
        "source_files": [
            {
                "role": "v2v3umi_label_annotation",
                "path": str(candidate["label_path"]),
                "sha256": label_sha256,
            },
            {
                "role": "episode_task_instruction",
                "path": str(candidate["task_path"]),
                "sha256": task_sha256,
            },
            {
                "role": "benchmark3_holdout_manifest",
                "path": str(BENCHMARK3),
                "sha256": benchmark_sha256,
            },
        ],
        "provenance": {
            "selection": "benchmark3_holdout",
            "selection_seed": SELECTION_SEED,
            "duration_stratum": candidate["duration_stratum"],
            "duration_rank": candidate["duration_rank"],
            "selection_hash": stable_rank(candidate["uid"]),
            "benchmark3_manifest": str(BENCHMARK3),
            "benchmark3_manifest_sha256": benchmark_sha256,
            "benchmark3_uid": candidate["uid"],
            "manifest_source_group": "x2_numeric",
            "label_annotation_path": str(candidate["label_path"]),
            "label_annotation_sha256": label_sha256,
            "task_instruction_source": task_source,
            "task_instruction_source_path": str(candidate["task_path"]),
            "decode_preflight": "full_decode_all_three_views_equal_frame_count_and_fps",
        },
    }
    write_json_new(spec_path, spec)
    batch = {
        "index": index,
        "name": name,
        "selection": "benchmark3_holdout",
        "spec": str(spec_path),
        "total_frames": total_frames,
        "duration_s": candidate["duration_s"],
        "duration_stratum": candidate["duration_stratum"],
    }
    audit = {
        **batch,
        "uid": candidate["uid"],
        "fps": fps,
        "action_count": len(actions),
        "segment_count": len(segments),
        "task_instruction": candidate["task_instruction"],
        "selection_hash": stable_rank(candidate["uid"]),
    }
    return batch, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    manifest_rows: list[dict[str, Any]] = []
    with BENCHMARK3.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    manifest_rows.append(value)
    light: list[dict[str, Any]] = []
    light_rejections: list[dict[str, str]] = []
    for row in manifest_rows:
        try:
            light.append(light_candidate(row))
        except Exception as exc:
            light_rejections.append({
                "uid": str(row.get("uid") or ""),
                "reason": f"{type(exc).__name__}: {exc}",
            })
    strata = duration_strata(light)
    benchmark_sha256 = sha256_file(BENCHMARK3)
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    decode_rejections: list[dict[str, Any]] = []
    for stratum_index, stratum in enumerate(strata, 1):
        ranked = sorted(stratum, key=lambda item: (stable_rank(item["uid"]), item["uid"]))
        accepted = 0
        for candidate_rank, candidate in enumerate(ranked, 1):
            if accepted == 5:
                break
            try:
                frozen = freeze_candidate(
                    candidate,
                    index=len(selected) + 1,
                    bundle=bundle,
                    benchmark_sha256=benchmark_sha256,
                )
            except Exception as exc:
                decode_rejections.append({
                    "uid": candidate["uid"],
                    "duration_stratum": stratum_index,
                    "candidate_rank_in_stratum": candidate_rank,
                    "selection_hash": stable_rank(candidate["uid"]),
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                continue
            selected.append(frozen)
            accepted += 1
        if accepted != 5:
            raise RuntimeError(
                f"duration stratum {stratum_index} yielded only {accepted} episodes"
            )
    batch_entries = [item[0] for item in selected]
    audit_entries = [item[1] for item in selected]
    if len(batch_entries) != 20 or len({row["uid"] for row in audit_entries}) != 20:
        raise RuntimeError("selection is not 20 unique episodes")
    write_json_new(bundle / "batch_spec.json", {
        "schema_version": "v5_3_progress_parity_batch_v1",
        "selection_policy": {
            "source": "Benchmark3",
            "source_group": "x2_numeric",
            "path_status": "available",
            "camera_count": 3,
            "profile": "action_only",
            "minimum_action_count": 4,
            "duration_strata": 4,
            "episodes_per_stratum": 5,
            "rank": f"sha256({SELECTION_SEED!r} + uid)",
            "decode_replacement": "next_same_stratum_rank_with_logged_rejection",
        },
        "episodes": batch_entries,
    })
    write_json_new(bundle / "episode_selection_audit.json", {
        "schema_version": "v5_3_progress_parity_selection_audit_v1",
        "benchmark3_manifest": str(BENCHMARK3),
        "benchmark3_manifest_sha256": benchmark_sha256,
        "manifest_rows": len(manifest_rows),
        "light_eligible_candidates": len(light),
        "light_rejection_count": len(light_rejections),
        "duration_stratum_sizes": [len(value) for value in strata],
        "decode_rejections": decode_rejections,
        "episode_count": len(audit_entries),
        "episodes": audit_entries,
    })
    print(json.dumps({
        "batch_spec": str(bundle / "batch_spec.json"),
        "selection_audit": str(bundle / "episode_selection_audit.json"),
        "episodes": len(batch_entries),
        "light_eligible_candidates": len(light),
        "decode_rejections": len(decode_rejections),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
