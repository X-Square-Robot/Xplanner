"""Materialize RoboDojo V5.3 rows into one independently weighted file."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

from .bucket_io import AtomicJsonlWriter, BucketWriter, atomic_json
from .materialize_episode import materialize_episode
from .holdout import EvaluationHoldout, DEFAULT_EVALUATION_MANIFEST, DEFAULT_EVALUATION_SHA256
from .robodojo_adapter import (
    DEFAULT_LABEL_ROOT,
    DEFAULT_MEDIA_LAYOUT,
    DEFAULT_MEDIA_ROOT,
    DEFAULT_OFFICIAL_SPLIT,
    MEDIA_LAYOUTS,
    scan_robodojo,
)


def materialize(
    *,
    output_root: Path,
    media_root: Path,
    label_root: Path,
    official_split: Path,
    holdout: EvaluationHoldout,
    max_episodes: int | None = None,
    media_layout: str = DEFAULT_MEDIA_LAYOUT,
) -> dict[str, Any]:
    scan = scan_robodojo(
        media_root=media_root,
        label_root=label_root,
        official_split_path=official_split,
        include_splits=("train",),
        media_layout=media_layout,
    )
    episodes = scan.episodes[:max_episodes] if max_episodes is not None else scan.episodes
    writer = BucketWriter(output_root, contract_examples_per_key=1)
    quarantine = AtomicJsonlWriter(output_root / "buckets" / "robodojo" / "quarantine.jsonl")
    reasons: Counter[str] = Counter()
    benchmark_excluded = 0
    accepted_episodes = 0
    missing_context_variants = 0
    try:
        for episode in episodes:
            videos = dict(episode.videos)
            pseudo = {
                "images": list(videos.values()),
                "provenance": {
                    "episode_key": episode.canonical_episode_id,
                    "episode_path": str(Path(next(iter(videos.values()))).parent),
                    "raw_video_paths": videos,
                },
            }
            matches = holdout.match_sample(pseudo)
            if matches:
                benchmark_excluded += 1
                quarantine.write({
                    "episode_key": episode.canonical_episode_id,
                    "reason": "evaluation_holdout_overlap",
                    "matches": matches,
                })
                continue
            try:
                actions = [
                    {
                        "start_frame": action.start_frame,
                        "end_frame": action.end_frame,
                        "caption": action.caption,
                    }
                    for action in episode.actions
                ]
                samples, missing = materialize_episode(
                    episode_key=episode.canonical_episode_id,
                    source="robodojo",
                    source_group="robodojo",
                    task_instruction=episode.task_instruction,
                    actions=actions,
                    segments=(),
                    videos=videos,
                    total_frames=episode.total_frames,
                    profiles=("action_only",),
                    split="train",
                    provenance={
                        "task_name": episode.task_name,
                        "trajectory_name": episode.trajectory_name,
                        "official_split": episode.split,
                        "media_instruction_file": episode.media_instruction_file,
                        "action_annotation_file": episode.action_annotation_file,
                        "task_instruction_source": episode.task_instruction_source,
                        "task_instruction_source_path": episode.media_instruction_file,
                        "task_instruction_policy": "explicit_task_caption_or_instruction_only_v1",
                        "total_frames_source": episode.total_frames_source,
                        "media_layout": episode.media_layout,
                    },
                )
                if not samples:
                    reasons["subtask_count_le_3"] += 1
                    quarantine.write({
                        "episode_key": episode.canonical_episode_id,
                        "reason": "subtask_count_le_3",
                        "action_count": len(actions),
                    })
                    continue
                for sample in samples:
                    writer.write(sample)
                for row in missing:
                    quarantine.write({
                        "episode_key": episode.canonical_episode_id,
                        "reason": "unavailable_noisy_context",
                        **row,
                    })
                missing_context_variants += len(missing)
                accepted_episodes += 1
            except Exception as exc:
                reason = type(exc).__name__
                reasons[reason] += 1
                quarantine.write({
                    "episode_key": episode.canonical_episode_id,
                    "reason": reason,
                    "detail": str(exc),
                })
        provenance = {
            "source": "robotwin30_x2/arx_x5",
            "media_root": str(media_root.resolve()),
            "label_root": str(label_root.resolve()),
            "media_layout": media_layout,
            "official_split": str(official_split.resolve()),
            "partial": max_episodes is not None,
            "max_episodes": max_episodes,
            "scan": scan.summary(),
            "eligible_episodes_seen": len(episodes),
            "accepted_episodes": accepted_episodes,
            "evaluation_holdout_excluded": benchmark_excluded,
            "missing_context_variants": missing_context_variants,
            "quarantine_reasons": dict(sorted(reasons.items())),
            "evaluation_holdout": holdout.metadata(),
            "single_training_file": "buckets/robodojo/train.jsonl",
            "recommended_exposure_fraction": 0.20,
        }
        root_manifest = writer.close(provenance=provenance)
        quarantine.close()
        bucket_manifest_path = output_root / "buckets" / "robodojo" / "manifest.json"
        bucket_manifest = json.loads(bucket_manifest_path.read_text(encoding="utf-8"))
        bucket_manifest["quarantine_records"] = quarantine.count
        bucket_manifest["recommended_exposure_fraction"] = 0.20
        bucket_manifest["single_training_file"] = True
        atomic_json(bucket_manifest_path, bucket_manifest)
        return {**root_manifest, "robodojo": provenance, "quarantine_records": quarantine.count}
    except BaseException:
        writer.abort()
        quarantine.close(publish=False)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--official-split", type=Path, default=DEFAULT_OFFICIAL_SPLIT)
    parser.add_argument(
        "--media-layout",
        choices=sorted(MEDIA_LAYOUTS),
        default=DEFAULT_MEDIA_LAYOUT,
        help="Media tree layout: view file names plus frame-count source.",
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_EVALUATION_MANIFEST)
    parser.add_argument("--evaluation-sha256", default=DEFAULT_EVALUATION_SHA256)
    args = parser.parse_args(argv)
    holdout = EvaluationHoldout.load(
        args.evaluation_manifest, expected_sha256=args.evaluation_sha256
    )
    report = materialize(
        output_root=args.output_root,
        media_root=args.media_root,
        label_root=args.label_root,
        official_split=args.official_split,
        holdout=holdout,
        max_episodes=args.max_episodes,
        media_layout=args.media_layout,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["materialize"]
