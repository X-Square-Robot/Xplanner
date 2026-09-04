"""Materialize Baseline Action/Segment buckets from a label-first V5.3 index."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

from .bucket_io import AtomicJsonlWriter, BucketWriter, file_sha256
from .materialize_episode import materialize_episode
from .holdout import EvaluationHoldout, DEFAULT_EVALUATION_MANIFEST, DEFAULT_EVALUATION_SHA256
from .video_probe import video_frame_count
from .task_instruction import TaskInstructionError, require_index_task_instruction


def infer_task_instruction(row: Mapping[str, Any]) -> tuple[str, str]:
    """Return only scanner-resolved source text; slug fallbacks are forbidden."""

    return require_index_task_instruction(row)


class FrameCountCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.rows: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        self.rows[str(row["path"])] = row
        self.dirty = False

    def get(self, path: str) -> int:
        stat = Path(path).stat()
        with self._lock:
            cached = self.rows.get(path)
        if (
            cached is not None
            and cached.get("size") == stat.st_size
            and cached.get("mtime_ns") == stat.st_mtime_ns
        ):
            return int(cached["frame_count"])
        count = video_frame_count(path)
        with self._lock:
            self.rows[path] = {
                "path": path,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "frame_count": count,
            }
            self.dirty = True
        return count

    def prefetch(
        self,
        paths: Sequence[str],
        *,
        executor: ThreadPoolExecutor,
    ) -> None:
        """Warm unique frame counts concurrently without changing row policy.

        A bad media file must remain a per-episode quarantine condition.  A
        speculative warm-up error is therefore swallowed here and is raised
        again by the normal synchronous ``get`` call for the affected row.
        """

        unique_paths = tuple(dict.fromkeys(str(path) for path in paths if path))

        def warm(path: str) -> None:
            try:
                self.get(path)
            except Exception:
                return

        tuple(executor.map(warm, unique_paths))

    def synchronized(self, paths: Sequence[str]) -> tuple[int, dict[str, int]]:
        counts = {path: self.get(path) for path in paths}
        return min(counts.values()), counts

    def save(self) -> None:
        if not self.dirty and self.path.is_file():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        os.fchmod(descriptor, 0o644)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                for path in sorted(self.rows):
                    handle.write(json.dumps(self.rows[path], separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        self.dirty = False


def _prefetch_paths(
    rows: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    require_profiles: Sequence[str],
    include_robodojo: bool,
) -> list[str]:
    """Return only media paths that can reach the expensive frame probe."""

    result: list[str] = []
    required = set(require_profiles)
    for _line_number, row in rows:
        source_group = str(row.get("source_group") or "")
        if source_group == "robodojo" and not include_robodojo:
            continue
        profiles = list(row.get("eligible_profiles") or ())
        if required:
            profiles = [profile for profile in profiles if profile in required]
        videos = [str(path) for path in (row.get("camera_videos") or ()) if path]
        if row.get("evaluation_holdout_excluded") or not profiles or not videos:
            continue
        try:
            infer_task_instruction(row)
        except TaskInstructionError:
            continue
        result.extend(videos)
    return result


def _iter_prefetched_rows(
    handle: Any,
    *,
    frame_cache: FrameCountCache,
    executor: ThreadPoolExecutor | None,
    batch_size: int,
    require_profiles: Sequence[str],
    include_robodojo: bool,
    line_start: int = 1,
    line_end: int | None = None,
) -> Iterator[tuple[int, Mapping[str, Any]]]:
    batch: list[tuple[int, Mapping[str, Any]]] = []
    for line_number, line in enumerate(handle, 1):
        if line_number < line_start:
            continue
        if line_end is not None and line_number > line_end:
            break
        if not line.strip():
            continue
        batch.append((line_number, json.loads(line)))
        if len(batch) < batch_size:
            continue
        if executor is not None:
            frame_cache.prefetch(
                _prefetch_paths(
                    batch,
                    require_profiles=require_profiles,
                    include_robodojo=include_robodojo,
                ),
                executor=executor,
            )
        yield from batch
        batch.clear()
    if batch:
        if executor is not None:
            frame_cache.prefetch(
                _prefetch_paths(
                    batch,
                    require_profiles=require_profiles,
                    include_robodojo=include_robodojo,
                ),
                executor=executor,
            )
        yield from batch


def materialize(
    *,
    instruction_index: Path,
    output_root: Path,
    holdout: EvaluationHoldout,
    max_episodes: int | None = None,
    include_robodojo: bool = False,
    require_profiles: Sequence[str] = (),
    frame_probe_workers: int = 1,
    materialize_batch_size: int = 256,
    progress_every_rows: int = 10_000,
    line_start: int = 1,
    line_end: int | None = None,
) -> dict[str, Any]:
    if frame_probe_workers <= 0:
        raise ValueError("frame_probe_workers must be positive")
    if materialize_batch_size <= 0:
        raise ValueError("materialize_batch_size must be positive")
    if progress_every_rows < 0:
        raise ValueError("progress_every_rows must be non-negative")
    if line_start <= 0:
        raise ValueError("line_start must be positive")
    if line_end is not None and line_end < line_start:
        raise ValueError("line_end must be greater than or equal to line_start")
    writer = BucketWriter(output_root, contract_examples_per_key=1)
    quarantine = AtomicJsonlWriter(output_root / "quarantine.jsonl")
    frame_cache = FrameCountCache(output_root / "media_frame_cache.jsonl")
    read_rows = 0
    accepted_episodes = 0
    emitted_samples = 0
    benchmark_excluded = 0
    reasons: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    missing_context_variants = 0
    started = time.monotonic()
    next_progress = progress_every_rows
    # Preserve exact max_episodes stopping semantics for bounded smoke calls.
    effective_batch_size = 1 if max_episodes is not None else materialize_batch_size
    probe_pool = (
        ThreadPoolExecutor(
            max_workers=frame_probe_workers,
            thread_name_prefix="v53-frame-probe",
        )
        if frame_probe_workers > 1
        else None
    )
    try:
        with instruction_index.open(encoding="utf-8") as handle:
            rows = _iter_prefetched_rows(
                handle,
                frame_cache=frame_cache,
                executor=probe_pool,
                batch_size=effective_batch_size,
                require_profiles=require_profiles,
                include_robodojo=include_robodojo,
                line_start=line_start,
                line_end=line_end,
            )
            for line_number, row in rows:
                read_rows += 1
                if progress_every_rows and read_rows >= next_progress:
                    elapsed = max(time.monotonic() - started, 1e-9)
                    print(json.dumps({
                        "event": "v53_index_materialize_progress",
                        "read_rows": read_rows,
                        "accepted_episodes": accepted_episodes,
                        "emitted_samples": emitted_samples,
                        "quarantine_records": quarantine.count,
                        "frame_cache_paths": len(frame_cache.rows),
                        "frame_probe_workers": frame_probe_workers,
                        "elapsed_seconds": round(elapsed, 3),
                        "rows_per_second": round(read_rows / elapsed, 3),
                    }, sort_keys=True), flush=True)
                    while next_progress <= read_rows:
                        next_progress += progress_every_rows
                source_group = str(row.get("source_group") or "")
                if source_group == "robodojo" and not include_robodojo:
                    continue
                if max_episodes is not None and accepted_episodes >= max_episodes:
                    break
                profiles = list(row.get("eligible_profiles") or ())
                if require_profiles:
                    profiles = [profile for profile in profiles if profile in require_profiles]
                    if not profiles:
                        continue
                videos = list(row.get("camera_videos") or ())
                instruction: str | None = None
                instruction_source: str | None = None
                instruction_error: str | None = None
                try:
                    instruction, instruction_source = infer_task_instruction(row)
                except TaskInstructionError as exc:
                    instruction_error = str(exc)
                if row.get("evaluation_holdout_excluded") or instruction_error or not profiles or not videos:
                    reason = (
                        "evaluation_holdout_overlap" if row.get("evaluation_holdout_excluded")
                        else "missing_task_instruction" if instruction_error
                        else "subtask_count_le_3" if not profiles
                        else "missing_media"
                    )
                    reasons[reason] += 1
                    benchmark_excluded += int(reason == "evaluation_holdout_overlap")
                    quarantine.write({
                        "source_index_line": line_number,
                        "episode_key": row.get("episode_key"),
                        "reason": reason,
                        "detail": instruction_error,
                        "task_instruction_source": row.get("task_instruction_source"),
                        "task_instruction_source_path": row.get("task_instruction_source_path"),
                        "evaluation_holdout_matches": row.get("evaluation_holdout_matches") or [],
                    })
                    continue
                pseudo = {
                    "images": videos,
                    "provenance": {
                        "episode_key": row.get("episode_key"),
                        "episode_path": row.get("resolved_episode_path"),
                        "raw_video_paths": {str(i): value for i, value in enumerate(videos)},
                    },
                }
                matches = holdout.match_sample(pseudo)
                if matches:
                    reasons["evaluation_holdout_overlap"] += 1
                    benchmark_excluded += 1
                    quarantine.write({
                        "source_index_line": line_number,
                        "episode_key": row.get("episode_key"),
                        "reason": "evaluation_holdout_overlap",
                        "matches": matches,
                    })
                    continue
                try:
                    total_frames, per_view_frames = frame_cache.synchronized(videos)
                    assert instruction is not None and instruction_source is not None
                    samples, missing = materialize_episode(
                        episode_key=str(row["episode_key"]),
                        source="baseline_v2v3umi",
                        source_group=source_group,
                        task_instruction=instruction,
                        actions=list(row.get("actions") or ()),
                        segments=list(row.get("segments") or ()),
                        videos=videos,
                        total_frames=total_frames,
                        profiles=profiles,
                        split="train",
                        provenance={
                            "instruction_index": str(instruction_index.resolve()),
                            "instruction_index_line": line_number,
                            "instruction_relative": row.get("instruction_relative"),
                            "resolved_episode_path": row.get("resolved_episode_path"),
                            "raw_video_paths": {str(i): value for i, value in enumerate(videos)},
                            "per_view_frame_counts": per_view_frames,
                            "total_frames_source": "mp4_video_sample_table_min_across_views",
                            "task_instruction_source": instruction_source,
                            "task_instruction_source_path": row.get("task_instruction_source_path"),
                            "task_instruction_source_field": row.get("task_instruction_source_field"),
                            "task_instruction_policy": "authoritative_source_only_v1",
                        },
                    )
                    if not samples:
                        raise ValueError("no samples remained after the <=3-unit policy")
                    for sample in samples:
                        writer.write(sample, source_record={
                            "instruction_relative": row.get("instruction_relative"),
                            "episode_key": row.get("episode_key"),
                        })
                    for missing_row in missing:
                        quarantine.write({
                            "source_index_line": line_number,
                            "episode_key": row.get("episode_key"),
                            "reason": "unavailable_noisy_context",
                            **missing_row,
                        })
                    missing_context_variants += len(missing)
                    accepted_episodes += 1
                    emitted_samples += len(samples)
                    source_counts[source_group] += 1
                except Exception as exc:
                    reason = type(exc).__name__
                    reasons[reason] += 1
                    quarantine.write({
                        "source_index_line": line_number,
                        "episode_key": row.get("episode_key"),
                        "reason": reason,
                        "detail": str(exc),
                    })
        frame_cache.save()
        provenance = {
            "source": "v2v3umi_instruction_index",
            "instruction_index": str(instruction_index.resolve()),
            "instruction_index_sha256": file_sha256(instruction_index),
            "partial": (
                max_episodes is not None or line_start != 1 or line_end is not None
            ),
            "max_episodes": max_episodes,
            "line_start": line_start,
            "line_end": line_end,
            "include_robodojo": include_robodojo,
            "require_profiles": list(require_profiles),
            "read_rows": read_rows,
            "accepted_episodes": accepted_episodes,
            "emitted_samples": emitted_samples,
            "source_counts": dict(sorted(source_counts.items())),
            "evaluation_holdout_excluded": benchmark_excluded,
            "missing_context_variants": missing_context_variants,
            "missing_task_instruction": reasons.get("missing_task_instruction", 0),
            "quarantine_reasons": dict(sorted(reasons.items())),
            "evaluation_holdout": holdout.metadata(),
            "media_scan_policy": "index_paths_only_no_recursive_media_walk",
            "end_policy": "exact_min_video_sample_count_minus_one",
            "frame_probe_workers": frame_probe_workers,
            "materialize_batch_size": effective_batch_size,
            "progress_every_rows": progress_every_rows,
        }
        manifest = writer.close(provenance=provenance)
        quarantine.close()
        return {**manifest, "index_materialization": provenance, "quarantine_records": quarantine.count}
    except BaseException:
        writer.abort()
        quarantine.close(publish=False)
        raise
    finally:
        if probe_pool is not None:
            probe_pool.shutdown(wait=True, cancel_futures=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruction-index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--include-robodojo", action="store_true")
    parser.add_argument("--frame-probe-workers", type=int, default=1)
    parser.add_argument("--materialize-batch-size", type=int, default=256)
    parser.add_argument("--progress-every-rows", type=int, default=10_000)
    parser.add_argument("--line-start", type=int, default=1)
    parser.add_argument("--line-end", type=int)
    parser.add_argument(
        "--require-profile",
        action="append",
        choices=("action_only", "segment_only", "action_segment_joint"),
        default=[],
    )
    parser.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_EVALUATION_MANIFEST)
    parser.add_argument("--evaluation-sha256", default=DEFAULT_EVALUATION_SHA256)
    args = parser.parse_args(argv)
    holdout = EvaluationHoldout.load(args.evaluation_holdout, expected_sha256=args.evaluation_holdout_sha256)
    report = materialize(
        instruction_index=args.instruction_index,
        output_root=args.output_root,
        holdout=holdout,
        max_episodes=args.max_episodes,
        include_robodojo=args.include_robodojo,
        require_profiles=tuple(args.require_profile),
        frame_probe_workers=args.frame_probe_workers,
        materialize_batch_size=args.materialize_batch_size,
        progress_every_rows=args.progress_every_rows,
        line_start=args.line_start,
        line_end=args.line_end,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FrameCountCache", "infer_task_instruction", "materialize"]
