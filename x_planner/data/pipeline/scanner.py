"""Incremental supervised multiprocess scanner."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import queue
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .adapters import AnnotationValidationError, EpisodeJob, adapt_job, discover_jobs
from .constants import (
    DEFAULT_MIN_INTERVAL_FRAMES,
    DEFAULT_MIN_L0_COUNT,
    DEFAULT_MIN_L1_COUNT,
)
from .hierarchy import EpisodeValidationError, build_samples, canonicalize_episode
from .snapshot import (
    canonical_json,
    publish_catalog_snapshot,
    publish_training_snapshot,
    write_episode_shard,
)
from .state import ScanLock, ScanState
from .video import VideoValidationError, validate_views


def split_for(episode_key: str, *, seed: int, validation_ratio: float) -> str:
    digest = hashlib.sha256(f"{seed}\0{episode_key}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "validation" if value < validation_ratio else "train"


def scan_episode(job: EpisodeJob, settings: Mapping[str, Any]) -> dict[str, Any]:
    try:
        adapted = adapt_job(job)
        videos, dropped_views, probes = validate_views(
            adapted.video_candidates,
            adapted.num_frames,
            max_camera_views=int(settings["max_camera_views"]),
        )
        split = split_for(
            job.episode_key,
            seed=int(settings["seed"]),
            validation_ratio=float(settings["validation_ratio"]),
        )
        metadata = dict(adapted.metadata)
        metadata.update({"dropped_views": dropped_views, "video_probes": probes})
        episode = canonicalize_episode(
            source=job.source,
            episode_key=job.episode_key,
            episode_name=job.episode_name,
            split=split,
            num_frames=adapted.num_frames,
            task_caption=adapted.task_caption,
            raw_levels=adapted.raw_levels,
            videos=videos,
            annotation_sources=adapted.annotation_sources,
            metadata=metadata,
            min_interval_frames=int(settings["min_interval_frames"]),
            min_l1_count=int(settings["min_l1_count"]),
            min_l0_count=int(settings["min_l0_count"]),
        )
        samples = build_samples(episode)
        sampled_frames = {image.frame for sample in samples for image in sample.images}
        if max(sampled_frames) >= episode.num_frames:
            raise EpisodeValidationError("sample_frame_out_of_bounds")
        return {
            "status": "accepted",
            "job": job.to_dict(),
            "episode": episode.to_dict(),
            "samples": [sample.to_dict() for sample in samples],
        }
    except EpisodeValidationError as exc:
        reason, detail = exc.reason, exc.detail
    except VideoValidationError as exc:
        reason, detail = "video_validation_failed", str(exc)
    except FileNotFoundError as exc:
        reason, detail = "missing_file", str(exc)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        reason, detail = "annotation_decode_failed", f"{type(exc).__name__}: {exc}"
    except AnnotationValidationError as exc:
        reason, detail = "invalid_annotation", str(exc)
    except Exception as exc:
        reason = "worker_exception"
        detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}"
    return {
        "status": "rejected",
        "job": job.to_dict(),
        "reason": reason,
        "detail": detail[:4000],
    }


def _worker_loop(job_queue: Any, result_queue: Any, settings: dict[str, Any]) -> None:
    pid = os.getpid()
    while True:
        payload = job_queue.get()
        if payload is None:
            return
        job = EpisodeJob.from_dict(payload)
        result_queue.put(("started", pid, job.episode_key, time.monotonic()))
        result = scan_episode(job, settings)
        result_queue.put(("result", pid, job.episode_key, result))


def _start_worker(context: Any, job_queue: Any, result_queue: Any, settings: dict[str, Any]):
    process = context.Process(target=_worker_loop, args=(job_queue, result_queue, settings))
    process.daemon = False
    process.start()
    return process


def scan_fingerprint(config: Mapping[str, Any], settings: Mapping[str, Any]) -> str:
    semantic = {
        "config": config,
        "settings": {
            key: settings[key]
            for key in (
                "seed", "validation_ratio", "min_interval_frames", "min_l1_count",
                "min_l0_count", "max_camera_views", "rule_version",
            )
        },
    }
    return hashlib.sha256(canonical_json(semantic).encode()).hexdigest()


def _pending_jobs(
    config: Mapping[str, Any], state: ScanState, max_episodes: int
) -> Iterator[EpisodeJob]:
    for job in discover_jobs(config, max_episodes=max_episodes):
        if not state.is_terminal(job.episode_key):
            yield job


def run_scan(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    settings: dict[str, Any],
    workers: int,
    episode_timeout: float,
    max_episodes: int = 0,
    publish_every: int = 100,
    publish_seconds: float = 60.0,
    auto_training_snapshot: bool = True,
    final_on_complete: bool = True,
) -> Path:
    fingerprint = scan_fingerprint(config, settings)
    run_root = output_root.resolve() / f"scan-{fingerprint[:12]}"
    run_root.mkdir(parents=True, exist_ok=True)
    state = ScanState(
        run_root / "scan_state.sqlite3",
        fingerprint=fingerprint,
        config_json=canonical_json(config),
    )
    try:
        with ScanLock(run_root / ".scan.lock"):
            context = mp.get_context("spawn")
            job_queue = context.Queue(maxsize=max(2, workers * 2))
            result_queue = context.Queue()
            processes = {
                process.pid: process
                for process in (
                    _start_worker(context, job_queue, result_queue, settings)
                    for _ in range(max(1, workers))
                )
            }
            iterator = iter(_pending_jobs(config, state, max_episodes))
            buffered: EpisodeJob | None = None
            exhausted = False
            outstanding: dict[str, EpisodeJob] = {}
            active: dict[int, tuple[str, float]] = {}
            completed_since_publish = 0
            last_publish = time.monotonic()
            early_snapshot_done = any((run_root / "training_snapshots").glob("early-*")) if (run_root / "training_snapshots").exists() else False

            def fill_queue() -> None:
                nonlocal buffered, exhausted
                while not exhausted and len(outstanding) < max(2, workers * 3):
                    if buffered is None:
                        try:
                            buffered = next(iterator)
                        except StopIteration:
                            exhausted = True
                            return
                    try:
                        job_queue.put_nowait(buffered.to_dict())
                    except queue.Full:
                        return
                    outstanding[buffered.episode_key] = buffered
                    buffered = None

            fill_queue()
            try:
                while not exhausted or outstanding:
                    fill_queue()
                    try:
                        message = result_queue.get(timeout=1.0)
                    except queue.Empty:
                        message = None
                    if message is not None:
                        kind, pid, episode_key, payload = message
                        if kind == "started":
                            active[int(pid)] = (episode_key, float(payload))
                        else:
                            active.pop(int(pid), None)
                            job = outstanding.pop(episode_key, None)
                            if job is None:
                                continue
                            result = payload
                            if result["status"] == "accepted":
                                from .models import CanonicalEpisode, V10Sample

                                episode = CanonicalEpisode.from_dict(result["episode"])
                                samples = tuple(V10Sample.from_dict(item) for item in result["samples"])
                                shard = write_episode_shard(run_root, episode, samples)
                                state.record_accepted(
                                    episode_key=episode.episode_key,
                                    source=episode.source,
                                    topic=job.topic,
                                    shard_path=str(shard),
                                    profile=episode.profile,
                                    split=episode.split,
                                    num_samples=len(samples),
                                    views=list(episode.videos),
                                )
                            else:
                                state.record_rejected(
                                    episode_key=job.episode_key,
                                    source=job.source,
                                    topic=job.topic,
                                    reason=result["reason"],
                                    detail=result["detail"],
                                )
                            completed_since_publish += 1

                    now = time.monotonic()
                    for pid, process in list(processes.items()):
                        current = active.get(pid)
                        timed_out = current is not None and now - current[1] > episode_timeout
                        crashed = not process.is_alive()
                        if not timed_out and not crashed:
                            continue
                        if process.is_alive():
                            process.terminate()
                        process.join(timeout=5)
                        processes.pop(pid, None)
                        active.pop(pid, None)
                        if current is not None:
                            episode_key = current[0]
                            job = outstanding.pop(episode_key, None)
                            if job is not None:
                                state.record_rejected(
                                    episode_key=job.episode_key,
                                    source=job.source,
                                    topic=job.topic,
                                    reason="worker_timeout" if timed_out else "worker_crash",
                                    detail=f"pid={pid} timeout={episode_timeout}",
                                )
                                completed_since_publish += 1
                        replacement = _start_worker(context, job_queue, result_queue, settings)
                        processes[replacement.pid] = replacement

                    if (
                        completed_since_publish >= publish_every
                        or now - last_publish >= publish_seconds
                    ) and state.stats()["terminal"]:
                        publish_catalog_snapshot(state, run_root)
                        completed_since_publish = 0
                        last_publish = now

                    if auto_training_snapshot and not early_snapshot_done:
                        stats = state.stats()
                        if (
                            stats["splits"].get("train", 0) >= 100
                            and stats["splits"].get("validation", 0) >= 10
                            and len(stats["profiles"]) >= 2
                        ):
                            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                            publish_training_snapshot(
                                state, run_root, version=f"early-{stamp}", complete=False
                            )
                            early_snapshot_done = True
            finally:
                for _pid in processes:
                    try:
                        job_queue.put_nowait(None)
                    except queue.Full:
                        break
                for process in processes.values():
                    process.join(timeout=5)
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=5)

            publish_catalog_snapshot(state, run_root)
            if final_on_complete and not max_episodes:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                publish_training_snapshot(
                    state, run_root, version=f"final-{stamp}", complete=True
                )
            return run_root
    finally:
        state.close()
