"""Deterministic shard-parallel V2 scanner with cumulative retry attempts."""

from __future__ import annotations

import json
import fcntl
import os
import signal
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from .artifact_ledger import register_artifacts
from .common.atomic import BatchedJsonlWriter, mark_done, read_json, write_json
from .common.hashing import config_hash, sampling_config_hash
from .shard_state import (
    Inventory,
    ShardPlan,
    completed_attempts,
    iter_plan,
    load_attempt_records,
    next_attempt,
    version_root,
)
from .validate_episode import process_episode


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with BatchedJsonlWriter(str(path)) as writer:
        for row in rows:
            writer.write(row)


def _sample_keys_by_episode(
    samples: Mapping[str, dict[str, Any]],
) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    for sample_key, row in samples.items():
        index[str(row.get("global_episode_key") or "")].add(sample_key)
    return dict(index)


def _replace_episode_samples(
    samples: dict[str, dict[str, Any]],
    index: dict[str, set[str]],
    episode_key: str,
    rows: list[dict[str, Any]],
) -> None:
    for sample_key in index.pop(episode_key, ()):
        samples.pop(sample_key, None)
    replacement: set[str] = set()
    for row in rows:
        sample_key = str(row["sample_key"])
        samples[sample_key] = row
        replacement.add(sample_key)
    if replacement:
        index[episode_key] = replacement


def _statistics(
    episodes: Mapping[str, dict[str, Any]],
    samples: Mapping[str, dict[str, Any]],
    *,
    shard_id: int,
    plan_hash: str,
) -> dict[str, Any]:
    status = Counter(row["status"] for row in episodes.values())
    by_source: dict[str, Counter] = {}
    errors = Counter()
    retryable = 0
    for row in episodes.values():
        source_counter = by_source.setdefault(str(row["source_id"]), Counter())
        source_counter[row["status"]] += 1
        source_counter["samples"] += int(row.get("sample_count", 0))
        if row["status"] == "failed":
            error_type = str(row.get("error", {}).get("error_type", "unknown_error"))
            errors[error_type] += 1
            retryable += int(bool(row.get("retryable")))
    return {
        "shard_id": shard_id,
        "plan_hash": plan_hash,
        "episodes": len(episodes),
        "status": dict(status),
        "samples": len(samples),
        "by_source": {key: dict(value) for key, value in sorted(by_source.items())},
        "by_error_type": dict(errors.most_common()),
        "retryable_failures": retryable,
    }


def _compatible_attempts(root: Path, completion_hash: str) -> list[Path]:
    return [
        attempt for attempt in completed_attempts(root)
        if (read_json(str(attempt / ".done"), {}) or {}).get("completion_hash") == completion_hash
    ]


def _latest_validation_failure(
    run_root: Path, plan: ShardPlan, completion_hash: str
) -> dict[str, dict[str, Any]]:
    attempts = _compatible_attempts(
        version_root(run_root, plan, stage="validate"), completion_hash
    )
    if not attempts:
        return {}
    episodes, _samples = load_attempt_records(attempts[-1])
    return {
        key: value for key, value in episodes.items()
        if value["status"] == "failed" and not bool(value.get("retryable"))
    }


def _latest_validation_success(
    run_root: Path, plan: ShardPlan, completion_hash: str
) -> dict[str, dict[str, Any]]:
    attempts = _compatible_attempts(
        version_root(run_root, plan, stage="validate"), completion_hash
    )
    if not attempts:
        return {}
    episodes, _samples = load_attempt_records(attempts[-1])
    return {
        key: value for key, value in episodes.items()
        if value["status"] == "success" and isinstance(value.get("canonical_episode"), dict)
    }


def _load_journal(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    episodes: dict[str, dict[str, Any]] = {}
    samples: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return episodes, samples
    sample_index = _sample_keys_by_episode(samples)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                break
            result = dict(event["result"])
            key = str(result["global_episode_key"])
            result_samples = (
                list(result.pop("samples", []))
                if result["status"] == "success" else []
            )
            _replace_episode_samples(samples, sample_index, key, result_samples)
            episodes[key] = result
    return episodes, samples


def _archive_incomplete_attempts(root: Path) -> None:
    """Preserve, but remove from the active namespace, non-durable attempts."""
    incomplete = [
        path for path in root.glob("attempt-*")
        if path.is_dir() and not (path / ".done").is_file()
    ]
    if not incomplete:
        return
    archive = root / "aborted_attempts"
    archive.mkdir(exist_ok=True)
    for path in incomplete:
        target = archive / f"{path.name}-pid{os.getpid()}-{time.time_ns()}"
        os.replace(path, target)


def _run_shard(payload: dict[str, Any]) -> dict[str, Any]:
    """Serialize a shard across duplicate CLI invocations and worker pools."""
    plan = ShardPlan(**payload["plan"])
    run_root = Path(payload["run_root"])
    root = version_root(run_root, plan, stage=str(payload["stage"]))
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".worker.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        _archive_incomplete_attempts(root)
        return _run_shard_locked(payload)


def _run_shard_locked(payload: dict[str, Any]) -> dict[str, Any]:
    plan = ShardPlan(**payload["plan"])
    run_root = Path(payload["run_root"])
    stage = str(payload["stage"])
    mode = str(payload["mode"])
    quick = stage == "validate"
    root = version_root(run_root, plan, stage=stage)
    completion_hash = str(payload["completion_hash"])
    previous_attempts = _compatible_attempts(root, completion_hash)
    if previous_attempts and mode != "resume":
        episodes, samples = load_attempt_records(previous_attempts[-1])
        return {
            "shard_id": plan.shard_id,
            "skipped": True,
            "attempt": str(previous_attempts[-1]),
            "statistics": _statistics(episodes, samples, shard_id=plan.shard_id, plan_hash=plan.plan_hash),
        }

    baseline_episodes: dict[str, dict[str, Any]] = {}
    baseline_samples: dict[str, dict[str, Any]] = {}
    if previous_attempts:
        baseline_episodes, baseline_samples = load_attempt_records(previous_attempts[-1])
    journal = root / f"journal-{completion_hash}.jsonl"
    journal_episodes, journal_samples = _load_journal(journal)
    baseline_sample_index = _sample_keys_by_episode(baseline_samples)
    for key in journal_episodes:
        _replace_episode_samples(baseline_samples, baseline_sample_index, key, [])
    baseline_episodes.update(journal_episodes)
    baseline_samples.update(journal_samples)
    jobs = {item.global_episode_key: item for item in iter_plan(plan)}
    max_attempts = int(payload["settings"].get("max_attempts", 3))
    exhausted_normalized = False
    for key, row in list(baseline_episodes.items()):
        if (
            row.get("status") == "failed"
            and bool(row.get("retryable"))
            and int(row.get("attempts", 1)) >= max_attempts
        ):
            # `retryable` means that another resume invocation has useful work
            # left to do. Once the configured budget is exhausted the row is
            # terminal, while the nested error retains the original classifier
            # for provenance and diagnosis.
            terminal = dict(row)
            terminal["retryable"] = False
            terminal["retry_exhausted"] = True
            baseline_episodes[key] = terminal
            exhausted_normalized = True
    if mode == "resume" and previous_attempts:
        work_keys = {
            key for key, row in baseline_episodes.items()
            if row["status"] == "failed"
            and bool(row.get("retryable"))
            and int(row.get("attempts", 1)) < max_attempts
        }
    else:
        work_keys = set(jobs) - set(baseline_episodes)

    validation_failures = (
        {} if quick else _latest_validation_failure(run_root, plan, completion_hash)
    )
    validation_success = (
        {} if quick else _latest_validation_success(run_root, plan, completion_hash)
    )
    if not work_keys and previous_attempts and not exhausted_normalized:
        return {
            "shard_id": plan.shard_id,
            "skipped": True,
            "attempt": str(previous_attempts[-1]),
            "statistics": _statistics(
                baseline_episodes, baseline_samples,
                shard_id=plan.shard_id, plan_hash=plan.plan_hash,
            ),
        }

    episodes = dict(baseline_episodes)
    samples = dict(baseline_samples)
    sample_index = _sample_keys_by_episode(samples)
    worker_id = f"pid-{os.getpid()}"
    sampling_hash = str(payload["sampling_hash"])
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_episodes = max(1, int(payload["settings"].get("checkpoint_episodes", 25)))
    journal_events = 0
    with journal.open("a", encoding="utf-8") as journal_handle:
        for key in sorted(work_keys):
            item = jobs[key]
            previous_count = int(episodes.get(key, {}).get("attempts", 0))
            while True:
                if key in validation_failures:
                    result = dict(validation_failures[key])
                    result["run_id"] = payload["run_id"]
                else:
                    timeout = float(payload["settings"].get("episode_timeout_seconds", 300))
                    previous_handler = signal.getsignal(signal.SIGALRM)
                    signal.signal(
                        signal.SIGALRM,
                        lambda _signum, _frame: (_ for _ in ()).throw(
                            TimeoutError(f"Episode exceeded {timeout} seconds")
                        ),
                    )
                    signal.setitimer(signal.ITIMER_REAL, timeout)
                    try:
                        result = process_episode(
                            item,
                            run_id=payload["run_id"],
                            shard_id=plan.shard_id,
                            worker_id=worker_id,
                            settings=payload["settings"],
                            view_config=payload["view_config"],
                            sampling_hash=sampling_hash,
                            quick=quick,
                            cached_episode=(
                                validation_success.get(key, {}).get("canonical_episode")
                                if key in validation_success else None
                            ),
                        )
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        signal.signal(signal.SIGALRM, previous_handler)
                result["attempts"] = previous_count + 1
                if (
                    result.get("status") == "failed"
                    and bool(result.get("retryable"))
                    and int(result["attempts"]) >= max_attempts
                ):
                    result["retryable"] = False
                    result["retry_exhausted"] = True
                journal_result = dict(result)
                journal_handle.write(json.dumps(
                    {"global_episode_key": key, "result": journal_result},
                    ensure_ascii=False, separators=(",", ":"),
                ) + "\n")
                journal_events += 1
                if journal_events % checkpoint_episodes == 0:
                    journal_handle.flush()
                    os.fsync(journal_handle.fileno())
                result_samples = (
                    list(result.pop("samples", []))
                    if result["status"] == "success" else []
                )
                _replace_episode_samples(samples, sample_index, key, result_samples)
                episodes[key] = result
                should_retry = (
                    mode == "resume"
                    and result["status"] == "failed"
                    and bool(result.get("retryable"))
                    and int(result["attempts"]) < max_attempts
                )
                if not should_retry:
                    break
                previous_count = int(result["attempts"])
        journal_handle.flush()
        os.fsync(journal_handle.fileno())

    attempt = next_attempt(root)
    attempt.mkdir(parents=True, exist_ok=False)
    success_rows = sorted(
        (row for row in episodes.values() if row["status"] == "success"),
        key=lambda row: row["global_episode_key"],
    )
    failure_rows = sorted(
        (row for row in episodes.values() if row["status"] == "failed"),
        key=lambda row: row["global_episode_key"],
    )
    if not quick:
        _write_jsonl(attempt / "catalog.jsonl", [samples[key] for key in sorted(samples)])
    _write_jsonl(attempt / "episodes_success.jsonl", success_rows)
    _write_jsonl(attempt / "episodes_failed.jsonl", failure_rows)
    _write_jsonl(
        attempt / "errors.jsonl",
        [row["error"] for row in failure_rows if isinstance(row.get("error"), dict)],
    )
    stats = _statistics(episodes, samples, shard_id=plan.shard_id, plan_hash=plan.plan_hash)
    write_json(str(attempt / "statistics.json"), stats)
    mark_done(attempt, {
        "run_id": payload["run_id"],
        "stage": stage,
        "shard_id": plan.shard_id,
        "plan_hash": plan.plan_hash,
        "sampling_config_hash": sampling_hash,
        "completion_hash": completion_hash,
        "episode_count": len(episodes),
        "retryable_remaining": sum(
            1 for row in failure_rows
            if row.get("retryable") and int(row.get("attempts", 1)) < max_attempts
        ),
    })
    write_json(str(root / "current.json"), {"attempt": str(attempt), "statistics": stats})
    # The completed attempt is now the durable source of truth.  Keeping the
    # recovery journal would roughly double catalog storage for successful runs.
    try:
        journal.unlink()
    except FileNotFoundError:
        pass
    return {
        "shard_id": plan.shard_id,
        "skipped": False,
        "attempt": str(attempt),
        "statistics": stats,
    }


def run_inventory(
    inventory: Inventory,
    run_root: Path,
    *,
    run_id: str,
    stage: str,
    mode: str,
    settings: Mapping[str, Any],
    view_config: Mapping[str, Any],
    num_workers: int,
    shard_id: int | None,
    ledger_path: str | None,
    fail_fast: bool,
) -> dict[str, Any]:
    if stage not in {"validate", "scan"}:
        raise ValueError(f"invalid stage: {stage}")
    if mode not in {"scan", "resume"}:
        raise ValueError(f"invalid mode: {mode}")
    selected = [
        plan for plan in inventory.shards
        if shard_id is None or plan.shard_id == shard_id
    ]
    sampling_hash = sampling_config_hash(settings.get("sampling") or {})
    completion_hash = config_hash({
        "sampling_config_hash": sampling_hash,
        "rule_version": settings.get("rule_version"),
        "seed": settings.get("seed"),
        "validation_ratio": settings.get("validation_ratio"),
        "min_interval_frames": settings.get("min_interval_frames"),
        "min_l1_count": settings.get("min_l1_count"),
        "min_l0_count": settings.get("min_l0_count"),
        "max_camera_views": settings.get("max_camera_views"),
        "view_config": view_config,
        "video_validation": settings.get("video_validation", "metadata"),
    })
    payloads = [{
        "plan": plan.to_dict(),
        "run_root": str(run_root),
        "run_id": run_id,
        "stage": stage,
        "mode": mode,
        "settings": dict(settings),
        "view_config": dict(view_config),
        "sampling_hash": sampling_hash,
        "completion_hash": completion_hash,
        "ledger_path": ledger_path,
    } for plan in selected]
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    if num_workers == 1:
        for payload in payloads:
            try:
                results.append(_run_shard(payload))
            except BaseException as exc:
                failures.append(f"shard={payload['plan']['shard_id']}: {type(exc).__name__}: {exc}")
                if fail_fast:
                    raise
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_payload = {executor.submit(_run_shard, item): item for item in payloads}
            for future in as_completed(future_to_payload):
                payload = future_to_payload[future]
                try:
                    results.append(future.result())
                except BaseException as exc:
                    failures.append(f"shard={payload['plan']['shard_id']}: {type(exc).__name__}: {exc}")
                    if fail_fast:
                        for pending in future_to_payload:
                            pending.cancel()
                        raise
    aggregate = Counter()
    samples = 0
    for result in results:
        aggregate.update(result["statistics"].get("status") or {})
        samples += int(result["statistics"].get("samples", 0))
    # A single stage-root entry is enough provenance. Registering every file
    # from every worker rewrote the multi-megabyte Markdown ledger thousands of
    # times under a global lock and materially throttled shard completion.
    shard_root = run_root / ("validation_shards" if stage == "validate" else "shards")
    register_artifacts(
        ledger_path,
        [shard_root],
        purpose=f"V2 {stage} shard outputs",
        source_id=",".join(sorted(inventory.by_source)),
        run_id=run_id,
    )
    return {
        "run_id": run_id,
        "stage": stage,
        "mode": mode,
        "inventory_hash": inventory.inventory_hash,
        "shards_selected": len(selected),
        "shards_succeeded": len(results),
        "shards_failed": len(failures),
        "episodes": dict(aggregate),
        "samples": samples,
        "failures": failures,
        "results": sorted(results, key=lambda item: item["shard_id"]),
    }
