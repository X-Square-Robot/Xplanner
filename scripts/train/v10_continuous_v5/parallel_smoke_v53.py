"""Timeout-isolated parallel smoke for physical V5.3 bucket files."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time
import traceback
from typing import Any

from .bucket_io_v53 import atomic_json, file_sha256
from .holdout_v5 import Benchmark3Holdout, DEFAULT_BENCHMARK3_MANIFEST, DEFAULT_BENCHMARK3_SHA256
from .prompt_v5 import normalized_execution_instruction, render_user
from .schema_v5 import dumps_assistant, validate_sample


def publish_audit_permissions(roots: Sequence[Path]) -> None:
    """Make worker-owned JSON evidence readable by the controller.

    Older atomic writers published ``mkstemp`` files as 0600.  The smoke runs
    under the same DLC UID that owns those files, so it is the earliest safe
    point at which a running pipeline can repair both the materialized buckets
    and its sibling label-scan evidence without weakening write permissions.
    """

    targets = list(roots)
    targets.extend(root.parent / "label_scan" for root in roots)
    seen: set[Path] = set()
    for target in targets:
        if target in seen or not target.exists():
            continue
        seen.add(target)
        if target.is_dir():
            target.chmod(0o755)
        for path in target.rglob("*"):
            if path.is_dir():
                path.chmod(0o755)
            elif path.is_file():
                path.chmod(0o644)


def discover_files(roots: Sequence[Path]) -> list[Path]:
    result: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        result.update(path.resolve() for path in root.glob("**/buckets/*/train.jsonl"))
        result.update(path.resolve() for path in root.glob("**/buckets/*/test.jsonl"))
    return sorted(path for path in result if path.is_file())


def _media_paths(sample: Mapping[str, Any]) -> list[str]:
    result = []
    for item in sample.get("images") or ():
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, Mapping):
            value = item.get("video") or item.get("path")
            if isinstance(value, str):
                result.append(value)
    return result


def _worker(
    path: str,
    max_rows: int,
    media_checks: int,
    benchmark_path: str,
    benchmark_sha256: str | None,
    output: mp.Queue,
) -> None:
    started = time.monotonic()
    result: dict[str, Any] = {
        "path": path,
        "passed": False,
        "rows": 0,
        "media_checked": 0,
        "benchmark3_overlap": 0,
        "errors": [],
        "categories": {},
        "contexts": {},
        "splits": {},
        "episodes_by_split": {"train": [], "test": []},
        "execution_suffixes": {},
    }
    try:
        holdout = Benchmark3Holdout.load(benchmark_path, expected_sha256=benchmark_sha256)
        categories: Counter[str] = Counter()
        contexts: Counter[str] = Counter()
        splits: Counter[str] = Counter()
        episodes: dict[str, set[str]] = defaultdict(set)
        suffixes: dict[str, set[str]] = defaultdict(set)
        checked_media: set[str] = set()
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if result["rows"] >= max_rows:
                    break
                if not line.strip():
                    continue
                try:
                    outer = json.loads(line)
                    raw = outer.get("v5_sample") if isinstance(outer, Mapping) else None
                    sample = validate_sample(raw)
                    prompt = render_user(sample)
                    dumps_assistant(sample["target"], sample["category"], sample["output_spec"])
                    matches = holdout.match_sample(sample)
                    if matches:
                        result["benchmark3_overlap"] += 1
                        raise ValueError(f"Benchmark3 overlap: {matches[:2]}")
                    if sample["category"] != "initial_plan":
                        suffixes[sample["output_profile_id"]].add(
                            normalized_execution_instruction(prompt)
                        )
                    split = sample["split"]
                    episode = str(sample["provenance"].get("episode_key") or "")
                    if episode:
                        episodes[split].add(episode)
                    categories[sample["category"]] += 1
                    contexts[sample["context_variant"]] += 1
                    splits[split] += 1
                    for media in _media_paths(sample):
                        if len(checked_media) >= media_checks or media in checked_media:
                            continue
                        stat = Path(media).stat()
                        if stat.st_size <= 0:
                            raise ValueError(f"empty media: {media}")
                        checked_media.add(media)
                    result["rows"] += 1
                except Exception as exc:
                    result["errors"].append({
                        "line": line_number,
                        "error": type(exc).__name__,
                        "detail": str(exc),
                    })
                    if len(result["errors"]) >= 20:
                        break
        result.update({
            "passed": result["rows"] > 0 and not result["errors"],
            "media_checked": len(checked_media),
            "categories": dict(sorted(categories.items())),
            "contexts": dict(sorted(contexts.items())),
            "splits": dict(sorted(splits.items())),
            "episodes_by_split": {
                split: sorted(values) for split, values in episodes.items()
            },
            "execution_suffixes": {
                profile: sorted(values) for profile, values in suffixes.items()
            },
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
    except BaseException as exc:
        result["errors"].append({
            "line": 0,
            "error": type(exc).__name__,
            "detail": str(exc),
            "traceback": traceback.format_exc(limit=8),
        })
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    output.put(result)


def run_smoke(
    *,
    artifact_roots: Sequence[Path],
    output_path: Path,
    workers: int,
    timeout_seconds: float,
    heartbeat_seconds: float,
    max_rows_per_file: int,
    media_checks_per_file: int,
    benchmark_path: Path,
    benchmark_sha256: str | None,
) -> dict[str, Any]:
    publish_audit_permissions(artifact_roots)
    files = discover_files(artifact_roots)
    if not files:
        raise ValueError("no physical V5.3 train/test bucket files found")
    context = mp.get_context("spawn")
    outputs: mp.Queue = context.Queue()
    pending = list(files)
    running: dict[int, tuple[mp.Process, Path, float]] = {}
    results: list[dict[str, Any]] = []
    last_heartbeat = time.monotonic()
    started = last_heartbeat

    while pending or running:
        while pending and len(running) < max(1, workers):
            path = pending.pop(0)
            process = context.Process(
                target=_worker,
                args=(
                    str(path), max_rows_per_file, media_checks_per_file,
                    str(benchmark_path), benchmark_sha256, outputs,
                ),
                name=f"v53-smoke-{path.parent.name}-{path.stem}",
            )
            process.start()
            running[process.pid] = (process, path, time.monotonic())

        while True:
            try:
                results.append(outputs.get_nowait())
            except queue.Empty:
                break

        now = time.monotonic()
        for pid, (process, path, task_started) in list(running.items()):
            if not process.is_alive():
                process.join(timeout=0.1)
                if process.exitcode != 0 and not any(row["path"] == str(path) for row in results):
                    results.append({
                        "path": str(path), "passed": False, "rows": 0,
                        "errors": [{"error": "WorkerExit", "detail": f"exitcode={process.exitcode}"}],
                    })
                del running[pid]
            elif now - task_started > timeout_seconds:
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5)
                results.append({
                    "path": str(path), "passed": False, "rows": 0,
                    "timed_out": True,
                    "errors": [{
                        "error": "Timeout",
                        "detail": f"isolated worker exceeded {timeout_seconds:.1f}s",
                    }],
                })
                del running[pid]
        if now - last_heartbeat >= heartbeat_seconds:
            print(json.dumps({
                "event": "heartbeat",
                "completed": len(results),
                "running": [str(value[1]) for value in running.values()],
                "pending": len(pending),
                "elapsed_seconds": round(now - started, 1),
            }, ensure_ascii=False), flush=True)
            last_heartbeat = now
        if pending or running:
            time.sleep(0.1)

    # A child can exit after queueing and before the parent drains the queue.
    time.sleep(0.05)
    while True:
        try:
            row = outputs.get_nowait()
            if not any(existing.get("path") == row.get("path") for existing in results):
                results.append(row)
        except queue.Empty:
            break
    by_path = {row["path"]: row for row in results}
    results = [by_path.get(str(path), {
        "path": str(path), "passed": False, "rows": 0,
        "errors": [{"error": "MissingResult", "detail": "worker produced no result"}],
    }) for path in files]

    suffixes: dict[str, set[str]] = defaultdict(set)
    train_episodes: set[str] = set()
    test_episodes: set[str] = set()
    for result in results:
        for profile, values in (result.get("execution_suffixes") or {}).items():
            suffixes[profile].update(values)
        episode_map = result.get("episodes_by_split") or {}
        train_episodes.update(episode_map.get("train") or ())
        test_episodes.update(episode_map.get("test") or ())
    inconsistent_profiles = {
        profile: sorted(values) for profile, values in suffixes.items() if len(values) != 1
    }
    overlap = sorted(train_episodes & test_episodes)
    report = {
        "schema_version": "v10_action_segment_v5_3_parallel_smoke_v1",
        "complete": True,
        "passed": all(result.get("passed") is True for result in results)
        and not inconsistent_profiles and not overlap,
        "workers": workers,
        "per_file_timeout_seconds": timeout_seconds,
        "heartbeat_seconds": heartbeat_seconds,
        "max_rows_per_file": max_rows_per_file,
        "media_checks_per_file": media_checks_per_file,
        "files": results,
        "file_count": len(files),
        "validated_rows": sum(int(result.get("rows") or 0) for result in results),
        "timed_out_files": sum(bool(result.get("timed_out")) for result in results),
        "benchmark3_overlap": sum(int(result.get("benchmark3_overlap") or 0) for result in results),
        "execution_prompt_profile_inconsistencies": inconsistent_profiles,
        "sampled_train_test_episode_overlap": overlap[:100],
        "artifact_roots": [str(path.resolve()) for path in artifact_roots],
        "benchmark3_manifest": str(benchmark_path.resolve()),
        "benchmark3_sha256": benchmark_sha256,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    atomic_json(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    parser.add_argument("--max-rows-per-file", type=int, default=64)
    parser.add_argument("--media-checks-per-file", type=int, default=3)
    parser.add_argument("--benchmark3", type=Path, default=DEFAULT_BENCHMARK3_MANIFEST)
    parser.add_argument("--benchmark3-sha256", default=DEFAULT_BENCHMARK3_SHA256)
    args = parser.parse_args(argv)
    report = run_smoke(
        artifact_roots=args.artifact_root,
        output_path=args.output,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
        max_rows_per_file=args.max_rows_per_file,
        media_checks_per_file=args.media_checks_per_file,
        benchmark_path=args.benchmark3,
        benchmark_sha256=args.benchmark3_sha256,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["discover_files", "publish_audit_permissions", "run_smoke"]
