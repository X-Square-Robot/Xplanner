"""Dependency-light CPU resource monitor for Memory V4 transforms.

The monitor reads Linux ``/proc`` only.  It never changes worker affinity or
process counts itself: every sample contains a deterministic concurrency
recommendation which the V4 build controller may choose to apply.  Keeping
observation and actuation separate makes a resumed transform auditable.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import signal
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "memory_v4_resource_monitor_v1"
DEFAULT_INTERVAL_SECONDS = 5.0
GIB = 1024**3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write one file durably, then atomically publish it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def _append_jsonl_durable(path: Path, value: Mapping[str, Any]) -> None:
    """Append a complete JSONL record with a process lock and durable fsync.

    If a previous writer died during an append, the incomplete trailing record
    is removed while holding the same lock before the next record is written.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        end = handle.tell()
        if end:
            handle.seek(end - 1)
            if handle.read(1) != b"\n":
                cursor = end
                newline = -1
                while cursor > 0 and newline < 0:
                    begin = max(0, cursor - 65536)
                    handle.seek(begin)
                    chunk = handle.read(cursor - begin)
                    found = chunk.rfind(b"\n")
                    if found >= 0:
                        newline = begin + found
                    cursor = begin
                handle.truncate(newline + 1 if newline >= 0 else 0)
        handle.seek(0, os.SEEK_END)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class ConcurrencyPolicy:
    start_concurrency: int = 96
    max_concurrency: int = 160
    min_concurrency: int = 1
    adjustment_step: int = 8
    target_cpu_lower_percent: float = 85.0
    target_cpu_upper_percent: float = 92.0
    load1_cap: float = 165.0
    iowait_soft_cap_percent: float = 10.0
    iowait_hard_cap_percent: float = 15.0
    reserve_logical_cpus: int = 20
    reserve_memory_gib: float = 512.0

    def validate(self) -> "ConcurrencyPolicy":
        if not (1 <= self.min_concurrency <= self.start_concurrency <= self.max_concurrency):
            raise ValueError("require min_concurrency <= start_concurrency <= max_concurrency")
        if self.adjustment_step <= 0:
            raise ValueError("adjustment_step must be positive")
        if not (0 <= self.target_cpu_lower_percent < self.target_cpu_upper_percent <= 100):
            raise ValueError("invalid target CPU range")
        if not (0 <= self.iowait_soft_cap_percent < self.iowait_hard_cap_percent <= 100):
            raise ValueError("invalid iowait caps")
        if self.load1_cap <= 0 or self.reserve_memory_gib < 0 or self.reserve_logical_cpus < 0:
            raise ValueError("load and reserve values must be non-negative")
        return self

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ConcurrencyPolicy":
        nested = value
        for key in ("concurrency_policy", "resource_policy", "resource_monitor"):
            candidate = value.get(key)
            if isinstance(candidate, Mapping):
                nested = candidate
                break
        aliases = {
            "start": "start_concurrency",
            "max": "max_concurrency",
            "target_cpu_lower": "target_cpu_lower_percent",
            "target_cpu_upper": "target_cpu_upper_percent",
            "load_cap": "load1_cap",
            "iowait_soft_cap": "iowait_soft_cap_percent",
            "iowait_hard_cap": "iowait_hard_cap_percent",
        }
        fields = set(cls.__dataclass_fields__)
        normalized: dict[str, Any] = {}
        for key, item in nested.items():
            canonical = aliases.get(str(key), str(key))
            if canonical in fields:
                normalized[canonical] = item
        return cls(**normalized).validate()


def load_policy(path: str | Path | None) -> ConcurrencyPolicy:
    if path is None:
        return ConcurrencyPolicy().validate()
    policy_path = Path(path)
    text = policy_path.read_text(encoding="utf-8")
    if policy_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as error:  # pragma: no cover - repository environment has PyYAML
            raise RuntimeError("PyYAML is required to read a YAML concurrency policy") from error
        value = yaml.safe_load(text) or {}
    else:
        value = json.loads(text)
    if not isinstance(value, Mapping):
        raise ValueError(f"concurrency policy must be an object: {policy_path}")
    return ConcurrencyPolicy.from_mapping(value)


def _read_cpu_times(proc_root: Path) -> dict[str, int]:
    first = (proc_root / "stat").read_text(encoding="utf-8").splitlines()[0].split()
    if not first or first[0] != "cpu":
        raise ValueError(f"invalid {proc_root / 'stat'}")
    names = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")
    values = [int(item) for item in first[1:1 + len(names)]]
    values.extend([0] * (len(names) - len(values)))
    result = dict(zip(names, values, strict=True))
    result["total"] = sum(values)
    return result


def _read_pressure(path: Path) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        values: dict[str, float | int] = {}
        for field in fields[1:]:
            name, raw = field.split("=", 1)
            values[name] = int(raw) if name == "total" else float(raw)
        result[fields[0]] = values
    return result


def _read_meminfo(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        fields = raw.split()
        if not fields:
            continue
        value = int(fields[0])
        if len(fields) > 1 and fields[1].lower() == "kb":
            value *= 1024
        result[key] = value
    return result


def read_proc_snapshot(proc_root: str | Path = "/proc") -> dict[str, Any]:
    root = Path(proc_root)
    load_fields = (root / "loadavg").read_text(encoding="utf-8").split()
    memory = _read_meminfo(root / "meminfo")
    return {
        "cpu_times": _read_cpu_times(root),
        "load": {
            "load1": float(load_fields[0]),
            "load5": float(load_fields[1]),
            "load15": float(load_fields[2]),
            "running_processes": int(load_fields[3].split("/", 1)[0]),
            "total_processes": int(load_fields[3].split("/", 1)[1]),
        },
        "pressure": {
            name: _read_pressure(root / "pressure" / name)
            for name in ("cpu", "io", "memory")
        },
        "memory": {
            "total_bytes": memory.get("MemTotal", 0),
            "available_bytes": memory.get("MemAvailable", 0),
            "free_bytes": memory.get("MemFree", 0),
            "cached_bytes": memory.get("Cached", 0),
            "swap_total_bytes": memory.get("SwapTotal", 0),
            "swap_free_bytes": memory.get("SwapFree", 0),
        },
    }


def derive_sample(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    before = previous["cpu_times"]
    after = current["cpu_times"]
    total_delta = int(after["total"]) - int(before["total"])
    if total_delta <= 0:
        raise ValueError("non-positive /proc/stat CPU delta")
    idle_delta = (
        int(after["idle"]) + int(after["iowait"])
        - int(before["idle"]) - int(before["iowait"])
    )
    iowait_delta = int(after["iowait"]) - int(before["iowait"])
    cpu_busy = 100.0 * max(0, total_delta - idle_delta) / total_delta
    iowait = 100.0 * max(0, iowait_delta) / total_delta
    return {
        "cpu_busy_percent": round(cpu_busy, 3),
        "iowait_percent": round(iowait, 3),
        "load": dict(current["load"]),
        "pressure": current["pressure"],
        "memory": current["memory"],
    }


def recommend_concurrency(
    desired: int,
    metrics: Mapping[str, Any],
    policy: ConcurrencyPolicy,
    *,
    logical_cpus: int | None = None,
) -> dict[str, Any]:
    policy.validate()
    cpus = logical_cpus if logical_cpus is not None else (os.cpu_count() or 1)
    effective_max = max(policy.min_concurrency, min(
        policy.max_concurrency,
        max(policy.min_concurrency, cpus - policy.reserve_logical_cpus),
    ))
    current = max(policy.min_concurrency, min(int(desired), effective_max))
    cpu_busy = float(metrics["cpu_busy_percent"])
    iowait = float(metrics["iowait_percent"])
    load1 = float(metrics["load"]["load1"])
    available_gib = float(metrics["memory"]["available_bytes"]) / GIB
    step = policy.adjustment_step

    if available_gib < policy.reserve_memory_gib:
        proposed, reason = current - 2 * step, "memory_reserve_breached"
    elif iowait >= policy.iowait_hard_cap_percent:
        proposed, reason = current - 2 * step, "iowait_hard_cap"
    elif load1 >= policy.load1_cap:
        proposed, reason = current - step, "load1_cap"
    elif cpu_busy > policy.target_cpu_upper_percent:
        proposed, reason = current - step, "cpu_above_target"
    elif iowait >= policy.iowait_soft_cap_percent:
        proposed, reason = current - step, "iowait_soft_cap"
    elif cpu_busy < policy.target_cpu_lower_percent:
        proposed, reason = current + step, "cpu_below_target"
    else:
        proposed, reason = current, "within_target"
    recommended = max(policy.min_concurrency, min(proposed, effective_max))
    return {
        "desired_concurrency": int(desired),
        "effective_max_concurrency": effective_max,
        "recommended_concurrency": recommended,
        "action": "increase" if recommended > current else "decrease" if recommended < current else "hold",
        "reason": reason,
    }


class RunningSummary:
    def __init__(self) -> None:
        self.count = 0
        self.started_at = _utc_now()
        self.sums = {"cpu_busy_percent": 0.0, "iowait_percent": 0.0, "load1": 0.0}
        self.minimums = {key: math.inf for key in self.sums}
        self.maximums = {key: -math.inf for key in self.sums}
        self.last: dict[str, Any] | None = None

    def add(self, sample: Mapping[str, Any]) -> None:
        values = {
            "cpu_busy_percent": float(sample["cpu_busy_percent"]),
            "iowait_percent": float(sample["iowait_percent"]),
            "load1": float(sample["load"]["load1"]),
        }
        self.count += 1
        for key, value in values.items():
            self.sums[key] += value
            self.minimums[key] = min(self.minimums[key], value)
            self.maximums[key] = max(self.maximums[key], value)
        self.last = dict(sample)

    def as_dict(self, *, state: str, policy: ConcurrencyPolicy) -> dict[str, Any]:
        aggregates = {}
        if self.count:
            aggregates = {
                key: {
                    "mean": round(self.sums[key] / self.count, 3),
                    "min": round(self.minimums[key], 3),
                    "max": round(self.maximums[key], 3),
                }
                for key in self.sums
            }
        return {
            "schema_version": SCHEMA_VERSION,
            "state": state,
            "started_at": self.started_at,
            "updated_at": _utc_now(),
            "sample_count": self.count,
            "aggregates": aggregates,
            "last_sample": self.last,
            "policy": asdict(policy),
        }


def monitor(
    *,
    output_dir: str | Path,
    policy: ConcurrencyPolicy,
    desired_concurrency: int,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    max_samples: int | None = None,
    stop_file: str | Path | None = None,
    proc_root: str | Path = "/proc",
) -> dict[str, Any]:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples_path = output / "resource_monitor.jsonl"
    summary_path = output / "resource_summary.json"
    state_path = output / "build_state.json"
    stop_path = Path(stop_file) if stop_file else None
    summary = RunningSummary()
    stopped = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = True

    old_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    state = "running"
    try:
        previous = read_proc_snapshot(proc_root)
        while not stopped and (max_samples is None or summary.count < max_samples):
            if stop_path is not None and stop_path.exists():
                state = "stopped_by_file"
                break
            time.sleep(interval_seconds)
            current = read_proc_snapshot(proc_root)
            metrics = derive_sample(previous, current)
            previous = current
            recommendation = recommend_concurrency(desired_concurrency, metrics, policy)
            sample = {
                "schema_version": SCHEMA_VERSION,
                "sampled_at": _utc_now(),
                "interval_seconds": interval_seconds,
                "logical_cpus": os.cpu_count() or 1,
                **metrics,
                "concurrency": recommendation,
            }
            _append_jsonl_durable(samples_path, sample)
            summary.add(sample)
            current_summary = summary.as_dict(state="running", policy=policy)
            _atomic_write_json(summary_path, current_summary)
            _atomic_write_json(state_path, {
                "schema_version": SCHEMA_VERSION,
                "state": "running",
                "updated_at": current_summary["updated_at"],
                "samples": summary.count,
                "recommended_concurrency": recommendation["recommended_concurrency"],
                "reason": recommendation["reason"],
            })
        else:
            state = "complete" if not stopped else "stopped_by_signal"
        if stopped:
            state = "stopped_by_signal"
    except BaseException as error:
        state = "failed"
        failed = summary.as_dict(state=state, policy=policy)
        failed["error"] = f"{type(error).__name__}: {error}"
        _atomic_write_json(summary_path, failed)
        _atomic_write_json(state_path, {
            "schema_version": SCHEMA_VERSION,
            "state": state,
            "updated_at": _utc_now(),
            "error": failed["error"],
        })
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)

    final = summary.as_dict(state=state, policy=policy)
    _atomic_write_json(summary_path, final)
    _atomic_write_json(state_path, {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "updated_at": final["updated_at"],
        "samples": summary.count,
        "recommended_concurrency": (
            summary.last["concurrency"]["recommended_concurrency"] if summary.last else None
        ),
    })
    return final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "one-shot"):
        command = subparsers.add_parser(name)
        command.add_argument("--output-dir", required=True)
        command.add_argument("--policy", help="JSON/YAML policy or full V4 config")
        command.add_argument("--desired-concurrency", type=int)
        command.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
        command.add_argument("--proc-root", default="/proc", help=argparse.SUPPRESS)
    start = subparsers.choices["start"]
    start.add_argument("--max-samples", type=int)
    start.add_argument("--stop-file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    policy = load_policy(args.policy)
    desired = args.desired_concurrency or policy.start_concurrency
    result = monitor(
        output_dir=args.output_dir,
        policy=policy,
        desired_concurrency=desired,
        interval_seconds=args.interval_seconds,
        max_samples=1 if args.command == "one-shot" else args.max_samples,
        stop_file=None if args.command == "one-shot" else args.stop_file,
        proc_root=args.proc_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

