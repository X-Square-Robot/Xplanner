"""Bounded deterministic CPU parallelism for V5 source materialization."""

from __future__ import annotations

import os
import multiprocessing as mp
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar


DEFAULT_CHUNK_BYTES = 256 * 1024 * 1024
HARD_MAX_WORKERS = 64
DEFAULT_NICE = 5

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class JsonlChunk:
    path: str
    start: int
    end: int
    first_line: int
    num_lines: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_workers(requested: int | None = None) -> int:
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    automatic = min(32, max(4, affinity // 4))
    value = automatic if requested in (None, 0) else requested
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("workers must be a positive integer or zero for auto")
    if value > HARD_MAX_WORKERS:
        raise ValueError(f"workers cannot exceed {HARD_MAX_WORKERS}")
    return min(value, max(1, affinity))


def plan_jsonl_chunks(
    paths: Sequence[Path | str],
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    max_rows_per_path: int | None = None,
) -> tuple[JsonlChunk, ...]:
    """Return newline-aligned chunks with exact original line numbers."""
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if max_rows_per_path is not None and max_rows_per_path <= 0:
        raise ValueError("max_rows_per_path must be positive")
    chunks: list[JsonlChunk] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        start = 0
        first_line = 1
        lines_in_chunk = 0
        total_lines = 0
        with path.open("rb") as handle:
            while True:
                line = handle.readline()
                if not line:
                    break
                total_lines += 1
                if max_rows_per_path is not None and total_lines > max_rows_per_path:
                    break
                if not line.strip():
                    continue
                lines_in_chunk += 1
                end = handle.tell()
                if end - start >= chunk_bytes:
                    chunks.append(JsonlChunk(
                        path=str(path),
                        start=start,
                        end=end,
                        first_line=first_line,
                        num_lines=lines_in_chunk,
                    ))
                    start = end
                    first_line = total_lines + 1
                    lines_in_chunk = 0
            end = handle.tell()
        if lines_in_chunk:
            chunks.append(JsonlChunk(
                path=str(path),
                start=start,
                end=end,
                first_line=first_line,
                num_lines=lines_in_chunk,
            ))
    return tuple(chunks)


def _worker_init(nice: int) -> None:
    if nice:
        try:
            os.nice(nice)
        except OSError:
            pass


def bounded_ordered_map(
    function: Callable[[T], R],
    values: Iterable[T],
    *,
    workers: int,
    max_in_flight: int | None = None,
    nice: int = DEFAULT_NICE,
) -> Iterator[R]:
    """Map with spawn workers, bounded submissions, and deterministic ordering."""
    worker_count = resolve_workers(workers)
    if worker_count == 1:
        yield from map(function, values)
        return
    bound = max_in_flight or worker_count * 2
    if bound < worker_count or bound > worker_count * 2:
        raise ValueError("max_in_flight must be between workers and 2 * workers")
    iterator = iter(values)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=_worker_init,
        initargs=(nice,),
    ) as executor:
        pending: deque[Any] = deque()
        for _ in range(bound):
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                break
        while pending:
            future = pending.popleft()
            yield future.result()
            try:
                pending.append(executor.submit(function, next(iterator)))
            except StopIteration:
                pass


__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "DEFAULT_NICE",
    "HARD_MAX_WORKERS",
    "JsonlChunk",
    "bounded_ordered_map",
    "plan_jsonl_chunks",
    "resolve_workers",
]
