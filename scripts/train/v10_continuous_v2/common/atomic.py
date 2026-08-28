"""Crash-safe writes: tmp -> flush -> fsync -> rename -> .done (spec section 7)."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TextIO

from .constants_v2 import DEFAULT_BATCH_FLUSH

DONE_MARKER = ".done"
TMP_SUFFIX = ".tmp"


def _fsync_dir(path: str) -> None:
    """Make a rename durable. Best-effort: not all filesystems allow this."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


@contextmanager
def atomic_write(path: str, *, mode: str = "w") -> Iterator[TextIO]:
    """Yield a handle to ``path + '.tmp'``, atomically renamed on clean exit."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = path + TMP_SUFFIX
    handle = open(tmp_path, mode, encoding="utf-8")
    try:
        yield handle
        handle.flush()
        os.fsync(handle.fileno())
    except BaseException:
        handle.close()
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    else:
        handle.close()
        os.replace(tmp_path, path)
        _fsync_dir(directory)


def write_json(path: str, value: Any) -> None:
    with atomic_write(path) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: str, default: Any = None) -> Any:
    if not os.path.isfile(path):
        return default
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


class BatchedJsonlWriter:
    """Append-only JSONL writer that flushes every ``batch_size`` records.

    Writes go to ``<path>.tmp`` and are promoted on ``close()``, so a crashed
    shard never leaves a half-written artifact that looks final.
    """

    def __init__(self, path: str, *, batch_size: int = DEFAULT_BATCH_FLUSH) -> None:
        self.path = path
        self.batch_size = max(1, batch_size)
        self.count = 0
        self._pending = 0
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, exist_ok=True)
        self._tmp_path = path + TMP_SUFFIX
        self._handle: TextIO | None = open(self._tmp_path, "w", encoding="utf-8")

    def write(self, record: Any) -> None:
        if self._handle is None:
            raise RuntimeError(f"writer already closed: {self.path}")
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count += 1
        self._pending += 1
        if self._pending >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if self._handle is None or self._pending == 0:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._pending = 0

    def close(self) -> None:
        if self._handle is None:
            return
        self.flush()
        self._handle.close()
        self._handle = None
        os.replace(self._tmp_path, self.path)
        _fsync_dir(os.path.dirname(os.path.abspath(self.path)) or ".")

    def abort(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None
        try:
            os.unlink(self._tmp_path)
        except OSError:
            pass

    def __enter__(self) -> "BatchedJsonlWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def iter_jsonl(path: str) -> Iterator[Any]:
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def clear_stale_tmp(directory: str) -> list[str]:
    """Remove leftover ``*.tmp`` files from a crashed run."""
    removed: list[str] = []
    if not os.path.isdir(directory):
        return removed
    for name in os.listdir(directory):
        if name.endswith(TMP_SUFFIX):
            target = os.path.join(directory, name)
            try:
                os.unlink(target)
                removed.append(target)
            except OSError:
                pass
    return removed


def is_done(directory: str) -> bool:
    return os.path.isfile(os.path.join(directory, DONE_MARKER))


def mark_done(directory: str, payload: Any = None) -> None:
    """Write ``.done`` last, after every artifact is durable."""
    marker = os.path.join(directory, DONE_MARKER)
    with atomic_write(marker) as handle:
        if payload is None:
            handle.write("done\n")
        else:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
