"""Idempotent append-only registration in the requested Markdown cc ledger."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import Iterable

from .common.atomic import atomic_write


START = "<!-- V10_V2_ARTIFACT_LEDGER_START -->"
END = "<!-- V10_V2_ARTIFACT_LEDGER_END -->"
HEADER = "| file_path | purpose | source_id | run_id |\n|---|---|---|---|"


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def register_artifacts(
    ledger_path: str | os.PathLike[str] | None,
    paths: Iterable[str | os.PathLike[str]],
    *,
    purpose: str,
    source_id: str,
    run_id: str,
) -> None:
    if not ledger_path:
        return
    ledger = Path(ledger_path).resolve()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger.with_suffix(ledger.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        text = ledger.read_text(encoding="utf-8") if ledger.is_file() else ""
        if START not in text:
            if text and not text.endswith("\n"):
                text += "\n"
            text += f"\n## V10 V2 Artifact Ledger\n\n{START}\n{HEADER}\n{END}\n"
        prefix, rest = text.split(START, 1)
        body, suffix = rest.split(END, 1)
        lines = [line for line in body.strip().splitlines() if line]
        if not lines:
            lines = HEADER.splitlines()
        existing = set()
        for line in lines[2:]:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) >= 4:
                existing.add((cells[3], cells[0]))
        additions = []
        for raw_path in sorted({str(Path(path).resolve()) for path in paths}):
            key = (run_id, raw_path)
            if key in existing:
                continue
            additions.append(
                f"| {_cell(raw_path)} | {_cell(purpose)} | {_cell(source_id)} | {_cell(run_id)} |"
            )
            existing.add(key)
        if not additions:
            return
        rendered = "\n".join(lines + additions)
        updated = f"{prefix}{START}\n{rendered}\n{END}{suffix}"
        with atomic_write(str(ledger)) as handle:
            handle.write(updated)
