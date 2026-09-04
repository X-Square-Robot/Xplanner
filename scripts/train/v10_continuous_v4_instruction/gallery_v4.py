"""Build a deterministic, exact Prompt/GT review gallery from Memory V4 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .prompt_v4 import render_user
from .schema_v4 import loads_assistant


SCHEMA_VERSION = "memory_v4_gallery_v1"
DEFAULT_OUTPUT = Path(
    "/mnt/cpfs/zbl-cpfs-new/USERS/luhao/APlan/0811/memory_v4/examples_prompt_v2"
)
DEFAULT_SEED = 20260811
STAGES = ("initial", "start", "process", "terminal")
_EXCLUDED_NAMES = (
    "episode", "oversize", "terminal_ref", "statistics", "resource_monitor",
    "status", "manifest", "failure", "error", "list",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def _completed_marker_for(path: Path, boundary: Path) -> Path | None:
    current = path.parent
    while current == boundary or boundary in current.parents:
        marker = current / "_SUCCESS"
        if marker.is_file():
            return marker
        if current == boundary:
            break
        current = current.parent
    return None


def _looks_like_sample_jsonl(path: Path) -> bool:
    lowered = path.name.lower()
    if path.suffix != ".jsonl" or any(token in lowered for token in _EXCLUDED_NAMES):
        return False
    # A separate terminal materialization is a hardlink/copy of continuous.
    if "terminal" in lowered:
        return False
    return True


@dataclass(frozen=True)
class InputFile:
    path: Path
    success_marker: Path


def discover_input_files(roots: Iterable[str | Path]) -> list[InputFile]:
    discovered: dict[Path, InputFile] = {}
    for raw_root in roots:
        root = Path(raw_root).resolve()
        if root.is_file():
            marker = _completed_marker_for(root, root.parent)
            if marker is None:
                raise ValueError(f"explicit JSONL is not under a completed directory: {root}")
            discovered[root] = InputFile(root, marker)
            continue
        if not root.is_dir():
            raise FileNotFoundError(root)

        # Immutable Snapshot layout.  Do not read datasets/terminal because it
        # intentionally duplicates terminal windows from continuous.
        datasets = root / "datasets"
        if datasets.is_dir():
            marker = root / "_SUCCESS"
            if not marker.is_file():
                raise ValueError(f"snapshot has no _SUCCESS marker: {root}")
            for task in ("continuous", "initial_plan"):
                for split in ("train", "validation"):
                    path = datasets / task / split / "data.jsonl"
                    if path.is_file():
                        discovered[path.resolve()] = InputFile(path.resolve(), marker.resolve())
            continue

        markers = sorted(root.rglob("_SUCCESS"))
        for marker in markers:
            completed = marker.parent
            for path in sorted(completed.rglob("*.jsonl")):
                # Do not let an outer marker claim files owned by a nested part.
                nearest = _completed_marker_for(path, completed)
                if nearest != marker or not _looks_like_sample_jsonl(path):
                    continue
                discovered[path.resolve()] = InputFile(path.resolve(), marker.resolve())
    if not discovered:
        raise ValueError("no completed V4 sample JSONL files found")
    return [discovered[path] for path in sorted(discovered)]


def _sample_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("v4_sample")
    return value if isinstance(value, Mapping) else {}


def classify_stage(row: Mapping[str, Any]) -> str:
    sample = _sample_payload(row)
    explicit = str(row.get("stage") or sample.get("stage") or "").lower()
    explicit = {"initial_plan": "initial", "middle": "process"}.get(explicit, explicit)
    if explicit in STAGES:
        return explicit
    task_type = str(row.get("task_type") or sample.get("task_type") or "")
    if task_type == "initial_plan":
        return "initial"
    terminal = task_type == "terminal" or bool(
        row.get("is_terminal_window", sample.get("is_terminal_window", False))
    )
    if terminal:
        return "terminal"
    if task_type not in {"continuous", "terminal"}:
        raise ValueError(f"unsupported task_type {task_type!r}")
    grid_index = sample.get("anchor_grid_index", row.get("anchor_grid_index"))
    anchor = sample.get("anchor_frame", row.get("anchor_frame"))
    episode_start = sample.get("episode_start_frame", row.get("episode_start_frame", 0))
    if grid_index == 0 or (anchor is not None and int(anchor) == int(episode_start)):
        return "start"
    return "process"


def _turn_text(row: Mapping[str, Any], role: str) -> str:
    turns = row.get("text")
    if not isinstance(turns, list):
        raise ValueError("row.text must be a list")
    matches = [turn.get("text") for turn in turns if isinstance(turn, Mapping) and turn.get("role") == role]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise ValueError(f"row must contain exactly one string {role!r} turn")
    return matches[0]


def _images(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    images = row.get("image")
    if not isinstance(images, list):
        images = _sample_payload(row).get("images")
    if not isinstance(images, list) or not all(isinstance(item, Mapping) for item in images):
        raise ValueError("row must contain image references")
    return images


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    sample = _sample_payload(row)
    source = str(row.get("source_id") or sample.get("source_id") or "").strip()
    profile = str(row.get("profile") or sample.get("profile") or "").strip()
    sample_key = str(row.get("sample_key") or row.get("data_id") or "").strip()
    if not source or not profile or not sample_key:
        raise ValueError("row requires source_id, profile, and sample_key/data_id")
    return source, profile, sample_key


def _selection_score(seed: int, source: str, profile: str, stage: str, sample_key: str) -> int:
    encoded = f"{seed}\0{source}\0{profile}\0{stage}\0{sample_key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest(), "big")


@dataclass
class Candidate:
    score: int
    sample_key: str
    source: str
    profile: str
    stage: str
    row: dict[str, Any]
    raw_line: str
    input_file: str
    line_number: int


def _retain(
    cells: dict[tuple[str, str, str], dict[str, Candidate]],
    candidate: Candidate,
    count: int,
) -> None:
    cell = cells.setdefault((candidate.source, candidate.profile, candidate.stage), {})
    if candidate.sample_key in cell:
        return
    cell[candidate.sample_key] = candidate
    if len(cell) > count:
        worst = max(cell.values(), key=lambda item: (item.score, item.sample_key))
        del cell[worst.sample_key]


def _candidate_from_line(
    raw_line: str, *, path: str, line_number: int, seed: int
) -> Candidate:
    row = json.loads(raw_line)
    if not isinstance(row, dict):
        raise ValueError(f"non-object row at {path}:{line_number}")
    source, profile, sample_key = _identity(row)
    stage = classify_stage(row)
    _turn_text(row, "user")
    _turn_text(row, "assistant")
    _images(row)
    return Candidate(
        score=_selection_score(seed, source, profile, stage, sample_key),
        sample_key=sample_key,
        source=source,
        profile=profile,
        stage=stage,
        row=row,
        raw_line=raw_line,
        input_file=path,
        line_number=line_number,
    )


def _scan_file(payload: tuple[str, int, int]) -> tuple[int, list[Candidate]]:
    path, seed, samples_per_cell = payload
    cells: dict[tuple[str, str, str], dict[str, Candidate]] = {}
    rows = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw_line = line.rstrip("\r\n")
            if not raw_line:
                continue
            candidate = _candidate_from_line(
                raw_line, path=path, line_number=line_number, seed=seed
            )
            _retain(cells, candidate, samples_per_cell)
            rows += 1
    return rows, [candidate for cell in cells.values() for candidate in cell.values()]


def _slug(value: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.").lower() or "value"
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{readable[:64]}--{digest}"


def _candidate_metadata(candidate: Candidate) -> dict[str, Any]:
    row = candidate.row
    sample = _sample_payload(row)
    prompt = _turn_text(row, "user")
    assistant = _turn_text(row, "assistant")
    instruction = str(sample.get("task_instruction") or "")
    task_type = str(row.get("task_type") or sample.get("task_type") or "")
    if sample.get("schema_version") != "memory_v4" or not instruction:
        raise ValueError(f"selected row is not a canonical Memory V4 sample: {candidate.sample_key}")
    expected_prompt = render_user(sample)
    if prompt != expected_prompt:
        raise ValueError(f"selected row prompt disagrees with V4 renderer: {candidate.sample_key}")
    loads_assistant(
        assistant,
        candidate.profile,
        task_type,
        instruction=instruction,
        is_terminal_window=bool(row.get("is_terminal_window", False)),
    )
    expected_images = [
        {"video": image["video"], "frame": image["frame"], "view": image["view"]}
        for image in sample["images"]
    ]
    if list(_images(row)) != expected_images:
        raise ValueError(f"selected row image references disagree with V4 sample: {candidate.sample_key}")
    return {
        "source_id": candidate.source,
        "profile": candidate.profile,
        "stage": candidate.stage,
        "sample_key": candidate.sample_key,
        "global_episode_key": row.get("global_episode_key"),
        "split": row.get("split"),
        "task_type": row.get("task_type"),
        "is_terminal_window": bool(row.get("is_terminal_window", False)),
        "task_instruction": instruction,
        "anchor_frame": sample.get("anchor_frame", row.get("anchor_frame")),
        "input_file": candidate.input_file,
        "line_number": candidate.line_number,
        "selection_score_hex": f"{candidate.score:064x}",
        "raw_row_sha256": hashlib.sha256(candidate.raw_line.encode("utf-8")).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "assistant_sha256": hashlib.sha256(assistant.encode("utf-8")).hexdigest(),
        "image_count": len(_images(row)),
    }


def _markdown(candidate: Candidate, metadata: Mapping[str, Any]) -> str:
    prompt = _turn_text(candidate.row, "user")
    assistant = _turn_text(candidate.row, "assistant")
    images = _images(candidate.row)
    meta = json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True, indent=2)
    image_text = json.dumps(images, ensure_ascii=False, indent=2)
    return f"""# {candidate.source} / {candidate.profile} / {candidate.stage} / {candidate.sample_key}

> Prompt、Assistant GT 和图片引用直接来自选中的 V4 JSONL 行；说明性元数据不属于模型输入。

## 样本元数据

````json
{meta}
````

## 图片/视频帧引用

````json
{image_text}
````

## 实际 User Prompt

````text
{prompt}
````

## 实际 Assistant GT

````json
{assistant}
````
"""


def _write_status(output: Path, **values: Any) -> None:
    _atomic_write_json(output / "status.json", {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _utc_now(),
        **values,
    })


def build_gallery(
    *,
    input_roots: Iterable[str | Path],
    output_root: str | Path = DEFAULT_OUTPUT,
    seed: int = DEFAULT_SEED,
    samples_per_cell: int = 3,
    max_rows: int | None = None,
    status_interval_seconds: float = 10.0,
    workers: int = 1,
) -> dict[str, Any]:
    if samples_per_cell <= 0:
        raise ValueError("samples_per_cell must be positive")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    files = discover_input_files(input_roots)
    cells: dict[tuple[str, str, str], dict[str, Candidate]] = {}
    rows_scanned = 0
    files_completed = 0
    started_at = _utc_now()
    last_status = time.monotonic()
    _write_status(
        output, state="running", started_at=started_at, files_total=len(files),
        files_completed=0, rows_scanned=0, cells_observed=0,
    )
    try:
        reached_limit = False
        if max_rows is None and workers > 1:
            with ProcessPoolExecutor(max_workers=min(workers, len(files))) as pool:
                futures = {
                    pool.submit(
                        _scan_file,
                        (str(input_file.path), seed, samples_per_cell),
                    ): input_file
                    for input_file in files
                }
                for future in as_completed(futures):
                    input_file = futures[future]
                    scanned, candidates = future.result()
                    for candidate in candidates:
                        _retain(cells, candidate, samples_per_cell)
                    rows_scanned += scanned
                    files_completed += 1
                    _write_status(
                        output, state="running", started_at=started_at,
                        files_total=len(files), files_completed=files_completed,
                        current_file=str(input_file.path), rows_scanned=rows_scanned,
                        cells_observed=len(cells), retained=sum(map(len, cells.values())),
                    )
        else:
          for input_file in files:
            with input_file.path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    raw_line = line.rstrip("\r\n")
                    if not raw_line:
                        continue
                    candidate = _candidate_from_line(
                        raw_line, path=str(input_file.path), line_number=line_number,
                        seed=seed,
                    )
                    _retain(cells, candidate, samples_per_cell)
                    rows_scanned += 1
                    if max_rows is not None and rows_scanned >= max_rows:
                        reached_limit = True
                        break
                    if time.monotonic() - last_status >= status_interval_seconds:
                        _write_status(
                            output, state="running", started_at=started_at,
                            files_total=len(files), files_completed=files_completed,
                            current_file=str(input_file.path), rows_scanned=rows_scanned,
                            cells_observed=len(cells), retained=sum(map(len, cells.values())),
                        )
                        last_status = time.monotonic()
            if reached_limit:
                break
            files_completed += 1
            _write_status(
                output, state="running", started_at=started_at,
                files_total=len(files), files_completed=files_completed,
                rows_scanned=rows_scanned, cells_observed=len(cells),
                retained=sum(map(len, cells.values())),
            )
    except BaseException as error:
        _write_status(
            output, state="failed", started_at=started_at, files_total=len(files),
            files_completed=files_completed, rows_scanned=rows_scanned,
            cells_observed=len(cells), error=f"{type(error).__name__}: {error}",
        )
        raise

    selections = []
    cell_summaries = []
    for key in sorted(cells):
        source, profile, stage = key
        candidates = sorted(cells[key].values(), key=lambda item: (item.score, item.sample_key))
        cell_dir = output / _slug(source) / _slug(profile) / stage
        records = []
        for rank, candidate in enumerate(candidates, 1):
            stem = f"rank-{rank:02d}--{_slug(candidate.sample_key)}"
            raw_path = cell_dir / f"{stem}.json"
            markdown_path = cell_dir / f"{stem}.md"
            metadata = _candidate_metadata(candidate)
            _atomic_write_text(raw_path, candidate.raw_line + "\n")
            _atomic_write_text(markdown_path, _markdown(candidate, metadata))
            record = {
                **metadata,
                "rank": rank,
                "raw_file": str(raw_path.relative_to(output)),
                "markdown_file": str(markdown_path.relative_to(output)),
            }
            selections.append(record)
            records.append(record)
        cell_summaries.append({
            "source_id": source,
            "profile": profile,
            "stage": stage,
            "available_selected": len(records),
            "requested": samples_per_cell,
            "complete": len(records) == samples_per_cell,
            "samples": records,
        })

    state = "partial" if max_rows is not None and rows_scanned >= max_rows else "complete"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "created_at": _utc_now(),
        "started_at": started_at,
        "seed": seed,
        "samples_per_cell": samples_per_cell,
        "selection_rule": (
            "Lowest SHA-256(seed\\0source\\0profile\\0stage\\0sample_key); "
            "unique sample_key within each observed cell."
        ),
        "input_roots": [str(Path(item).resolve()) for item in input_roots],
        "input_files": [
            {
                "path": str(item.path),
                "bytes": item.path.stat().st_size,
                "success_marker": str(item.success_marker),
            }
            for item in files
        ],
        "rows_scanned": rows_scanned,
        "files_completed": files_completed,
        "cells_observed": len(cells),
        "selections": len(selections),
        "selected_samples": selections,
        "incomplete_cells": sum(not item["complete"] for item in cell_summaries),
        "cells": cell_summaries,
    }
    _atomic_write_json(output / "selection_manifest.json", manifest)
    readme = (
        "# Memory V4 Prompt / GT 样例画廊\n\n"
        f"- 状态：`{state}`\n"
        f"- 固定随机种子：`{seed}`\n"
        f"- 已扫描行数：`{rows_scanned}`\n"
        f"- 已观察 source×profile×stage：`{len(cells)}`\n"
        f"- 保存样本：`{len(selections)}`\n"
        f"- 不足 {samples_per_cell} 条的已观察分组：`{manifest['incomplete_cells']}`\n\n"
        "每条 `.json` 是源 JSONL 行的逐字副本；同名 `.md` 展示其准确 Prompt、"
        "Assistant GT 和图片引用。以 `selection_manifest.json` 为权威索引。\n"
    )
    _atomic_write_text(output / "README.md", readme)
    _write_status(
        output, state=state, started_at=started_at, files_total=len(files),
        files_completed=files_completed, rows_scanned=rows_scanned,
        cells_observed=len(cells), retained=len(selections),
        manifest=str(output / "selection_manifest.json"),
    )
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--samples-per-cell", type=int, default=3)
    parser.add_argument("--max-rows", type=int, help="bounded provisional gallery/debug only")
    parser.add_argument("--status-interval-seconds", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_gallery(
        input_roots=args.input_root,
        output_root=args.output_root,
        seed=args.seed,
        samples_per_cell=args.samples_per_cell,
        max_rows=args.max_rows,
        status_interval_seconds=args.status_interval_seconds,
        workers=args.workers,
    )
    print(json.dumps({
        key: result[key]
        for key in ("state", "rows_scanned", "cells_observed", "selections", "incomplete_cells")
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
