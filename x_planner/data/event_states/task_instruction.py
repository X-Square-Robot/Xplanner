"""Authoritative, fail-closed task-instruction resolution for V5.3."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any


_CJK = re.compile(r"[\u3400-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")
_SPACE = re.compile(r"\s+")
INSTRUCTION_FIELDS = ("task_caption", "instruction")


class TaskInstructionError(ValueError):
    """Raised when no authoritative natural-language task instruction exists."""


@dataclass(frozen=True, slots=True)
class TaskInstructionResolution:
    text: str | None
    source: str | None
    source_path: str | None
    source_field: str | None
    status: str
    checked_paths: tuple[str, ...]
    rejected_candidates: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_task_instruction(value: Any) -> str:
    """Return normalized English text or an empty string for an unusable value."""

    if not isinstance(value, str):
        return ""
    text = _SPACE.sub(" ", value).strip()
    if not text or not _LATIN.search(text) or _CJK.search(text):
        return ""
    return text


def select_record_instruction(
    record: Mapping[str, Any],
    *,
    source_prefix: str,
    source_path: Path | str | None = None,
) -> tuple[str, str, str | None, str]:
    """Select only explicit ``task_caption``/``instruction`` source fields."""

    rejected: list[str] = []
    for field in INSTRUCTION_FIELDS:
        raw = record.get(field)
        if raw is None:
            continue
        text = clean_task_instruction(raw)
        if text:
            return (
                text,
                f"{source_prefix}.{field}",
                str(Path(source_path).resolve()) if source_path is not None else None,
                field,
            )
        rejected.append(field)
    suffix = f"; rejected non-English/empty fields={rejected}" if rejected else ""
    raise TaskInstructionError(
        f"no authoritative task_caption/instruction in {source_prefix}{suffix}"
    )


def _load_mapping(
    path: Path,
    cache: MutableMapping[str, Mapping[str, Any] | None],
    errors: list[str],
) -> Mapping[str, Any] | None:
    key = str(path.resolve())
    if key in cache:
        return cache[key]
    if not path.is_file():
        cache[key] = None
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("root is not an object")
    except Exception as exc:
        errors.append(f"{key}: {type(exc).__name__}: {exc}")
        cache[key] = None
        return None
    cache[key] = value
    return value


def resolve_episode_task_instruction(
    *,
    annotation: Mapping[str, Any],
    annotation_path: Path | str | None = None,
    episode_key: str,
    resolved_episode_path: Path | None,
    json_cache: MutableMapping[str, Mapping[str, Any] | None] | None = None,
) -> TaskInstructionResolution:
    """Resolve a Baseline instruction without deriving text from directory names.

    The compact v2v3umi label is checked first.  If it has no explicit task
    field, the exact episode entry in the media-side ``instruction.json`` is
    used.  The task-directory file is checked before the episode-local file so
    many episodes share a single cached metadata read.
    """

    rejected: list[str] = []
    checked_paths: list[str] = []
    cache = json_cache if json_cache is not None else {}
    try:
        text, source, source_path, field = select_record_instruction(
            annotation,
            source_prefix="label_annotation",
            source_path=annotation_path,
        )
        return TaskInstructionResolution(
            text=text,
            source=source,
            source_path=source_path,
            source_field=field,
            status="resolved",
            checked_paths=(),
            rejected_candidates=(),
        )
    except TaskInstructionError as exc:
        rejected.append(str(exc))

    if resolved_episode_path is not None:
        metadata_paths = (
            resolved_episode_path.parent / "instruction.json",
            resolved_episode_path / "instruction.json",
        )
        seen: set[str] = set()
        for position, path in enumerate(metadata_paths):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            checked_paths.append(key)
            value = _load_mapping(path, cache, rejected)
            if value is None:
                continue
            exact = value.get(episode_key)
            if isinstance(exact, Mapping):
                prefix = (
                    "media_task_instruction_json.episode"
                    if position == 0
                    else "media_episode_instruction_json.episode"
                )
                try:
                    text, source, source_path, field = select_record_instruction(
                        exact,
                        source_prefix=prefix,
                        source_path=path,
                    )
                    return TaskInstructionResolution(
                        text=text,
                        source=source,
                        source_path=source_path,
                        source_field=field,
                        status="resolved",
                        checked_paths=tuple(checked_paths),
                        rejected_candidates=tuple(rejected),
                    )
                except TaskInstructionError as exc:
                    rejected.append(str(exc))
            # Some episode-local files store instruction directly at root.
            prefix = (
                "media_task_instruction_json.root"
                if position == 0
                else "media_episode_instruction_json.root"
            )
            try:
                text, source, source_path, field = select_record_instruction(
                    value,
                    source_prefix=prefix,
                    source_path=path,
                )
                return TaskInstructionResolution(
                    text=text,
                    source=source,
                    source_path=source_path,
                    source_field=field,
                    status="resolved",
                    checked_paths=tuple(checked_paths),
                    rejected_candidates=tuple(rejected),
                )
            except TaskInstructionError as exc:
                rejected.append(str(exc))

    return TaskInstructionResolution(
        text=None,
        source=None,
        source_path=None,
        source_field=None,
        status="missing_task_instruction",
        checked_paths=tuple(checked_paths),
        rejected_candidates=tuple(rejected),
    )


def require_index_task_instruction(row: Mapping[str, Any]) -> tuple[str, str]:
    """Reject stale V5.3 indexes that used a slug or generic fallback."""

    text = clean_task_instruction(row.get("task_instruction"))
    source = str(row.get("task_instruction_source") or "")
    status = str(row.get("task_instruction_status") or "")
    if not text or status != "resolved":
        raise TaskInstructionError("instruction index has no resolved task instruction")
    if not source.endswith((".task_caption", ".instruction")):
        raise TaskInstructionError(f"untrusted task instruction source: {source!r}")
    if source in {"episode_slug", "task_slug", "label_path", "grounded_generic", "index"}:
        raise TaskInstructionError(f"derived task instruction source is forbidden: {source}")
    source_path = row.get("task_instruction_source_path")
    if not isinstance(source_path, str) or not source_path.strip():
        raise TaskInstructionError("instruction index has no authoritative source path")
    return text, source


__all__ = [
    "INSTRUCTION_FIELDS",
    "TaskInstructionError",
    "TaskInstructionResolution",
    "clean_task_instruction",
    "require_index_task_instruction",
    "resolve_episode_task_instruction",
    "select_record_instruction",
]
