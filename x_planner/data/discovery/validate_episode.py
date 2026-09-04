"""Two-stage Episode adaptation, validation, canonicalization, and sampling."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from ..pipeline.adapters import adapt_job
from ..pipeline.captions import normalize_caption
from ..pipeline.hierarchy import EpisodeValidationError, canonicalize_episode
from ..pipeline.models import CanonicalEpisode
from ..pipeline.constants import UNIT_LEVEL
from .common.constants import (
    DEFAULT_MIN_INTERVAL_FRAMES,
    DEFAULT_MIN_L0_COUNT,
    DEFAULT_MIN_L1_COUNT,
)
from .common.errors import ScanError, error_record
from .common.constants import ERROR_TYPES
from .source_discovery import DiscoveredEpisode
from .sampling import build_samples_v2
from .validate_views import resolve_view_candidates, validate_episode_views


def _split_for(key: str, *, seed: int, validation_ratio: float) -> str:
    digest = hashlib.sha256(f"{seed}\0{key}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "validation" if value < validation_ratio else "train"


def _adapt_or_classify(item: DiscoveredEpisode):
    try:
        return adapt_job(item.job)
    except FileNotFoundError as exc:
        path = str(exc.filename or exc)
        if path.endswith(".mp4"):
            error_type = "missing_video"
        elif path == item.job.metadata_path:
            error_type = "missing_required_field"
        else:
            error_type = "missing_annotation"
        raise ScanError(error_type, path, stage="adapt", input_paths=(path,)) from exc
    except ValueError as exc:
        message = str(exc)
        if "missing episode annotation" in message:
            raise ScanError(
                "missing_annotation",
                message,
                stage="adapt",
                input_paths=tuple(item.job.annotation_paths),
            ) from exc
        if "total" in message:
            raise ScanError(
                "missing_required_field",
                message,
                stage="adapt",
                input_paths=(item.job.metadata_path,),
            ) from exc
        raise ScanError("annotation_parse_error", message, stage="adapt") from exc


_WORD_RE = re.compile(r"[a-z0-9]+")
_TASK_STOP_WORDS = frozenset({
    "a", "an", "and", "at", "by", "do", "for", "from", "in", "into", "of",
    "on", "the", "to", "with", "vr", "master", "slave", "mode",
})
_TOKEN_ALIASES = {
    "boxes": "box",
    "folding": "fold",
    "lemonade": "lemon",
    "making": "make",
    "packaging": "package",
    "scallions": "scallion",
}


def _token_forms(token: str) -> set[str]:
    forms = {token, _TOKEN_ALIASES.get(token, token)}
    if len(token) > 4 and token.endswith("ies"):
        forms.add(token[:-3] + "y")
    elif len(token) > 4 and token.endswith("es"):
        forms.add(token[:-2])
    elif len(token) > 3 and token.endswith("s"):
        forms.add(token[:-1])
    if len(token) > 5 and token.endswith("ing"):
        forms.add(token[:-3])
    return forms


def _caption_tokens(value: str) -> set[str]:
    normalized = normalize_caption(value.replace("_", " ").replace("-", " "))
    return {
        form
        for token in _WORD_RE.findall(normalized)
        if token not in _TASK_STOP_WORDS
        for form in _token_forms(token)
    }


def _resolve_l3_caption(
    fallback: str, metadata: Mapping[str, Any]
) -> tuple[str, dict[str, Any] | None]:
    """Prefer the candidate best supported by the explicit task identifier.

    Lexical disagreement is not safe evidence for quarantine: real captions
    use synonyms, hypernyms, and source placeholders.  When the task/path and
    detailed instruction share more normalized terms than the short
    instruction, select the detailed caption and retain the decision as
    provenance.  Ties preserve the adapter's established choice.
    """
    raw = metadata.get("l3_candidates") or {}
    if not isinstance(raw, Mapping):
        return fallback, None
    instruction = str(raw.get("instruction") or "")
    detailed = str(raw.get("detailed_instruction") or "")
    support = str(raw.get("task") or raw.get("task_caption") or "")
    if not instruction or not detailed or not support:
        return fallback, None
    instruction_tokens = _caption_tokens(instruction)
    detailed_tokens = _caption_tokens(detailed)
    support_tokens = _caption_tokens(support)
    instruction_score = len(instruction_tokens.intersection(support_tokens))
    detailed_score = len(detailed_tokens.intersection(support_tokens))
    selected = detailed if detailed_score > instruction_score else fallback
    if selected == fallback:
        return fallback, None
    return selected, {
        "strategy": "task_token_support",
        "selected_source": "detailed_instruction",
        "original_caption": fallback,
        "selected_caption": selected,
        "task_support": support,
        "instruction_score": instruction_score,
        "detailed_instruction_score": detailed_score,
    }


def process_episode(
    item: DiscoveredEpisode,
    *,
    run_id: str,
    shard_id: int,
    worker_id: str,
    settings: Mapping[str, Any],
    view_config: Mapping[str, Any],
    sampling_hash: str,
    quick: bool = False,
    cached_episode: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    failure_stage = "adapt"
    media_realpath = item.media_realpath
    try:
        view_warnings: list[str] = []
        if cached_episode is not None:
            episode = CanonicalEpisode.from_dict(dict(cached_episode))
        else:
            adapted = _adapt_or_classify(item)
            task_caption, l3_resolution = _resolve_l3_caption(
                adapted.task_caption, adapted.metadata
            )
            canonical_args = {
                "source": item.source_id,
                "episode_key": item.episode_key,
                "episode_name": item.job.episode_name,
                "split": _split_for(
                    item.global_episode_key,
                    seed=int(settings.get("seed", 42)),
                    validation_ratio=float(settings.get("validation_ratio", 0.05)),
                ),
                "num_frames": adapted.num_frames,
                "task_caption": task_caption,
                "raw_levels": adapted.raw_levels,
                "annotation_sources": adapted.annotation_sources,
                "min_interval_frames": int(settings.get("min_interval_frames", DEFAULT_MIN_INTERVAL_FRAMES)),
                "min_l1_count": int(settings.get("min_l1_count", DEFAULT_MIN_L1_COUNT)),
                "min_l0_count": int(settings.get("min_l0_count", DEFAULT_MIN_L0_COUNT)),
            }
            # A sentinel mapped view lets hierarchy/profile rejection happen
            # before any video container is opened.
            failure_stage = "canonicalize"
            canonicalize_episode(videos={"head": "__annotation_preflight__"}, **canonical_args)
            failure_stage = "validate"
            candidates, view_warnings = resolve_view_candidates(
                item.job.episode_dir, adapted.video_candidates, view_config
            )
            videos, dropped, probes = validate_episode_views(
                candidates,
                adapted.num_frames,
                max_camera_views=int(settings.get("max_camera_views", 3)),
                quick=bool(settings.get("video_validation", "metadata") == "metadata"),
            )
            failure_stage = "canonicalize"
            episode = canonicalize_episode(
                videos=videos,
                metadata={
                    **adapted.metadata,
                    **({"l3_resolution": l3_resolution} if l3_resolution else {}),
                    "global_episode_key": item.global_episode_key,
                    "dataset_name": item.dataset_name,
                    "dropped_views": dropped,
                    "video_probes": probes,
                    "discovery_warnings": list(item.discovery_warnings),
                    "view_warnings": view_warnings,
                    "media_realpath": media_realpath,
                },
                **canonical_args,
            )
        input_paths = tuple(dict.fromkeys(
            tuple(episode.annotation_sources) + tuple(episode.videos.values())
        ))
        failure_stage = "sample"
        samples = () if quick else build_samples_v2(
            episode,
            source_id=item.source_id,
            global_episode_key=item.global_episode_key,
            dataset_name=item.dataset_name,
            sampling=settings.get("sampling") or {},
            sampling_hash=sampling_hash,
            input_paths=input_paths,
        )
        if not quick:
            for sample in samples:
                sample["media_realpath"] = media_realpath
        return {
            "status": "success",
            "run_id": run_id,
            "source_id": item.source_id,
            "episode_key": item.episode_key,
            "global_episode_key": item.global_episode_key,
            "dataset_name": item.dataset_name,
            "profile": episode.profile,
            "unit_level": UNIT_LEVEL[episode.profile],
            "views": list(episode.videos),
            "media_realpath": media_realpath,
            "input_paths": list(input_paths),
            "sample_count": len(samples),
            "samples": list(samples),
            "attempts": 1,
            **({"canonical_episode": episode.to_dict()} if quick else {}),
        }
    except BaseException as exc:
        if isinstance(exc, EpisodeValidationError):
            error_type = exc.reason if exc.reason in ERROR_TYPES else "sampling_error"
            exc = ScanError(error_type, exc.detail or str(exc), stage=failure_stage)
        record = error_record(
            exc,
            run_id=run_id,
            source_id=item.source_id,
            episode_key=item.episode_key,
            global_episode_key=item.global_episode_key,
            stage=failure_stage,
            worker_id=worker_id,
            shard_id=shard_id,
            input_paths=tuple(item.job.annotation_paths) + (item.job.metadata_path, item.job.episode_dir),
        ).to_dict()
        return {
            "status": "failed",
            "run_id": run_id,
            "source_id": item.source_id,
            "episode_key": item.episode_key,
            "global_episode_key": item.global_episode_key,
            "dataset_name": item.dataset_name,
            "media_realpath": media_realpath,
            "input_paths": record["input_paths"],
            "error": record,
            "retryable": bool(record["retryable"]),
            "attempts": 1,
        }
