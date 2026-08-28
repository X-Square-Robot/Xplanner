"""V10-only resilient video-frame loading with whole-view degradation.

The scanner proves that every video can be decoded sequentially.  Training,
however, requests a short random-access window through ``X2RobotFrameDecoder``;
irregular timestamps can therefore still make one camera fail at runtime.  This
processor keeps the two contracts aligned:

* explicitly quarantined samples fail before media I/O and are handled by the
  dataset's DDP-safe bad-sample fallback;
* a failed camera removes that camera's complete temporal sequence;
* at least one fully decoded real camera is still required;
* retained reference positions are passed to the V10 text processor so Prompt
  view declarations and ``<image>`` markers exactly match the tensors.
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from PIL import Image

from scripts.train.final_vqa_smoke import (
    VideoFrameVisionProcessor,
    iter_multimodal_image_refs,
)
from x2robot_dataset_v2.processors.vision.base import register_vision_processor
from x2robot_dataset_v2.processors.vision.multimodal_jsonl_vision_processor import (
    VIDEO_META_KEY,
    VIDEO_OBSERVATIONS_KEY,
    _resolve_multimodal_sample_idx,
)
from x2robot_dataset_v2.readers.multimodal_jsonl_reader import load_indexed_jsonl_item


class V10QuarantinedSampleError(RuntimeError):
    """Raised before decode for a sample explicitly listed in quarantine."""


def load_quarantined_sample_ids(path: str | os.PathLike[str] | None) -> frozenset[str]:
    """Load ``{"sample_ids": [...]}`` or ``{"samples": [...]}`` quarantine JSON."""
    if not path:
        return frozenset()
    quarantine = Path(path).expanduser()
    if not quarantine.is_file():
        return frozenset()
    value = json.loads(quarantine.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        values = value.get("sample_ids", value.get("samples", ()))
    elif isinstance(value, list):
        values = value
    else:
        raise TypeError(f"invalid V10 quarantine document: {type(value).__name__}")
    if not isinstance(values, list):
        raise TypeError("V10 quarantine sample_ids must be a list")
    sample_ids: set[str] = set()
    for item in values:
        if isinstance(item, str):
            sample_id = item
        elif isinstance(item, dict):
            sample_id = item.get("sample_id", "")
        else:
            raise TypeError(f"invalid V10 quarantine entry: {item!r}")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"invalid V10 quarantine sample_id: {sample_id!r}")
        sample_ids.add(sample_id.strip())
    return frozenset(sample_ids)


def decode_refs_by_complete_view(
    refs: Iterable[Any],
    jsonl_path: str,
    load_fn: Callable[[Iterable[Any], str], list[Image.Image]],
) -> tuple[list[int], list[Image.Image], dict[str, str]]:
    """Decode one complete view at a time, restoring time-major input order."""
    refs = list(refs)
    grouped: "OrderedDict[str, list[tuple[int, Any]]]" = OrderedDict()
    for position, ref in enumerate(refs):
        if not isinstance(ref, dict):
            raise TypeError(
                f"V10 media ref at position {position} must be a mapping, "
                f"got {type(ref).__name__}"
            )
        view = ref.get("view")
        if not isinstance(view, str) or not view.strip():
            raise ValueError(f"V10 media ref at position {position} has no view")
        grouped.setdefault(view.strip(), []).append((position, ref))

    decoded_by_position: list[tuple[int, Image.Image]] = []
    failures: dict[str, str] = {}
    for view, entries in grouped.items():
        # Static sampled frames (notably reviewed Replan CoT frames) keep a
        # mandatory ``view`` for whole-view failure handling and put their
        # image path in ``path``.  The lower-level loader already supports
        # string image paths, while video-frame mappings stay unchanged.
        view_refs = [
            ref["path"]
            if (
                "video" not in ref
                and isinstance(ref.get("path"), str)
                and ref["path"].strip()
            )
            else ref
            for _position, ref in entries
        ]
        try:
            view_images = load_fn(view_refs, jsonl_path)
            if len(view_images) != len(entries):
                raise RuntimeError(
                    f"view {view} returned {len(view_images)} images for "
                    f"{len(entries)} references"
                )
            decoded_by_position.extend(
                (position, image)
                for (position, _ref), image in zip(entries, view_images)
            )
        except Exception as exc:
            failures[view] = f"{type(exc).__name__}: {exc}"

    decoded_by_position.sort(key=lambda pair: pair[0])
    positions = [position for position, _image in decoded_by_position]
    images = [image for _position, image in decoded_by_position]
    return positions, images, failures


def _append_jsonl(path: str, record: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass
        try:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        finally:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


@register_vision_processor("v10_resilient_video_frame")
class V10ResilientVideoFrameVisionProcessor(VideoFrameVisionProcessor):
    """Decode V10 media while allowing a broken camera, never fake imagery."""

    def __init__(
        self,
        *args: Any,
        quarantine_path: str = "",
        media_failure_report_path: str = "",
        drop_failed_views: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.quarantine_path = str(quarantine_path or "")
        self.media_failure_report_path = str(media_failure_report_path or "")
        self.drop_failed_views = bool(drop_failed_views)
        self._quarantine_ids: frozenset[str] | None = None
        self._reported_media_events: set[tuple[str, str, str]] = set()

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["_reported_media_events"] = set()
        return state

    def _quarantined_ids(self) -> frozenset[str]:
        if self._quarantine_ids is None:
            self._quarantine_ids = load_quarantined_sample_ids(self.quarantine_path)
        return self._quarantine_ids

    def _report_once(
        self,
        *,
        sample_id: str,
        episode_path: str,
        event: str,
        view: str = "",
        error: str = "",
    ) -> None:
        key = (sample_id, event, view)
        if key in self._reported_media_events:
            return
        self._reported_media_events.add(key)
        _append_jsonl(
            self.media_failure_report_path,
            {
                "event": event,
                "ts": time.time(),
                "pid": os.getpid(),
                "rank": os.environ.get("RANK"),
                "local_rank": os.environ.get("LOCAL_RANK"),
                "sample_id": sample_id,
                "episode_path": episode_path,
                "view": view,
                "error": error,
            },
        )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        episode = data.get("_episode")
        if episode is None:
            return data

        frame_local_idx = data.get("_frame_local_idx", 0)
        sample_idx = _resolve_multimodal_sample_idx(episode, frame_local_idx)
        data["frame_idx"] = sample_idx
        data["uid"] = episode.path
        item = load_indexed_jsonl_item(episode.path, sample_idx)
        raw_sample = item.get("v10_sample")
        sample_id = str(
            item.get("data_id")
            or (raw_sample.get("sample_id") if isinstance(raw_sample, dict) else "")
            or f"{episode.path}:{sample_idx}"
        )
        if sample_id in self._quarantined_ids():
            self._report_once(
                sample_id=sample_id,
                episode_path=str(episode.path),
                event="quarantined_sample",
            )
            raise V10QuarantinedSampleError(f"V10 sample is quarantined: {sample_id}")

        refs = list(iter_multimodal_image_refs(item.get("image")))
        positions, images, failures = decode_refs_by_complete_view(
            refs, episode.path, self.load_image_refs
        )
        for view, error in failures.items():
            self._report_once(
                sample_id=sample_id,
                episode_path=str(episode.path),
                event="dropped_view",
                view=view,
                error=error,
            )
        if failures and not self.drop_failed_views:
            detail = "; ".join(f"{view}={error}" for view, error in failures.items())
            raise RuntimeError(f"V10 view decode failed: {detail}")
        if not images:
            detail = "; ".join(f"{view}={error}" for view, error in failures.items())
            raise RuntimeError(
                f"V10 sample has no fully decodable camera view: {sample_id}; {detail}"
            )

        data["_v10_kept_image_positions"] = positions
        data["_v10_dropped_views"] = sorted(failures)
        is_train = data.get("_is_train", True)
        rng = data.get("_augmentation_rng") or data.get("_rng")
        self._aug_seed = rng.getrandbits(32) if rng is not None else None
        saved_max_pixels = self.max_pixels
        if self.max_pixels_split_by_images and len(images) > 1:
            self.max_pixels = max(self.min_pixels, self.max_pixels // len(images))
        try:
            result = self.process_multimodal(
                images,
                episode_type=episode.episode_type,
                is_train=is_train,
            )
        finally:
            self.max_pixels = saved_max_pixels
            self._aug_seed = None
        data.update(result)

        if result.get("orig_height", 0) > 0 and result.get("orig_width", 0) > 0:
            data["_grounding_resize_info"] = {
                "orig_height": result["orig_height"],
                "orig_width": result["orig_width"],
                "resized_height": result["resized_height"],
                "resized_width": result["resized_width"],
            }

        videos, video_metas = self._process_videos(episode, item)
        data[VIDEO_OBSERVATIONS_KEY] = videos
        data[VIDEO_META_KEY] = json.dumps(video_metas)
        return data


__all__ = [
    "V10QuarantinedSampleError",
    "V10ResilientVideoFrameVisionProcessor",
    "decode_refs_by_complete_view",
    "load_quarantined_sample_ids",
]
