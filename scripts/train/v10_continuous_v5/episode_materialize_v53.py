"""Build V5.3 initial/execution/end samples from one labelled episode.

This module is intentionally source-agnostic.  Source scanners resolve media
and provide Action/Segment intervals; this module owns the model-visible wire
contract, context variants, and exact video-end End target.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
from typing import Any

from .context_v53 import clean_context_variants, noisy_context
from .schema_v5 import SCHEMA_VERSION_V53, output_profile_id, validate_sample


PROFILES = ("action_only", "segment_only", "action_segment_joint")


def _stable_id(*parts: object) -> str:
    payload = "\0".join(map(str, parts)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def _unit(caption: str, progress: int) -> dict[str, Any]:
    return {
        "available": True,
        "caption": caption,
        "progress_percent": max(0, min(100, int(progress))),
    }


def _progress(frame: int, interval: Mapping[str, Any]) -> int:
    start = int(interval["start_frame"])
    end = int(interval["end_frame"])
    return max(0, min(100, round((frame - start) * 100 / max(end - start - 1, 1))))


def _task_progress(frame: int, total_frames: int) -> int:
    return max(0, min(100, round(frame * 100 / max(total_frames - 1, 1))))


def _overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    return max(
        0,
        min(int(left["end_frame"]), int(right["end_frame"]))
        - max(int(left["start_frame"]), int(right["start_frame"])),
    )


def _joint_units(
    actions: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for action in actions:
        candidates = [segment for segment in segments if _overlap(action, segment)]
        if not candidates:
            continue
        # A single synchronized anchor must have both labels.  Prefer the
        # longest overlap and use ordering only as a deterministic tie break.
        segment = max(
            candidates,
            key=lambda item: (
                _overlap(action, item),
                -int(item["start_frame"]),
                -int(item["end_frame"]),
            ),
        )
        start = max(int(action["start_frame"]), int(segment["start_frame"]))
        end = min(int(action["end_frame"]), int(segment["end_frame"]))
        result.append({
            "action": dict(action),
            "segment": dict(segment),
            "start_frame": start,
            "end_frame": end,
        })
    return result


def _profile_units(
    profile: str,
    actions: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if profile == "action_only":
        return [
            {
                "action": dict(item),
                "start_frame": int(item["start_frame"]),
                "end_frame": int(item["end_frame"]),
            }
            for item in actions
        ]
    if profile == "segment_only":
        return [
            {
                "segment": dict(item),
                "start_frame": int(item["start_frame"]),
                "end_frame": int(item["end_frame"]),
            }
            for item in segments
        ]
    if profile == "action_segment_joint":
        return _joint_units(actions, segments)
    raise ValueError(f"unknown output profile: {profile!r}")


def _plan(
    profile: str,
    actions: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]] | None:
    if profile == "segment_only":
        return None
    result: list[dict[str, Any]] = []
    for index, action in enumerate(actions, 1):
        value: dict[str, Any] = {"caption": str(action["caption"])}
        if profile == "action_segment_joint":
            overlapping = [segment for segment in segments if _overlap(action, segment)]
            value["segments"] = [
                {
                    "index": segment_index,
                    "segment": {"caption": str(segment["caption"])},
                }
                for segment_index, segment in enumerate(overlapping, 1)
            ]
        result.append({"index": index, "action": value})
    return result or None


def _history(units: Sequence[Mapping[str, Any]], current_offset: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for raw in units[:current_offset]:
        item: dict[str, Any] = {"index": 0}
        identity: list[str] = []
        for name in ("action", "segment"):
            unit = raw.get(name)
            if isinstance(unit, Mapping):
                caption = str(unit["caption"])
                item[name] = caption
                identity.append(caption.casefold())
            else:
                identity.append("")
        key = tuple(identity)
        if key in identities:
            continue
        identities.add(key)
        result.append(item)
    result = result[-8:]
    for index, item in enumerate(result, 1):
        item["index"] = index
    return result


def _short_memory(
    unit: Mapping[str, Any], frame: int, total_frames: int
) -> dict[str, Any]:
    prediction: dict[str, Any] = {}
    for name in ("action", "segment"):
        value = unit.get(name)
        if isinstance(value, Mapping):
            prediction[name] = _unit(str(value["caption"]), _progress(frame, value))
    return {
        "task_progress_percent": _task_progress(frame, total_frames),
        "prediction1": prediction,
    }


def _views(videos: Mapping[str, str] | Sequence[str], frame: int) -> list[dict[str, Any]]:
    if isinstance(videos, Mapping):
        pairs = sorted((str(name), str(path)) for name, path in videos.items())
    else:
        pairs = []
        for offset, raw in enumerate(videos):
            path = str(raw)
            filename = path.rsplit("/", 1)[-1].casefold()
            view = (
                "head" if "face" in filename
                else "left_wrist" if "left" in filename
                else "right_wrist" if "right" in filename
                else f"view_{offset}"
            )
            pairs.append((view, path))
        pairs.sort()
    return [{"video": path, "frame": frame, "view": view} for view, path in pairs]


def _prediction(
    index: int,
    unit: Mapping[str, Any] | None,
    frame: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "index": index,
        "role": "current" if index == 1 else "next",
    }
    if unit is None:
        return result
    for name in ("action", "segment"):
        value = unit.get(name)
        if isinstance(value, Mapping):
            result[name] = _unit(
                str(value["caption"]),
                _progress(frame, value) if index == 1 else 0,
            )
    return result


def _output_spec(
    unit: Mapping[str, Any],
    following: Mapping[str, Any] | None,
    *,
    profile: str,
) -> dict[str, list[str]]:
    present = [name for name in ("action", "segment") if isinstance(unit.get(name), Mapping)]
    next_present = (
        [name for name in ("action", "segment") if isinstance(following.get(name), Mapping)]
        if following is not None else []
    )
    return {
        "prediction1_units": present,
        "prediction2_units": next_present,
        "plan_units": ["action", "segment"] if profile == "action_segment_joint" else ["action"],
    }


def _sample(
    *,
    base_id: str,
    source: str,
    bucket: str,
    category: str,
    context_variant: str,
    split: str,
    output_spec: Mapping[str, Any],
    task_instruction: str,
    images: list[dict[str, Any]],
    prompt_context: Mapping[str, Any],
    target: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION_V53,
        "sample_id": f"{base_id}_{context_variant}",
        "base_sample_id": base_id,
        "source": source,
        "training_bucket": bucket,
        "category": category,
        "context_variant": context_variant,
        "split": split,
        "output_spec": dict(output_spec),
        "output_profile_id": output_profile_id(output_spec),
        "task_instruction": task_instruction,
        "images": images,
        "prompt_context": copy.deepcopy(dict(prompt_context)),
        "target": copy.deepcopy(dict(target)),
        "supervision": {"loss_mask_paths": []},
        "provenance": dict(provenance),
    }
    return validate_sample(value)


def _context_samples(
    *,
    base_id: str,
    source: str,
    bucket: str,
    category: str,
    split: str,
    output_spec: Mapping[str, Any],
    task_instruction: str,
    images: list[dict[str, Any]],
    target: Mapping[str, Any],
    initial_plan: Sequence[Mapping[str, Any]] | None,
    long_memory: Sequence[Mapping[str, Any]],
    short_memory: Mapping[str, Any] | None,
    provenance: Mapping[str, Any],
    missing_variants: list[dict[str, str]],
) -> list[dict[str, Any]]:
    contexts = clean_context_variants(
        initial_plan=initial_plan,
        long_memory=long_memory,
        short_memory=short_memory,
    )
    result: list[dict[str, Any]] = []
    for variant, context in contexts.items():
        clean = _sample(
            base_id=base_id,
            source=source,
            bucket=bucket,
            category=category,
            context_variant=variant,
            split=split,
            output_spec=output_spec,
            task_instruction=task_instruction,
            images=images,
            prompt_context=context,
            target=target,
            provenance=provenance,
        )
        result.append(clean)
        if not variant.startswith("with_memory"):
            continue
        noisy_variant = variant + "_noisy"
        try:
            noisy, noise_meta = noisy_context(
                context,
                context_variant=noisy_variant,
                sample_id=base_id,
                episode_key=str(provenance["episode_key"]),
                output_profile_id=output_profile_id(output_spec),
            )
            result.append(_sample(
                base_id=base_id,
                source=source,
                bucket=bucket,
                category=category,
                context_variant=noisy_variant,
                split=split,
                output_spec=output_spec,
                task_instruction=task_instruction,
                images=images,
                prompt_context=noisy,
                target=target,
                provenance={**dict(provenance), "context_noise": noise_meta},
            ))
        except ValueError as exc:
            missing_variants.append({
                "base_sample_id": base_id,
                "context_variant": noisy_variant,
                "reason": str(exc),
            })
    return result


def materialize_episode(
    *,
    episode_key: str,
    source: str,
    source_group: str,
    task_instruction: str,
    actions: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    videos: Mapping[str, str] | Sequence[str],
    total_frames: int,
    profiles: Sequence[str],
    split: str = "train",
    provenance: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Return all valid physical context rows for one episode.

    ``total_frames`` must come from the video/container metadata (or an
    official media index), not from the final label boundary.  Consequently an
    emitted End anchor is always exactly ``total_frames - 1``.
    """

    if not episode_key or not videos or total_frames <= 0:
        raise ValueError("episode identity, videos, and positive total_frames are required")
    unknown = sorted(set(profiles) - set(PROFILES))
    if unknown:
        raise ValueError(f"unknown profiles: {unknown}")
    common_provenance = {
        "episode_key": episode_key,
        "source_group": source_group,
        "split": split,
        "total_frames": total_frames,
        **dict(provenance or {}),
    }
    samples: list[dict[str, Any]] = []
    missing_variants: list[dict[str, str]] = []
    bucket_override = "robodojo" if source_group == "robodojo" else None

    for profile in profiles:
        units = _profile_units(profile, actions, segments)
        if len(units) <= 3:
            continue
        initial_plan = _plan(profile, actions, segments)
        profile_id = _stable_id(episode_key, profile)
        profile_provenance = {**common_provenance, "label_profile": profile}

        if initial_plan is not None:
            spec = {
                "prediction1_units": [],
                "prediction2_units": [],
                "plan_units": ["action", "segment"] if profile == "action_segment_joint" else ["action"],
            }
            base_id = f"v53_{profile_id}_initial"
            samples.append(_sample(
                base_id=base_id,
                source=source,
                bucket=bucket_override or "initial_plan",
                category="initial_plan",
                context_variant="no_memory_no_initial",
                split=split,
                output_spec=spec,
                task_instruction=task_instruction,
                images=_views(videos, 0),
                prompt_context={},
                target={"initial_plan": initial_plan},
                provenance={**profile_provenance, "anchor_frame": 0, "video_end_exact": False},
            ))

        previous_short: dict[str, Any] | None = None
        for offset, unit in enumerate(units):
            following = units[offset + 1] if offset + 1 < len(units) else None
            anchor = (int(unit["start_frame"]) + int(unit["end_frame"]) - 1) // 2
            spec = _output_spec(unit, following, profile=profile)
            target = {
                "task_progress_percent": _task_progress(anchor, total_frames),
                "predictions": [
                    _prediction(1, unit, anchor),
                    _prediction(2, following, int(following["start_frame"]) if following else anchor),
                ],
                "execution_decision": "Continue",
                "decision_detail": None,
            }
            base_id = f"v53_{profile_id}_ongoing_{offset + 1:04d}"
            samples.extend(_context_samples(
                base_id=base_id,
                source=source,
                bucket=bucket_override or "ongoing",
                category="ongoing",
                split=split,
                output_spec=spec,
                task_instruction=task_instruction,
                images=_views(videos, anchor),
                target=target,
                initial_plan=initial_plan,
                long_memory=_history(units, offset),
                short_memory=previous_short,
                provenance={
                    **profile_provenance,
                    "anchor_frame": anchor,
                    "label_offset": offset,
                    "video_end_exact": False,
                },
                missing_variants=missing_variants,
            ))
            previous_short = _short_memory(unit, anchor, total_frames)

        final = units[-1]
        end_frame = total_frames - 1
        # End is defined by media exhaustion.  The final available labels are
        # carried forward to that exact frame with completed progress.
        end_unit = copy.deepcopy(final)
        for name in ("action", "segment"):
            value = end_unit.get(name)
            if isinstance(value, dict):
                value["end_frame"] = max(end_frame + 1, int(value["end_frame"]))
                value["start_frame"] = min(int(value["start_frame"]), end_frame)
        spec = _output_spec(end_unit, None, profile=profile)
        target = {
            "task_progress_percent": 100,
            "predictions": [
                _prediction(1, end_unit, end_frame),
                _prediction(2, None, end_frame),
            ],
            "execution_decision": "End",
            "decision_detail": {"outcome": "completed"},
        }
        base_id = f"v53_{profile_id}_end"
        samples.extend(_context_samples(
            base_id=base_id,
            source=source,
            bucket=bucket_override or "end",
            category="end",
            split=split,
            output_spec=spec,
            task_instruction=task_instruction,
            images=_views(videos, end_frame),
            target=target,
            initial_plan=initial_plan,
            long_memory=_history(units, len(units) - 1),
            short_memory=previous_short,
            provenance={
                **profile_provenance,
                "anchor_frame": end_frame,
                "label_offset": len(units) - 1,
                "video_end_exact": True,
                "end_frame_source": "media_total_frames",
            },
            missing_variants=missing_variants,
        ))
    return samples, missing_variants


__all__ = ["PROFILES", "materialize_episode"]
