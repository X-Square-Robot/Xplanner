#!/usr/bin/env python3
"""Run causal V10 inference over one complete episode and render a video demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Iterator

from .constants import (
    DEFAULT_DONE_THRESHOLD,
    DEFAULT_SHORT_MEMORY_K,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_STABLE_STEPS,
    DEFAULT_VISIBLE_LONG_MEMORY_LIMIT,
    UNIT_LEVEL,
)
from .hierarchy import EpisodeValidationError, build_target
from .memory import MemoryBank, MemoryCodec
from .metrics import ScoreAccumulator, TargetScore, score_target_text
from .models import CanonicalEpisode, V10Sample
from .schema import dumps_target, loads_target
from .snapshot import atomic_write_json
from .validation_runtime import (
    ModelGenerator,
    OracleGenerator,
    UNIT_FIELD,
    _observation,
)


USER_ROOT = Path("/mnt/cpfs/zbl-cpfs-new/USERS/luhao")
DEFAULT_CATALOG = USER_ROOT / (
    "checkpoint/v10_continuous/merged/catalog_snapshots/"
    "merged-allvalid-zhengwei008463-open624214-20260806T1431Z"
)
DEFAULT_RUNS_ROOT = USER_ROOT / "checkpoint/v10_continuous/runs"
DEFAULT_PROCESSOR = USER_ROOT / "models/Qwen3.5-9B"
DEFAULT_EPISODE_KEY = (
    "robotwin30_arx_x5/cpfs/zbl-cpfs-new/share/qudelin/DATA/"
    "robotwin30_x2/arx_x5/build_tower/trajectory_44"
)
REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "v10_checkpoint_meta.json",
    "x2_sampler_state.json",
)


@dataclass(frozen=True, slots=True)
class EpisodeBundle:
    episode: CanonicalEpisode
    samples: tuple[V10Sample, ...]
    catalog_row: dict[str, Any]
    shard_path: Path


@dataclass(frozen=True, slots=True)
class EpisodeMemoryConfig:
    """Inference-time memory controls shared by prompt rendering and MemoryBank."""

    done_threshold: int = DEFAULT_DONE_THRESHOLD
    stable_steps: int = DEFAULT_STABLE_STEPS
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    short_memory_k: int = DEFAULT_SHORT_MEMORY_K
    visible_long_memory_limit: int = DEFAULT_VISIBLE_LONG_MEMORY_LIMIT
    vocabulary: str = "open"

    def __post_init__(self) -> None:
        if not 0 <= self.done_threshold <= 100:
            raise ValueError("memory done_threshold must be in [0, 100]")
        if self.stable_steps < 2:
            raise ValueError("memory stable_steps must be at least 2")
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("memory similarity_threshold must be in [0, 1]")
        if self.short_memory_k not in {1, 2}:
            raise ValueError("memory short_memory_k must be 1 or 2")
        if self.visible_long_memory_limit <= 0:
            raise ValueError("memory visible_long_memory_limit must be positive")
        if self.vocabulary not in {"open", "closed"}:
            raise ValueError("memory vocabulary must be 'open' or 'closed'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "done_threshold": self.done_threshold,
            "stable_steps": self.stable_steps,
            "similarity_threshold": self.similarity_threshold,
            "short_memory_k": self.short_memory_k,
            "visible_long_memory_limit": self.visible_long_memory_limit,
            "vocabulary": self.vocabulary,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EpisodeMemoryConfig":
        return cls(
            done_threshold=int(value["done_threshold"]),
            stable_steps=int(value["stable_steps"]),
            similarity_threshold=float(value["similarity_threshold"]),
            short_memory_k=int(value["short_memory_k"]),
            visible_long_memory_limit=int(value["visible_long_memory_limit"]),
            vocabulary=str(value["vocabulary"]),
        )

    def codec(self) -> MemoryCodec:
        return MemoryCodec(
            short_memory_k=self.short_memory_k,
            visible_long_memory_limit=self.visible_long_memory_limit,
            similarity_threshold=self.similarity_threshold,
        )

    def canonical_captions(self, episode: CanonicalEpisode) -> tuple[str, ...]:
        if self.vocabulary == "open":
            return ()
        level = UNIT_LEVEL[episode.profile]
        return tuple(unit.caption for unit in episode.levels[level])

    def bank(self, episode: CanonicalEpisode) -> MemoryBank:
        bank = MemoryBank(
            codec=self.codec(),
            done_threshold=self.done_threshold,
            stable_steps=self.stable_steps,
        )
        bank.reset(
            episode.episode_key,
            canonical_captions=self.canonical_captions(episode),
        )
        return bank


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_lines(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"expected JSON object in {path}")
                yield value


def load_episode_bundle(catalog_snapshot: Path, episode_key: str) -> EpisodeBundle:
    """Resolve one accepted catalog record to its canonical episode shard."""

    catalog_snapshot = catalog_snapshot.resolve()
    accepted = catalog_snapshot / "accepted.jsonl"
    if not accepted.is_file():
        raise FileNotFoundError(accepted)
    matches = [
        row for row in _json_lines(accepted)
        if str(row.get("episode_key", "")) == episode_key
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one accepted row for {episode_key!r}, got {len(matches)}"
        )
    row = matches[0]
    if row.get("status") != "accepted":
        raise ValueError(f"catalog row is not accepted: {row.get('status')!r}")
    shard_path = Path(str(row.get("shard_path", ""))).resolve()
    if not shard_path.is_file():
        raise FileNotFoundError(shard_path)
    payload = json.loads(shard_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"episode shard must be a JSON object: {shard_path}")
    episode = CanonicalEpisode.from_dict(payload["episode"])
    samples = tuple(
        sorted(
            (V10Sample.from_dict(value) for value in payload["samples"]),
            key=lambda sample: (sample.unit_index, sample.current_frame, sample.sample_id),
        )
    )
    if episode.episode_key != episode_key:
        raise ValueError("catalog episode key differs from shard episode key")
    if not samples:
        raise ValueError("episode shard contains no V10 samples")
    if any(sample.episode_key != episode_key for sample in samples):
        raise ValueError("episode shard contains a sample from another episode")
    if any(sample.profile != episode.profile for sample in samples):
        raise ValueError("episode/sample profile mismatch")
    if any(image.frame > sample.current_frame for sample in samples for image in sample.images):
        raise ValueError("episode sample contains a future image")
    expected_samples = int(row.get("num_samples", -1))
    if expected_samples != len(samples):
        raise ValueError(
            f"catalog/shard sample mismatch: catalog={expected_samples} shard={len(samples)}"
        )
    return EpisodeBundle(episode, samples, row, shard_path)


def _checkpoint_paths(runs_root: Path) -> tuple[Path, ...]:
    paths = [
        *runs_root.glob("merged_*/train/checkpoint-*"),
        *runs_root.glob("aihc/*/train/checkpoint-*"),
    ]
    # Do not stat candidates here: a root-owned AIHC checkpoint may be visible
    # from its parent but deny traversal.  _checkpoint_record converts that
    # candidate into an auditable rejection instead of aborting discovery.
    return tuple(sorted({path.resolve() for path in paths}))


def _checkpoint_record(
    path: Path,
    *,
    min_step: int,
    minimum_large_file_bytes: int,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        step = int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        return None, "invalid checkpoint directory name"
    if step < min_step:
        return None, f"step {step} is below minimum {min_step}"
    try:
        missing = [
            name for name in REQUIRED_CHECKPOINT_FILES
            if not (path / name).is_file()
        ]
        rng_files = sorted(path.glob("rng_state*.pth"))
    except OSError as exc:
        return None, f"checkpoint files are inaccessible: {type(exc).__name__}: {exc}"
    if not rng_files:
        missing.append("rng_state*.pth")
    if missing:
        return None, f"missing: {missing}"
    try:
        for name in ("model.safetensors", "optimizer.pt"):
            if (path / name).stat().st_size < minimum_large_file_bytes:
                return None, f"{name} is too small"
        for name in ("config.json", "model.safetensors"):
            if not os.access(path / name, os.R_OK):
                return None, f"{name} is not readable by the current user"
    except OSError as exc:
        return None, f"checkpoint metadata is inaccessible: {type(exc).__name__}: {exc}"

    metadata_readable = False
    metadata_step: int | None = None
    try:
        metadata = json.loads((path / "v10_checkpoint_meta.json").read_text(encoding="utf-8"))
        metadata_step = int(metadata["global_step"])
        metadata_readable = True
    except PermissionError:
        metadata = None
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return None, f"invalid v10 metadata: {type(exc).__name__}: {exc}"
    if metadata_step is not None and metadata_step != step:
        return None, f"metadata step mismatch: {metadata_step} != {step}"
    try:
        completion_stat = (path / "v10_checkpoint_meta.json").stat()
        model_stat = (path / "model.safetensors").stat()
    except OSError as exc:
        return None, f"checkpoint stat failed: {type(exc).__name__}: {exc}"
    return {
        "path": str(path),
        "global_step": step,
        "completion_mtime_ns": completion_stat.st_mtime_ns,
        "completion_time_utc": datetime.fromtimestamp(
            completion_stat.st_mtime_ns / 1_000_000_000, tz=timezone.utc
        ).isoformat(),
        "model_size": model_stat.st_size,
        "model_mtime_ns": model_stat.st_mtime_ns,
        "rng_files": len(rng_files),
        "metadata_readable": metadata_readable,
    }, None


def select_latest_checkpoint(
    runs_root: Path,
    *,
    min_step: int = 1000,
    minimum_large_file_bytes: int = 1_000_000_000,
) -> tuple[Path, dict[str, Any]]:
    """Select the most recently completed, locally readable V10 checkpoint."""

    complete: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    for path in _checkpoint_paths(runs_root.resolve()):
        record, reason = _checkpoint_record(
            path,
            min_step=min_step,
            minimum_large_file_bytes=minimum_large_file_bytes,
        )
        if record is None:
            rejected.append({"path": str(path), "reason": str(reason)})
        else:
            complete.append(record)
    if not complete:
        raise RuntimeError(f"no complete readable V10 checkpoint under {runs_root}")
    selected = max(
        complete,
        key=lambda row: (int(row["completion_mtime_ns"]), int(row["global_step"])),
    )
    report = {
        "selected_at": _utc_now(),
        "selection_rule": (
            "newest complete readable v10_checkpoint_meta mtime; global_step tie-breaker"
        ),
        "selected": selected,
        "complete_candidate_count": len(complete),
        "rejected_candidate_count": len(rejected),
        "rejected": rejected[:100],
    }
    return Path(str(selected["path"])), report


def validate_explicit_checkpoint(path: Path) -> tuple[Path, dict[str, Any]]:
    path = path.resolve()
    record, reason = _checkpoint_record(
        path,
        min_step=0,
        minimum_large_file_bytes=1_000_000_000,
    )
    if record is None:
        raise RuntimeError(f"explicit checkpoint is not complete/readable: {path}: {reason}")
    return path, {
        "selected_at": _utc_now(),
        "selection_rule": "explicit checkpoint after full V10 completeness validation",
        "selected": record,
        "complete_candidate_count": 1,
        "rejected_candidate_count": 0,
        "rejected": [],
    }


def _partial_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".partial")


def _load_prior_rows(output_path: Path, resume: bool) -> list[dict[str, Any]]:
    partial = _partial_path(output_path)
    source = partial if partial.is_file() else output_path if output_path.is_file() else None
    if source is None:
        return []
    if not resume:
        raise FileExistsError(f"output already exists and --no-resume was requested: {source}")
    return list(_json_lines(source))


def _memory_update_dict(update: Any) -> dict[str, Any]:
    return {
        "committed": update.committed,
        "transitioned": update.transitioned,
        "reason": update.reason,
        "active_caption": update.active_caption,
        "active_stable_steps": update.active_stable_steps,
        "long_memory": list(update.long_memory),
    }


def _apply_saved_row(
    bank: MemoryBank,
    sample: V10Sample,
    row: dict[str, Any],
    memory_config: EpisodeMemoryConfig,
    legacy_memory_config: dict[str, Any] | None,
) -> None:
    expected_memory = list(bank.long_memory)
    if row.get("sample_id") != sample.sample_id:
        raise ValueError(
            f"resume sample mismatch: {row.get('sample_id')!r} != {sample.sample_id!r}"
        )
    if row.get("input_long_memory") != expected_memory:
        raise ValueError(
            f"resume memory mismatch for {sample.sample_id}: "
            f"saved={row.get('input_long_memory')} replayed={expected_memory}"
        )
    saved_config = row.get("memory_config")
    if saved_config is None:
        if legacy_memory_config is not None:
            if legacy_memory_config != memory_config.to_dict():
                raise ValueError(
                    f"resume memory configuration mismatch for {sample.sample_id}: "
                    f"saved={legacy_memory_config} requested={memory_config.to_dict()}"
                )
        elif memory_config != EpisodeMemoryConfig():
            raise ValueError(
                "legacy prediction row has no memory configuration; "
                "only the default configuration can resume it"
            )
    elif saved_config != memory_config.to_dict():
        raise ValueError(
            f"resume memory configuration mismatch for {sample.sample_id}: "
            f"saved={saved_config} requested={memory_config.to_dict()}"
        )
    if row.get("valid_json"):
        parsed = loads_target(str(row["assistant_json"]), sample.profile)
        bank.step(sample.episode_key, _observation(parsed, sample.unit_type))


def run_episode_inference(
    *,
    bundle: EpisodeBundle,
    output_path: Path,
    generator: Any,
    memory_config: EpisodeMemoryConfig | None = None,
    resume: bool = True,
    stop_after: int = 0,
) -> dict[str, Any]:
    """Run or resume one episode without ever feeding ground truth into MemoryBank."""

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prior_rows = _load_prior_rows(output_path, resume)
    if len(prior_rows) > len(bundle.samples):
        raise ValueError("resume output has more rows than the selected episode")
    memory_config = memory_config or EpisodeMemoryConfig()
    legacy_memory_config: dict[str, Any] | None = None
    metrics_path = output_path.with_suffix(".metrics.json")
    if prior_rows and metrics_path.is_file():
        prior_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if isinstance(prior_metrics.get("memory_config"), dict):
            legacy_memory_config = dict(prior_metrics["memory_config"])
    generator_codec = getattr(generator, "memory_codec", None)
    if isinstance(generator_codec, MemoryCodec):
        expected_codec = memory_config.codec()
        actual = (
            generator_codec.short_memory_k,
            generator_codec.visible_long_memory_limit,
            generator_codec.similarity_threshold,
        )
        expected = (
            expected_codec.short_memory_k,
            expected_codec.visible_long_memory_limit,
            expected_codec.similarity_threshold,
        )
        if actual != expected:
            raise ValueError(
                f"generator and MemoryBank codec configurations differ: "
                f"generator={actual} memory_bank={expected}"
            )
    bank = memory_config.bank(bundle.episode)
    accumulator = ScoreAccumulator()
    for index, row in enumerate(prior_rows):
        sample = bundle.samples[index]
        _apply_saved_row(
            bank,
            sample,
            row,
            memory_config,
            legacy_memory_config,
        )
        accumulator.add(TargetScore(**row["score"]))

    processed = len(prior_rows)
    partial = _partial_path(output_path)
    if processed < len(bundle.samples):
        if output_path.is_file() and not partial.exists():
            os.replace(output_path, partial)
        mode = "a" if partial.is_file() else "w"
        with partial.open(mode, encoding="utf-8") as handle:
            for sample in bundle.samples[processed:]:
                if stop_after > 0 and processed >= stop_after:
                    break
                input_memory = tuple(bank.long_memory)
                started = time.monotonic()
                assistant_json = generator.generate(sample, input_memory)
                generation_seconds = time.monotonic() - started
                score = score_target_text(assistant_json, sample.target, sample.profile)
                field = UNIT_FIELD[sample.unit_type]
                truth = str(sample.target["predictions"][0][field]["caption"])
                text1 = "INVALID JSON"
                update_value: dict[str, Any] | None = None
                parse_error = score.error
                if score.valid_json:
                    parsed = loads_target(assistant_json, sample.profile)
                    text1 = str(parsed["predictions"][0][field]["caption"])
                    update = bank.step(
                        sample.episode_key,
                        _observation(parsed, sample.unit_type),
                    )
                    update_value = _memory_update_dict(update)
                row = {
                    "row_index": processed,
                    "sample_id": sample.sample_id,
                    "episode_key": sample.episode_key,
                    "profile": sample.profile,
                    "unit_type": sample.unit_type,
                    "unit_index": sample.unit_index,
                    "current_frame": sample.current_frame,
                    "image_frames": sorted({image.frame for image in sample.images}),
                    "views": list(dict.fromkeys(image.view for image in sample.images)),
                    "input_long_memory": list(input_memory),
                    "memory_config": memory_config.to_dict(),
                    "assistant_json": assistant_json,
                    "target_json": dumps_target(sample.target, sample.profile),
                    "valid_json": score.valid_json,
                    "parse_error": parse_error,
                    "text1": text1,
                    "text2": truth,
                    "score": score.to_dict(),
                    "memory_update": update_value,
                    "generation_seconds": generation_seconds,
                }
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
                accumulator.add(score)
                processed += 1

    completed = processed == len(bundle.samples)
    if completed and partial.is_file():
        os.replace(partial, output_path)
    rows_path = output_path if completed else partial
    rows = list(_json_lines(rows_path))
    memory_reasons: dict[str, int] = {}
    commits = transitions = nonempty_inputs = 0
    total_generation_seconds = 0.0
    for row in rows:
        total_generation_seconds += float(row.get("generation_seconds", 0.0))
        nonempty_inputs += int(bool(row.get("input_long_memory")))
        update = row.get("memory_update")
        if isinstance(update, dict):
            reason = str(update.get("reason", "unknown"))
            memory_reasons[reason] = memory_reasons.get(reason, 0) + 1
            commits += int(bool(update.get("committed")))
            transitions += int(bool(update.get("transitioned")))
    result = {
        "episode_key": bundle.episode.episode_key,
        "profile": bundle.episode.profile,
        "unit_type": bundle.episode.unit_type,
        "completed": completed,
        "samples": processed,
        "expected_samples": len(bundle.samples),
        "output": str(rows_path),
        **accumulator.report(name="episode_rollout_score"),
        "memory_commits": commits,
        "memory_transitions": transitions,
        "nonempty_memory_inputs": nonempty_inputs,
        "memory_reasons": memory_reasons,
        "memory_config": memory_config.to_dict(),
        "generation_seconds": total_generation_seconds,
    }
    atomic_write_json(output_path.with_suffix(".metrics.json"), result)
    return result


def _font(size: int):
    from PIL import ImageFont

    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _wrap_text(draw: Any, text: str, font: Any, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        width = draw.textbbox((0, 0), candidate, font=font)[2]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _fit_image(image: Any, width: int, height: int):
    from PIL import Image

    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), "black")
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def _frame_overlay_json(
    episode: CanonicalEpisode,
    frame_index: int,
    active: dict[str, Any] | None,
) -> tuple[str, str]:
    if active is None:
        prediction = json.dumps(
            {"status": "waiting_for_first_causal_prediction"},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    elif active.get("valid_json"):
        prediction = str(active["assistant_json"]).strip()
        json.loads(prediction)
    else:
        prediction = json.dumps(
            {
                "error": "invalid_model_json",
                "raw": str(active.get("assistant_json", "")),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    try:
        target, _unit_index = build_target(episode, frame_index)
        truth = dumps_target(target, episode.profile)
    except EpisodeValidationError:
        truth = json.dumps(
            {"error": "unannotated_frame", "frame": frame_index},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return prediction, truth


def _video_rate(stream: Any) -> Fraction:
    if stream.average_rate is not None:
        return Fraction(stream.average_rate)
    return Fraction(25, 1)


def render_episode_demo(
    *,
    bundle: EpisodeBundle,
    rows: Iterable[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Render every source frame with causal prediction and per-frame ground truth."""

    import av
    from PIL import Image, ImageDraw

    rows = tuple(sorted(rows, key=lambda row: int(row["current_frame"])))
    if len(rows) != len(bundle.samples):
        raise ValueError(
            f"video render requires all episode predictions: {len(rows)} != {len(bundle.samples)}"
        )
    views = tuple(bundle.episode.videos)
    if not views:
        raise ValueError("episode has no videos")
    containers = [av.open(bundle.episode.videos[view], mode="r") for view in views]
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".partial" + output_path.suffix)
    try:
        streams = [container.streams.video[0] for container in containers]
        fps = _video_rate(streams[0])
        for stream in streams[1:]:
            if abs(float(_video_rate(stream)) - float(fps)) > 1e-6:
                raise ValueError("episode view FPS values differ")
        decoders = [iter(container.decode(stream)) for container, stream in zip(containers, streams)]
        first_frames = [next(decoder) for decoder in decoders]
        first_images = [frame.to_image().convert("RGB") for frame in first_frames]
        cell_width = max(image.width for image in first_images)
        cell_height = max(image.height for image in first_images)
        panel_height = 420
        canvas_width = cell_width * len(views)
        canvas_height = cell_height + panel_height
        canvas_width += canvas_width % 2
        canvas_height += canvas_height % 2
        output = av.open(
            str(temporary), mode="w", format="mp4", options={"movflags": "+faststart"}
        )
        try:
            stream_out = output.add_stream("libx264", rate=fps)
            stream_out.width = canvas_width
            stream_out.height = canvas_height
            stream_out.pix_fmt = "yuv420p"
            stream_out.options = {"crf": "20", "preset": "medium"}
            font = _font(18)
            small_font = _font(22)
            row_index = -1
            active: dict[str, Any] | None = None
            for frame_index in range(bundle.episode.num_frames):
                if frame_index == 0:
                    video_frames = first_frames
                    images = first_images
                else:
                    try:
                        video_frames = [next(decoder) for decoder in decoders]
                    except StopIteration as exc:
                        raise ValueError(
                            f"a source video ended before frame {bundle.episode.num_frames}"
                        ) from exc
                    images = [frame.to_image().convert("RGB") for frame in video_frames]
                while (
                    row_index + 1 < len(rows)
                    and int(rows[row_index + 1]["current_frame"]) <= frame_index
                ):
                    row_index += 1
                    active = rows[row_index]
                prediction, truth = _frame_overlay_json(
                    bundle.episode, frame_index, active
                )
                canvas = Image.new("RGB", (canvas_width, canvas_height), "black")
                draw = ImageDraw.Draw(canvas)
                for index, (view, image) in enumerate(zip(views, images)):
                    fitted = _fit_image(image, cell_width, cell_height)
                    left = index * cell_width
                    canvas.paste(fitted, (left, 0))
                    draw.rectangle((left + 8, 8, left + 190, 43), fill=(0, 0, 0))
                    draw.text((left + 14, 10), view, font=small_font, fill=(255, 255, 255))
                panel_top = cell_height
                draw.rectangle(
                    (0, panel_top, canvas_width, canvas_height), fill=(16, 16, 16)
                )
                info = (
                    f"frame {frame_index + 1}/{bundle.episode.num_frames} | "
                    f"{float(fps):.2f} FPS | anchor "
                    f"{active['current_frame'] if active is not None else 'pending'}"
                )
                draw.text((20, panel_top + 14), info, font=small_font, fill=(170, 170, 170))
                column_width = canvas_width // 2
                draw.line(
                    (column_width, panel_top + 48, column_width, canvas_height - 12),
                    fill=(80, 80, 80),
                    width=2,
                )
                draw.text(
                    (20, panel_top + 52),
                    "text1 (prediction JSON)",
                    font=small_font,
                    fill=(80, 220, 255),
                )
                draw.text(
                    (column_width + 20, panel_top + 52),
                    "text2 (ground-truth JSON)",
                    font=small_font,
                    fill=(255, 220, 80),
                )
                prediction_lines = _wrap_text(
                    draw, prediction, font, column_width - 40
                )
                truth_lines = _wrap_text(
                    draw, truth, font, column_width - 40
                )
                line_height = 24
                body_y = panel_top + 86
                max_lines = (canvas_height - body_y - 12) // line_height
                if len(prediction_lines) > max_lines or len(truth_lines) > max_lines:
                    raise ValueError(
                        "full JSON overlay exceeds panel capacity: "
                        f"prediction={len(prediction_lines)} truth={len(truth_lines)} "
                        f"max={max_lines}"
                    )
                for index, line in enumerate(prediction_lines):
                    draw.text(
                        (20, body_y + index * line_height),
                        line,
                        font=font,
                        fill=(80, 220, 255),
                    )
                for index, line in enumerate(truth_lines):
                    draw.text(
                        (column_width + 20, body_y + index * line_height),
                        line,
                        font=font,
                        fill=(255, 220, 80),
                    )
                encoded = av.VideoFrame.from_image(canvas)
                encoded.pts = frame_index
                encoded.time_base = Fraction(1, 1) / fps
                for packet in stream_out.encode(encoded):
                    output.mux(packet)
            for decoder in decoders:
                try:
                    next(decoder)
                except StopIteration:
                    pass
                else:
                    raise ValueError("a source video contains more frames than episode.num_frames")
            for packet in stream_out.encode():
                output.mux(packet)
        finally:
            output.close()
    finally:
        for container in containers:
            container.close()
    os.replace(temporary, output_path)
    with av.open(str(output_path), mode="r") as verification:
        stream = verification.streams.video[0]
        decoded = sum(1 for _frame in verification.decode(stream))
        if decoded != bundle.episode.num_frames:
            raise RuntimeError(
                f"rendered video frame mismatch: {decoded} != {bundle.episode.num_frames}"
            )
        report = {
            "output": str(output_path),
            "views": list(views),
            "frames": decoded,
            "fps": float(_video_rate(stream)),
            "width": stream.width,
            "height": stream.height,
            "codec": stream.codec_context.name,
            "pixel_format": stream.codec_context.format.name,
            "size": output_path.stat().st_size,
            "overlay": {
                "text1": "full prediction JSON",
                "text2": "full per-frame ground-truth JSON",
            },
        }
    atomic_write_json(output_path.with_suffix(".video.json"), report)
    return report


def _run_fingerprint(
    *,
    bundle: EpisodeBundle,
    checkpoint: Path,
    processor_path: Path,
    max_new_tokens: int,
    memory_config: EpisodeMemoryConfig | None,
) -> str:
    value = {
        "episode_key": bundle.episode.episode_key,
        "shard_path": str(bundle.shard_path),
        "checkpoint": str(checkpoint),
        "processor_path": str(processor_path),
        "max_new_tokens": max_new_tokens,
        "samples": [sample.sample_id for sample in bundle.samples],
    }
    if memory_config is not None:
        value["memory"] = memory_config.to_dict()
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _default_output() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return USER_ROOT / f"checkpoint/v10_continuous/evaluations/episode_demo_{stamp}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-snapshot", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--episode-key", default=DEFAULT_EPISODE_KEY)
    parser.add_argument("--checkpoint", default="auto")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--processor-path", type=Path, default=DEFAULT_PROCESSOR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--memory-done-threshold", type=int, default=DEFAULT_DONE_THRESHOLD,
        help="commit eligibility progress threshold in [0, 100] (default: 80)",
    )
    parser.add_argument(
        "--memory-stable-steps", type=int, default=DEFAULT_STABLE_STEPS,
        help="repeated observations required for a stable unit/transition (default: 2)",
    )
    parser.add_argument(
        "--memory-similarity-threshold", type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="caption equivalence/dedup threshold in [0, 1] (default: 0.8)",
    )
    parser.add_argument(
        "--short-memory-k", type=int, choices=(1, 2), default=DEFAULT_SHORT_MEMORY_K,
        help="number of Long Memory tail entries repeated as Short Memory (default: 1)",
    )
    parser.add_argument(
        "--visible-long-memory-limit", type=int,
        default=DEFAULT_VISIBLE_LONG_MEMORY_LIMIT,
        help="maximum Long Memory entries rendered in the prompt (default: 8)",
    )
    parser.add_argument(
        "--memory-vocabulary", choices=("open", "closed"), default="open",
        help="open is deployable; closed uses GT episode captions and is evaluation-only",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()

    bundle = load_episode_bundle(args.catalog_snapshot, args.episode_key)
    if bundle.episode.split != "validation":
        raise ValueError(
            f"episode demo requires a validation episode, got {bundle.episode.split!r}"
        )
    output_dir = (args.output_dir or _default_output()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    previous: dict[str, Any] | None = None
    run_max_new_tokens = args.max_new_tokens
    memory_config = EpisodeMemoryConfig(
        done_threshold=args.memory_done_threshold,
        stable_steps=args.memory_stable_steps,
        similarity_threshold=args.memory_similarity_threshold,
        short_memory_k=args.short_memory_k,
        visible_long_memory_limit=args.visible_long_memory_limit,
        vocabulary=args.memory_vocabulary,
    )
    if args.render_only:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"--render-only requires an existing run manifest: {manifest_path}"
            )
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = Path(str(previous["checkpoint"])).resolve()
        selection = dict(previous["checkpoint_selection"])
        processor_path = Path(str(previous["processor_path"])).resolve()
        run_max_new_tokens = int(previous["max_new_tokens"])
        if isinstance(previous.get("memory"), dict):
            memory_config = EpisodeMemoryConfig.from_dict(previous["memory"])
    elif args.checkpoint == "auto":
        checkpoint, selection = select_latest_checkpoint(args.runs_root)
        processor_path = args.processor_path.resolve()
    else:
        checkpoint, selection = validate_explicit_checkpoint(Path(args.checkpoint))
        processor_path = args.processor_path.resolve()
    atomic_write_json(output_dir / "checkpoint_selection.json", selection)
    if not processor_path.is_dir():
        raise FileNotFoundError(processor_path)
    if manifest_path.is_file():
        previous = previous or json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint_memory_config: EpisodeMemoryConfig | None = memory_config
    if previous is not None and "memory" not in previous:
        if not args.render_only and memory_config != EpisodeMemoryConfig():
            raise ValueError(
                "legacy output directory has no memory configuration; use a new output directory"
            )
        fingerprint_memory_config = None
    fingerprint = _run_fingerprint(
        bundle=bundle,
        checkpoint=checkpoint,
        processor_path=processor_path,
        max_new_tokens=run_max_new_tokens,
        memory_config=fingerprint_memory_config,
    )
    if manifest_path.is_file():
        assert previous is not None
        if previous.get("run_fingerprint") != fingerprint:
            raise ValueError(
                "output directory belongs to a different episode/checkpoint/configuration"
            )
    else:
        module_path = Path(__file__).resolve()
        manifest = {
            "schema_version": "v10_episode_demo_v2",
            "created_at": _utc_now(),
            "run_fingerprint": fingerprint,
            "episode_key": bundle.episode.episode_key,
            "split": bundle.episode.split,
            "profile": bundle.episode.profile,
            "unit_type": bundle.episode.unit_type,
            "num_frames": bundle.episode.num_frames,
            "samples": len(bundle.samples),
            "views": list(bundle.episode.videos),
            "catalog_snapshot": str(args.catalog_snapshot.resolve()),
            "catalog_manifest_sha256": _sha256(args.catalog_snapshot.resolve() / "manifest.json"),
            "shard_path": str(bundle.shard_path),
            "checkpoint": str(checkpoint),
            "checkpoint_selection": selection,
            "processor_path": str(processor_path),
            "device": args.device,
            "max_new_tokens": run_max_new_tokens,
            "generation": {"do_sample": False, "attention": "sdpa", "thinking": False},
            "memory": memory_config.to_dict(),
            "memory_vocabulary": (
                "open_predictions_only"
                if memory_config.vocabulary == "open"
                else "closed_gt_episode_vocabulary_evaluation_only"
            ),
            "cadence": "all stored training anchors",
            "module": str(module_path),
            "module_sha256": _sha256(module_path),
            "oracle": args.oracle,
        }
        atomic_write_json(manifest_path, manifest)

    predictions_path = output_dir / "predictions.jsonl"
    if args.render_only:
        if not predictions_path.is_file():
            raise FileNotFoundError(
                f"--render-only requires completed predictions: {predictions_path}"
            )
        metrics_path = predictions_path.with_suffix(".metrics.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        generator = (
            OracleGenerator()
            if args.oracle
            else ModelGenerator(
                checkpoint,
                processor_path,
                max_new_tokens=run_max_new_tokens,
                device=args.device,
                attn_implementation="sdpa",
                memory_codec=memory_config.codec(),
            )
        )
        metrics = run_episode_inference(
            bundle=bundle,
            output_path=predictions_path,
            generator=generator,
            memory_config=memory_config,
            resume=args.resume,
        )
    video = None
    if args.render:
        if not metrics["completed"]:
            raise RuntimeError("refusing to render an incomplete episode rollout")
        video = render_episode_demo(
            bundle=bundle,
            rows=_json_lines(predictions_path),
            output_path=output_dir / "demo.mp4",
        )
    result = {
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint),
        "metrics": metrics,
        "video": video,
    }
    atomic_write_json(output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
