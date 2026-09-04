"""Build Memory V3 JSON from immutable V1/V2 Episode artifacts.

This stage deliberately never calls source discovery or video validation.  V1
Episodes are loaded from their accepted episode shards.  V2-only Episodes are
reconstructed from the saved inventory plan plus the already validated view
paths in ``episodes_success.jsonl`` and must match the formal V2 snapshot
Episode contract before any V3 row is emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..pipeline.adapters import adapt_job
from ..pipeline.constants import (
    DEFAULT_MIN_INTERVAL_FRAMES,
    DEFAULT_MIN_L0_COUNT,
    DEFAULT_MIN_L1_COUNT,
)
from ..pipeline.hierarchy import canonicalize_episode
from ..pipeline.models import CanonicalEpisode
from ..discovery.common.atomic import BatchedJsonlWriter
from ..discovery.merge_catalogs import _legacy_roots
from ..discovery.source_discovery import DiscoveredEpisode
from ..discovery.validate_episode import _resolve_l3_caption
from .common import (
    canonical_digest,
    file_sha256,
    iter_jsonl,
    load_config,
    load_episode_contracts,
    mark_success,
    read_success,
    stable_bucket,
    validate_formal_snapshot,
    write_json,
)
from .prompt import render_initial_plan_user
from .sampling import continuous_samples, initial_plan_sample
from .schema import dumps_assistant


PACKAGE_ROOT = Path(__file__).resolve().parent
NORMALIZATION_ROOT = PACKAGE_ROOT.parent / "pipeline"
DISCOVERY_ROOT = PACKAGE_ROOT.parent / "discovery"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _directory_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(file_sha256(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def preflight(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    formal = validate_formal_snapshot(config["input_snapshot"])
    output = Path(str(config["output_root"])).resolve()
    protected = [
        NORMALIZATION_ROOT.resolve(),
        DISCOVERY_ROOT.resolve(),
        Path(formal["root"]),
    ]
    if any(output == root or root in output.parents for root in protected):
        raise ValueError(f"V3 output overlaps protected input: {output}")
    result = {
        "schema_version": "memory_v3_preflight_v1",
        "checked_at": _utc_now(),
        "input_snapshot": formal,
        "protected_inputs": [str(path) for path in protected],
        "mutation_policy": "read_only_inputs; context_output_only",
        "normalization_code_digest": _directory_digest(NORMALIZATION_ROOT),
        "discovery_code_digest": _directory_digest(DISCOVERY_ROOT),
        "config_digest": canonical_digest(config),
        "output_root": str(output),
        "ok": True,
    }
    return result


def _candidate_v1_keys(row: Mapping[str, Any]) -> tuple[str, ...]:
    source = str(row.get("source") or "")
    raw = str(row.get("episode_key") or "").lstrip("/")
    candidates = []
    # The immutable V1 accepted Catalog predates normalized source ids.  In
    # particular, OpenAction rows are labelled v2v3umi and carry a sanitized
    # absolute path. Reuse the exact legacy roots frozen by V2 merge instead
    # of rediscovering the source.
    for normalized_source, root in _legacy_roots():
        root_text = root.as_posix().rstrip("/")
        markers = [root_text.lstrip("/")]
        if "/data/" in root_text:
            markers.append(root_text.split("/data/", 1)[1])
        for marker in markers:
            needle = marker.rstrip("/") + "/"
            position = raw.find(needle)
            if position >= 0:
                relative = raw[position + len(needle):].lstrip("/")
                if relative:
                    candidates.append(f"{normalized_source}:{relative}")
    candidates.append(f"{source}:{raw}")
    if raw.startswith(source + "/"):
        candidates.insert(0, f"{source}:{raw[len(source) + 1:]}")
    return tuple(dict.fromkeys(candidates))


def _plan_path_for_catalog(path: Path) -> Path:
    resolved = path.resolve()
    if len(resolved.parents) < 5:
        raise ValueError(f"unrecognized V2 catalog path: {path}")
    attempt = resolved.parent
    plan_hash = attempt.parent.name
    shard_name = attempt.parent.parent.name
    run_root = attempt.parent.parent.parent.parent
    candidate = run_root / "plans" / shard_name / f"{plan_hash}.jsonl"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _build_episode_index(
    *,
    config: Mapping[str, Any],
    build_root: Path,
    max_episodes: int | None,
    source_filter: set[str] | None,
    num_shards: int,
) -> dict[str, Any]:
    index_root = build_root / "episode_index"
    marker = read_success(index_root)
    if marker is not None:
        return marker
    formal = validate_formal_snapshot(config["input_snapshot"])
    contracts = load_episode_contracts(formal["root"])
    selected: dict[str, dict[str, Any]] = {}

    metadata = json.loads(
        (Path(formal["root"]) / "snapshot_metadata.json").read_text(encoding="utf-8")
    )
    catalogs = [Path(str(value)) for value in metadata["input_catalogs"]]
    for catalog_index, catalog in enumerate(catalogs):
        if max_episodes is not None and len(selected) >= max_episodes:
            break
        if catalog_index == 0:
            for row in iter_jsonl(str(catalog)):
                key = next((item for item in _candidate_v1_keys(row) if item in contracts), None)
                if key is None or key in selected:
                    continue
                contract = contracts[key]
                if source_filter and contract["source_id"] not in source_filter:
                    continue
                accepted_contract = {
                    "profile": str(row.get("profile") or ""),
                    "views": list(row.get("views") or ()),
                    "split": str(row.get("split") or ""),
                }
                if accepted_contract != {
                    name: contract[name] for name in ("profile", "views", "split")
                }:
                    # This is the losing side of an inherited V2 duplicate-key
                    # split conflict.  A matching V2 artifact may be selected
                    # later; never let the wrong-side V1 row leak across split.
                    continue
                shard_path = Path(str(row.get("shard_path") or ""))
                if not shard_path.is_file():
                    continue
                selected[key] = {
                    "kind": "v1_episode_shard",
                    "global_episode_key": key,
                    "episode_shard": str(shard_path),
                    "contract": contract,
                }
                if max_episodes is not None and len(selected) >= max_episodes:
                    break
            continue

        resolved = catalog.resolve()
        success_path = resolved.parent / "episodes_success.jsonl"
        plan_path = _plan_path_for_catalog(catalog)
        success = {
            str(row["global_episode_key"]): row
            for row in iter_jsonl(str(success_path))
            if (
                str(row.get("global_episode_key") or "") in contracts
                and str(row.get("profile") or "")
                == contracts[str(row["global_episode_key"])]["profile"]
                and list(row.get("views") or ())
                == contracts[str(row["global_episode_key"])]["views"]
            )
        }
        if not success:
            continue
        for discovered in iter_jsonl(str(plan_path)):
            key = str(discovered.get("global_episode_key") or "")
            if key not in success or key in selected:
                continue
            contract = contracts[key]
            if source_filter and contract["source_id"] not in source_filter:
                continue
            selected[key] = {
                "kind": "v2_saved_plan",
                "global_episode_key": key,
                "discovered_episode": discovered,
                "success": success[key],
                "catalog": str(catalog),
                "plan": str(plan_path),
                "contract": contract,
            }
            if max_episodes is not None and len(selected) >= max_episodes:
                break

    index_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".episode-index-", dir=index_root.parent))
    writers: dict[int, BatchedJsonlWriter] = {}
    counts: Counter[int] = Counter()
    try:
        for key in sorted(selected):
            shard_id = stable_bucket(key, num_shards)
            writer = writers.get(shard_id)
            if writer is None:
                writer = BatchedJsonlWriter(str(temporary / f"shard-{shard_id:05d}.jsonl"))
                writers[shard_id] = writer
            writer.write(selected[key])
            counts[shard_id] += 1
        for writer in writers.values():
            writer.close()
        payload = {
            "schema_version": "memory_v3_episode_index_v1",
            "created_at": _utc_now(),
            "formal_snapshot_digest": formal["content_digest"],
            "formal_episode_count": len(contracts),
            "selected_episode_count": len(selected),
            "requested_max_episodes": max_episodes,
            "source_filter": sorted(source_filter or ()),
            "num_shards": num_shards,
            "nonempty_shards": len(counts),
            "shard_episode_counts": {str(key): counts[key] for key in sorted(counts)},
        }
        write_json(str(temporary / "manifest.json"), payload)
        mark_success(temporary, payload)
        if index_root.exists():
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, index_root)
        return payload
    except BaseException:
        for writer in writers.values():
            writer.abort()
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_v1_episode(reference: Mapping[str, Any]) -> CanonicalEpisode:
    payload = json.loads(Path(str(reference["episode_shard"])).read_text(encoding="utf-8"))
    raw = payload.get("episode")
    if not isinstance(raw, dict):
        raise ValueError("V1 accepted shard is missing canonical episode")
    return CanonicalEpisode.from_dict(raw)


def _load_v2_episode(reference: Mapping[str, Any]) -> CanonicalEpisode:
    discovered = DiscoveredEpisode.from_dict(reference["discovered_episode"])
    success = reference["success"]
    contract = reference["contract"]
    adapted = adapt_job(discovered.job)
    task_caption, resolution = _resolve_l3_caption(adapted.task_caption, adapted.metadata)
    views = list(success.get("views") or ())
    media_paths = [
        str(path) for path in success.get("input_paths") or ()
        if str(path).lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]
    if len(media_paths) < len(views):
        raise ValueError(f"saved success has {len(views)} views but {len(media_paths)} media paths")
    videos = dict(zip(views, media_paths[-len(views):], strict=True))
    return canonicalize_episode(
        source=discovered.source_id,
        episode_key=discovered.episode_key,
        episode_name=discovered.job.episode_name,
        split=str(contract["split"]),
        num_frames=adapted.num_frames,
        task_caption=task_caption,
        raw_levels=adapted.raw_levels,
        videos=videos,
        annotation_sources=adapted.annotation_sources,
        metadata={
            **adapted.metadata,
            **({"l3_resolution": resolution} if resolution else {}),
            "global_episode_key": discovered.global_episode_key,
            "dataset_name": discovered.dataset_name,
            "v3_reconstructed_from_validated_v2_success": True,
        },
        min_interval_frames=DEFAULT_MIN_INTERVAL_FRAMES,
        min_l1_count=DEFAULT_MIN_L1_COUNT,
        min_l0_count=DEFAULT_MIN_L0_COUNT,
    )


def _check_episode_contract(
    episode: CanonicalEpisode, reference: Mapping[str, Any]
) -> CanonicalEpisode:
    contract = reference["contract"]
    observed = {
        "profile": episode.profile,
        "views": list(episode.videos),
        "split": episode.split,
    }
    expected = {key: contract[key] for key in observed}
    if observed != expected:
        raise ValueError(f"formal Episode contract mismatch: expected={expected} observed={observed}")
    source_id = str(contract["source_id"])
    if episode.source == source_id:
        return episode
    if reference["kind"] != "v1_episode_shard":
        raise ValueError(
            "formal Episode source mismatch: "
            f"expected={source_id!r} observed={episode.source!r}"
        )
    # V1 shards predate the normalized V2 Catalog and legitimately retain
    # aliases such as public/v2v3umi.  The V2 Episode index already resolved
    # their identity, so preserve the legacy value as provenance while exposing
    # only the Catalog's canonical source_id to V3 prompts and metadata.
    return replace(
        episode,
        source=source_id,
        metadata={**episode.metadata, "legacy_source_id": episode.source},
    )


def _load_and_check_episode(reference: Mapping[str, Any]) -> CanonicalEpisode:
    if reference["kind"] == "v1_episode_shard":
        episode = _load_v1_episode(reference)
    elif reference["kind"] == "v2_saved_plan":
        episode = _load_v2_episode(reference)
    else:
        raise ValueError(f"unknown Episode reference kind: {reference['kind']}")
    return _check_episode_contract(episode, reference)


def _context_estimate(sample: Mapping[str, Any], settings: Mapping[str, Any]) -> int:
    prompt_chars = len(render_initial_plan_user(sample))
    answer_chars = len(dumps_assistant(sample["target"], sample["profile"], "initial_plan"))
    chars_per_token = float(settings.get("fallback_chars_per_token", 3.0))
    image_tokens = int(settings.get("image_tokens_per_image", 400)) * len(sample["images"])
    safety_margin = int(settings.get("safety_margin_tokens", 256))
    return (
        int((prompt_chars + answer_chars + chars_per_token - 1) // chars_per_token)
        + image_tokens
        + safety_margin
    )


def _build_shard(payload: Mapping[str, Any]) -> dict[str, Any]:
    shard_id = int(payload["shard_id"])
    shard_root = Path(str(payload["shard_root"]))
    marker = read_success(shard_root)
    if marker is not None:
        return {**marker, "skipped": True}
    shard_root.mkdir(parents=True, exist_ok=True)
    config = payload["config"]
    stride = int(config["anchor_stride_frames"])
    offsets = tuple(int(value) for value in config["history_offsets_frames"])
    initial_settings = config.get("initial_plan") or {}
    max_context = int(initial_settings.get("max_context_tokens", 4096))
    resize_id = str(config["resize"]["policy_id"])
    counts: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    continuous_path = shard_root / "continuous.jsonl"
    initial_path = shard_root / "initial_plan.jsonl"
    terminal_path = shard_root / "terminal_refs.jsonl"
    failure_path = shard_root / "failures.jsonl"
    oversize_path = shard_root / "initial_plan_oversize.jsonl"
    with (
        BatchedJsonlWriter(str(continuous_path)) as continuous_writer,
        BatchedJsonlWriter(str(initial_path)) as initial_writer,
        BatchedJsonlWriter(str(terminal_path)) as terminal_writer,
        BatchedJsonlWriter(str(failure_path)) as failure_writer,
        BatchedJsonlWriter(str(oversize_path)) as oversize_writer,
    ):
        for reference in iter_jsonl(str(payload["index_path"])):
            key = str(reference["global_episode_key"])
            try:
                episode = _load_and_check_episode(reference)
                rows = continuous_samples(
                    episode,
                    source_id=str(reference["contract"]["source_id"]),
                    global_episode_key=key,
                    stride=stride,
                    offsets=offsets,
                    terminal_caption=str(config["terminal_caption"]),
                    resize_policy_id=resize_id,
                    source_frame_rate_hz=(config.get("known_source_fps") or {}).get(
                        str(reference["contract"]["source_id"])
                    ),
                )
                grid_indices = [
                    int(row["anchor_grid_index"])
                    for row in rows if row.get("anchor_grid_index") is not None
                ]
                counts["unlabelled_grid_anchors_skipped"] += sum(
                    current - previous - 1
                    for previous, current in zip(grid_indices, grid_indices[1:])
                )
                first_images = rows[0]["images"]
                for row in rows:
                    continuous_writer.write(row)
                    counts["continuous"] += 1
                    if row["is_terminal_window"]:
                        terminal_writer.write({
                            "sample_key": row["sample_key"],
                            "shard_path": str(continuous_path.resolve()),
                            "line_number": continuous_writer.count,
                            "global_episode_key": key,
                            "split": row["split"],
                            "profile": row["profile"],
                            "task_type": "terminal",
                        })
                        counts["terminal"] += 1
                plan = initial_plan_sample(
                    episode,
                    source_id=str(reference["contract"]["source_id"]),
                    global_episode_key=key,
                    images=first_images,
                    resize_policy_id=resize_id,
                )
                estimate = _context_estimate(plan, initial_settings)
                plan["estimated_context_tokens"] = estimate
                plan["context_estimator"] = "chars_plus_fixed_image_tokens_v1"
                if estimate > max_context:
                    oversize_writer.write({
                        "sample_key": plan["sample_key"],
                        "global_episode_key": key,
                        "split": plan["split"],
                        "profile": plan["profile"],
                        "estimated_context_tokens": estimate,
                        "max_context_tokens": max_context,
                    })
                    counts["initial_plan_oversize"] += 1
                else:
                    initial_writer.write(plan)
                    counts["initial_plan"] += 1
                counts["episodes"] += 1
            except Exception as exc:
                reason = type(exc).__name__
                failures[reason] += 1
                failure_writer.write({
                    "global_episode_key": key,
                    "kind": reference.get("kind"),
                    "reason": reason,
                    "detail": str(exc),
                    "traceback": traceback.format_exc(limit=8),
                })
    statistics = {
        "schema_version": "memory_v3_shard_statistics_v1",
        "shard_id": shard_id,
        "created_at": _utc_now(),
        "counts": dict(sorted(counts.items())),
        "failures": dict(sorted(failures.items())),
        "files": {
            path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in (continuous_path, initial_path, terminal_path, oversize_path, failure_path)
        },
    }
    write_json(str(shard_root / "statistics.json"), statistics)
    marker = {
        "schema_version": (
            "memory_v3_shard_success_v1" if not failures
            else "memory_v3_shard_failed_v1"
        ),
        "shard_id": shard_id,
        "counts": statistics["counts"],
        "failure_count": sum(failures.values()),
        "statistics_sha256": file_sha256(shard_root / "statistics.json"),
    }
    # A shard containing even one failed Episode is diagnostic output, not a
    # stable publishable shard.  Keeping _SUCCESS absent makes partial merge and
    # resume semantics fail closed.
    if not failures:
        mark_success(shard_root, marker)
    return {**marker, "skipped": False}


def build(
    config_path: str | Path,
    *,
    resume: bool,
    max_episodes: int | None,
    workers: int | None,
    num_shards: int | None,
    source_filter: set[str] | None,
) -> dict[str, Any]:
    config = load_config(config_path)
    formal = validate_formal_snapshot(config["input_snapshot"])
    shard_count = int(num_shards or config["runtime"]["num_shards"])
    worker_count = int(workers or config["runtime"]["workers"])
    identity = {
        "builder_schema_revision": 5,
        "formal_snapshot_digest": formal["content_digest"],
        "config_digest": canonical_digest(config),
        "max_episodes": max_episodes,
        "source_filter": sorted(source_filter or ()),
        "num_shards": shard_count,
    }
    build_id = canonical_digest(identity)[:24]
    build_root = Path(str(config["output_root"])) / "builds" / build_id
    build_root.mkdir(parents=True, exist_ok=True)
    preflight_result = preflight(config_path)
    write_json(str(build_root / "preflight.json"), preflight_result)
    index = _build_episode_index(
        config=config,
        build_root=build_root,
        max_episodes=max_episodes,
        source_filter=source_filter,
        num_shards=shard_count,
    )
    active_state = {
        "schema_version": "memory_v3_active_build_v1",
        "build_id": build_id,
        "build_root": str(build_root.resolve()),
        "created_at": _utc_now(),
        "identity": identity,
        "episode_index": index,
        "expected_shards": int(index["nonempty_shards"]),
        "completed_shards": 0,
        "failure_count": 0,
        "counts": {},
        "active": True,
    }
    write_json(str(build_root / "build_state.json"), active_state)
    # Expose the active build only after its immutable Episode index exists.
    # Partial merge still admits exclusively shard directories with _SUCCESS.
    write_json(str(Path(str(config["output_root"])) / "current_build.json"), active_state)
    shards_root = build_root / "shards"
    shards_root.mkdir(exist_ok=True)
    tasks = []
    for index_path in sorted((build_root / "episode_index").glob("shard-*.jsonl")):
        shard_id = int(index_path.stem.split("-")[-1])
        shard_root = shards_root / f"shard-{shard_id:05d}"
        if not resume and shard_root.exists() and read_success(shard_root) is None:
            raise FileExistsError(f"incomplete shard exists; pass --resume: {shard_root}")
        tasks.append({
            "shard_id": shard_id,
            "index_path": str(index_path),
            "shard_root": str(shard_root),
            "config": config,
        })
    results = []
    with ProcessPoolExecutor(max_workers=max(1, min(worker_count, len(tasks) or 1))) as pool:
        futures = {pool.submit(_build_shard, task): task for task in tasks}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: int(item["shard_id"]))
    summary = {
        "schema_version": "memory_v3_build_v1",
        "build_id": build_id,
        "build_root": str(build_root.resolve()),
        "created_at": _utc_now(),
        "identity": identity,
        "episode_index": index,
        "expected_shards": len(tasks),
        "processed_shards": len(results),
        "completed_shards": sum(
            int(row.get("failure_count", 0)) == 0 for row in results
        ),
        "failure_count": sum(int(row.get("failure_count", 0)) for row in results),
        "counts": {
            name: sum(int(row.get("counts", {}).get(name, 0)) for row in results)
            for name in (
                "episodes", "continuous", "initial_plan", "terminal",
                "initial_plan_oversize", "unlabelled_grid_anchors_skipped",
            )
        },
        "shards": results,
        "active": False,
    }
    write_json(str(build_root / "build_manifest.json"), summary)
    write_json(str(Path(str(config["output_root"])) / "current_build.json"), summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PACKAGE_ROOT / "configs" / "history_context.yaml"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--resume", action="store_true")
    build_parser.add_argument("--max-episodes", type=int)
    build_parser.add_argument("--workers", type=int)
    build_parser.add_argument("--num-shards", type=int)
    build_parser.add_argument("--source-id", action="append", default=[])
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "preflight":
        result = preflight(args.config)
    else:
        result = build(
            args.config,
            resume=bool(args.resume),
            max_episodes=args.max_episodes,
            workers=args.workers,
            num_shards=args.num_shards,
            source_filter=set(args.source_id) or None,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
