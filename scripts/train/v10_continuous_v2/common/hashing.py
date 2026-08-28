"""Deterministic hashing helpers.

Everything here must be stable across processes and Python invocations, so
``hash()`` (PYTHONHASHSEED-dependent) is never used.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from typing import Any


def sha256_hex(*parts: Any) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def stable_hash(value: str) -> int:
    """Process-stable 64-bit hash of a string."""
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def shard_for(global_episode_key: str, num_shards: int) -> int:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    return stable_hash(global_episode_key) % num_shards


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sampling_config_hash(config: Mapping[str, Any]) -> str:
    """Hash only the fields that change sample content."""
    relevant = {
        "anchor_quantiles": list(config.get("anchor_quantiles", ())),
        "anchor_stride": config.get("anchor_stride"),
        "max_anchors_per_episode": config.get("max_anchors_per_episode"),
        "visual_stride": config.get("visual_stride"),
        "visual_timesteps": config.get("visual_timesteps"),
        "max_camera_views": config.get("max_camera_views"),
        "max_visual_inputs": config.get("max_visual_inputs"),
    }
    return config_hash(relevant)


def file_signature(path: str) -> dict[str, Any]:
    """Cache key ingredient: path + size + mtime (spec section 12.2)."""
    try:
        info = os.stat(path)
    except OSError:
        return {"path": path, "size": None, "mtime": None}
    return {"path": path, "size": info.st_size, "mtime": int(info.st_mtime)}


def global_episode_key(source_id: str, episode_key: str) -> str:
    return f"{source_id}:{episode_key}"
