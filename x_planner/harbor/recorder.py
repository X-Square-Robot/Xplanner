"""Write reproducible X-Planner rollout trials.

The recorder deliberately has no Harbor SDK dependency. A completed trial is
an immutable, content-addressed directory that a deployment-specific adapter
can upload later.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping


class TrialRecorder:
    def __init__(self, root: str | os.PathLike[str], config: Mapping[str, Any]):
        self.root = Path(root)
        self.config = dict(config)
        self._started = time.time()
        self._closed = False
        self.root.mkdir(parents=True, exist_ok=False)
        self._write_json("config.json", self.config)
        seed = self.config.get("seed")
        if seed is not None:
            self._write_json("environment/seed.json", {"seed": seed})

    def _write_json(self, relative: str, value: Any) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    def append(self, relative: str, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("trial is already closed")
        item = {"trial_id": self.config.get("trial_id"), "timestamp": time.time(), **dict(record)}
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    def finish(self, result: Mapping[str, Any]) -> Path:
        if self._closed:
            raise RuntimeError("trial is already closed")
        self._write_json("result.json", {"duration_s": time.time() - self._started, **dict(result)})
        files: dict[str, dict[str, Any]] = {}
        for path in sorted(p for p in self.root.rglob("*") if p.is_file() and p.name != "manifest.json"):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            files[str(path.relative_to(self.root))] = {"sha256": digest, "bytes": path.stat().st_size}
        self._write_json("manifest.json", {"schema_version": "harbor_trial_manifest_v1", "files": files})
        self._closed = True
        return self.root / "manifest.json"
