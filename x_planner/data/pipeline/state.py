"""Crash-safe SQLite scan state and process lock."""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterator


class ScanLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = None

    def __enter__(self) -> "ScanLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another scanner holds {self.path}") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(str(os.getpid()) + "\n")
        self.handle.flush()
        return self

    def __exit__(self, *_args: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


class ScanState:
    def __init__(self, path: Path, *, fingerprint: str, config_json: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS episodes (
                episode_key TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                topic TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('accepted', 'rejected')),
                reason TEXT NOT NULL DEFAULT '',
                detail TEXT NOT NULL DEFAULT '',
                shard_path TEXT NOT NULL DEFAULT '',
                profile TEXT NOT NULL DEFAULT '',
                split TEXT NOT NULL DEFAULT '',
                num_samples INTEGER NOT NULL DEFAULT 0,
                views_json TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS episodes_status_idx ON episodes(status);
            CREATE INDEX IF NOT EXISTS episodes_split_idx ON episodes(split);
            CREATE INDEX IF NOT EXISTS episodes_profile_idx ON episodes(profile);
            """
        )
        previous = self.get_meta("fingerprint")
        if previous is not None and previous != fingerprint:
            raise RuntimeError(
                f"scan fingerprint mismatch: state={previous}, requested={fingerprint}"
            )
        self.set_meta("fingerprint", fingerprint)
        self.set_meta("config_json", config_json)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def get_meta(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return str(row[0]) if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def is_terminal(self, episode_key: str) -> bool:
        return self.connection.execute(
            "SELECT 1 FROM episodes WHERE episode_key=?", (episode_key,)
        ).fetchone() is not None

    def record_accepted(
        self,
        *,
        episode_key: str,
        source: str,
        topic: str,
        shard_path: str,
        profile: str,
        split: str,
        num_samples: int,
        views: list[str],
    ) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO episodes(
                    episode_key,source,topic,status,shard_path,profile,split,
                    num_samples,views_json,updated_at
                ) VALUES(?,?,?,'accepted',?,?,?,?,?,CURRENT_TIMESTAMP)
                ON CONFLICT(episode_key) DO UPDATE SET
                    source=excluded.source,topic=excluded.topic,status='accepted',
                    reason='',detail='',shard_path=excluded.shard_path,
                    profile=excluded.profile,split=excluded.split,
                    num_samples=excluded.num_samples,views_json=excluded.views_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    episode_key,
                    source,
                    topic,
                    shard_path,
                    profile,
                    split,
                    num_samples,
                    json.dumps(views, separators=(",", ":")),
                ),
            )

    def record_rejected(
        self,
        *,
        episode_key: str,
        source: str,
        topic: str,
        reason: str,
        detail: str,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO episodes(
                    episode_key,source,topic,status,reason,detail,updated_at
                ) VALUES(?,?,?,'rejected',?,?,CURRENT_TIMESTAMP)
                ON CONFLICT(episode_key) DO UPDATE SET
                    source=excluded.source,topic=excluded.topic,status='rejected',
                    reason=excluded.reason,detail=excluded.detail,shard_path='',
                    profile='',split='',num_samples=0,views_json='[]',
                    updated_at=CURRENT_TIMESTAMP
                """,
                (episode_key, source, topic, reason, detail[:4000]),
            )

    def iter_rows(self, status: str | None = None) -> Iterator[dict[str, Any]]:
        if status is None:
            query, params = "SELECT * FROM episodes ORDER BY episode_key", ()
        else:
            query, params = "SELECT * FROM episodes WHERE status=? ORDER BY episode_key", (status,)
        for row in self.connection.execute(query, params):
            value = dict(row)
            value["views"] = json.loads(value.pop("views_json"))
            yield value

    def stats(self) -> dict[str, Any]:
        status = {
            row["status"]: int(row["count"])
            for row in self.connection.execute(
                "SELECT status,COUNT(*) AS count FROM episodes GROUP BY status"
            )
        }
        splits = {
            row["split"]: int(row["count"])
            for row in self.connection.execute(
                "SELECT split,COUNT(*) AS count FROM episodes "
                "WHERE status='accepted' GROUP BY split"
            )
        }
        profiles = {
            row["profile"]: int(row["count"])
            for row in self.connection.execute(
                "SELECT profile,COUNT(*) AS count FROM episodes "
                "WHERE status='accepted' GROUP BY profile"
            )
        }
        samples = self.connection.execute(
            "SELECT COALESCE(SUM(num_samples),0) FROM episodes WHERE status='accepted'"
        ).fetchone()[0]
        return {
            "status": status,
            "splits": splits,
            "profiles": profiles,
            "samples": int(samples),
            "terminal": sum(status.values()),
        }

