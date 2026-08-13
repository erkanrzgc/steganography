"""SQLite persistence for REST API scan jobs."""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class JobStore:
    def __init__(self, database: Path, *, retention_days: int = 30) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS scan_jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    result_json TEXT,
                    error TEXT
                )
                """
            )
            connection.execute(
                """
                UPDATE scan_jobs
                SET status = 'failed', updated_at = ?,
                    error = 'server restarted before the job completed'
                WHERE status IN ('queued', 'running')
                """,
                (_now(),),
            )
        self.cleanup()

    def create(self, job_id: str, profile: str) -> dict[str, Any]:
        timestamp = _now()
        retention_seconds = max(0, self.retention_days) * 86400
        expires_at = time.time() + retention_seconds
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO scan_jobs
                    (id, status, profile, created_at, updated_at, expires_at)
                VALUES (?, 'queued', ?, ?, ?, ?)
                """,
                (job_id, profile, timestamp, timestamp, expires_at),
            )
        return self.get(job_id) or {}

    def set_running(self, job_id: str) -> None:
        self._update(job_id, status="running")

    def complete(self, job_id: str, result: dict[str, Any]) -> None:
        self._update(
            job_id,
            status="completed",
            result_json=json.dumps(result, ensure_ascii=False),
            error=None,
        )

    def fail(self, job_id: str, error: str) -> None:
        self._update(job_id, status="failed", error=error)

    def _update(self, job_id: str, **values: Any) -> None:
        values["updated_at"] = _now()
        columns = ", ".join(f"{key} = ?" for key in values)
        parameters = [*values.values(), job_id]
        with self._connect() as connection:
            connection.execute(
                f"UPDATE scan_jobs SET {columns} WHERE id = ?",  # noqa: S608
                parameters,
            )

    def get(self, job_id: str) -> dict[str, Any] | None:
        self.cleanup()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM scan_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["job_id"] = result.pop("id")
        result.pop("expires_at", None)
        raw = result.pop("result_json", None)
        result["result"] = json.loads(raw) if raw else None
        return result

    def delete(self, job_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM scan_jobs WHERE id = ?", (job_id,))
        return cursor.rowcount > 0

    def cleanup(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM scan_jobs WHERE expires_at <= ?", (time.time(),)
            )
        return cursor.rowcount
