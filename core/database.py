"""Versioned SQLite persistence for local cases and analysis records."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class Database:
    """Small SQLite gateway with explicit migrations and an audit hash chain."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with suppress(OSError):
            self.path.parent.chmod(0o700)
        self._lock = threading.RLock()
        self._migrate()
        with suppress(OSError):
            self.path.chmod(0o600)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock, self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    def _migrate(self) -> None:
        with self._lock, self.connect() as connection:
            current = connection.execute("PRAGMA user_version").fetchone()[0]
            if current > SCHEMA_VERSION:
                raise RuntimeError(
                    f"database schema {current} is newer than supported {SCHEMA_VERSION}"
                )
            if current == 0:
                connection.executescript(_SCHEMA_V1)
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            elif current == 1:
                connection.executescript(_SCHEMA_V2)
                connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            connection.commit()

    def audit(
        self,
        action: str,
        *,
        object_type: str,
        object_id: str | None = None,
        details: dict[str, Any] | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> str:
        """Append a canonical event. The chain detects mutation and reordering."""
        if connection is None:
            with self.transaction() as owned:
                return self.audit(
                    action,
                    object_type=object_type,
                    object_id=object_id,
                    details=details,
                    connection=owned,
                )
        event_id = uuid.uuid4().hex
        created_at = utc_now()
        detail_json = json.dumps(details or {}, sort_keys=True, separators=(",", ":"))
        previous_row = connection.execute(
            "SELECT event_hash FROM audit_events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous_hash = previous_row[0] if previous_row else "0" * 64
        material = "\x1f".join(
            (previous_hash, event_id, created_at, action, object_type, object_id or "", detail_json)
        ).encode("utf-8")
        event_hash = hashlib.sha256(material).hexdigest()
        connection.execute(
            """INSERT INTO audit_events
               (id, created_at, action, object_type, object_id, details_json,
                previous_hash, event_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                created_at,
                action,
                object_type,
                object_id,
                detail_json,
                previous_hash,
                event_hash,
            ),
        )
        return event_id

    def verify_audit(self) -> tuple[bool, int, str | None]:
        previous_hash = "0" * 64
        count = 0
        with self.connect() as connection:
            rows = connection.execute(
                """SELECT sequence, id, created_at, action, object_type, object_id,
                          details_json, previous_hash, event_hash
                   FROM audit_events ORDER BY sequence"""
            )
            for row in rows:
                material = "\x1f".join(
                    (
                        previous_hash,
                        row["id"],
                        row["created_at"],
                        row["action"],
                        row["object_type"],
                        row["object_id"] or "",
                        row["details_json"],
                    )
                ).encode("utf-8")
                expected = hashlib.sha256(material).hexdigest()
                if row["previous_hash"] != previous_hash or row["event_hash"] != expected:
                    return False, count, f"audit chain break at sequence {row['sequence']}"
                previous_hash = row["event_hash"]
                count += 1
        return True, count, None

    def purge_expired_cases(self) -> int:
        now = datetime.now(UTC)
        purged = 0
        with self.transaction() as connection:
            rows = connection.execute(
                "SELECT id, created_at, retention_days FROM cases WHERE retention_days IS NOT NULL"
            ).fetchall()
            for row in rows:
                created = datetime.fromisoformat(row["created_at"].replace("Z", "+00:00"))
                if created + timedelta(days=row["retention_days"]) <= now:
                    connection.execute("DELETE FROM cases WHERE id=?", (row["id"],))
                    self.audit(
                        "case.purged",
                        object_type="case",
                        object_id=row["id"],
                        connection=connection,
                    )
                    purged += 1
        return purged


_SCHEMA_V1 = """
CREATE TABLE cases (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    retention_days INTEGER CHECK(retention_days IS NULL OR retention_days >= 0),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE vault_config (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    salt BLOB NOT NULL,
    memory_cost_kib INTEGER NOT NULL,
    iterations INTEGER NOT NULL,
    lanes INTEGER NOT NULL,
    check_blob BLOB NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE vault_objects (
    sha256 TEXT PRIMARY KEY,
    storage_name TEXT NOT NULL UNIQUE,
    size INTEGER NOT NULL CHECK(size >= 0),
    ref_count INTEGER NOT NULL DEFAULT 0 CHECK(ref_count >= 0),
    created_at TEXT NOT NULL
);

CREATE TABLE evidence (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    sha256 TEXT NOT NULL REFERENCES vault_objects(sha256),
    original_name TEXT NOT NULL,
    media_type TEXT NOT NULL,
    size INTEGER NOT NULL CHECK(size >= 0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(case_id, sha256)
);
CREATE INDEX evidence_case_idx ON evidence(case_id, created_at);

CREATE TRIGGER evidence_ref_insert AFTER INSERT ON evidence BEGIN
  UPDATE vault_objects SET ref_count=ref_count+1 WHERE sha256=NEW.sha256;
END;
CREATE TRIGGER evidence_ref_delete AFTER DELETE ON evidence BEGIN
  UPDATE vault_objects SET ref_count=ref_count-1 WHERE sha256=OLD.sha256;
END;

CREATE TABLE scans (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK(status IN ('queued','running','completed','cancelled','failed')),
    profile TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0 CHECK(progress >= 0 AND progress <= 1),
    total_files INTEGER NOT NULL DEFAULT 0,
    completed_files INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    result_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE analyses (
    id TEXT PRIMARY KEY,
    scan_id TEXT NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    evidence_id TEXT NOT NULL REFERENCES evidence(id) ON DELETE CASCADE,
    verdict TEXT NOT NULL,
    confidence REAL NOT NULL,
    duration_ms REAL NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE findings (
    id TEXT PRIMARY KEY,
    analysis_id TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    rule_id TEXT NOT NULL,
    verdict TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_strength TEXT NOT NULL,
    offset_start INTEGER,
    offset_end INTEGER,
    analyzer TEXT NOT NULL,
    analyzer_version TEXT NOT NULL,
    status TEXT NOT NULL,
    detail TEXT NOT NULL
);

CREATE TABLE artifacts (
    id TEXT PRIMARY KEY,
    case_id TEXT REFERENCES cases(id) ON DELETE CASCADE,
    scan_id TEXT REFERENCES scans(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    media_type TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size INTEGER NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE models (
    id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    version TEXT NOT NULL,
    state TEXT NOT NULL,
    path TEXT,
    sha256 TEXT,
    manifest_json TEXT NOT NULL,
    installed_at TEXT,
    UNIQUE(id, version)
);

CREATE TABLE audit_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    action TEXT NOT NULL,
    object_type TEXT NOT NULL,
    object_id TEXT,
    details_json TEXT NOT NULL,
    previous_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL UNIQUE
);
"""


_SCHEMA_V2 = """
PRAGMA foreign_keys=OFF;
PRAGMA legacy_alter_table=ON;
BEGIN;
ALTER TABLE scans RENAME TO scans_v1;
CREATE TABLE scans (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK(status IN ('queued','running','completed','cancelled','failed')),
    profile TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0 CHECK(progress >= 0 AND progress <= 1),
    total_files INTEGER NOT NULL DEFAULT 0,
    completed_files INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    result_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
INSERT INTO scans
SELECT id, case_id, status, profile, progress, total_files, completed_files,
       error, result_json, created_at, updated_at
FROM scans_v1;
DROP TABLE scans_v1;
COMMIT;
PRAGMA legacy_alter_table=OFF;
PRAGMA foreign_keys=ON;
"""
