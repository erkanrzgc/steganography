"""Case and evidence lifecycle services."""

from __future__ import annotations

import json
import mimetypes
import uuid
from pathlib import Path
from typing import Any

from core.database import Database, utc_now
from core.vault import VaultService


class CaseNotFoundError(LookupError):
    pass


class EvidenceNotFoundError(LookupError):
    pass


class CaseService:
    """Manage case metadata while storing all evidence through ``VaultService``."""

    def __init__(self, database: Database, vault: VaultService) -> None:
        self.database = database
        self.vault = vault

    def create_case(
        self,
        name: str,
        *,
        description: str = "",
        retention_days: int | None = None,
    ) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("case name must not be empty")
        if retention_days is not None and retention_days < 0:
            raise ValueError("retention_days must be non-negative")
        case_id = uuid.uuid4().hex
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO cases
                   (id, name, description, retention_days, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (case_id, name, description, retention_days, now, now),
            )
            self.database.audit(
                "case.created",
                object_type="case",
                object_id=case_id,
                details={"name": name, "retention_days": retention_days},
                connection=connection,
            )
        return self.get_case(case_id)

    def list_cases(self) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """SELECT c.*,
                          COUNT(DISTINCT e.id) AS evidence_count,
                          COUNT(DISTINCT s.id) AS scan_count
                   FROM cases c
                   LEFT JOIN evidence e ON e.case_id=c.id
                   LEFT JOIN scans s ON s.case_id=c.id
                   GROUP BY c.id ORDER BY c.updated_at DESC"""
            ).fetchall()
        return [dict(row) for row in rows]

    def get_case(self, case_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute(
                """SELECT c.*,
                          COUNT(DISTINCT e.id) AS evidence_count,
                          COUNT(DISTINCT s.id) AS scan_count
                   FROM cases c
                   LEFT JOIN evidence e ON e.case_id=c.id
                   LEFT JOIN scans s ON s.case_id=c.id
                   WHERE c.id=? GROUP BY c.id""",
                (case_id,),
            ).fetchone()
        if row is None:
            raise CaseNotFoundError(case_id)
        return dict(row)

    def delete_case(self, case_id: str) -> None:
        with self.database.transaction() as connection:
            row = connection.execute("SELECT name FROM cases WHERE id=?", (case_id,)).fetchone()
            if row is None:
                raise CaseNotFoundError(case_id)
            connection.execute("DELETE FROM cases WHERE id=?", (case_id,))
            self.database.audit(
                "case.deleted",
                object_type="case",
                object_id=case_id,
                details={"name": row["name"]},
                connection=connection,
            )
        self.vault.delete_unreferenced()

    def add_evidence(
        self,
        case_id: str,
        source: Path,
        *,
        original_name: str | None = None,
        media_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.get_case(case_id)
        source = Path(source)
        digest, size, vault_duplicate = self.vault.store(source)
        evidence_id = uuid.uuid4().hex
        name = Path(original_name or source.name).name
        guessed_type = media_type or mimetypes.guess_type(name)[0] or "application/octet-stream"
        now = utc_now()
        with self.database.transaction() as connection:
            existing = connection.execute(
                "SELECT id FROM evidence WHERE case_id=? AND sha256=?",
                (case_id, digest),
            ).fetchone()
            if existing:
                return self.get_evidence(existing["id"])
            connection.execute(
                """INSERT INTO evidence
                   (id, case_id, sha256, original_name, media_type, size,
                    metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    evidence_id,
                    case_id,
                    digest,
                    name,
                    guessed_type,
                    size,
                    json.dumps(metadata or {}, sort_keys=True),
                    now,
                ),
            )
            connection.execute("UPDATE cases SET updated_at=? WHERE id=?", (now, case_id))
            self.database.audit(
                "evidence.added",
                object_type="evidence",
                object_id=evidence_id,
                details={
                    "case_id": case_id,
                    "sha256": digest,
                    "size": size,
                    "vault_duplicate": vault_duplicate,
                },
                connection=connection,
            )
        return self.get_evidence(evidence_id)

    def list_evidence(self, case_id: str) -> list[dict[str, Any]]:
        self.get_case(case_id)
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM evidence WHERE case_id=? ORDER BY created_at",
                (case_id,),
            ).fetchall()
        return [_evidence_row(row) for row in rows]

    def get_evidence(self, evidence_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM evidence WHERE id=?", (evidence_id,)).fetchone()
        if row is None:
            raise EvidenceNotFoundError(evidence_id)
        return _evidence_row(row)


def _evidence_row(row: Any) -> dict[str, Any]:
    value = dict(row)
    value["metadata"] = json.loads(value.pop("metadata_json"))
    return value
