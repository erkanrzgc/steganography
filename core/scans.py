"""Persistent case scan orchestration."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from core.cases import CaseService
from core.database import Database, utc_now
from core.pipeline import AnalysisPipeline, PipelineReport
from core.vault import VaultService


class ScanNotFoundError(LookupError):
    pass


class ScanService:
    def __init__(
        self,
        database: Database,
        vault: VaultService,
        cases: CaseService,
        pipeline_factory: Callable[[str], AnalysisPipeline],
    ) -> None:
        self.database = database
        self.vault = vault
        self.cases = cases
        self.pipeline_factory = pipeline_factory

    def create(self, case_id: str, *, profile: str = "sensitive") -> dict[str, Any]:
        evidence = self.cases.list_evidence(case_id)
        scan_id = uuid.uuid4().hex
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO scans
                   (id, case_id, status, profile, total_files, created_at, updated_at)
                   VALUES (?, ?, 'queued', ?, ?, ?, ?)""",
                (scan_id, case_id, profile, len(evidence), now, now),
            )
            self.database.audit(
                "scan.queued",
                object_type="scan",
                object_id=scan_id,
                details={"case_id": case_id, "files": len(evidence)},
                connection=connection,
            )
        return self.get(scan_id)

    def run(
        self,
        scan_id: str,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Run a scan, optionally reporting progress and honoring cancellation.

        Cancellation is checked between evidence files so an analyzer is never
        interrupted while it owns a plaintext temporary file.
        """
        scan = self.get(scan_id)
        evidence = self.cases.list_evidence(scan["case_id"])
        pipeline = self.pipeline_factory(scan["profile"])
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE scans SET status='running', updated_at=? WHERE id=?",
                (now, scan_id),
            )
        reports: list[PipelineReport] = []
        try:
            for index, item in enumerate(evidence, start=1):
                if should_cancel and should_cancel():
                    with self.database.transaction() as connection:
                        connection.execute(
                            "UPDATE scans SET status='cancelled', updated_at=? WHERE id=?",
                            (utc_now(), scan_id),
                        )
                        self.database.audit(
                            "scan.cancelled",
                            object_type="scan",
                            object_id=scan_id,
                            details={"completed_files": len(reports)},
                            connection=connection,
                        )
                    return self.get(scan_id)
                suffix = Path(item["original_name"]).suffix
                with self.vault.materialize(item["sha256"], suffix=suffix) as path:
                    report = pipeline.analyze(path)
                # Do not expose the random plaintext staging path.
                report_dict = report.to_dict()
                report_dict["file"]["path"] = item["original_name"]
                report_dict["file"]["name"] = item["original_name"]
                reports.append(report)
                self._save_analysis(scan_id, item["id"], report, report_dict)
                with self.database.transaction() as connection:
                    connection.execute(
                        """UPDATE scans SET completed_files=?, progress=?, updated_at=?
                           WHERE id=?""",
                        (index, index / max(1, len(evidence)), utc_now(), scan_id),
                    )
                if progress_callback:
                    progress_callback(index, len(evidence))
            result = {
                "schema_version": "2.0",
                "scan_id": scan_id,
                "case_id": scan["case_id"],
                "profile": scan["profile"],
                "summary": {
                    "files": len(reports),
                    "confirmed": sum(r.verdict == "confirmed" for r in reports),
                    "likely": sum(r.verdict == "likely" for r in reports),
                    "suspicious": sum(r.verdict == "suspicious" for r in reports),
                    "no_indicators": sum(r.verdict == "no_indicators" for r in reports),
                    "inconclusive": sum(r.verdict == "inconclusive" for r in reports),
                },
                "files": self._stored_reports(scan_id),
            }
            with self.database.transaction() as connection:
                connection.execute(
                    """UPDATE scans SET status='completed', progress=1,
                       result_json=?, updated_at=? WHERE id=?""",
                    (json.dumps(result, separators=(",", ":")), utc_now(), scan_id),
                )
                self.database.audit(
                    "scan.completed",
                    object_type="scan",
                    object_id=scan_id,
                    details=result["summary"],
                    connection=connection,
                )
        except Exception as exc:
            with self.database.transaction() as connection:
                connection.execute(
                    "UPDATE scans SET status='failed', error=?, updated_at=? WHERE id=?",
                    (f"{type(exc).__name__}: {exc}", utc_now(), scan_id),
                )
                self.database.audit(
                    "scan.failed",
                    object_type="scan",
                    object_id=scan_id,
                    details={"error_type": type(exc).__name__},
                    connection=connection,
                )
        return self.get(scan_id)

    def get(self, scan_id: str) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute("SELECT * FROM scans WHERE id=?", (scan_id,)).fetchone()
        if row is None:
            raise ScanNotFoundError(scan_id)
        value = dict(row)
        value["result"] = json.loads(value.pop("result_json")) if value["result_json"] else None
        return value

    def list_for_case(self, case_id: str) -> list[dict[str, Any]]:
        self.cases.get_case(case_id)
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM scans WHERE case_id=? ORDER BY created_at DESC", (case_id,)
            ).fetchall()
        values = []
        for row in rows:
            value = dict(row)
            value["result"] = json.loads(value.pop("result_json")) if value["result_json"] else None
            values.append(value)
        return values

    def _save_analysis(
        self,
        scan_id: str,
        evidence_id: str,
        report: PipelineReport,
        report_dict: dict[str, Any],
    ) -> None:
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO analyses
                   (id, scan_id, evidence_id, verdict, confidence, duration_ms,
                    result_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    report.id,
                    scan_id,
                    evidence_id,
                    report.verdict,
                    report.confidence,
                    report.duration_ms,
                    json.dumps(report_dict, separators=(",", ":")),
                    utc_now(),
                ),
            )
            for finding in report.findings:
                connection.execute(
                    """INSERT INTO findings
                       (id, analysis_id, rule_id, verdict, confidence,
                        evidence_strength, offset_start, offset_end, analyzer,
                        analyzer_version, status, detail)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        finding.id,
                        report.id,
                        finding.rule_id,
                        finding.verdict,
                        finding.confidence,
                        finding.evidence_strength,
                        finding.offset_start,
                        finding.offset_end,
                        finding.analyzer,
                        finding.analyzer_version,
                        finding.status,
                        finding.detail,
                    ),
                )

    def _stored_reports(self, scan_id: str) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                "SELECT result_json FROM analyses WHERE scan_id=? ORDER BY created_at",
                (scan_id,),
            ).fetchall()
        return [json.loads(row["result_json"]) for row in rows]
