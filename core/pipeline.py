"""JSON v2 analysis pipeline built on the compatible v1 analysis service."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from core.result import FileAnalysis, Signal
from core.service import AnalysisService
from core.version import __version__

Verdict = Literal["confirmed", "likely", "suspicious", "no_indicators", "inconclusive"]


@dataclass(frozen=True, slots=True)
class Finding:
    id: str
    rule_id: str
    verdict: Verdict
    confidence: float
    evidence_strength: str
    detail: str
    analyzer: str
    analyzer_version: str
    status: str
    offset_start: int | None = None
    offset_end: int | None = None
    region: dict[str, Any] | None = None
    duration_ms: float = 0.0
    error: str | None = None
    artifacts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "evidence_strength": self.evidence_strength,
            "location": {
                "offset_start": self.offset_start,
                "offset_end": self.offset_end,
                "region": self.region,
            },
            "analyzer": {"name": self.analyzer, "version": self.analyzer_version},
            "duration_ms": self.duration_ms,
            "status": self.status,
            "detail": self.detail,
            "error": self.error,
            "artifacts": list(self.artifacts),
        }


@dataclass(frozen=True, slots=True)
class PipelineReport:
    id: str
    verdict: Verdict
    confidence: float
    analysis: FileAnalysis
    findings: tuple[Finding, ...]
    duration_ms: float
    execution_provider: str = "cpu"
    schema_version: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tool": {"name": "steganography", "version": __version__},
            "id": self.id,
            "generated_at": self.analysis.generated_at,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "execution_provider": self.execution_provider,
            "file": self.analysis.file.to_dict(),
            "profile": self.analysis.profile,
            "findings": [item.to_dict() for item in self.findings],
        }


class AnalysisPipeline:
    """Normalize analyzer signals into stable, evidence-aware v2 findings."""

    def __init__(self, service: AnalysisService | None = None) -> None:
        self.service = service or AnalysisService()

    def analyze(self, path: Path) -> PipelineReport:
        started = time.perf_counter()
        analysis = self.service.analyze_safe(path)
        findings: list[Finding] = []
        unavailable = 0
        errors = 0
        for result in analysis.results:
            if result.status in {"unavailable", "unsupported"}:
                unavailable += 1
            elif result.status == "error":
                errors += 1
            for signal in result.signals:
                findings.append(_finding(result.analyzer, result.status, signal))
            if result.status not in {"ok", "unsupported"} and not result.signals:
                findings.append(
                    Finding(
                        id=uuid.uuid4().hex,
                        rule_id=f"{_slug(result.analyzer)}.runtime",
                        verdict="inconclusive",
                        confidence=0.0,
                        evidence_strength="informational",
                        detail=result.error or result.explanation or result.status,
                        analyzer=result.analyzer,
                        analyzer_version=__version__,
                        status=result.status,
                        error=result.error,
                    )
                )
        verdict = _overall_verdict(analysis, findings, unavailable, errors)
        duration = round((time.perf_counter() - started) * 1000, 3)
        return PipelineReport(
            id=uuid.uuid4().hex,
            verdict=verdict,
            confidence=analysis.overall_score / 100.0,
            analysis=analysis,
            findings=tuple(findings),
            duration_ms=duration,
        )


def _finding(analyzer: str, status: str, signal: Signal) -> Finding:
    verdict: Verdict
    if signal.evidence == "verified" and signal.category == "known_marker":
        verdict = "confirmed"
    elif signal.score >= 70:
        verdict = "likely"
    elif signal.score >= 30:
        verdict = "suspicious"
    else:
        verdict = "no_indicators"
    return Finding(
        id=uuid.uuid4().hex,
        rule_id=f"{_slug(analyzer)}.{_slug(signal.name)}",
        verdict=verdict,
        confidence=signal.score / 100.0,
        evidence_strength=signal.evidence,
        detail=signal.detail,
        analyzer=analyzer,
        analyzer_version=__version__,
        status=status,
    )


def _overall_verdict(
    analysis: FileAnalysis,
    findings: list[Finding],
    unavailable: int,
    errors: int,
) -> Verdict:
    if any(item.verdict == "confirmed" for item in findings):
        return "confirmed"
    if analysis.overall_score >= 70:
        return "likely"
    if analysis.overall_score >= 30:
        return "suspicious"
    usable = sum(result.status == "ok" for result in analysis.results)
    if usable == 0 and (unavailable or errors):
        return "inconclusive"
    return "no_indicators"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "unknown"
