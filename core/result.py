"""Immutable result types returned by carriers, analyzers and services."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Signal:
    """A single heuristic finding from an analyzer."""
    name: str
    score: int          # 0-100, contribution to overall suspicion
    detail: str
    category: str = "general"
    evidence: str = "heuristic"  # heuristic | strong | verified | informational

    def __post_init__(self) -> None:
        if not 0 <= self.score <= 100:
            raise ValueError("signal score must be between 0 and 100")


@dataclass(frozen=True, slots=True)
class EmbedResult:
    carrier: str
    out_path: Path
    bytes_written: int
    encrypted: bool


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    analyzer: str
    suspicion: int
    signals: tuple[Signal, ...]
    explanation: str | None
    status: str = "ok"  # ok | error | unsupported | unavailable
    error: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.suspicion <= 100:
            raise ValueError("suspicion must be between 0 and 100")

    def to_dict(self) -> dict[str, Any]:
        return {
            "analyzer": self.analyzer,
            "suspicion": self.suspicion,
            "signals": [
                {
                    "name": signal.name,
                    "score": signal.score,
                    "detail": signal.detail,
                    "category": signal.category,
                    "evidence": signal.evidence,
                }
                for signal in self.signals
            ],
            "explanation": self.explanation,
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class FileInfo:
    """Stable evidence metadata for one analyzed file."""

    path: str
    name: str
    size: int
    sha256: str
    detected_type: str
    mime_type: str
    extension: str
    extension_mismatch: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "size": self.size,
            "sha256": self.sha256,
            "detected_type": self.detected_type,
            "mime_type": self.mime_type,
            "extension": self.extension,
            "extension_mismatch": self.extension_mismatch,
        }


@dataclass(frozen=True, slots=True)
class FileAnalysis:
    """Aggregated, serializable verdict for a single file."""

    file: FileInfo
    overall_score: int
    severity: str
    profile: str
    results: tuple[AnalysisResult, ...]
    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    schema_version: str = "1.0"

    def to_dict(self, *, tool_version: str = "0.2.0") -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tool_version": tool_version,
            "generated_at": self.generated_at,
            "file": self.file.to_dict(),
            "overall": {
                "score": self.overall_score,
                "severity": self.severity,
                "profile": self.profile,
            },
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True, slots=True)
class ScanReport:
    """A collection of file verdicts produced by one scan."""

    files: tuple[FileAnalysis, ...]
    profile: str
    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    schema_version: str = "1.0"

    def to_dict(self, *, tool_version: str = "0.2.0") -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tool_version": tool_version,
            "generated_at": self.generated_at,
            "profile": self.profile,
            "summary": {
                "files": len(self.files),
                "high": sum(item.severity == "high" for item in self.files),
                "medium": sum(item.severity == "medium" for item in self.files),
                "low": sum(item.severity == "low" for item in self.files),
                "errors": sum(
                    any(result.status == "error" for result in item.results)
                    for item in self.files
                ),
            },
            "files": [item.to_dict(tool_version=tool_version) for item in self.files],
        }
