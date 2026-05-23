"""Immutable result types returned by carriers and analyzers."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Signal:
    """A single heuristic finding from an analyzer."""
    name: str
    score: int          # 0-100, contribution to overall suspicion
    detail: str


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
