"""Analyzer ABC: analysis-only modules (signatures, AI triage)."""
from abc import ABC, abstractmethod
from pathlib import Path

from core.result import AnalysisResult


class Analyzer(ABC):
    name: str = ""

    @abstractmethod
    def analyze(self, src: Path) -> AnalysisResult: ...
