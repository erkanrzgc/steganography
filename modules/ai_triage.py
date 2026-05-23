"""Pluggable AI triage analyzer. Heuristic-only if no provider registered.

Providers are user-supplied callables. Any provider that needs an API key MUST
read it via `config.get_secret(...)` — no literal keys in this file.
"""
from collections.abc import Callable
from pathlib import Path

from core.analyzer import Analyzer
from core.result import AnalysisResult, Signal

AIProvider = Callable[[dict, list[Signal]], tuple[int, str]]
_provider: AIProvider | None = None


def set_provider(provider: AIProvider | None) -> None:
    """Register or clear the AI provider used by AITriage.analyze()."""
    global _provider
    _provider = provider


def get_provider() -> AIProvider | None:
    return _provider


class AITriage(Analyzer):
    name = "ai_triage"

    def score_from_signals(self, signals: tuple[Signal, ...]) -> AnalysisResult:
        if not signals:
            return AnalysisResult(self.name, 0, (), "no signals")
        suspicion = max(s.score for s in signals)
        explanation = "; ".join(f"{s.name}:{s.score}" for s in signals)
        return AnalysisResult(self.name, suspicion, signals, explanation)

    def analyze(self, src: Path) -> AnalysisResult:
        info = {"path": str(src), "size": src.stat().st_size if src.exists() else 0}
        signals: list[Signal] = []
        if _provider is not None:
            score, explanation = _provider(info, signals)
            return AnalysisResult(self.name, score, tuple(signals), explanation)
        return self.score_from_signals(tuple(signals))
