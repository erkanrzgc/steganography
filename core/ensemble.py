"""Correlation-aware deterministic and calibrated model score fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from core.result import AnalysisResult


@dataclass(frozen=True, slots=True)
class ModelScore:
    model_id: str
    domain: str
    probability: float | None
    status: Literal["ok", "unavailable", "unsupported", "error"]
    execution_provider: str | None = None
    calibrated: bool = False
    error: str | None = None

    def __post_init__(self) -> None:
        if self.probability is not None and not 0 <= self.probability <= 1:
            raise ValueError("model probability must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class EnsembleScore:
    score: int
    status: Literal["ok", "inconclusive"]
    deterministic_score: int
    model_score: int | None
    execution_provider: str | None
    correlation_groups: dict[str, int]


class HybridEnsemble:
    """Fuse independent groups while preventing correlated double counting."""

    def combine(
        self,
        results: tuple[AnalysisResult, ...],
        *,
        model: ModelScore | None = None,
    ) -> EnsembleScore:
        groups: dict[str, int] = {}
        verified = False
        for result in results:
            if result.status != "ok" or result.analyzer == "ai_triage":
                continue
            for signal in result.signals:
                if signal.evidence == "informational":
                    continue
                verified |= signal.evidence == "verified"
                group = signal.category if signal.category != "general" else result.analyzer
                groups[group] = max(groups.get(group, 0), signal.score)
        clean = math.prod(1 - value / 100 for value in groups.values()) if groups else 1.0
        deterministic = round(100 * (1 - clean))
        if verified:
            deterministic = max(95, deterministic)
        model_value: int | None = None
        provider = None
        if model and model.status == "ok" and model.probability is not None and model.calibrated:
            model_value = round(model.probability * 100)
            provider = model.execution_provider
        score = max(deterministic, model_value or 0)
        usable = any(result.status == "ok" for result in results)
        inconclusive_model = model is not None and model.status != "ok"
        status: Literal["ok", "inconclusive"] = (
            "inconclusive" if not usable and inconclusive_model else "ok"
        )
        return EnsembleScore(
            score=score,
            status=status,
            deterministic_score=deterministic,
            model_score=model_value,
            execution_provider=provider,
            correlation_groups=groups,
        )
