from dataclasses import replace

import pytest

from core.analyzer import Analyzer
from core.pipeline import AnalysisPipeline
from core.result import AnalysisResult, FileAnalysis, FileInfo, Signal
from core.service import AnalysisService, aggregate_score
from modules.ai_triage import AITriage
from registry import Registry


@pytest.mark.parametrize("score", [0, 50, 100])
@pytest.mark.parametrize("status", ["ok", "error", "unavailable"])
def test_triage_cannot_change_primary_findings_or_coverage(tmp_path, score, status):
    native = AnalysisResult("native", 0, (), None, status="unavailable")
    ai = AnalysisResult(
        "ai_triage",
        score,
        (
            Signal(
                "claimed_marker",
                100,
                "untrusted claim",
                category="known_marker",
                evidence="verified",
            ),
        ),
        "explanation only",
        status=status,
    )
    info = FileInfo("file.txt", "file.txt", 1, "a" * 64, "text", "text/plain", ".txt", False)
    original = FileAnalysis(info, 0, "low", "balanced", (native,))

    class FixedService:
        def __init__(self, analysis):
            self.analysis = analysis

        def analyze_safe(self, path):
            return self.analysis

    baseline = AnalysisPipeline(FixedService(original)).analyze(tmp_path / "file.txt")
    enriched = replace(original, results=(native, ai))
    result = AnalysisPipeline(FixedService(enriched)).analyze(tmp_path / "file.txt")
    assert aggregate_score(enriched.results, "balanced") == 0
    assert result.verdict == baseline.verdict == "inconclusive"
    assert result.confidence == baseline.confidence
    assert [f.rule_id for f in result.findings] == [f.rule_id for f in baseline.findings]
    assert result.analysis.results[-1].explanation == "explanation only"


def test_external_provider_score_does_not_raise_or_lower_native_score(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("plain text")

    class Native(Analyzer):
        name = "native"

        def analyze(self, path):
            return AnalysisResult("native", 40, (Signal("evidence", 40, "detail"),), None)

    registry = Registry()
    registry.register(Native())
    registry.register(AITriage())
    for score in (0, 100):
        service = AnalysisService(
            registry, ai_provider=lambda info, signals, value=score: (value, "triage")
        )
        result = service.analyze(path)
        assert result.overall_score == 40
        assert result.results[-1].suspicion == score
        report = AnalysisPipeline(service).analyze(path)
        assert report.verdict == "suspicious"
        assert report.confidence == 0.4
