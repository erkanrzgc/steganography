from dataclasses import replace

import pytest

from core.analyzer import Analyzer
from core.pipeline import AnalysisPipeline
from core.result import AnalysisResult, FileAnalysis, FileInfo, Signal
from core.service import AnalysisService, aggregate_score
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


def test_retired_analyzer_is_not_invoked_and_provider_is_rejected(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("plain text")

    class Native(Analyzer):
        name = "native"

        def analyze(self, path):
            return AnalysisResult("native", 40, (Signal("evidence", 40, "detail"),), None)

    registry = Registry()
    registry.register(Native())

    class Retired(Analyzer):
        name = "ai_triage"

        def analyze(self, path):
            pytest.fail("retired adapter must not be invoked")

    registry.register(Retired())
    service = AnalysisService(registry, ai_provider=None)
    result = service.analyze(path)
    assert result.overall_score == 40
    assert [r.analyzer for r in result.results] == ["native"]
    report = AnalysisPipeline(service).analyze(path)
    assert report.verdict == "suspicious" and report.confidence == 0.4
    with pytest.raises(ValueError, match="removed"):
        AnalysisService(registry, ai_provider=lambda *_: pytest.fail("provider called"))
