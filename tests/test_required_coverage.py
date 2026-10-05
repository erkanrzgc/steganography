from dataclasses import replace
from pathlib import Path

import pytest

from core.coverage import assess_coverage
from core.pipeline import AnalysisPipeline
from core.result import AnalysisResult, FileAnalysis, FileInfo, Signal
from core.service import AnalysisService
from report.v2 import html_v2, sarif_v2

REQUIRED = {
    "png": ("image_lsb", "image_lsb_scatter", "image_bitplane"),
    "bmp": ("image_lsb", "image_lsb_scatter", "image_bitplane"),
    "jpeg": ("image_jpeg", "image_jpeg_dct"),
    "gif": ("image_gif",),
    "wav": ("audio_wav",),
    "mp3": ("audio_mp3",),
    "pdf": ("file_pdf",),
    "text": ("text_whitespace", "text_zerowidth", "text_anomalies"),
}


def analysis(fmt, score=0):
    info = FileInfo(
        "host/file.bin", "file.bin", 1, "a" * 64, fmt, "application/octet-stream", ".bin", True
    )
    names = ("file_structure", "signatures", *REQUIRED.get(fmt, ()))
    return FileAnalysis(
        info, score, "low", "balanced", tuple(AnalysisResult(n, 0, (), None) for n in names)
    )


def report(value):
    class Fixed(AnalysisService):
        def analyze_safe(self, path):
            return value

    return AnalysisPipeline(Fixed()).analyze(Path("not-opened.bin"))


@pytest.mark.parametrize("fmt", REQUIRED)
@pytest.mark.parametrize("status", ["ok", "missing", "error", "unsupported", "unavailable"])
def test_every_native_format_requires_complete_execution(fmt, status):
    value = analysis(fmt)
    target = value.results[-1]
    value = replace(
        value,
        results=value.results[:-1]
        + (() if status == "missing" else (replace(target, status=status),)),
    )
    result = report(value)
    expected = status == "ok"
    assert assess_coverage(value).complete is expected
    assert result.verdict == ("no_indicators" if expected else "inconclusive")
    document = result.to_dict()
    assert document["schema_version"] == "2.0" and document["schema_revision"] == 2
    assert document["coverage_policy"]["complete"] is expected
    assert "host/" not in str(document)
    assert any(f.rule_id.endswith("required_coverage") for f in result.findings) is not expected
    if not expected:
        assert document["recommendations"][0]["action"].startswith("Restore")


@pytest.mark.parametrize("fmt", ["unknown", "tiff", "zip"])
def test_generic_or_unimplemented_format_is_not_coverage_complete(fmt):
    value = analysis(fmt)
    assert report(value).verdict == "inconclusive"
    requirement = assess_coverage(value).requirements[-1]
    assert requirement.component == "format_specific_analysis"
    assert requirement.status == "unsupported" and requirement.required


def test_duplicate_required_result_and_base_analyzer_absence_fail_closed():
    value = analysis("jpeg")
    duplicate = replace(value, results=(*value.results, value.results[-1]))
    assert report(duplicate).verdict == "inconclusive"
    assert assess_coverage(duplicate).requirements[-1].status == "failed"
    absent = replace(value, results=value.results[1:])
    assert not assess_coverage(absent).complete


@pytest.mark.parametrize("score,verdict", [(30, "suspicious"), (70, "likely"), (0, "confirmed")])
def test_incomplete_coverage_preserves_independent_positive_evidence(score, verdict):
    value = analysis("jpeg", score)
    marker = Signal("marker", 95, "verified marker", "known_marker", "verified")
    if verdict == "confirmed":
        value = replace(value, results=(AnalysisResult("signatures", 95, (marker,), None),))
    else:
        value = replace(value, results=())
    result = report(value)
    assert result.verdict == verdict
    assert not result.to_dict()["coverage_policy"]["complete"]


def test_optional_cloud_tools_and_failed_signals_cannot_fill_coverage():
    value = analysis("jpeg")
    marker = Signal("marker", 95, "untrusted", "known_marker", "verified")
    results = value.results[:-1] + (
        AnalysisResult("ai_triage", 100, (marker,), None),
        AnalysisResult("stegseek", 0, (), None),
        AnalysisResult("image_jpeg_dct", 95, (marker,), None, status="error"),
    )
    result = report(replace(value, results=results))
    assert result.verdict == "inconclusive"
    assert not any(f.verdict == "confirmed" for f in result.findings)
    assert not any(f.analyzer == "ai_triage" for f in result.findings)


def test_all_views_preserve_incomplete_coverage_without_a_positive_sarif_finding():
    document = report(analysis("unknown")).to_dict()
    rendered = html_v2(document)
    assert "format_specific_analysis" in rendered
    sarif = sarif_v2(document)
    assert not sarif["runs"][0]["results"]
    assert (
        sarif["runs"][0]["properties"]["analyses"][0]["coverage_policy"]
        == document["coverage_policy"]
    )


def test_real_jpeg_without_optional_dct_never_reports_no_indicators(tmp_path, monkeypatch):
    from PIL import Image

    from modules.file_structure import FileStructure
    from modules.image_jpeg import ImageJpeg
    from modules.image_jpeg_dct import ImageJpegDct
    from modules.signatures import Signatures
    from registry import Registry

    path = tmp_path / "cover.jpg"
    Image.new("RGB", (32, 32), (50, 60, 70)).save(path)
    monkeypatch.setattr(ImageJpegDct, "available", property(lambda self: False))
    registry = Registry()
    for plugin in (FileStructure(), Signatures(), ImageJpeg(), ImageJpegDct()):
        registry.register(plugin)
    result = AnalysisPipeline(AnalysisService(registry, profile="strict")).analyze(path)
    assert result.verdict == "inconclusive"
    assert result.analysis.overall_score == 0
    policy = result.to_dict()["coverage_policy"]
    assert not policy["complete"]
    assert policy["required"][-1]["component"] == "image_jpeg_dct"
    from core.ctf import CTFLimits, CTFService, _JobState
    from core.ctf_types import Artifact

    service = CTFService(analysis=AnalysisPipeline(AnalysisService(registry, profile="strict")))
    state = _JobState(tmp_path, CTFLimits())
    artifact = Artifact(
        "root", "cover.jpg", "image/jpeg", path.stat().st_size, "a" * 64, 0, "fixture", path=path
    )
    service._analyze(state, artifact)
    assert service._verdict(state, "completed")[0] == "inconclusive"
    assert any(
        c.component == "image_jpeg_dct" and c.required and c.status == "unavailable"
        for c in state.coverage
    )


def test_required_ctf_coverage_cannot_be_erased_by_a_successful_other_artifact():
    from core.ctf import _deduplicate_coverage
    from core.ctf_types import Coverage

    values = [
        Coverage("native", "available", True),
        Coverage("native", "unavailable", True, "child gap"),
        Coverage("optional", "unavailable"),
        Coverage("optional", "available"),
    ]
    for ordered in (values, list(reversed(values))):
        result = _deduplicate_coverage(ordered)
        assert result[0].status == "unavailable" and result[0].reason == "child gap"
        assert result[1].status == "available"


def test_html_escapes_coverage_and_complete_views_retain_noncertificate_scope():
    document = report(analysis("png")).to_dict()
    assert "Required native coverage: complete" in html_v2(document)
    assert "not a clean-file certificate" in document["verdict_reason"]
    document["coverage_policy"]["required"][0]["component"] = "<script>"
    assert "<script>" not in html_v2(document)
    assert "&lt;script&gt;" in html_v2(document)
