import json
from pathlib import Path

from core.result import AnalysisResult, FileAnalysis, FileInfo, ScanReport, Signal
from report.report import write_html, write_json_v1


def _report(path: str = "bad<script>alert(1)</script>.png") -> ScanReport:
    info = FileInfo(path, Path(path).name, 10, "abc", "png", "image/png", ".png", False)
    result = AnalysisResult(
        "<img src=x onerror=alert(1)>",
        90,
        (Signal("<script>x</script>", 90, "<b>detail</b>"),),
        None,
    )
    item = FileAnalysis(info, 90, "high", "sensitive", (result,))
    return ScanReport((item,), "sensitive")


def test_json_v1_schema_and_summary(tmp_path: Path):
    out = tmp_path / "report.json"
    write_json_v1(_report(), out)
    data = json.loads(out.read_text())
    assert data["schema_version"] == "1.0"
    assert data["summary"]["high"] == 1
    assert data["files"][0]["overall"]["score"] == 90


def test_html_v1_escapes_all_dynamic_fields(tmp_path: Path):
    out = tmp_path / "report.html"
    write_html(_report(), out)
    document = out.read_text()
    assert "<script>x</script>" not in document
    assert "<img src=x onerror=alert(1)>" not in document
    assert "&lt;script&gt;" in document
    assert "Filter files" in document
