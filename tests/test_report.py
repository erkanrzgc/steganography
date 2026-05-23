import json
from pathlib import Path

from core.result import AnalysisResult, Signal
from report.report import write_html, write_json

_FAKE_A = Path("/tmp/a.png")  # noqa: S108 — fake fixture path used only in assertions
_FAKE_B = Path("/tmp/b.png")  # noqa: S108


def _sample():
    return [
        (_FAKE_A, AnalysisResult("image_lsb", 80, (Signal("x", 80, "d"),), "high")),
        (_FAKE_B, AnalysisResult("image_lsb", 10, (), None)),
    ]


def test_write_json_writes_valid_file(tmp_path: Path):
    out = tmp_path / "r.json"
    write_json(_sample(), out)
    data = json.loads(out.read_text())
    assert len(data["files"]) == 2
    assert data["files"][0]["suspicion"] == 80


def test_write_html_writes_renderable(tmp_path: Path):
    out = tmp_path / "r.html"
    write_html(_sample(), out)
    text = out.read_text()
    assert "<html" in text.lower()
    assert "image_lsb" in text
    assert str(_FAKE_A) in text
