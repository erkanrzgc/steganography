import json
import zipfile
from html.parser import HTMLParser
from pathlib import Path

import pytest

from core.ctf import CTFService
from report.v2 import html_v2, json_v2, sarif_v2, write_evidence_bundle


class NormalizedReportParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_report = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "pre" and dict(attrs).get("id") == "normalized-report":
            self.in_report = True

    def handle_endtag(self, tag):
        if tag == "pre":
            self.in_report = False

    def handle_data(self, data):
        if self.in_report:
            self.parts.append(data)


def normalized_html(document):
    parser = NormalizedReportParser()
    parser.feed(document)
    return json.loads("".join(parser.parts))


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_ctf_complete_evidence_survives_every_view(tmp_path, status):
    hostile = '</pre><script>alert("injection")</script>&'
    value = {
        "schema_version": "2.0", "schema_revision": 1, "report_type": "ctf",
        "status": status, "verdict": "inconclusive", "verdict_reason": hostile,
        "analyses": [], "artifacts": [],
        "recommendations": [{"priority": 1, "action": hostile,
                             "rationale": "not calibrated", "method_family": "wav-lsb"}],
        "coverage": [{"component": hostile, "status": "failed", "required": True,
                      "reason": hostile}],
        "tool_executions": [{"tool": hostile, "version": "1.2", "status": "timed_out",
                             "exit_code": -9, "duration_ms": 30, "command": ["[REDACTED]"]}],
        "future_additive_field": {"unicode": "Türkçe", "nested": [1, None, hostile]},
    }
    document = html_v2(value)
    assert "<script>" not in document
    assert "Recommendations" in document and "wav-lsb" in document
    assert "required" in document and "timed_out" in document
    assert normalized_html(document) == value
    sarif = sarif_v2(value)
    assert sarif["runs"][0]["properties"]["report"] == value
    assert sarif["runs"][0]["results"] == []
    output = tmp_path / "evidence.zip"
    write_evidence_bundle(value, [], output)
    with zipfile.ZipFile(output) as bundle:
        assert json.loads(bundle.read("report.json")) == value
        assert normalized_html(bundle.read("report.html").decode()) == value
        assert json.loads(bundle.read("report.sarif"))["runs"][0]["properties"]["report"] == value


@pytest.mark.parametrize("report", [
    {}, {"files": [{"file": {"name": "empty.txt"}, "findings": []}]},
    {"report_type": "ctf", "analyses": []},
])
def test_empty_and_non_ctf_views_keep_additive_contract(report):
    assert normalized_html(html_v2(report)) == json.loads(json_v2(report))
    assert sarif_v2(report)["runs"][0]["properties"]["report"] == report


def test_actual_service_recommendations_and_coverage_roundtrip(tmp_path: Path):
    source = tmp_path / "input.txt"
    source.write_text("Example" + "\u200b" * 32, encoding="utf-8")
    value = CTFService().solve(source, tmp_path / "job", mode="quick").to_dict()
    assert value["recommendations"][0]["method_family"] == "text-zerowidth"
    assert normalized_html(html_v2(value)) == value
    assert sarif_v2(value)["runs"][0]["properties"]["report"] == value
    assert str(tmp_path) not in html_v2(value)
