from pathlib import Path

import pytest

from core.ctf import _METHOD_HINTS, CTFLimits, CTFService, _JobState
from core.ctf_types import Coverage


def state(tmp_path, findings=()):
    value = _JobState(tmp_path, CTFLimits())
    value.analyses.append({"findings": list(findings)})
    return value


def finding(rule, evidence="heuristic", score=0.7, status="ok"):
    return {"rule_id": rule, "evidence_strength": evidence, "confidence": score,
            "status": status, "detail": "/private/secret password=do-not-publish"}


@pytest.mark.parametrize("rule", sorted(_METHOD_HINTS))
def test_allowlisted_native_routes(tmp_path, rule):
    hints = CTFService._recommendations(state(tmp_path, [finding(rule)]), "balanced")
    assert hints[0].method_family == _METHOD_HINTS[rule][0]
    assert "not a calibrated" in hints[0].rationale
    assert "do-not-publish" not in repr(hints)


@pytest.mark.parametrize("override", [
    {"rule_id": "unknown.plugin"}, {"rule_id": []},
    {"status": "failed"}, {"status": "unsupported"},
    {"evidence_strength": "informational"}, {"evidence_strength": []},
    {"confidence": None}, {"confidence": True}, {"confidence": "1"},
    {"confidence": float("nan")}, {"confidence": float("inf")},
    {"confidence": -1}, {"confidence": 1.01}, {"confidence": 0.29},
    {"evidence_strength": "verified"},
])
def test_invalid_and_non_evidence_do_not_diagnose(tmp_path, override):
    value = finding("audio_wav.wav_lsb_bias") | override
    hints = CTFService._recommendations(state(tmp_path, [value]), "deep")
    assert len(hints) == 1 and hints[0].method_family is None


def test_strength_order_deduplication_and_stable_ties(tmp_path):
    values = [finding("image_lsb.lsb_channel_bias", score=1),
              finding("text_whitespace.trailing_whitespace_ratio", score=0.9),
              finding("text_whitespace.validated_whitespace_payload", "verified", 0.98),
              finding("text_zerowidth.zero_width_chars", "strong", 0.7),
              finding("image_gif.gif_comment_payload", "strong", 0.7),
              finding("image_gif.gif_comment_payload", score=0.3)]
    hints = CTFService._recommendations(state(tmp_path, values), "quick")
    assert [h.method_family for h in hints] == ["text-whitespace", "gif-comment", "text-zerowidth"]
    assert [h.priority for h in hints] == [1, 2, 3]
    assert "reports a verified marker" in hints[0].rationale


@pytest.mark.parametrize("status", ["unavailable", "failed", "unsupported"])
def test_native_gap_not_misrepresented_as_optional_tools(tmp_path, status):
    value = state(tmp_path)
    value.coverage = [Coverage("/private/secret", status, True)]
    hints = CTFService._recommendations(value, "quick")
    assert hints[0].action == "Restore required native analyzer coverage"
    assert "external tools alone" in hints[0].rationale
    assert "/private" not in repr(hints)


def test_optional_absence_is_not_method_proof(tmp_path):
    value = state(tmp_path)
    value.coverage = [Coverage("steghide", "unavailable", reason="secret password")]
    hints = CTFService._recommendations(value, "balanced")
    assert hints[0].method_family == "external-tools"
    assert "secret" not in repr(hints)
    assert all(h.method_family != "steghide" for h in hints)


def test_empty_deep_analysis_and_extension_do_not_invent_method(tmp_path):
    value = state(tmp_path)
    value.analyses[0]["file"] = {"name": "secret.jpg", "detected_type": "jpeg"}
    assert [h.method_family for h in CTFService._recommendations(value, "deep")] == [None]


def test_text_hint_without_payload_recovery(tmp_path: Path):
    # Independent, undecodable zero-width text: triage must not imply recovery.
    source = tmp_path / "input.txt"
    source.write_text("Ordinary text." + "\u200b" * 32, encoding="utf-8")
    report = CTFService().solve(source, tmp_path / "job", mode="quick")
    assert report.recommendations[0].method_family == "text-zerowidth"
    assert report.verdict != "confirmed"
    assert len(report.artifacts) == 1
    assert "method_family" in report.to_dict()["recommendations"][0]
