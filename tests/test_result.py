from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from core.result import AnalysisResult, EmbedResult, Signal


def test_embed_result_is_frozen():
    r = EmbedResult(
        carrier="image_lsb", out_path=Path("/tmp/x.png"), bytes_written=42, encrypted=True  # noqa: S108
    )
    with pytest.raises(FrozenInstanceError):
        r.bytes_written = 99


def test_signal_is_frozen_and_has_score():
    s = Signal(name="lsb_chi_square", score=87, detail="strong LSB anomaly")
    assert s.score == 87
    with pytest.raises(FrozenInstanceError):
        s.score = 0


def test_analysis_result_aggregates_signals():
    sigs = (Signal("a", 10, "x"), Signal("b", 90, "y"))
    r = AnalysisResult(analyzer="image_lsb", suspicion=50, signals=sigs, explanation=None)
    assert r.signals[1].name == "b"
    assert r.suspicion == 50
