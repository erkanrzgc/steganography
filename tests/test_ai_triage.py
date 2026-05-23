from pathlib import Path

from core.result import Signal
from modules.ai_triage import AITriage, set_provider


def test_heuristic_fallback_uses_signals(tmp_path: Path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"x")
    t = AITriage()
    # No provider registered → heuristic-only.
    signals = (Signal("a", 30, ""), Signal("b", 80, ""))
    r = t.score_from_signals(signals)
    assert r.suspicion == 80
    assert r.explanation is not None


def test_pluggable_provider_overrides_score(tmp_path: Path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"x")

    def fake_provider(file_info: dict, signals: list[Signal]) -> tuple[int, str]:
        return 42, "ai says 42"

    set_provider(fake_provider)
    try:
        r = AITriage().analyze(p)
        assert r.suspicion == 42
        assert r.explanation == "ai says 42"
    finally:
        set_provider(None)
