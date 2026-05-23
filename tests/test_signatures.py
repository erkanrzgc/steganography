from pathlib import Path

from modules.signatures import Signatures


def test_detects_openstego_marker(tmp_path: Path):
    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100 + b"OPENSTEGO" + b"\x00" * 50)
    assert Signatures().analyze(p).suspicion >= 80


def test_clean_returns_zero(tmp_path: Path):
    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)
    assert Signatures().analyze(p).suspicion == 0
