from pathlib import Path

from modules.text_zerowidth import TextZeroWidth


def test_roundtrip(tmp_path: Path, tmp_outdir: Path):
    src = tmp_path / "cover.txt"
    src.write_text("Hello, world! This is cover text.\n", encoding="utf-8")
    c = TextZeroWidth()
    out = tmp_outdir / "out.txt"
    payload = b"hi"
    c.embed(src, payload, out)
    assert c.extract(out) == payload


def test_analyze_clean_low(tmp_path: Path):
    src = tmp_path / "clean.txt"
    src.write_text("plain ASCII only", encoding="utf-8")
    assert TextZeroWidth().analyze(src).suspicion < 20


def test_analyze_stego_high(tmp_path: Path, tmp_outdir: Path):
    src = tmp_path / "cover.txt"
    src.write_text("some carrier text", encoding="utf-8")
    out = tmp_outdir / "out.txt"
    TextZeroWidth().embed(src, b"abc", out)
    assert TextZeroWidth().analyze(out).suspicion >= 70
