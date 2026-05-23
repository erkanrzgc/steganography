from pathlib import Path

from modules.text_whitespace import TextWhitespace


def test_roundtrip(tmp_path: Path, tmp_outdir: Path):
    src = tmp_path / "cover.txt"
    src.write_text(
        "line one\nline two\nline three\nline four\n"
        "line five\nline six\nline seven\nline eight\n",
        encoding="utf-8",
    )
    c = TextWhitespace()
    out = tmp_outdir / "out.txt"
    payload = b"x"
    c.embed(src, payload, out)
    assert c.extract(out) == payload


def test_analyze_detects_trailing_whitespace(tmp_path: Path, tmp_outdir: Path):
    src = tmp_path / "cover.txt"
    src.write_text("a\nb\nc\nd\ne\nf\ng\nh\n", encoding="utf-8")
    out = tmp_outdir / "out.txt"
    TextWhitespace().embed(src, b"!", out)
    assert TextWhitespace().analyze(out).suspicion >= 50
