from pathlib import Path

from modules.image_jpeg import ImageJpeg


def test_roundtrip_appended(jpeg_64x64: Path, tmp_outdir: Path):
    c = ImageJpeg()
    payload = b"appended jpeg secret"
    out = tmp_outdir / "out.jpg"
    c.embed(jpeg_64x64, payload, out)
    assert c.extract(out) == payload


def test_clean_jpeg_analyze_low(jpeg_64x64: Path):
    c = ImageJpeg()
    assert c.analyze(jpeg_64x64).suspicion < 30


def test_stego_jpeg_analyze_high(jpeg_64x64: Path, tmp_outdir: Path):
    c = ImageJpeg()
    out = tmp_outdir / "stego.jpg"
    c.embed(jpeg_64x64, b"S" * 200, out)
    assert c.analyze(out).suspicion >= 70
