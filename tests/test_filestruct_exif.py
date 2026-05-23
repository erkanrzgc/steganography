from pathlib import Path

from modules.filestruct_exif import FilestructExif


def test_roundtrip_jpeg_exif_usercomment(jpeg_64x64: Path, tmp_outdir: Path):
    c = FilestructExif()
    out = tmp_outdir / "out.jpg"
    payload = b"exif secret"
    c.embed(jpeg_64x64, payload, out)
    assert c.extract(out) == payload


def test_analyze_flags_present_usercomment(jpeg_64x64: Path, tmp_outdir: Path):
    c = FilestructExif()
    out = tmp_outdir / "out.jpg"
    c.embed(jpeg_64x64, b"x" * 30, out)
    assert c.analyze(out).suspicion >= 50
