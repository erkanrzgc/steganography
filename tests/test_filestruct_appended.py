from pathlib import Path

from modules.filestruct_appended import FilestructAppended


def test_detects_zip_after_png(png_64x64: Path, tmp_path: Path):
    poly = tmp_path / "poly.png"
    poly.write_bytes(png_64x64.read_bytes() + b"PK\x03\x04" + b"\x00" * 30)
    r = FilestructAppended().analyze(poly)
    assert r.suspicion >= 80


def test_clean_png_no_appended(png_64x64: Path):
    r = FilestructAppended().analyze(png_64x64)
    assert r.suspicion < 30
