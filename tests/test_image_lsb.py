from pathlib import Path

import pytest

from core.carrier import InsufficientCapacityError
from modules.image_lsb import ImageLsb


def test_roundtrip_embed_extract(png_64x64: Path, tmp_outdir: Path):
    c = ImageLsb()
    payload = b"hello stego world"
    out = tmp_outdir / "out.png"
    res = c.embed(png_64x64, payload, out)
    assert res.bytes_written == len(payload)
    assert out.exists()
    assert c.extract(out) == payload


def test_capacity_is_pixels_times_3_bits_div_8(png_64x64: Path):
    c = ImageLsb()
    # 64*64*3 bits = 12288 bits = 1536 bytes; usable = 1536 - 4 (length prefix)
    assert c.capacity(png_64x64) == 64 * 64 * 3 // 8 - 4


def test_oversized_payload_rejected(png_64x64: Path, tmp_outdir: Path):
    c = ImageLsb()
    too_big = b"x" * (c.capacity(png_64x64) + 1)
    with pytest.raises(InsufficientCapacityError):
        c.embed(png_64x64, too_big, tmp_outdir / "out.png")


def test_analyze_clean_image_low_suspicion(png_64x64: Path):
    c = ImageLsb()
    r = c.analyze(png_64x64)
    assert 0 <= r.suspicion <= 100
    # Random noise should not score very high.
    assert r.suspicion < 70


def test_analyze_stego_image_higher_than_clean(png_64x64: Path, tmp_outdir: Path):
    c = ImageLsb()
    out = tmp_outdir / "stego.png"
    c.embed(png_64x64, b"A" * 800, out)
    clean = c.analyze(png_64x64).suspicion
    stego = c.analyze(out).suspicion
    assert stego >= clean
