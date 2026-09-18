"""Tests for JPEG DCT and DQT steganalysis detectors."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.service import AnalysisService
from modules.image_jpeg_dct import ImageJpegDct

jpeglib = pytest.importorskip("jpeglib", reason="jpeglib is required for DCT steganalysis")


def _generate_clean_jpeg(path: Path, quality: int = 85) -> Path:
    x = np.linspace(0, 4 * np.pi, 256)
    y = np.linspace(0, 4 * np.pi, 256)
    xx, yy = np.meshgrid(x, y)
    base = ((np.sin(xx) * np.cos(yy) + 1.0) * 110.0 + 15.0).astype(np.uint8)
    image = Image.fromarray(np.stack([base, base, base], axis=2), "RGB")
    image.save(path, format="JPEG", quality=quality)
    return path


def _simulate_jsteg(src: Path, out: Path, rate: float = 0.5, seed: int = 42) -> Path:
    """Simulate JSteg by equalizing AC DCT coefficient LSBs on eligible positions."""
    image = jpeglib.read_dct(str(src))
    coeffs = image.Y.reshape(-1)
    pos = np.arange(coeffs.size, dtype=np.int64)
    ac = pos % 64 != 0
    el = pos[ac & (np.abs(coeffs.astype(np.int32)) >= 2)]

    rng = np.random.default_rng(seed)
    mask = rng.random(len(el)) < rate
    target_pos = el[mask]

    values = coeffs[target_pos].astype(np.int32)
    mags = np.abs(values)
    random_bits = rng.integers(0, 2, size=len(values), dtype=np.int32)
    new_mags = (mags & ~1) | random_bits
    coeffs[target_pos] = np.where(values < 0, -new_mags, new_mags).astype(coeffs.dtype)
    image.Y = coeffs.reshape(image.Y.shape)

    image.write_dct(str(out))
    return out


def test_clean_jpeg_dct_has_low_suspicion(tmp_path: Path):
    clean = _generate_clean_jpeg(tmp_path / "clean.jpg")
    service = AnalysisService(profile="balanced")
    analysis = service.analyze(clean)

    assert analysis.severity == "low"
    assert analysis.overall_score < 50


def test_jsteg_dct_chi_square_detection(tmp_path: Path):
    clean = _generate_clean_jpeg(tmp_path / "clean_base.jpg")
    stego = _simulate_jsteg(clean, tmp_path / "stego_jsteg.jpg", rate=0.60)

    analyzer = ImageJpegDct()
    result = analyzer.analyze(stego)

    assert result.status == "ok"
    assert result.suspicion >= 50
    signal_names = {s.name for s in result.signals}
    assert "westfeld_dct_chi_square" in signal_names

    service = AnalysisService(profile="balanced")
    analysis = service.analyze(stego)
    assert analysis.overall_score >= 50


def test_jpeg_dqt_flat_quantization_detection(tmp_path: Path):
    flat_jpg = tmp_path / "flat_q100.jpg"
    _generate_clean_jpeg(flat_jpg, quality=100)

    service = AnalysisService(profile="balanced")
    analysis = service.analyze(flat_jpg)

    dqt_signals = [
        s for r in analysis.results for s in r.signals if s.name == "jpeg_quantization_tables"
    ]
    assert len(dqt_signals) >= 1
    assert "flat" in dqt_signals[0].detail.lower() or "q100" in dqt_signals[0].detail.lower()


def test_jsteg_header_detection(tmp_path: Path):
    import struct

    clean = _generate_clean_jpeg(tmp_path / "clean_base.jpg")
    image = jpeglib.read_dct(str(clean))
    coeffs = image.Y.reshape(-1)
    pos = np.arange(coeffs.size, dtype=np.int64)
    ac = pos % 64 != 0
    el = pos[ac & (np.abs(coeffs.astype(np.int32)) >= 2)]

    payload = b"SECRET_CTF_FLAG{jsteg_dct_flag}"
    header = b"jsteg" + struct.pack("<I", len(payload)) + payload
    bits = []
    for b in header:
        for bit_idx in range(8):
            bits.append((b >> bit_idx) & 1)

    target_pos = el[: len(bits)]
    values = coeffs[target_pos].astype(np.int32)
    mags = np.abs(values)
    new_mags = (mags & ~1) | np.array(bits, dtype=np.int32)
    coeffs[target_pos] = np.where(values < 0, -new_mags, new_mags).astype(coeffs.dtype)
    image.Y = coeffs.reshape(image.Y.shape)
    stego_path = tmp_path / "stego_header.jpg"
    image.write_dct(str(stego_path))

    analyzer = ImageJpegDct()
    result = analyzer.analyze(stego_path)
    assert result.status == "ok"
    assert result.suspicion >= 90
    signal_names = {s.name for s in result.signals}
    assert "jsteg_header" in signal_names


def test_jsteg_native_extraction(tmp_path: Path):
    import struct

    from modules.image_jpeg_dct import extract_jsteg

    clean = _generate_clean_jpeg(tmp_path / "clean_base.jpg")
    image = jpeglib.read_dct(str(clean))
    coeffs = image.Y.reshape(-1)
    pos = np.arange(coeffs.size, dtype=np.int64)
    ac = pos % 64 != 0
    el = pos[ac & (np.abs(coeffs.astype(np.int32)) >= 2)]

    payload = b"FLAG{jsteg_native_extraction_works}"
    header = b"jsteg" + struct.pack("<I", len(payload)) + payload
    bits = []
    for b in header:
        for bit_idx in range(8):
            bits.append((b >> bit_idx) & 1)

    target_pos = el[: len(bits)]
    values = coeffs[target_pos].astype(np.int32)
    mags = np.abs(values)
    new_mags = (mags & ~1) | np.array(bits, dtype=np.int32)
    coeffs[target_pos] = np.where(values < 0, -new_mags, new_mags).astype(coeffs.dtype)
    image.Y = coeffs.reshape(image.Y.shape)
    stego_path = tmp_path / "stego_extract.jpg"
    image.write_dct(str(stego_path))

    recovered = extract_jsteg(stego_path)
    assert recovered == payload


def test_ctf_solves_jsteg(tmp_path: Path):
    import struct

    from core.ctf import CTFService

    clean = _generate_clean_jpeg(tmp_path / "clean_base.jpg")
    image = jpeglib.read_dct(str(clean))
    coeffs = image.Y.reshape(-1)
    pos = np.arange(coeffs.size, dtype=np.int64)
    ac = pos % 64 != 0
    el = pos[ac & (np.abs(coeffs.astype(np.int32)) >= 2)]

    flag = b"flag{automated_ctf_jsteg_solved}"
    header = b"jsteg" + struct.pack("<I", len(flag)) + flag
    bits = []
    for b in header:
        for bit_idx in range(8):
            bits.append((b >> bit_idx) & 1)

    target_pos = el[: len(bits)]
    values = coeffs[target_pos].astype(np.int32)
    mags = np.abs(values)
    new_mags = (mags & ~1) | np.array(bits, dtype=np.int32)
    coeffs[target_pos] = np.where(values < 0, -new_mags, new_mags).astype(coeffs.dtype)
    image.Y = coeffs.reshape(image.Y.shape)
    stego_path = tmp_path / "stego_ctf.jpg"
    image.write_dct(str(stego_path))

    engine = CTFService()
    result = engine.solve(stego_path, output_dir=tmp_path / "ctf_out")
    assert result.verdict == "confirmed"
    recovered_artifacts = [a for a in result.artifacts if "recovered.bin" in a.name]
    assert len(recovered_artifacts) >= 1
    assert (tmp_path / "ctf_out" / "artifacts" / recovered_artifacts[0].name).read_bytes() == flag


