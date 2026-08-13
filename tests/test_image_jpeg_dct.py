import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import modules.image_jpeg_dct as dct_module
from core.payload import pack, unpack
from modules.image_jpeg_dct import ImageJpegDct


@pytest.fixture
def jpeg_256x256(tmp_path: Path) -> Path:
    path = tmp_path / "cover-large.jpg"
    array = np.random.default_rng(99).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    Image.fromarray(array, "RGB").save(path, format="JPEG", quality=92)
    return path


def test_dct_roundtrip_capacity_and_analysis(jpeg_256x256: Path, tmp_path: Path):
    carrier = ImageJpegDct()
    assert carrier.available
    payload = pack(
        payload=b"DCT secret",
        encrypted=False,
        salt=b"\x00" * 16,
        nonce=b"\x00" * 12,
    )
    assert carrier.capacity(jpeg_256x256) > len(payload)
    out = tmp_path / "dct.jpg"
    carrier.embed_with_options(jpeg_256x256, payload, out, steg_key="placement")
    recovered = carrier.extract_with_options(out, steg_key="placement")
    assert unpack(recovered).payload == b"DCT secret"
    assert carrier.analyze(out).suspicion >= 95
    assert Image.open(out).size == (256, 256)


def test_dct_requires_key_and_rejects_small_capacity(
    jpeg_64x64: Path, tmp_path: Path
):
    carrier = ImageJpegDct()
    with pytest.raises(ValueError, match="requires"):
        carrier.embed(jpeg_64x64, b"x", tmp_path / "x.jpg")
    with pytest.raises(ValueError, match="requires"):
        carrier.extract(jpeg_64x64)
    with pytest.raises(ValueError, match="requires"):
        carrier.embed_with_options(jpeg_64x64, b"x", tmp_path / "x.jpg")
    with pytest.raises(Exception, match="capacity"):
        carrier.embed_with_options(
            jpeg_64x64,
            b"x" * 10000,
            tmp_path / "x.jpg",
            steg_key="key",
        )


def test_progressive_jpeg_reports_unsupported(tmp_path: Path):
    path = tmp_path / "progressive.jpg"
    array = np.random.default_rng(2).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(array, "RGB").save(path, progressive=True)
    result = ImageJpegDct().analyze(path)
    assert result.status == "unsupported"
    assert result.suspicion == 0


def test_dct_worker_timeout_isolated(jpeg_64x64: Path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("worker", 15)

    monkeypatch.setattr(dct_module.subprocess, "run", timeout)
    result = ImageJpegDct().analyze(jpeg_64x64)
    assert result.status == "error"
    assert "exceeded" in (result.error or "")


def test_dct_worker_crash_is_structured(jpeg_64x64: Path, monkeypatch):
    def failed(*args, **kwargs):
        return subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr(dct_module.subprocess, "run", failed)
    result = ImageJpegDct().analyze(jpeg_64x64)
    assert result.status == "error"
    assert "boom" in (result.error or "")
