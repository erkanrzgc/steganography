# tests/conftest.py
"""Shared pytest fixtures."""
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tmp_outdir(tmp_path: Path) -> Path:
    """Throwaway output directory for embed tests."""
    out = tmp_path / "out"
    out.mkdir()
    return out


@pytest.fixture(autouse=True)
def _reset_config_load_state(monkeypatch):
    """Force config to re-load .env on each test so test ordering is irrelevant."""
    try:
        import config
        monkeypatch.setattr(config, "_ENV_LOADED", False)
    except ImportError:
        pass


@pytest.fixture
def png_64x64(tmp_path: Path) -> Path:
    """64x64 RGB PNG with deterministic noise."""
    rng = np.random.default_rng(seed=42)
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    p = tmp_path / "cover.png"
    Image.fromarray(arr, "RGB").save(p, format="PNG")
    return p


@pytest.fixture
def jpeg_64x64(tmp_path: Path) -> Path:
    """64x64 RGB JPEG with deterministic noise."""
    rng = np.random.default_rng(seed=7)
    arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    p = tmp_path / "cover.jpg"
    Image.fromarray(arr, "RGB").save(p, format="JPEG", quality=85)
    return p


@pytest.fixture
def wav_pcm16(tmp_path: Path) -> Path:
    """1 second of 16-bit mono PCM @ 8 kHz, deterministic noise."""
    rng = np.random.default_rng(seed=11)
    samples = rng.integers(-1000, 1000, size=8000, dtype=np.int16)
    p = tmp_path / "cover.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(samples.tobytes())
    return p
