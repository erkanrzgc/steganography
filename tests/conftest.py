# tests/conftest.py
"""Shared pytest fixtures."""
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
