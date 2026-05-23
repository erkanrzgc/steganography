# tests/conftest.py
"""Shared pytest fixtures."""
from pathlib import Path

import pytest


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
