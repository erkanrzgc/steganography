"""Kernel-reported headroom is an observation, not a replacement tenant hard cap."""

from pathlib import Path

import pytest

from core import colab_memory as m
from core.spatial_colab_readiness import ResourceMetadata


@pytest.fixture(autouse=True)
def unavailable_cgroup(monkeypatch):
    monkeypatch.setattr(m.metadata, "resources", lambda: ResourceMetadata("unavailable", "fixture"))


def sample(tmp_path, total=16 * 1024**2, available=12 * 1024**2):
    path = tmp_path / "meminfo"
    path.write_text(f"MemTotal: {total} kB\nMemAvailable: {available} kB\n")
    return path


@pytest.mark.parametrize("gib", [4, 8, 16, 32, 96])
def test_runtime_capacity_scales_without_eight_gib_ceiling(tmp_path, gib):
    observed = m.observe(meminfo=sample(tmp_path, gib * 1024**2, gib * 1024**2 * 3 // 4))
    m.check(observed)
    assert observed["capacity_bytes"] == gib * 1024**3
    assert observed["headroom_bytes"] == gib * 1024**3 // 10
    assert observed["custom_data_or_rss_ceiling_bytes"] is None
    assert not observed["tenant_limit_verified"]


@pytest.mark.parametrize("raw", ["", "MemTotal: 5 kB\n", "MemTotal: 16 MB\nMemAvailable: 8 kB\n",
                                 "MemTotal: -16 kB\nMemAvailable: 8 kB\n",
                                 "MemTotal: 4 kB\nMemAvailable: 8 kB\n",
                                 "MemTotal: 16 kB\nMemAvailable: 0 kB\n",
                                 "MemTotal: 16 kB\nMemTotal: 16 kB\nMemAvailable: 8 kB\n"])
def test_invalid_or_missing_observation_never_invents_capacity(tmp_path, raw):
    path = tmp_path / "bad"
    path.write_text(raw)
    with pytest.raises(ValueError):
        m.observe(meminfo=path)


def test_observed_cgroup_counters_do_not_mislabel_available_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(m.metadata, "resources", lambda: ResourceMetadata(
        "observed_metadata_only", "not_enforcement", "v2",
        (("memory.max", 8 * 1024**3), ("memory.current", 7 * 1024**3)),
    ))
    observed = m.observe(meminfo=sample(tmp_path))
    assert observed["available_bytes"] == 12 * 1024**3
    assert observed["observed_cgroup_counters"]["memory.max"] == 8 * 1024**3
    assert not observed["tenant_limit_verified"]


def test_headroom_and_changed_capacity_stop_the_job(tmp_path):
    before = m.observe(meminfo=sample(tmp_path))
    with pytest.raises(MemoryError):
        m.check({**before, "available_bytes": before["headroom_bytes"]})
    for field, value in (("capacity_bytes", 32 * 1024**3), ("scope", "different")):
        with pytest.raises(ValueError):
            m.check({**before, field: value}, initial=before)


def test_untrusted_symlink_and_missing_metadata_are_not_followed(tmp_path):
    path = sample(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        m.observe(meminfo=link)
    with pytest.raises(FileNotFoundError):
        m.observe(meminfo=Path(tmp_path / "missing"))
