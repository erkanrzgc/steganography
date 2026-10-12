"""Portable preparation evidence is not a detector or blind recovery score."""

import json
import re
from pathlib import Path


def test_complete_nonjpeg_preparation_keeps_original_denominators_and_gaps():
    root = Path(__file__).resolve().parents[1]
    raw = (root / "benchmarks/nonjpeg-preparation-20261012.json").read_text()
    data = json.loads(raw)
    assert data["status"] == "completed"
    assert data["source_commit"] == "4271395c1ec74c4170acfe24ac21c5d887d59534"
    assert data["original_count"] == 160
    assert sum(data["original_counts"].values()) == 160
    assert set(data["original_counts"].values()) == {32, 8}
    assert data["artifact_count"] == 4080
    assert data["independent_scalar_recoveries"] == 3840
    assert sum(data["by_format_label"].values()) == 4080
    assert sum(v for k, v in data["by_format_label"].items() if k.endswith(":stego")) == 3840
    assert sum(data["cells"].values()) == 3840
    assert len(data["cells"]) == 192
    assert set(data["cells"].values()) == {8, 32}
    assert len(data["source_bindings"]) == 5
    assert len(data["source_sha256"]) == 6
    assert set(data["research_manifests"]) == {"spatial-research.json", "audio-research.json"}
    for field in ("config_sha256", "manifest_sha256", "report_sha256", "audit_sha256"):
        assert re.fullmatch("[a-f0-9]{64}", data[field])
    assert data["test_media_opened"] is False
    assert data["model_trained"] is data["detector_scored"] is data["deployed"] is False
    assert data["gpu_used"] is False and data["cloud_spending"] == 0
    assert data["optimizer_updates"] == 0
    assert data["blind"] is data["autonomous_recovery_measured"] is False
    assert data["accuracy_qualification"] == "unavailable"
    assert data["camera_scene_independence_verified"] is False
    assert data["cover_cleanliness_verified"] is False
    assert all(
        "unavailable" in data["method_readiness"][family]
        for family in (
            "gif-native-palette-frame",
            "text-natural-multilingual",
            "container-blind-polyglot-archive",
            "wav-stereo-frequency",
            "compressed-audio-and-upstream-tools",
        )
    )
    assert "/home/" not in raw and "/tmp/" not in raw  # noqa: S108 - privacy assertion
    assert "password" not in raw and "api_key" not in raw
