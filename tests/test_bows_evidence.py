"""Portable original evidence, without private corpus or any live download."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def evidence(name, digest):
    raw = (ROOT / "benchmarks" / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    assert b"/home/" not in raw and b"password" not in raw and b".ssh" not in raw
    return json.loads(raw)


def test_diversity_counts_are_originals_not_accuracy():
    acquired = evidence(
        "bows-diversity-acquisition-20261010.json",
        "9b64ee28ef81819927343476862c8acaf5d7b61421b578309ab98fa68873d7a0",
    )
    audit = evidence(
        "bows-diversity-independent-audit-20261010.json",
        "5f3f945a606fd5f6cf3cffe5bfb632b2732523ae9adfecbfc80241e0974b5536",
    )
    assert acquired["status"] == audit["status"] == "completed"
    assert acquired["originals"] == audit["originals"] == 1001
    assert acquired["unique_train_originals"] == audit["split"]["train"] == 807
    assert acquired["unique_validation_originals"] == audit["split"]["validation"] == 194
    assert acquired["source_manifest_sha256"] == audit["source_manifest_sha256"]
    assert acquired["archive_sha256"] == audit["archive_sha256"]
    assert len(acquired["reserved_manifest_sha256"]) == 8
    assert acquired["reserved_manifest_sha256"] == audit["reserved_manifest_sha256"]
    assert audit["prior_boss_originals_pixel_checked"] == 3000
    assert audit["exact_identity_overlap"] == audit["decoded_boss_pixel_overlap"] == 0
    assert audit["camera_scene_independence"] == "unverified"
    assert not acquired["model_trained"] and not audit["model_trained"]
    assert not audit["WIFD_training_used"] and audit["prepared_training_rows"] == 0
    assert acquired["accuracy_qualification"] == audit["accuracy_qualification"] == "unavailable"
    assert not acquired["upstream_sha256_verified"]
    protocol = ROOT / "docs/BOWS_VERIFIED_LAYOUT_PROTOCOL.md"
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == acquired["protocol_sha256"]


def test_failed_attempts_are_not_retroactively_passed():
    for name, protocol, digest in (
        (
            "bows-first-acquisition-failure-20261010.json",
            "BOWS_DIVERSITY_PROTOCOL.md",
            "5892a7e112a82e8ab346d2ba859c1b7116075f6a9d9692a887d4b7f3b86d4dd7",
        ),
        (
            "bows-native-name-failure-20261010.json",
            "BOWS_NATIVE_RETRY_PROTOCOL.md",
            "e11f0c8837fdf1a4d0f5730e7f9ec3a65758d57706b4a7380e0d1b6357ccee74",
        ),
    ):
        report = json.loads((ROOT / "benchmarks" / name).read_bytes())
        assert report["status"] == "failed" and report["retained_originals"] == 0
        assert not report["model_trained"] and report["accuracy_qualification"] == "unavailable"
        assert (
            report["protocol_sha256"]
            == hashlib.sha256((ROOT / "docs" / protocol).read_bytes()).hexdigest()
            == digest
        )
