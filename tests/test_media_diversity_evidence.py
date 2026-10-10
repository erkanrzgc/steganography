"""Portable complete acquisition evidence, never live downloads or accuracy."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def evidence(name, digest):
    raw = (ROOT / "benchmarks" / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    for forbidden in (b"/home/", b"password", b".ssh", b"api_key", b"access_token"):
        assert forbidden not in raw
    return json.loads(raw)


def test_complete_media_acquisition_and_independent_audit_match():
    acquired = evidence(
        "media-diversity-acquisition-20261010.json",
        "45d8ecc3c2a3c41e85b607dd49ba5f4430dedfb117980c436f6092e9413b81ce",
    )
    audited = evidence(
        "media-diversity-independent-audit-20261010.json",
        "3f6f5839bf0bba04f98085812475644ca61e61ee12612a0e575d8d0149778d24",
    )
    assert acquired["status"] == audited["status"] == "completed"
    assert acquired["originals"] == audited["originals_audited"] == 2900
    expected = {
        "div2k-train": (800, {"train": 800}),
        "div2k-validation": (100, {"validation": 100}),
        "esc50": (2000, {"train": 1200, "validation": 398, "test": 402}),
    }
    original_records = {r["dataset"]: r for r in acquired["sources"]}
    audit_records = {r["dataset"]: r for r in audited["sources"]}
    assert original_records.keys() == audit_records.keys() == expected.keys()
    for name, (count, splits) in expected.items():
        original, independent = original_records[name], audit_records[name]
        assert original["originals"] == independent["originals"] == count
        assert original["splits"] == independent["splits"] == splits
        assert original["source_manifest_sha256"] == independent["manifest_sha256"]
        for field in ("archive_sha256", "archive_bytes", "license_evidence_sha256", "license"):
            assert original[field] == independent[field]
        assert original["total_media_bytes"] == independent["media_bytes"]
        assert original["prior_manifest_sha256"] == audited["prior_manifest_sha256"]
        assert len(original["prior_manifest_sha256"]) == 10
        assert not original["upstream_sha256_verified"]
        assert not original["model_trained"] and not original["cover_cleanliness_verified"]
    assert audited["prior_covers_reread"] == 10225
    assert audit_records["esc50"]["original_recording_groups"] == 1524
    for field in (
        "prior_exact_overlap",
        "prior_decoded_overlap",
        "new_duplicate_originals",
        "original_group_split_overlap",
    ):
        assert audited[field] == 0
    for report in (acquired, audited):
        assert report["model_trained"] is False
        assert report["accuracy_qualification"] == "unavailable"
    assert not audited["camera_scene_independence_verified"]
    assert not audited["cover_cleanliness_verified"]


def test_execution_source_and_unchanged_frozen_protocol_bindings():
    acquired = evidence(
        "media-diversity-acquisition-20261010.json",
        "45d8ecc3c2a3c41e85b607dd49ba5f4430dedfb117980c436f6092e9413b81ce",
    )
    for name, digest in acquired["protocol_sha256"].items():
        assert hashlib.sha256((ROOT / "docs" / name).read_bytes()).hexdigest() == digest
    assert len(acquired["protocol_sha256"]) == 3
    assert acquired["protocol_sha256"]["MEDIA_DIVERSITY_PROTOCOL.md"] == (
        "e4b15841887761c974962e9edf07af33c38a3c4be2e4b9ee00f2ab351ac07c83"
    )
    first, second, grouped = acquired["sources"]
    assert (
        first["implementation_sha256"]
        == second["implementation_sha256"]
        == ("b273d24f3fe8403f15df4f780b5dd8daa782a130e2c5a17ed8de71ce7927e8c1")
    )
    assert first["execution_revision"] == second["execution_revision"]
    assert grouped["execution_revision"] != first["execution_revision"]
    audited = evidence(
        "media-diversity-independent-audit-20261010.json",
        "3f6f5839bf0bba04f98085812475644ca61e61ee12612a0e575d8d0149778d24",
    )
    assert audited["audit_source_sha256"] == (
        "66c2f88fb402db2429493c8a468034eeff8d1a90cfbc907496cbae0b11d7d226"
    )


def test_real_failures_preserved_before_grouped_retry():
    legacy = json.loads(
        (ROOT / "benchmarks/media-diversity-first-failure-20261010.json").read_text()
    )
    native = json.loads((ROOT / "benchmarks/esc50-native-fold-failure-20261010.json").read_text())
    assert legacy["status"] == native["status"] == "failed"
    assert not legacy["media_download_started"] and not legacy["output_directories_created"]
    assert native["originals_written"] == 0 and not native["success_manifest_written"]
    assert native["cross_fold_groups"] == {
        "131943": [2, 3],
        "134049": [2, 3],
        "209698": [4, 5],
        "234879": [4, 5],
    }
    assert not legacy["model_trained"] and not native["model_trained"]
    assert legacy["accuracy_qualification"] == native["accuracy_qualification"] == "unavailable"
