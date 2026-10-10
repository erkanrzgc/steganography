"""Published real preparation is not measured GPU timing or accuracy."""

import hashlib
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    raw = (ROOT / "benchmarks" / name).read_bytes()
    for forbidden in (b"/home/", b".benchmark/", b"password", b"api_key", b"access_token"):
        assert forbidden not in raw
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def test_complete_portable_evidence_and_source_binding():
    prepared, _ = read("jpeg-real-timing-preparation-20261010.json")
    audited, checksum = read("jpeg-real-timing-independent-audit-20261010.json")
    protocol, protocol_sha = read("jpeg-real-timing-protocol-20261010.json")
    assert prepared["independent_audit_sha256"] == checksum
    assert prepared["protocol_sha256"] == audited["protocol_sha256"] == protocol_sha
    assert prepared["manifest_sha256"] == audited["manifest_sha256"]
    assert (
        prepared["rows_per_source"]
        == audited["rows_per_source"]
        == {"ALASKA2": 96, "BOSSbase-1.01": 192, "BOWS2": 192}
    )
    assert prepared["originals"] == audited["originals"] == 96
    assert prepared["jpeg_rows"] == audited["jpeg_rows"] == audited["independent_IDCT_rows"] == 480
    assert prepared["tensor_bytes"] == audited["tensor_bytes"] == 480 * 256 * 256 * 4
    assert prepared["timing_kit_epoch_updates"] == protocol["kit_epoch_updates"] == 192
    assert (
        prepared["prospective_full_corpus_epoch_updates"]
        == protocol["prospective_full_corpus_epoch_updates"]
        == 4932
    )
    assert audited["BOWS_coefficient_pairs_checked"] == 128
    assert audited["maximum_pixel_difference"] <= audited["absolute_tolerance"] == 1e-4
    assert (
        audited["relative_tolerance"] == 0 and not audited["native_parser_independently_verified"]
    )
    for result in (prepared, audited):
        assert result["status"] == "completed"
        assert result["validation_pixels_read"] == result["cloud_resources_created"] == 0
        assert result["model_trained"] is False
        assert result["accuracy_qualification"] == result["GPU_timing"] == "unavailable"
    for name, expected in prepared["execution_source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    assert (
        hashlib.sha256((ROOT / "scripts/audit-jpeg-timing.py").read_bytes()).hexdigest()
        == audited["audit_script_sha256"]
    )


def test_audit_failure_not_erased():
    failure, _ = read("jpeg-real-timing-audit-bound-failure-20261010.json")
    audited, _ = read("jpeg-real-timing-independent-audit-20261010.json")
    assert failure["status"] == "failed"
    assert failure["manifest_sha256"] == audited["manifest_sha256"]
    assert not failure["model_trained"]
    assert failure["accuracy_qualification"] == "unavailable"


def test_independent_auditor_paths_regular_files_and_limits(tmp_path):
    code = runpy.run_path(str(ROOT / "scripts/audit-jpeg-timing.py"))
    for unsafe in ("../file", "/absolute", "back\\slash", "", None):
        with pytest.raises(ValueError):
            code["name"](unsafe)
    assert code["name"]("jpeg/digest.jpg") == "jpeg/digest.jpg"
    data = tmp_path / "data.json"
    data.write_bytes(b"{}")
    assert code["read"](data, 2) == b"{}"
    assert code["doc"](data)[0] == {}
    with pytest.raises(ValueError, match="limit"):
        code["read"](data, 1)
    with pytest.raises(ValueError, match="binding"):
        code["doc"](data, "a" * 64)
    data.write_bytes(b"[]")
    with pytest.raises(ValueError, match="object"):
        code["doc"](data)
    link = tmp_path / "link"
    link.symlink_to(data)
    with pytest.raises(ValueError, match="symlink"):
        code["read"](link, 10)
    with pytest.raises(ValueError, match="regular"):
        code["read"](tmp_path, 10)
