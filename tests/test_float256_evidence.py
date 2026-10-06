"""Published preprocessing evidence cannot masquerade as real model accuracy."""

import hashlib
import json
from pathlib import Path


def test_float_preparation_evidence():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks/float256-preparation-20261006.json"
    record = json.loads(path.read_bytes())
    assert (
        record["protocol_sha256"]
        == hashlib.sha256((root / "docs/JPEG_FLOAT256_PROTOCOL.md").read_bytes()).hexdigest()
    )
    assert record["execution"]["implementation_protocol_commit"].startswith("467f02f")
    assert record["files_rehashed"] == 3750 and record["mathematical_audit_passed"]
    assert (
        record["qualification"]
        == record["real_training"]
        == record["accuracy_metrics"]
        == "unavailable"
    )
    assert not record["native_parser_independently_verified"]
    assert not record["deployed"] and not record["primary_detection_changed"]
    assert (
        record["splits"]["train"]["rows"] == 2985 and record["splits"]["validation"]["rows"] == 765
    )
    assert len(record["examples"]) == 18
    assert len({(e["split"], e["quality_factor"], e["method"]) for e in record["examples"]}) == 18
    assert all(e["passed"] and e["maximum_pixel_difference"] == 0 for e in record["examples"])
    assert record["absolute_tolerance"] == 1e-4 and record["relative_tolerance"] == 0
    for split, info in record["splits"].items():
        assert info["bytes"] == info["rows"] * 256 * 256 * 4
        assert info["data_sha256"] == record["execution"]["splits"][split]["data_sha256"]
    assert "/home/" not in path.read_text() and ".benchmark/" not in path.read_text()
