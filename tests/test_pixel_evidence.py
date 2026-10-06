"""Pixel preparation proves integrity/crop parity, never a model accuracy claim."""

import hashlib
import json
from pathlib import Path


def test_preparation_evidence_integrity():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks/pixel-residual-preparation-20261005.json"
    record = json.loads(path.read_bytes())
    assert (
        record["protocol_sha256"]
        == hashlib.sha256(
            (root / "docs/PIXEL_RESIDUAL_PREPARATION_PROTOCOL.md").read_bytes()
        ).hexdigest()
    )
    assert record["preregistered_implementation_protocol_commit"] == "a6d0f2b"
    assert record["execution"]["implementation_protocol_commit"].startswith("a6d0f2b")
    assert record["execution"]["protocol_sha256"] == record["protocol_sha256"]
    previous = json.loads((root / "benchmarks/jrm-transfer-development-20261005.json").read_bytes())
    assert record["manifest_sha256"] == previous["manifest_sha256"]
    assert not any(
        record[k] for k in ("trained", "deployed", "calibrated", "primary_detection_changed")
    )
    assert record["accuracy"] == record["qualification"] == "unavailable"
    for split, rows in (("train", 2985), ("validation", 765)):
        cache = record["caches"][split]
        assert cache["rows"] == rows and cache["shape"] == [rows, 1, 128, 128]
        assert cache["bytes"] == rows * 16384
        assert cache["descriptor_sha256"] == record["artifact_sha256"][f"{split}/cache.json"]
        assert cache["data_sha256"] == record["artifact_sha256"][f"{split}/pixels.u8"]
    audit = record["independent_audit"]
    assert audit["passed"] and audit["files_rehashed"] == 3750
    assert audit["crop_examples_checked"] == len(audit["crop_examples"]) == 18
    assert {e["split"] for e in audit["crop_examples"]} == {"train", "validation"}
    assert {e["method"] for e in audit["crop_examples"]} == {None, "JUNIWARD", "UERD"}
    assert {e["quality_factor"] for e in audit["crop_examples"]} == {None, 75, 95}
    assert not audit["independent_jpeg_decoder"]
    assert "/home/" not in path.read_text() and ".benchmark/" not in path.read_text()
