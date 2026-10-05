"""Published development evidence must retain failures and match frozen inputs."""

import hashlib
import json
from pathlib import Path

import pytest


def test_nonlinear_evidence_integrity():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks/jpeg-nonlinear-development-20261005.json"
    record = json.loads(path.read_bytes())
    assert (
        record["protocol_sha256"]
        == hashlib.sha256((root / "docs/JPEG_NONLINEAR_PROTOCOL.md").read_bytes()).hexdigest()
    )
    reference = root / "benchmarks/jpeg-residual-development-20261005.json"
    assert record["reference_record_sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest()
    assert record["support_status"] == "experimental" and record["qualification"] == "failed"
    assert not any(record[k] for k in ("calibrated", "deployed", "primary_detection_changed"))
    assert record["source_separation"]["unseen_validation_source_groups"] == 0
    cells = record["comparison_cells"]
    assert len(cells) == 14
    assert sum(c["status"] == "unavailable" for c in cells) == 2
    baseline = json.loads(reference.read_bytes())["comparison_cells"]
    for cell, old in zip(cells, baseline, strict=True):
        assert cell["reference"] == old["new"]
        if cell["new"] is None:
            continue
        assert cell["numeric_failures"]
        for side in ("reference", "new"):
            m = cell[side]
            c = m["confusion"]
            assert c["tp"] + c["fn"] == cell["stego"]
            assert c["fp"] + c["tn"] == cell["covers"]
            assert m["recall"] == pytest.approx(c["tp"] / cell["stego"], abs=5.01e-7)
            assert m["false_positive_rate"] == pytest.approx(c["fp"] / cell["covers"], abs=5.01e-7)
        for key, delta in cell["delta_new_minus_reference"].items():
            assert delta == pytest.approx(cell["new"][key] - cell["reference"][key])
            lo, hi = cell["paired_delta_bootstrap_95_percent"][key]
            assert lo <= hi
    audit = record["independent_audit"]
    assert audit["files_rehashed"] == 3750 and audit["metric_cells_independently_checked"] == 12
    assert audit["passed"] and audit["relative_tolerance"] == 0
    assert set(audit["onnx_batches"]) == {"1", "17", "765"}
    assert all(
        b["passed"] and b["threshold_decisions_equal"] for b in audit["onnx_batches"].values()
    )
    serialized = path.read_text()
    assert "/home/" not in serialized and ".benchmark/" not in serialized
