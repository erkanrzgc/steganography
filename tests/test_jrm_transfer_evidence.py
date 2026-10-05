"""Frozen source-transfer evidence keeps all failed controls and cross-origin cells."""

import hashlib
import json
from pathlib import Path

import pytest


def test_transfer_evidence_integrity():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks/jrm-transfer-development-20261005.json"
    record = json.loads(path.read_bytes())
    reference_path = root / "benchmarks/jrm-reference-development-20261005.json"
    reference = json.loads(reference_path.read_bytes())
    assert (
        record["reference_record_sha256"] == hashlib.sha256(reference_path.read_bytes()).hexdigest()
    )
    assert (
        record["protocol_sha256"]
        == hashlib.sha256((root / "docs/JRM_TRANSFER_PROTOCOL.md").read_bytes()).hexdigest()
    )
    assert record["preregistered_implementation_protocol_commit"] == "f8a4660"
    assert record["manifest_sha256"] == reference["manifest_sha256"]
    assert record["qualification"] == "unavailable"
    assert record["numeric_gate_status"] == "failed in all 8 cells"
    assert not any(record[k] for k in ("deployed", "calibrated", "primary_detection_changed"))
    cells = record["comparison_cells"]
    assert len(cells) == 8 and sum(c["training_excluded_target"] for c in cells) == 4
    assert (
        len(
            {
                (c["training_source_id"], c["validation_source_id"], c["method_family"])
                for c in cells
            }
        )
        == 8
    )
    for cell in cells:
        assert cell["numeric_failures"]
        old = next(
            c
            for c in reference["comparison_cells"]
            if (
                c["dimension"] == "source"
                and c["context"] == cell["validation_source_id"]
                and c["method_family"] == cell["method_family"]
            )
        )
        assert cell["reference"] == old["new"]
        m = cell["new"]
        c = m["confusion"]
        assert c["tp"] + c["fn"] == m["positives"]
        assert c["fp"] + c["tn"] == m["negatives"]
        assert m["recall"] == pytest.approx(c["tp"] / m["positives"], abs=5.01e-7)
        assert m["false_positive_rate"] == pytest.approx(c["fp"] / m["negatives"], abs=5.01e-7)
        for key, delta in cell["delta_new_minus_reference"].items():
            assert delta == pytest.approx(m[key] - old["new"][key])
            lo, hi = cell["paired_delta_bootstrap_95_percent"][key]
            assert lo <= hi
    for source, model in record["models"].items():
        scope = model["training"]["training_scope"]
        assert scope["source_ids"] == [source] and len(scope["excluded_source_ids"]) == 1
        assert scope["rows"] in (618, 2367)
        assert model["training"]["paired_training_rows"] in (412, 1578)
    audit = record["independent_audit"]
    assert audit["passed"] and audit["models_checked"] == 2
    assert audit["files_rehashed"] == 3750 and audit["metric_cells_checked"] == 8
    assert audit["scalar_vote_rows_per_model"] == 765
    for batches in audit["numeric_reload_batches"].values():
        assert set(batches) == {"1", "17", "765"}
        assert all(
            b["passed"] and b["decisions_equal"] and b["max_score_difference"] == 0
            for b in batches.values()
        )
    assert "/home/" not in path.read_text() and ".benchmark/" not in path.read_text()
