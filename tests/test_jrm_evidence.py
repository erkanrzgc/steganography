"""Published JRM development evidence binds the frozen protocol and all failures."""

import hashlib
import json
from pathlib import Path

import pytest


def test_jrm_evidence_integrity():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks/jrm-reference-development-20261005.json"
    record = json.loads(path.read_bytes())
    protocol = root / "docs/JRM_REFERENCE_PROTOCOL.md"
    reference = root / "benchmarks/jpeg-nonlinear-development-20261005.json"
    assert record["protocol_sha256"] == hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert record["reference_record_sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest()
    assert record["preregistered_implementation_protocol_commit"] == "2a9d005"
    assert record["support_status"] == "experimental"
    assert record["qualification"].startswith("unavailable;")
    assert record["numeric_gate_status"] == "failed in all 12 measurable cells"
    assert not any(record[k] for k in ("deployed", "calibrated", "primary_detection_changed"))
    assert record["source_separation"]["unseen_validation_source_groups"] == 0
    assert record["training"]["paired_training_rows"] == 1990
    assert record["training"]["upstream_training_vote_parity"]
    assert record["training"]["learners"] == 31 and record["training"]["subspace"] == 256
    cells = record["comparison_cells"]
    assert len(cells) == 14
    assert sum(c["status"] == "unavailable" for c in cells) == 2
    baseline = json.loads(reference.read_bytes())["comparison_cells"]
    for cell, old in zip(cells, baseline, strict=True):
        assert cell["reference"] == old["new"]
        if cell["new"] is None:
            continue
        assert set(cell["numeric_failures"]) == {
            "roc_auc", "balanced_accuracy", "recall", "false_positive_rate",
            "expected_calibration_error",
        }
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
    assert audit["passed"] and audit["files_rehashed"] == 3750
    assert audit["metric_cells_checked"] == 12 and audit["scalar_fld_vote_rows"] == 765
    assert audit["upstream_jrm_parity_examples"] == 18
    assert not audit["jrm_algorithm_independent_oracle"]
    assert set(record["artifact_sha256"]) == {
        "execution-summary.json", "model/model-card.json", "model/model.npz",
        "predictions.json", "train/cache.json", "train/features.f32",
        "validation/cache.json", "validation/features.f32",
    }
    assert set(audit["numeric_reload_batches"]) == {"1", "17", "765"}
    assert all(
        b["passed"] and b["decisions_equal"] and b["max_score_difference"] == 0
        for b in audit["numeric_reload_batches"].values()
    )
    assert "/home/" not in path.read_text() and ".benchmark/" not in path.read_text()
