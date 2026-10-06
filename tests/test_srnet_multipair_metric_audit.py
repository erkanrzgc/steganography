"""Independent scalar replay checks provenance, rounding and failed metrics."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def audit_script():
    path = Path(__file__).resolve().parents[1] / "scripts/verify-srnet-multipair-results.py"
    spec = importlib.util.spec_from_file_location("multipair_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def artifacts(tmp_path):
    rows, cells = [], []
    for source, q, count in (("A", None, 205), ("B", 75, 25), ("B", 95, 25)):
        for _i in range(count):
            for label, method, score in (
                ("cover", None, 0.1),
                ("stego", "JUNIWARD", 0.9),
                ("stego", "UERD", 0.9),
            ):
                rows.append(
                    {
                        "source_id": source,
                        "quality_factor": q,
                        "label": label,
                        "method": method,
                        "score": score,
                    }
                )
        for method in ("JUNIWARD", "UERD"):
            cells.append(
                {
                    "source_id": source,
                    "quality_factor": q,
                    "method": method,
                    "metrics": {
                        "roc_auc": 1.0,
                        "balanced_accuracy": 1.0,
                        "recall": 1.0,
                        "false_positive_rate": 0.0,
                        "expected_calibration_error": 0.1,
                        "confusion": {"tp": count, "tn": count, "fp": 0, "fn": 0},
                    },
                }
            )
    hashes = {
        "model_sha256": "1" * 64,
        "model_card_sha256": "2" * 64,
        "training_plan_sha256": "3" * 64,
    }
    prediction = tmp_path / "predictions.json"
    prediction.write_text(json.dumps({"status": "completed", "predictions": rows, **hashes}))
    record = {
        **hashes,
        "diagnostics": {
            "status": "completed",
            "validation_rows": 765,
            "cells": cells,
            "predictions_sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
        },
    }
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(record))
    return evidence, prediction, record


def test_independent_perfect_fixture_is_audit_not_accuracy(audit_script, artifacts):
    evidence, prediction, _ = artifacts
    result = audit_script.verify(evidence, prediction)
    assert result["passed"] and result["cells_audited"] == 6
    assert result["maximum_metric_difference"] < 1e-12
    assert result["accuracy_qualification"] == "unavailable"
    assert not result["model_weights_loaded"] and not result["primary_detection_changed"]


@pytest.mark.parametrize(
    "edit", ["hash", "status", "duplicate", "extra", "auc", "confusion", "nan"]
)
def test_changed_or_incomplete_metric_records_rejected(audit_script, artifacts, edit):
    evidence, prediction, record = artifacts
    diagnostic = record["diagnostics"]
    if edit == "hash":
        diagnostic["predictions_sha256"] = "0" * 64
    elif edit == "status":
        diagnostic["status"] = "failed_numerical_gate"
    elif edit == "duplicate":
        diagnostic["cells"][1] = diagnostic["cells"][0]
    elif edit == "extra":
        diagnostic["cells"].append(diagnostic["cells"][0])
    elif edit == "auc":
        diagnostic["cells"][0]["metrics"]["roc_auc"] = 0.5
    elif edit == "confusion":
        diagnostic["cells"][0]["metrics"]["confusion"]["tp"] = 0
    else:
        diagnostic["cells"][0]["metrics"]["roc_auc"] = float("nan")
    evidence.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        audit_script.verify(evidence, prediction)


def test_cli_output_exclusive(audit_script, artifacts, tmp_path, monkeypatch):
    evidence, prediction, _ = artifacts
    out = tmp_path / "audit.json"
    monkeypatch.setattr(
        audit_script.sys,
        "argv",
        ["audit", "--evidence", str(evidence), "--predictions", str(prediction), "--out", str(out)],
    )
    assert audit_script.main() == 0
    with pytest.raises(FileExistsError):
        audit_script.main()
