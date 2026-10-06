"""Residual-domain trial retains complete paired controls and frozen provenance."""

import hashlib
import json
from pathlib import Path

import pytest


def test_pixel_scale_evidence():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks/pixel-cnn-scale-development-20261006.json"
    record = json.loads(path.read_bytes())
    previous_path = root / "benchmarks/pixel-cnn-development-20261006.json"
    previous = json.loads(previous_path.read_bytes())
    assert (
        record["previous_pixel_reference_sha256"]
        == hashlib.sha256(previous_path.read_bytes()).hexdigest()
    )
    assert (
        record["protocol_sha256"]
        == hashlib.sha256((root / "docs/PIXEL_CNN_SCALE_PROTOCOL.md").read_bytes()).hexdigest()
    )
    assert record["preregistered_implementation_protocol_commit"].startswith("e1fea41")
    assert record["manifest_sha256"] == previous["manifest_sha256"]
    assert record["qualification"] == "unavailable"
    assert not any(record[k] for k in ("deployed", "calibrated", "primary_detection_changed"))
    cells = record["comparison_cells"]
    assert len(cells) == 12
    assert len({(c["fit"], c["validation_source_id"], c["method_family"]) for c in cells}) == 12
    assert sum(c["training_excluded_target"] for c in cells) == 4
    for cell in cells:
        old = next(
            c
            for c in previous["comparison_cells"]
            if all(c[k] == cell[k] for k in ("fit", "validation_source_id", "method_family"))
        )
        assert cell["previous_pixel_reference"] == old["new"]
        m = cell["new"]
        c = m["confusion"]
        assert c["tp"] + c["fn"] == m["positives"]
        assert c["fp"] + c["tn"] == m["negatives"]
        assert m["recall"] == pytest.approx(c["tp"] / m["positives"], abs=5.01e-7)
        assert m["false_positive_rate"] == pytest.approx(c["fp"] / m["negatives"], abs=5.01e-7)
        for lo, hi in cell["paired_delta_vs_previous_pixel_95_percent"].values():
            assert lo <= hi
        limits = {
            "roc_auc": 0.90,
            "balanced_accuracy": 0.85,
            "recall": 0.80,
            "false_positive_rate": 0.03,
            "expected_calibration_error": 0.05,
        }
        failures = [
            k
            for k, v in limits.items()
            if (
                m[k] > v if k in ("false_positive_rate", "expected_calibration_error") else m[k] < v
            )
        ]
        assert cell["numeric_failures"] == failures
    for name, model in record["models"].items():
        card = model["training"]
        assert (
            card["architecture"]
            == record["architecture"]
            == "jpeg-center128-pixel-residual-cnn8-v1"
        )
        assert card["settings"] == previous["models"][name]["training"]["settings"]
        assert card["training_scope"] == previous["models"][name]["training"]["training_scope"]
        assert len(card["epoch_losses"]) == 10
        numeric = model["numeric_audit"]
        assert all(numeric[k]["passed"] for k in ("native_reload", "numpy_forward", "onnx"))
        assert set(numeric["onnx"]["batches"]) == {"1", "17", "64"}
    assert record["independent_audit"]["files_rehashed"] == 3750
    assert record["independent_audit"]["metric_cells_checked"] == 12
    assert "/home/" not in path.read_text() and ".benchmark/" not in path.read_text()
