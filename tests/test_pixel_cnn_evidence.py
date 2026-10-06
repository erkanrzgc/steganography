"""Published complete trial preserves failures and numerical provenance."""

import hashlib
import json
from pathlib import Path

import pytest


def test_pixel_cnn_evidence():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks/pixel-cnn-development-20261006.json"
    record = json.loads(path.read_bytes())
    reference_path = root / "benchmarks/jrm-reference-development-20261005.json"
    reference = json.loads(reference_path.read_bytes())
    assert (
        record["reference_record_sha256"] == hashlib.sha256(reference_path.read_bytes()).hexdigest()
    )
    assert (
        record["protocol_sha256"]
        == hashlib.sha256((root / "docs/PIXEL_CNN_TRAINING_PROTOCOL.md").read_bytes()).hexdigest()
    )
    assert record["preregistered_implementation_protocol_commit"].startswith("ac377b4")
    assert record["manifest_sha256"] == reference["manifest_sha256"]
    assert record["qualification"] == "unavailable"
    assert not any(record[k] for k in ("deployed", "calibrated", "primary_detection_changed"))
    cells = record["comparison_cells"]
    assert len(cells) == 12
    assert len({(c["fit"], c["validation_source_id"], c["method_family"]) for c in cells}) == 12
    assert sum(c["training_excluded_target"] for c in cells) == 4
    for cell in cells:
        m = cell["new"]
        assert m["recall"] == m["false_positive_rate"] == 0
        assert m["balanced_accuracy"] == 0.5
        assert m["confusion"] == {"tp": 0, "tn": m["negatives"], "fp": 0, "fn": m["positives"]}
        assert set(cell["numeric_failures"]) == {"roc_auc", "balanced_accuracy", "recall"}
        old = next(
            c["new"]
            for c in reference["comparison_cells"]
            if c["dimension"] == "source"
            and c["context"] == cell["validation_source_id"]
            and c["method_family"] == cell["method_family"]
        )
        assert cell["reference"] == old
        for key, delta in cell["delta_new_minus_reference"].items():
            assert delta == pytest.approx(m[key] - old[key])
            lo, hi = cell["paired_delta_bootstrap_95_percent"][key]
            assert lo <= hi
    for name, model in record["models"].items():
        training = model["training"]
        scope = training["training_scope"]
        assert scope["rows"] in (618, 2367, 2985)
        assert len(training["epoch_losses"]) == training["settings"]["epochs"] == 10
        assert not training["deployed"] and not training["calibrated"]
        if name != "all":
            assert scope["source_ids"] == [name] and len(scope["excluded_source_ids"]) == 1
        numeric = model["numeric_audit"]
        for key in ("native_reload", "numpy_forward"):
            assert numeric[key]["passed"] and numeric[key]["decisions_equal"]
        assert numeric["onnx"]["passed"] and numeric["onnx"]["status"] == "completed"
        assert set(numeric["onnx"]["batches"]) == {"1", "17", "64"}
        assert all(
            b["passed"] and b["decisions_equal"] for b in numeric["onnx"]["batches"].values()
        )
    assert record["independent_audit"]["files_rehashed"] == 3750
    assert record["independent_audit"]["metric_cells_checked"] == 12
    assert "/home/" not in path.read_text() and ".benchmark/" not in path.read_text()
