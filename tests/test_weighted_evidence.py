"""Portable published evidence stays honest, internally consistent and path-free."""

import hashlib
import json
import re
from pathlib import Path

import pytest


def test_weighted_publication_contract_and_confusions():
    root = Path(__file__).resolve().parents[1]
    record = json.loads(
        (root / "benchmarks/spatial-weighted-development-20261005.json").read_bytes()
    )
    assert record["protocol_sha256"] == hashlib.sha256(
        (root / "docs/SPATIAL_WEIGHTING_PROTOCOL.md").read_bytes()
    ).hexdigest()
    assert not record["deployed"] and not record["calibrated"]
    assert record["support_status"] == "experimental"
    assert record["cross_source_gate"] == "unavailable"
    assert record["counts"]["test"] == 0
    assert record["counts"]["validation"] == 184 * 7
    assert record["counts"]["train"] == 816 * 7
    assert record["counts"]["source_groups"] == ["BOSSbase-1.01"]
    recipe = record["training"]["sample_weighting"]
    assert recipe["positive_mass"] == 816 * 12
    assert recipe["negative_mass"] == 816
    assert recipe["positive_class_weight"] == pytest.approx(1 / 12)
    assert set(record["by_method_rate"]) == {
        f"{m}-{r}" for m in ("sequential", "scattered") for r in (5, 20, 40)
    }
    for cell in record["by_method_rate"].values():
        for side in ("reference", "new"):
            metrics = cell[side]
            c = metrics["confusion"]
            assert c["tp"] + c["fn"] == c["tn"] + c["fp"] == 184
            assert metrics["recall"] == pytest.approx(c["tp"] / 184, abs=5e-7)
            assert metrics["false_positive_rate"] == pytest.approx(c["fp"] / 184, abs=5e-7)
            assert metrics["balanced_accuracy"] == pytest.approx(
                (c["tp"] + c["tn"]) / 368, abs=5e-7
            )
        for metric, delta in cell["delta"].items():
            assert delta == pytest.approx(cell["new"][metric] - cell["reference"][metric])
        assert cell["new"]["confusion"]["fp"] == 6
        assert cell["new"]["expected_calibration_error"] > 0.05
    assert record["onnx_parity"]["passed"]
    assert record["onnx_parity"]["absolute_tolerance"] == 1e-6
    assert record["independent_audit"]["all_six_pairwise_auc_and_confusion_cells"]

    def check(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.endswith("sha256"):
                    assert isinstance(child, str) and re.fullmatch(r"[a-f0-9]{64}", child)
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
        elif isinstance(value, str):
            assert not value.startswith("/") and "/home/" not in value

    check(record)
