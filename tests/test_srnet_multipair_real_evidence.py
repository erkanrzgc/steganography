"""Completed real fitting and numerical parity are not detector qualification."""

import hashlib
import json
from pathlib import Path

import pytest


def test_complete_real_trial_keeps_every_failed_cell_and_original_comparison():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks/srnet-multipair-real-20261007.json"
    record = json.loads(path.read_bytes())
    protocol = root / "docs/SRNET_MULTIPAIR_PROTOCOL.md"
    assert record["protocol_sha256"] == hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert record["protocol_commit"] == "837cb79"
    assert record["fit_evaluation_base_commit"] == "1b0c80c"
    assert record["execution_source_sha256"]["scripts/audit-srnet-multipair-real.py"] == (
        "7c71d711317d403182c82915cd1d01ef16e8c9a13312ed3438ce4ef19c7256ed"
    )
    card = record["fit_card"]
    assert (
        record["model_card_sha256"]
        == hashlib.sha256((json.dumps(card, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
    )
    assert card["training"] == "completed" and card["schema_version"] == "srnet-fit-v2"
    assert card["batch_size"] == 4 and card["training_scope"]["rows"] == 2985
    assert card["epoch_training"][0]["updates"] == 1580
    assert card["epoch_schedule"][0]["pairs"] == 3160
    assert card["epoch_batch_schedule"][0]["optimizer_updates"] == 1580
    assert not card["validation_used"] and not card["deployed"]
    assert 0 < card["seconds"] < 1920 and record["evaluation_seconds"] > 0
    diagnostic = record["diagnostics"]
    assert diagnostic["status"] == "completed" and diagnostic["validation_rows"] == 765
    assert len(diagnostic["independent_forward_audit"]) == 9
    assert all(
        a["passed"] and a["decisions_equal"] for a in diagnostic["independent_forward_audit"]
    )
    assert len(diagnostic["cells"]) == len(record["comparison"]) == 6
    prior_path = root / "benchmarks/srnet-real-pilot-20261006.json"
    assert hashlib.sha256(prior_path.read_bytes()).hexdigest() == record["prior_evidence_sha256"]
    prior = json.loads(prior_path.read_bytes())

    def key(cell):
        return cell["source_id"], cell["quality_factor"], cell["method"]

    old = {key(c): c for c in prior["diagnostics"]["cells"]}
    changes = {key(c): c["point_change"] for c in record["comparison"]}
    for cell in diagnostic["cells"]:
        metrics = cell["metrics"]
        assert 0.5 <= metrics["balanced_accuracy"] <= 0.52
        assert 0.5008 <= metrics["roc_auc"] <= 0.5072
        assert metrics["false_positive_rate"] >= 0.404878
        assert not cell["metric_gates_passed"] and not cell["sample_size_gate"]
        assert not cell["qualified"] and cell["support_status"] == "experimental"
        for name, difference in changes[key(cell)].items():
            assert difference == pytest.approx(metrics[name] - old[key(cell)]["metrics"][name])
        for interval in cell["bootstrap_95_percent"].values():
            assert 0 <= interval[0] <= interval[1] <= 1
    assert not diagnostic["qualified"] and not diagnostic["deployed"]
    assert record["qualification"] == "unavailable" and not record["deployed"]
    assert not record["raw_data_and_model_published"]
    observation = record["runtime_observation"]
    assert observation["address_bytes_hard"] == 8 * 1024**3
    assert observation["cpu_seconds_soft"] == 3660 and observation["core_bytes_hard"] == 0
    assert not observation["filesystem_or_network_sandbox"]
    assert "/home/" not in path.read_text() and ".benchmark/" not in path.read_text()
    metric_audit = json.loads(
        (root / "benchmarks/srnet-multipair-metric-audit-20261007.json").read_bytes()
    )
    assert metric_audit["evidence_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metric_audit["predictions_sha256"] == diagnostic["predictions_sha256"]
    assert metric_audit["passed"] and metric_audit["cells_audited"] == 6
    assert metric_audit["maximum_metric_difference"] <= metric_audit["absolute_tolerance"] == 1e-6
    assert metric_audit["confusion_counts_exact"] and not metric_audit["model_weights_loaded"]
    assert metric_audit["accuracy_qualification"] == "unavailable"
