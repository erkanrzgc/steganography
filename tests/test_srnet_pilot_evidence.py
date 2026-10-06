"""Real failed pilot evidence must not be promoted to qualified accuracy."""

import hashlib
import json
from pathlib import Path


def test_frozen_real_fit_evidence_and_failed_context_gates():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks/srnet-real-pilot-20261006.json"
    record = json.loads(path.read_bytes())
    assert (
        record["protocol_sha256"]
        == hashlib.sha256((root / "docs/SRNET_REAL_PILOT_PROTOCOL.md").read_bytes()).hexdigest()
    )
    assert record["protocol_commit"] == "ff24d10"
    assert record["fit_evaluation_base_commit"] == "8a73ffa"
    # Historical execution versions, not a restriction on future code changes.
    assert record["diagnostic_service_sha256"] == (
        "50a18b313f3be94dc4d415e513860bf7fd79d7fab66cb9255ed95add0d0db8c2"
    )
    assert record["audit_script_sha256"] == (
        "31726433a89c29d69d90ff857402569ffc280cdef0a6a24cb90c6432abb91562"
    )
    assert record["replay"]["runs"] == 2 and record["replay"]["validation_rows_per_run"] == 765
    assert record["replay"]["predictions_identical"]
    assert record["replay"]["metrics_and_intervals_identical"]
    assert record["replay"]["independent_audits_identical"]
    assert record["evaluation_seconds"] > 0
    card = record["fit_card"]
    assert (
        record["model_card_sha256"]
        == hashlib.sha256((json.dumps(card, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
    )
    assert card["training"] == "completed" and card["epoch_training"][0]["updates"] == 412
    assert card["sampling_settings"] == {"epochs": 1, "seed": 20261012}
    assert card["training_scope"]["rows"] == 618
    assert card["epoch_schedule"][0]["ordered_pair_sha256"] == (
        "57affc332b551ff55afe45bfeff890daf53c7f20201d2a0b5fcd29276fbe8122"
    )
    assert card["seconds"] > 0 and not card["validation_used"] and not card["deployed"]
    diagnostic = record["diagnostics"]
    assert diagnostic["status"] == "completed" and diagnostic["validation_rows"] == 765
    assert len(diagnostic["independent_forward_audit"]) == 9
    assert all(
        a["passed"] and a["decisions_equal"] for a in diagnostic["independent_forward_audit"]
    )
    assert len(diagnostic["cells"]) == 6
    assert (
        not diagnostic["qualified"] and not diagnostic["deployed"] and not diagnostic["calibrated"]
    )
    for cell in diagnostic["cells"]:
        metrics = cell["metrics"]
        assert metrics["balanced_accuracy"] == 0.5 and metrics["roc_auc"] < 0.51
        assert metrics["false_positive_rate"] >= 0.24
        assert metrics["expected_calibration_error"] > 0.49
        assert (
            not cell["metric_gates_passed"]
            and not cell["sample_size_gate"]
            and not cell["qualified"]
        )
        assert cell["support_status"] == "experimental"
        assert set(cell["bootstrap_95_percent"]) == {
            "roc_auc",
            "balanced_accuracy",
            "recall",
            "false_positive_rate",
            "expected_calibration_error",
        }
        assert all(
            len(bounds) == 2 and 0 <= bounds[0] <= bounds[1] <= 1
            for bounds in cell["bootstrap_95_percent"].values()
        )
    assert record["qualification"] == "unavailable" and not record["deployed"]
    assert not record["raw_data_and_model_published"]
    assert "/home/" not in path.read_text() and ".benchmark/" not in path.read_text()
