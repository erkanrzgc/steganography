"""Published failures and context-only scores cannot become detector accuracy."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from core import srnet_bn_refresh as bn
from core import srnet_multibatch, srnet_sanity

ROOT = Path(__file__).resolve().parents[1]


def test_public_refresh_complete_provenance_scalar_replay_and_preserved_failures():
    raw = (ROOT / "benchmarks/srnet-bn-refresh-20261007.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "db9d0be00d87b9a2d6bd230fa41e1ebfd8eca62aab33afc63e0ab0d329cf4a73"
    )
    report = json.loads(raw)
    baseline_raw = (ROOT / "benchmarks/srnet-signal-strength-20261007.json").read_bytes()
    baseline = json.loads(baseline_raw)
    assert report["baseline_report_sha256"] == hashlib.sha256(baseline_raw).hexdigest()
    assert (
        report["protocol_sha256"]
        == hashlib.sha256((ROOT / "docs/SRNET_BN_REFRESH_PROTOCOL.md").read_bytes()).hexdigest()
    )
    assert report["protocol_commit"] == "71f3569" and report["status"] == "completed"
    assert report["selected_rows"] == baseline["selected_rows"]
    assert [a["factor"] for a in report["arms"]] == [1, 32]
    rows = [{**r, "format": "JPEG", "split": "train"} for r in report["selected_rows"]]
    batches, schedule = srnet_multibatch.epoch_batches(rows, seed=20261012, epoch=19)
    for arm, original in zip(report["arms"], baseline["arms"], strict=True):
        assert arm["source_model_sha256"] == original["model_sha256"]
        assert arm["derived_tensor_sha256"] == original["derived_tensor_sha256"]
        assert arm["baseline_own_metrics"] == original["own_input_train_metrics"]
        assert arm["baseline_original_metrics"] == original["original_input_train_metrics"]
        assert arm["refresh"]["source_state_unchanged"]
        assert arm["refresh"]["learned_parameters_bit_identical"]
        assert arm["refresh"]["bn_refresh_batches"] == 8 and arm["refresh"]["optimizer_steps"] == 0
        assert not arm["refresh"]["exact_population_variance"]
        assert arm["refresh"]["epoch_batch_schedule"] == schedule
        assert len(arm["refresh"]["changed_buffers"]) == 78
        assert all(
            k.endswith(("running_mean", "running_var", "num_batches_tracked"))
            for k in arm["refresh"]["changed_buffers"]
        )
        for prefix in ("own_input", "original_input"):
            metric = srnet_sanity.metrics(np.array(arm[prefix + "_singleton_logits"]), rows)
            assert metric == arm[prefix + "_train_metrics"]
            contrast = arm["contrast"][prefix]
            assert contrast["state_unchanged"] and contrast["epoch_batch_schedule"] == schedule
            for mode in ("stored_bn", "batch_statistics_diagnostic_only"):
                assert contrast[mode]["presented_rows"] == 32
                assert [b["indices"] for b in contrast[mode]["batches"]] == batches.tolist()
                metrics = [
                    srnet_sanity.metrics(np.array(b["logits"]), [rows[i] for i in b["indices"]])
                    for b in contrast[mode]["batches"]
                ]
                for k in ("cross_entropy", "balanced_accuracy", "recall", "false_positive_rate"):
                    assert contrast[mode]["metrics"][k] == pytest.approx(
                        sum(m[k] for m in metrics) / 8
                    )
        assert arm["in_sample_objectives"] == bn.objectives(
            arm["own_input_train_metrics"], arm["baseline_own_metrics"]
        )
        assert not any(arm["in_sample_objectives"].values())
        assert not arm["in_sample_objectives_passed"]
        assert arm["original_input_train_metrics"]["balanced_accuracy"] == 0.5
        audit = arm["audit"]
        assert audit["numerical_gates_passed"] and len(audit["independent_oracles"]) == 9
        assert all(r["passed"] and r["decisions_equal"] for r in audit["independent_oracles"])
    assert (
        report["arms"][1]["contrast"]["own_input"]["batch_statistics_diagnostic_only"]["metrics"][
            "balanced_accuracy"
        ]
        == 0.96875
    )
    assert report["arms"][1]["own_input_train_metrics"]["balanced_accuracy"] == 0.625
    assert report["optimizer_steps"] == 0 and not report["validation_pixels_loaded"]
    assert report["in_sample_only"] and report["accuracy_qualification"] == "unavailable"
    assert not report["deployed"] and not report["primary_detection_changed"]
    assert b"/home/" not in raw and b"/tmp/" not in raw
