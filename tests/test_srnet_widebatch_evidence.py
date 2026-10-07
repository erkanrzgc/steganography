"""Portable evidence keeps exposure confounds and learning failures visible."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from core import srnet_multibatch, srnet_widebatch

ROOT = Path(__file__).resolve().parents[1]


def test_published_wide_control_complete_provenance_and_independent_scalar_replay():
    raw = (ROOT / "benchmarks/srnet-wide-batch-20261007.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "ae828ea18337fcff82f40506717c7fe5f33d664e380537632a4ac1ca4dcf9427"
    )
    report = json.loads(raw)
    baseline_raw = (ROOT / "benchmarks/srnet-signal-strength-20261007.json").read_bytes()
    baseline = json.loads(baseline_raw)
    assert report["baseline_report_sha256"] == hashlib.sha256(baseline_raw).hexdigest()
    assert (
        report["protocol_sha256"]
        == hashlib.sha256((ROOT / "docs/SRNET_WIDE_BATCH_PROTOCOL.md").read_bytes()).hexdigest()
    )
    assert report["protocol_commit"] == "c78a9eb" and report["status"] == "completed"
    assert report["selected_rows"] == baseline["selected_rows"]
    assert report["train_data_sha256"] == baseline["train_data_sha256"]
    assert [a["factor"] for a in report["arms"]] == [1, 32]
    rows = [{**r, "format": "JPEG", "split": "train"} for r in report["selected_rows"]]
    for name, digest in report["execution_source_sha256"].items():
        assert name.startswith(("core/", "steganography/")) and name.endswith(".py")
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
    for arm, old in zip(report["arms"], baseline["arms"], strict=True):
        assert arm["derived_tensor_sha256"] == old["derived_tensor_sha256"]
        assert arm["baseline_own_metrics"] == old["own_input_train_metrics"]
        assert arm["baseline_original_metrics"] == old["original_input_train_metrics"]
        assert arm["optimizer_updates"] == 80 and arm["presented_rows"] == 640
        assert arm["baseline_optimizer_updates"] == 160 and arm["baseline_presented_rows"] == 640
        assert not arm["optimizer_steps_matched"]
        for e in range(20):
            batches, schedule = srnet_widebatch.epoch_batches(rows, seed=20261012, epoch=e)
            base, _ = srnet_multibatch.epoch_batches(rows, seed=20261012, epoch=e)
            assert sorted(batches.ravel().tolist()) == sorted(base.ravel().tolist())
            assert schedule == arm["epoch_batch_schedule"][e]
            assert arm["epoch_training"][e]["updates"] == 4
        expected_labels = np.array([int(r["label"] == "stego") for r in rows])
        for prefix in ("own_input", "original_input"):
            logits = np.array(arm[prefix + "_singleton_logits"], dtype=np.float64)
            assert logits.shape == (24, 2) and np.isfinite(logits).all()
            predicted = (logits[:, 1] >= logits[:, 0]).astype(int)
            recall = np.mean(predicted[expected_labels == 1])
            fpr = np.mean(predicted[expected_labels == 0])
            shifted = logits - logits.max(axis=1, keepdims=True)
            loss = np.mean(
                np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(24), expected_labels]
            )
            metrics = arm[prefix + "_train_metrics"]
            assert metrics["recall"] == pytest.approx(recall)
            assert metrics["false_positive_rate"] == pytest.approx(fpr)
            assert metrics["balanced_accuracy"] == pytest.approx((recall + 1 - fpr) / 2)
            assert metrics["cross_entropy"] == pytest.approx(loss)
        goals = arm["learning_objectives"]
        first, last = (
            arm["epoch_training"][0]["mean_pair_loss"],
            arm["epoch_training"][-1]["mean_pair_loss"],
        )
        reduction = (first - last) / first if first else 0
        assert goals["final_batch_loss_at_most_0_35"] is (last <= 0.35)
        assert goals["relative_loss_reduction_at_least_0_25"] is (reduction >= 0.25)
        assert goals["stored_bn_train_balanced_accuracy_at_least_0_90"] is (
            arm["own_input_train_metrics"]["balanced_accuracy"] >= 0.90
        )
        assert goals["relative_loss_reduction"] == pytest.approx(reduction)
        assert arm["learning_objectives_passed"] is all(
            v for k, v in goals.items() if k != "relative_loss_reduction"
        )
        assert not arm["learning_objectives_passed"]
        assert arm["original_input_train_metrics"]["balanced_accuracy"] == 0.5
        assert arm["audit"]["all_own_and_original_singletons_replayed"]
        assert arm["audit"]["numerical_gates_passed"]
        assert len(arm["audit"]["independent_oracles"]) == 9
        assert all(
            r["passed"] and r["decisions_equal"] for r in arm["audit"]["independent_oracles"]
        )
    assert report["row_exposure_matched"] and not report["optimizer_steps_matched"]
    assert report["in_sample_only"] and report["accuracy_qualification"] == "unavailable"
    assert not report["validation_pixels_loaded"] and not report["deployed"]
    assert not report["primary_detection_changed"]
    assert b"/home/" not in raw and b"/tmp/" not in raw
