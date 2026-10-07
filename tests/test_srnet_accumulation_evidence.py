"""Published control matches both exposures and steps without claiming accuracy."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from core import srnet_widebatch

ROOT = Path(__file__).resolve().parents[1]


def test_portable_accumulation_exact_baseline_schedule_and_independent_scalar_metrics():
    raw = (ROOT / "benchmarks/srnet-accumulation-20261007.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "91cd7442d3f235fd962c7fbece67c660ba17804f32262cceb832346f7b9b2588"
    )
    report = json.loads(raw)
    baseline_raw = (ROOT / "benchmarks/srnet-wide-batch-20261007.json").read_bytes()
    baseline = json.loads(baseline_raw)
    assert report["baseline_report_sha256"] == hashlib.sha256(baseline_raw).hexdigest()
    assert (
        report["protocol_sha256"]
        == hashlib.sha256((ROOT / "docs/SRNET_ACCUMULATION_PROTOCOL.md").read_bytes()).hexdigest()
    )
    assert report["protocol_commit"] == "47217aa" and report["status"] == "completed"
    assert report["selected_rows"] == baseline["selected_rows"]
    assert report["train_data_sha256"] == baseline["train_data_sha256"]
    assert [a["factor"] for a in report["arms"]] == [1, 32]
    rows = [{**r, "split": "train", "format": "JPEG"} for r in report["selected_rows"]]
    labels = np.array([r["label"] == "stego" for r in rows])
    for name, digest in report["execution_source_sha256"].items():
        assert name.startswith(("core/", "steganography/")) and name.endswith(".py")
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
    for arm, old in zip(report["arms"], baseline["arms"], strict=True):
        assert arm["derived_tensor_sha256"] == old["derived_tensor_sha256"]
        assert arm["epoch_batch_schedule"] == old["epoch_batch_schedule"]
        assert arm["epoch_pair_schedule"] == old["epoch_pair_schedule"]
        assert arm["baseline_own_metrics"] == old["own_input_train_metrics"]
        assert arm["baseline_original_metrics"] == old["original_input_train_metrics"]
        assert (
            arm["optimizer_updates"]
            == arm["baseline_optimizer_updates"]
            == old["optimizer_updates"]
            == 80
        )
        assert (
            arm["presented_rows"] == arm["baseline_presented_rows"] == old["presented_rows"] == 640
        )
        assert arm["optimizer_steps_matched"] and arm["microbatch_size"] == 4
        assert arm["gradient_accumulation_steps"] == 2 and arm["bn_forward_batches"] == 160
        for e in range(20):
            batches, record = srnet_widebatch.epoch_batches(rows, seed=20261012, epoch=e)
            assert record == arm["epoch_batch_schedule"][e]
            assert (
                arm["epoch_training"][e]["epoch"] == e and arm["epoch_training"][e]["updates"] == 4
            )
            micros = batches.reshape(8, 4)
            assert micros.size == 32 and all(
                [rows[i]["label"] for i in m] == ["cover", "stego"] * 2 for m in micros
            )
        for prefix in ("own_input", "original_input"):
            logits = np.array(arm[prefix + "_singleton_logits"], dtype=np.float64)
            assert logits.shape == (24, 2) and np.isfinite(logits).all()
            predictions = logits[:, 1] >= logits[:, 0]
            recall = float(predictions[labels].mean())
            fpr = float(predictions[~labels].mean())
            shifted = logits - logits.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
            loss = float(
                np.mean(
                    np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(24), labels.astype(int)]
                )
            )
            metrics = arm[prefix + "_train_metrics"]
            assert metrics["cross_entropy"] == pytest.approx(loss)
            assert metrics["balanced_accuracy"] == pytest.approx((recall + 1 - fpr) / 2)
            assert metrics["recall"] == pytest.approx(recall)
            assert metrics["false_positive_rate"] == pytest.approx(fpr)
            np.testing.assert_allclose(metrics["scores"], probabilities[:, 1], atol=1e-12, rtol=0)
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
        assert (
            len(arm["audit"]["independent_oracles"]) == 9 and arm["audit"]["numerical_gates_passed"]
        )
        assert all(
            r["passed"] and r["decisions_equal"] for r in arm["audit"]["independent_oracles"]
        )
    assert report["row_exposure_matched"] and report["optimizer_steps_matched"]
    assert report["in_sample_only"] and report["accuracy_qualification"] == "unavailable"
    assert not report["validation_pixels_loaded"] and not report["deployed"]
    assert not report["primary_detection_changed"]
    assert b"/home/" not in raw and b"/tmp/" not in raw
