"""Preserve amplified learning failures separately from real detection metrics."""

import hashlib
import json
from pathlib import Path

import numpy as np

from core import srnet_positive, srnet_sanity, srnet_signal
from core.srnet_sampling import epoch_pairs

ROOT = Path(__file__).resolve().parents[1]


def test_published_signal_control_complete_arm_failures_provenance_and_scalar_metrics():
    raw = (ROOT / "benchmarks/srnet-signal-strength-20261007.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "a7ce1b3f61a22accb45b73ae8d69ebd3e47e3ee53b428a7e49aabc32d67666cf"
    )
    report = json.loads(raw)
    assert (
        hashlib.sha256((ROOT / "docs/SRNET_SIGNAL_STRENGTH_PROTOCOL.md").read_bytes()).hexdigest()
        == report["protocol_sha256"]
    )
    assert report["protocol_commit"] == "09e28da" and report["status"] == "completed"
    assert [a["factor"] for a in report["arms"]] == [1, 32]
    rows = [{**s, "split": "train", "format": "JPEG"} for s in report["selected_rows"]]
    assert len(rows) == 24
    expected_pairs = [epoch_pairs(rows, seed=20261012, epoch=e)[1] for e in range(20)]
    previous = json.loads((ROOT / "benchmarks/srnet-tiny-sanity-20261007.json").read_bytes())
    assert report["arms"][0]["epoch_training"] == previous["epoch_training"][:20]
    first, second = report["arms"]
    assert first["epoch_batch_schedule"] == second["epoch_batch_schedule"]
    assert first["epoch_pair_schedule"] == second["epoch_pair_schedule"] == expected_pairs
    for arm in report["arms"]:
        assert arm["optimizer_updates"] == sum(r["updates"] for r in arm["epoch_training"]) == 160
        assert arm["epoch_batch_schedule"] == [
            srnet_signal.srnet_multibatch.epoch_batches(rows, seed=20261012, epoch=e)[1]
            for e in range(20)
        ]
        assert len(arm["derived_tensor_sha256"]) == 24
        for prefix in ("own_input", "original_input"):
            metrics = srnet_sanity.metrics(np.array(arm[prefix + "_singleton_logits"]), rows)
            assert metrics == arm[prefix + "_train_metrics"]
        assert arm["learning_objectives"] == srnet_positive.objectives(
            arm["epoch_training"], arm["own_input_train_metrics"]
        )
        assert not arm["learning_objectives_passed"]
        assert not arm["learning_objectives"]["stored_bn_train_balanced_accuracy_at_least_0_90"]
        assert arm["original_input_train_metrics"]["balanced_accuracy"] == 0.5
        audit = arm["audit"]
        assert audit["numerical_gates_passed"] and audit["all_own_and_original_singletons_replayed"]
        assert len(audit["independent_oracles"]) == 9
        for oracle in audit["independent_oracles"]:
            i = oracle["row"]
            assert oracle["source_sha256"] == rows[i]["sha256"]
            assert oracle["derived_tensor_sha256"] == arm["derived_tensor_sha256"][i]
            assert oracle["passed"] and oracle["decisions_equal"]
    for i, row in enumerate(rows):
        assert (first["derived_tensor_sha256"][i] == second["derived_tensor_sha256"][i]) == (
            row["label"] == "cover"
        )
    assert first["own_input_train_metrics"] == first["original_input_train_metrics"]
    assert first["own_input_singleton_logits"] == first["original_input_singleton_logits"]
    assert second["own_input_train_metrics"]["balanced_accuracy"] == 0.6875
    assert report["sampler_sha256_is_original_jpeg_identity_not_derived_tensor"]
    assert report["amplified_inputs_not_encoded_jpeg_or_actual_embedding"]
    assert report["accuracy_qualification"] == "unavailable" and report["in_sample_only"]
    assert not report["validation_pixels_loaded"] and not report["deployed"]
    assert not report["primary_detection_changed"]
    assert b"/home/" not in raw and b"/tmp/" not in raw
