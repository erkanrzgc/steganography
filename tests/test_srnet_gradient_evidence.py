"""Frozen gradient records cannot be relabeled detector accuracy evidence."""

import hashlib
import json
import math
from pathlib import Path

import pytest

from core import srnet_gradients, srnet_multibatch, srnet_positive
from core.srnet_sampling import epoch_pairs

ROOT = Path(__file__).resolve().parents[1]


def test_published_gradient_provenance_coverage_and_scalar_norm_checks():
    raw = (ROOT / "benchmarks/srnet-gradient-diagnostic-20261007.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "1921a57a86c506ff1254842d2efe75f9db1b0649b6f45a736420a66e736f5486"
    )
    report = json.loads(raw)
    assert (
        report["protocol_sha256"]
        == hashlib.sha256((ROOT / "docs/SRNET_GRADIENT_PROTOCOL.md").read_bytes()).hexdigest()
    )
    assert report["status"] == "completed" and report["protocol_commit"] == "7c365fc"
    assert report["trained_model_sha256"] == (
        "78dfff3e6da836fcabd3a6e65dc103b0a7508f4fcd9d30ca15ce961984c4b16e"
    )
    rows = [{**r, "split": "train", "format": "JPEG"} for r in report["selected_rows"]]
    batches, schedule = srnet_gradients.select_batches(rows)
    assert batches == report["selected_batches"] and schedule == report["epoch_batch_schedule"]
    assert len(rows) == 24 and len(batches) == 4 and len(report["probes"]) == 9
    epoch_pairs(rows, seed=20261012, epoch=0)
    _, generated_rows = srnet_positive.generate()
    generated_batch = srnet_multibatch.epoch_batches(generated_rows, seed=20261012, epoch=0)[0][0]
    assert report["probes"][-1]["rows"] == [generated_rows[i] for i in generated_batch]
    for j, batch in enumerate(batches):
        assert report["probes"][2 * j]["rows"] == [report["selected_rows"][i] for i in batch]
        assert report["probes"][2 * j + 1]["rows"] == report["probes"][2 * j]["rows"]
        assert report["probes"][2 * j]["kind"] == "real_fresh"
        assert report["probes"][2 * j + 1]["kind"] == "real_failed_tiny_model"
    for probe in report["probes"]:
        params = probe["parameters"]
        assert len({r["parameter"] for r in params}) == len(params)
        assert probe["true_gradient_l2"] == pytest.approx(
            math.sqrt(sum(r["true_l2"] ** 2 for r in params))
        )
        assert probe["null_gradient_l2"] == pytest.approx(
            math.sqrt(sum(r["null_l2"] ** 2 for r in params))
        )
        assert probe["relative_gradient_difference"] == pytest.approx(
            probe["gradient_difference_l2"] / probe["true_gradient_l2"]
        )
        check = probe["classifier_directional_check"]
        assert check["absolute_error"] == abs(check["analytic"] - check["numeric"])
        assert check["passed"] == (check["absolute_error"] <= 0.002 + 0.02 * abs(check["analytic"]))
        classifier = next(r["true_l2"] for r in params if r["parameter"] == "classifier.weight")
        assert check["analytic"] == pytest.approx(classifier, rel=1e-6)
        assert probe["state_unchanged"] and probe["batch_statistics_training_diagnostic_only"]
        assert probe["input_difference_rms"] > 0 and probe["input_gradient_l2"] > 0
    assert all(p["classifier_directional_check"]["passed"] for p in report["probes"])
    assert report["classifier_directional_checks_passed"]
    assert report["optimizer_steps"] == 0 and not report["model_saved"]
    assert not report["validation_pixels_loaded"] and not report["primary_detection_changed"]
    assert report["accuracy_qualification"] == "unavailable"
    assert b"/home/" not in raw and b"/tmp/" not in raw
