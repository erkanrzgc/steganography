"""Portable, immutable evidence checks without corpora, Torch or CUDA."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "jpeg-"
FILES = {
    "controls": (
        "owned-controls-results",
        "2eaf02f2f78ab1040886d06f1b5dfb041ff8078abf2a1c6390618cb17b1b0304",
    ),
    "fit": (
        "fit-development",
        "e5b7f776b929603ec2d6b44313a0ed6c5c0d2429df17b4b73d5a6e3d8d099f82",
    ),
    "paired": (
        "all-paired-diagnosis",
        "b3567c91c30485dba72394badaeadd906da3939dd2e19f28f052d764b9bdbdb3",
    ),
    "normalization": (
        "normalization-contrast",
        "24bc12beda15492a3122571ef2f19b3ec92650e506f83c3770a4e9d9294c4f5f",
    ),
}
ARMS = ("legacy", "reference-adamax", "convolution-l2", "genuine-batch")
SEEDS = (20261012, 20261013)
ORIGINS = ("BOSSbase-1.01", "BOWS2")


def receipt(name):
    stem, digest = FILES[name]
    raw = (ROOT / "benchmarks" / f"{PREFIX}{stem}-20261014.json").read_bytes()
    assert len(raw) < 512 * 1024
    assert hashlib.sha256(raw).hexdigest() == digest
    return json.loads(raw)


def key(trial):
    return trial["arm"], trial["seed"]


@pytest.mark.parametrize("name", FILES)
def test_completed_receipts_are_portable_unqualified_and_immutable(name):
    value = receipt(name)
    assert value["status"] == "completed"
    assert value["accuracy_qualification"] == "unavailable"
    assert value["deployed"] is False
    raw = json.dumps(value, allow_nan=False)
    for token in ("/home/", "/mnt/", "192.168.", "authorized_keys", "password", "api_key"):
        assert token not in raw


def test_all_controls_have_full_exposure_and_matched_initialization():
    value = receipt("controls")
    trials = value["trials"]
    assert len(trials) == 8
    assert {key(t) for t in trials} == {(a, s) for a in ARMS for s in SEEDS}
    assert value["inherited_complete_controls"] == 2
    assert value["newly_completed_controls"] == 6
    assert value["verified_numeric_snapshots"] == 16
    assert len(value["source_sha256"]) == 52
    assert len(value["controller_sha256"]) == 3
    assert value["originals"] == {
        "fit": 512, "development": 128, "development_pairs_per_source": 64,
    }
    assert value["capacity"]["32"].startswith("unavailable:")
    assert value["real_original_covers"] is True
    assert value["positive_ground_truth"] == "embedding simulation"
    assert value["simulation_contract"] == "conseal-2025.11-juniward-bounded-entropy-refinement-v1"
    for t in trials:
        baseline = next(b for b in trials if b["arm"] == "legacy" and b["seed"] == t["seed"])
        assert t["status"] == "completed"
        assert t["batch_rows"] == (16 if t["arm"] == "genuine-batch" else 4)
        assert t["presented_rows"] == 32768
        assert t["optimizer_updates"] == t["total_optimizer_updates"] == 32768 // t["batch_rows"]
        assert [c["presented_rows"] for c in t["checkpoints"]] == [16384, 32768]
        assert t["initial_learned_sha256"] == baseline["initial_learned_sha256"]
        assert t["schedule_sha256"] == baseline["schedule_sha256"]
        assert t["execution"]["device"] == "cuda:0"
        assert t["execution"]["cpu_fallback"] is False
        assert t["execution"]["mixed_precision"] is False
        assert t["inherited_complete_control"] == (
            t["seed"] == 20261012 and t["arm"] in ARMS[:2])


def test_all_source_cells_retain_failed_fixed_threshold_learning_gates():
    value = receipt("controls")
    assert value["learning_gate"] is False and value["passing_arms"] == []
    passed = []
    for t in value["trials"]:
        assert tuple(c["source_group"] for c in t["cells"]) == ORIGINS
        for c in t["cells"]:
            m = c["metrics"]
            assert c["original_pairs"] == m["positives"] == m["negatives"] == 64
            assert m["samples"] == 128 and m["threshold"] == 50
            assert m["recommended_threshold"] is None
            assert c["support_status"] == "experimental"
            assert c["method"] == "JUNIWARD" and c["quality_factor"] == 75
            assert c["payload_rate"] == 0.4
            counts = m["confusion"]
            assert sum(counts.values()) == 128
            assert m["accuracy"] == round((counts["tp"] + counts["tn"]) / 128, 6)
            assert m["false_positive_rate"] == round(counts["fp"] / 64, 6)
            assert 0 <= m["roc_auc"] <= 1 and 0 <= c["ece"] <= 1
            delta = c["paired_auc_delta_ci95"]
            expected = bool(m["roc_auc"] >= .7 and c["auc_ci95"][0] > .5
                            and delta and delta[0] > 0)
            assert c["learning_gate"] == expected
            if expected:
                passed.append((key(t), c["source_group"]))
    assert passed == [(("genuine-batch", 20261012), "BOSSbase-1.01")]


def test_readonly_fit_metrics_match_development_and_snapshot_ancestry():
    controls, fit = receipt("controls"), receipt("fit")
    assert fit["study_report_sha256"] == controls["report_sha256"]
    assert fit["source_sha256"] == controls["source_sha256"]
    assert fit["controller_sha256"] == controls["controller_sha256"]
    assert fit["optimizer_updates_added"] == 0 and fit["models_modified"] is False
    assert fit["threshold_selected"] is False and fit["reserved_source_used"] is False
    assert len(fit["results"]) == 8
    for r in fit["results"]:
        t = next(t for t in controls["trials"] if key(t) == key(r))
        assert r["status"] == "completed" and r["optimizer_updates_added"] == 0
        assert r["model_digest_before"] == r["model_digest_after"]
        assert r["snapshot_binding"] == t["binding"]
        assert r["snapshot_sha256"] == t["checkpoints"][1]["sha256"]
        assert len(r["cells"]) == 2
        for c, development in zip(r["cells"], t["cells"], strict=True):
            assert c["source_group"] == development["source_group"]
            assert c["development_metrics"] == development["metrics"]
            assert c["fit_metrics"]["samples"] == 512
            assert c["fit_metrics"]["threshold"] == 50
            assert c["fit_metrics"]["recommended_threshold"] is None
            assert math.isclose(c["fit_minus_development_auc"],
                                c["fit_metrics"]["roc_auc"] - development["metrics"]["roc_auc"])


def test_normalization_uses_same_fit_selection_and_unmodified_learned_states():
    fit, value = receipt("fit"), receipt("normalization")
    assert value["fit_evaluation_sha256"] == fit["native_receipt_sha256"]
    assert value["diagnostic_sha256"] == value["protocol_boundary"]["diagnostic_sha256"]
    protocol = ROOT / "docs" / "JPEG_NORMALIZATION_DIAGNOSTIC.md"
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == (
        value["protocol_boundary"]["protocol_sha256"])
    assert value["development_source_used"] is False and value["reserved_source_used"] is False
    assert value["learned_state_modified"] is False and value["optimizer_updates_added"] == 0
    assert value["requires_known_cover"] is True
    assert value["standalone_accuracy_computed"] is False and value["threshold_selected"] is False
    assert len(value["results"]) == 8
    assert len({r["selection_sha256"] for r in value["results"]}) == 1
    for r in value["results"]:
        prior = next(p for p in fit["results"] if key(p) == key(r))
        assert r["original_fit_pairs"] == 16 and r["original_fit_pairs_per_source"] == 8
        assert r["model_digest_before"] == r["model_digest_after"] == prior["model_digest_before"]
        assert r["snapshot_sha256"] == prior["snapshot_sha256"]
        assert r["snapshot_binding"] == prior["snapshot_binding"]
        assert set(r["modes"]) == {"stored_statistics", "paired_batch_statistics"}
        for mode in r["modes"].values():
            assert mode["mean_cross_entropy"] >= 0 and mode["mean_absolute_logit_margin"] >= 0
            assert math.isclose(mode["mean_within_pair_probability_difference"],
                                mode["mean_stego_probability"] - mode["mean_cover_probability"],
                                abs_tol=1e-12)
            assert "accuracy" not in mode and "roc_auc" not in mode


def test_pair_direction_does_not_become_standalone_accuracy():
    controls, paired = receipt("controls"), receipt("paired")
    assert paired["study_report_sha256"] == controls["report_sha256"]
    assert paired["study_source_sha256"] == controls["source_sha256"]
    assert len(paired["completed_controls"]) == 8
    for trial in paired["completed_controls"]:
        t = next(t for t in controls["trials"] if key(t) == key(trial))
        assert trial["snapshot_binding"] == t["binding"]
        diagnostic = trial["diagnostic"]
        assert diagnostic["requires_known_cover"] is True
        assert diagnostic["standalone_accuracy_computed"] is False
        assert diagnostic["scores_adjusted"] is False and diagnostic["threshold_selected"] is False
        for c in diagnostic["cells"]:
            assert c["original_pairs"] == 64
            assert c["higher_stego_pairs"] + c["tied_pairs"] + c["lower_stego_pairs"] == 64
            assert c["positive_pair_fraction"] == c["higher_stego_pairs"] / 64


def test_both_diagnoses_consume_original_cumulative_envelope():
    controls, fit, bn = receipt("controls"), receipt("fit"), receipt("normalization")
    budget = controls["budget"]
    assert budget["controls_budget_seconds"] == 21600
    assert budget["prior_controls_seconds"] == 5403.521319699999
    assert math.isclose(budget["seconds"],
                        budget["prior_controls_seconds"] + budget["new_controls_seconds"],
                        abs_tol=.01)
    assert math.isclose(fit["cumulative_controls_and_diagnostic_seconds"],
                        budget["seconds"] + fit["seconds"])
    assert math.isclose(bn["cumulative_controls_and_diagnostics_seconds"],
                        fit["cumulative_controls_and_diagnostic_seconds"] + bn["seconds"])
    assert bn["cumulative_controls_and_diagnostics_seconds"] < 21600
