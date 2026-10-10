"""Portable physical evidence integrity, not rerun training or accuracy gates."""

import hashlib
import json
from pathlib import Path

import pytest

from steganography.research_jpeg_norm_learning import sources

ROOT = Path(__file__).resolve().parents[1]


def evidence(name):
    return ROOT / "benchmarks" / f"jpeg-norm-learning-{name}-20261010.json"


@pytest.mark.parametrize("arm", ["bn", "gn"])
def test_physical_arm_is_complete_source_bound_and_not_qualified(arm):
    path = evidence(arm)
    raw = path.read_bytes()
    report = json.loads(raw)
    audit = json.loads(evidence("independent-metrics").read_bytes())
    assert hashlib.sha256(raw).hexdigest() == audit[f"{arm}_report_sha256"]
    assert report["source_sha256"] == sources()
    assert report["status"] == "completed" and report["arm"] == arm
    assert report["epochs"] == 8 and report["optimizer_updates"] == 1152
    assert len(report["epoch_losses"]) == 8
    assert all(r["updates"] == 144 for r in report["epoch_losses"])
    assert report["fit_rows"] == 360 and report["probe_rows"] == 120
    assert report["fit_originals"] == 72 and report["probe_originals"] == 24
    assert report["original_groups_disjoint"] is True
    assert report["probe_previously_inspected"] is True
    assert report["validation_test_used"] is False
    assert report["real_model_trained"] is True
    assert report["accuracy_qualification"] == "unavailable"
    assert report["deployed"] is False
    assert report["initial_parameter_sha256"] != report["learned_parameter_sha256"]
    assert report["execution"]["device"] == "cuda:0"
    assert report["execution"]["cpu_fallback"] is False
    assert report["singleton_batch_parity"]["passed"] is True
    assert len(report["probe_logits"]) == 120
    assert len(report["probe_cells"]) == 10
    assert all(c["balanced_accuracy"] == 0.5 for c in report["probe_cells"])
    assert b"/home/" not in raw and b"/tmp/" not in raw


def test_both_arms_and_independent_scalar_evidence_match():
    bn, gn = (json.loads(evidence(a).read_bytes()) for a in ("bn", "gn"))
    for key in (
        "initial_parameter_sha256",
        "seed",
        "optimizer",
        "schedules",
        "schedule_sha256",
        "manifest_sha256",
        "audit_sha256",
        "source_sha256",
    ):
        assert bn[key] == gn[key]
    assert bn["architecture"] != gn["architecture"]
    audit = json.loads(evidence("independent-metrics").read_bytes())
    assert audit["status"] == "completed"
    assert audit["matched_initialization_schedule_optimizer"] is True
    assert audit["independent_whole_model_oracle"] is False
    assert audit["accuracy_qualification"] == "unavailable" and audit["deployed"] is False
    for arm in ("bn", "gn"):
        assert len(audit[f"{arm}_cells"]) == 10
        assert all(c["counts_exact"] and c["scalar_metrics_passed"] for c in audit[f"{arm}_cells"])
    assert (
        audit["auditor_source_sha256"]["core/jpeg_norm_audit.py"]
        == hashlib.sha256((ROOT / "core/jpeg_norm_audit.py").read_bytes()).hexdigest()
    )
