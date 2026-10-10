"""Independently replay portable physical evidence; never qualify accuracy."""

import hashlib
import json
import math
from pathlib import Path

import pytest

from steganography.research_jpeg_learnability import sources

ROOT = Path(__file__).resolve().parents[1]


def evidence(arm):
    path = ROOT / "benchmarks" / f"jpeg-learnability-{arm}-20261010.json"
    raw = path.read_bytes()
    assert b"/home/" not in raw and b"/tmp/" not in raw
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("arm", ["legacy", "reference"])
def test_real_arm_accounting_and_independent_logsumexp(arm):
    report, checksum = evidence(arm)
    audit = json.loads((ROOT / "benchmarks/jpeg-learnability-audit-20261010.json").read_bytes())
    assert checksum == audit[f"{arm}_report_sha256"]
    assert report["source_sha256"] == sources()
    assert len(report["source_sha256"]) == 31
    assert report["schema_version"] == "jpeg-real-pair-learnability-v1"
    assert report["status"] == "completed" and report["arm"] == arm
    assert report["epochs"] == 8 and report["optimizer_updates"] == 1024
    assert len(report["epoch_losses"]) == 8
    assert all(row["updates"] == 128 for row in report["epoch_losses"])
    assert report["fit_originals"] == 2 and len(set(report["fit_rows"])) == 4
    assert report["source_group"] == "BOWS2"
    assert report["quality_factor"] == 75 and report["method"] == "JUNIWARD"
    assert report["selected_crops_have_signal"] is True
    assert report["real_model_trained"] is True and report["in_sample_only"] is True
    assert report["model_discarded"] is True
    assert report["probe_pixels_read"] == 0 and report["validation_test_used"] is False
    assert report["accuracy_qualification"] == "unavailable" and report["deployed"] is False
    assert report["initial_parameter_sha256"] != report["learned_parameter_sha256"]
    execution = report["execution"]
    assert execution["device"] == "cuda:0" and execution["cpu_fallback"] is False
    assert execution["tf32"] is False and execution["mixed_precision"] is False
    assert execution["host_memory_max_bytes"] <= 8 * 1024**3
    assert execution["torch_allocator_limit_bytes"] <= 4 * 1024**3
    assert report["process_peak_gpu_reserved_bytes"] <= 4 * 1024**3
    assert 0 < report["seconds"] < 1800
    for phase in ("initial", "final"):
        metrics = report[phase]
        assert report["singleton_batch_parity"][phase] is True
        assert [row["position"] for row in metrics["rows"]] == list(range(4))
        assert [row["label"] for row in metrics["rows"]] == [0, 1, 0, 1]
        correct, losses = 0, []
        for row in metrics["rows"]:
            logits = row["logits"]
            assert len(logits) == 2 and all(math.isfinite(v) for v in logits)
            top = max(logits)
            # Independent two-class logsumexp, not the training margin helper.
            losses.append(
                top + math.log(sum(math.exp(v - top) for v in logits)) - logits[row["label"]]
            )
            correct += int(logits[1] >= logits[0]) == row["label"]
        loss = sum(losses) / 4
        assert metrics["correct"] == correct
        assert metrics["cross_entropy"] == pytest.approx(loss, abs=1e-10)
        assert metrics["learnability_criterion_passed"] == (correct == 4 and loss < 0.2)
        assert audit[arm][phase]["correct"] == correct
        assert audit[arm][phase]["cross_entropy"] == pytest.approx(loss, abs=1e-10)


def test_both_real_arms_have_identical_initialization_and_exposure():
    legacy, reference = (evidence(arm)[0] for arm in ("legacy", "reference"))
    for name in (
        "initial_parameter_sha256",
        "seed",
        "fit_rows",
        "schedule_sha256",
        "manifest_sha256",
        "audit_sha256",
        "source_sha256",
        "epochs",
        "optimizer_updates",
    ):
        assert legacy[name] == reference[name]
    assert legacy["architecture"] != reference["architecture"]
    audit = json.loads((ROOT / "benchmarks/jpeg-learnability-audit-20261010.json").read_bytes())
    assert audit["matched_initialization_and_exposure"] is True
    assert audit["independent_whole_model_oracle"] is False
    assert audit["accuracy_qualification"] == "unavailable" and audit["deployed"] is False
    assert (
        audit["auditor_source_sha256"]["tests/test_learnability_evidence.py"]
        == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )
