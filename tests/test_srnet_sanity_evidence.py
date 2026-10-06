"""Real in-sample failure and numerical correctness must remain separate."""

import hashlib
import json
from pathlib import Path

import pytest


def test_real_train_only_sanity_fails_despite_complete_replay():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks/srnet-tiny-sanity-20261007.json"
    report = json.loads(path.read_bytes())
    audit = json.loads((root / "benchmarks/srnet-tiny-sanity-audit-20261007.json").read_bytes())
    assert report["protocol_commit"] == "64584b0"
    assert (
        report["protocol_sha256"]
        == hashlib.sha256((root / "docs/SRNET_TINY_SANITY_PROTOCOL.md").read_bytes()).hexdigest()
    )
    assert (
        report["execution_source_sha256"]["core/srnet_sanity.py"]
        == "0523685f2de210064edc7d1610626f863ac5a3197468d619af8f10714aaf6210"
    )
    assert report["status"] == "completed" and report["optimizer_updates"] == 400
    assert (
        len(report["epoch_training"])
        == len(report["epoch_pair_schedule"])
        == len(report["epoch_batch_schedule"])
        == 50
    )
    for i, record in enumerate(report["epoch_training"]):
        assert record["epoch"] == i and record["updates"] == 8
        assert report["epoch_pair_schedule"][i]["pairs"] == 16
        assert report["epoch_batch_schedule"][i]["optimizer_updates"] == 8
    selected = report["selected_rows"]
    assert len(selected) == 24 and sum(r["label"] == "cover" for r in selected) == 8
    assert len({(r["source_id"], r["lineage"]) for r in selected}) == 6
    assert {r["quality_factor"] for r in selected} == {None, 75, 95}
    assert not report["sanity_objectives_passed"]
    assert report["sanity_objectives"]["relative_loss_reduction_at_least_0_25"]
    assert not report["sanity_objectives"]["final_batch_loss_at_most_0_35"]
    assert not report["sanity_objectives"]["stored_bn_train_balanced_accuracy_at_least_0_90"]
    assert report["stored_bn_singleton_train_metrics"]["balanced_accuracy"] == 0.5
    assert report["epoch_training"][-1]["mean_pair_loss"] == pytest.approx(0.6930854693055153)
    assert report["in_sample_only"] and not report["validation_pixels_loaded"]
    assert report["accuracy_qualification"] == "unavailable" and not report["deployed"]
    assert audit["passed"] and audit["bn_updates_verified"] == 400
    assert audit["native_singleton_rows"] == 24 and len(audit["independent_forward_audit"]) == 9
    assert all(a["passed"] and a["decisions_equal"] for a in audit["independent_forward_audit"])
    assert len(audit["input_pair_differences"]) == 16
    assert all(a["input_difference_rms"] > 0 for a in audit["input_pair_differences"])
    assert audit["model_sha256"] == report["model_sha256"]
    original = {k: v for k, v in report.items() if k != "execution_source_sha256"}
    assert (
        hashlib.sha256(
            (json.dumps(original, indent=2, allow_nan=False) + "\n").encode()
        ).hexdigest()
        == audit["sanity_report_sha256"]
    )
    assert "/home/" not in path.read_text() and ".benchmark/" not in path.read_text()
