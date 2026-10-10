"""Immutable actual pilot evidence; no live training or accuracy qualification."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HASHES = [
    "e9ca5d1beecfacd5de74f88a079bc211b739f1a63c998deba56d557a3eb3fdaf",
    "904fd7fbf15777b09e4bc8073624cb83c2305860f85b48a6052a7f06880893d3",
    "8142a05f362715198c0a3a122e57440951775ee4c2442632cebd396d7c69c4e2",
    "cf3bbf41f41662da05e4678ac9195ffa1483207be293963059463bd412edd824",
    "b48686d96f9302b95380a830763c550f238ba9f378878ac4ccf5c6ad2b1e22c7",
]
PLAN_SHA = "13ea2f8d97e382e5a4c1ed2df8d84aeebe70c656948219dfed2407adeaf0ed90"


def pinned(name, digest):
    raw = (ROOT / "benchmarks" / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    assert not any(secret in raw for secret in (b"/home/", b"password", b".ssh"))
    return json.loads(raw)


def test_original_gpu_epoch_chain_is_complete_but_not_deployed():
    plan = pinned("jpeg-gpu-epoch-plan-20261010.json", PLAN_SHA)
    previous = None
    for epoch, digest in enumerate(HASHES):
        card = pinned(f"jpeg-gpu-epoch-{epoch:03d}-20261010.json", digest)
        assert card["status"] == "completed" and card["next_epoch"] == epoch + 1
        assert card["plan_sha256"] == PLAN_SHA
        assert card["parent_report_sha256"] == previous
        assert card["source_sha256"] == plan["source_sha256"]
        assert len(card["epoch_training"]) == epoch + 1
        assert all(item["updates"] == 3288 for item in card["epoch_training"])
        assert 0 < card["seconds"] < 1800
        assert card["execution"]["device"] == "cuda:0"
        assert not card["validation_used"] and not card["deployed"]
        assert card["accuracy_qualification"] == "unavailable"
        previous = digest


def test_original_development_scores_preserve_failed_detection_gates():
    report = pinned(
        "jpeg-epoch-development-validation-20261010.json",
        "652fdf708beceb70ef4fb864817fe8fd5dca6cb043cc67e21b4d48bad847e657",
    )
    assert report["status"] == "completed" and report["validation_rows"] == 1611
    assert report["plan_sha256"] == PLAN_SHA and report["final_card_sha256"] == HASHES[-1]
    assert len(report["independent_forward_audit"]) == 9
    assert all(item["passed"] for item in report["independent_forward_audit"])
    assert len(report["cells"]) == 6
    for cell in report["cells"]:
        metrics = cell["metrics"]
        assert cell["support"] == "experimental"
        assert metrics["positives"] == metrics["negatives"] < 1000
        assert metrics["roc_auc"] < 0.90 and metrics["balanced_accuracy"] < 0.85
        assert metrics["recall"] < 0.80 and metrics["false_positive_rate"] > 0.03
        assert metrics["recommended_threshold"] is None
    assert report["threshold"] == 0.5
    assert not report["threshold_tuned"] and not report["normalization_refreshed"]
    assert not report["cross_source_heldout"] and not report["deployed"]
    assert report["qualification"] == "development_only_not_supported"
