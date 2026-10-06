"""Portable real accounting must never masquerade as model accuracy."""

import hashlib
import json
from pathlib import Path


def test_frozen_protocol_complete_real_groups_and_unavailable_accuracy():
    root = Path(__file__).resolve().parents[1]
    evidence = json.loads(
        (root / "benchmarks/srnet-multipair-accounting-20261007.json").read_bytes()
    )
    protocol = root / "docs/SRNET_MULTIPAIR_PROTOCOL.md"
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == evidence["protocol_sha256"]
    assert evidence["protocol_commit"] == "837cb79" and evidence["passed"] is True
    plan = evidence["plan"]
    serialized = (json.dumps(plan, indent=2, allow_nan=False) + "\n").encode()
    assert hashlib.sha256(serialized).hexdigest() == evidence["plan_sha256"]
    assert plan["schema_version"] == "srnet-training-plan-v2"
    assert plan["settings"] == {
        "epochs": 1,
        "seed": 20261012,
        "batch_recipe": "two-source-two-lineage-pairs-v1",
    }
    batch = plan["epoch_batch_schedule"][0]
    assert batch["batch_size"] == 4 and batch["optimizer_updates"] == 1580
    assert batch["pairs"] == 3160 and batch["rows"] == 6320
    assert batch["ordered_batch_sha256"] == evidence["ordered_batch_sha256"]
    assert evidence["batches_audited"] == 1580
    assert evidence["train_rows"] == 2985 and evidence["validation_hashes_excluded"] == 765
    assert evidence["original_stegos_covered"] == 1990
    assert len(evidence["pairs_per_source"]) == 2
    assert set(evidence["pairs_per_source"].values()) == {1580}
    assert sorted(c["sampled_pairs"] for c in plan["epochs"][0]["cells"]) == [
        395,
        395,
        395,
        395,
        790,
        790,
    ]
    assert plan["validation_used"] is False and evidence["primary_detection_changed"] is False
    for record in (plan, evidence):
        assert record["real_training"] == record["accuracy_metrics"] == "unavailable"
        assert record["deployed"] is False
    serialized_evidence = json.dumps(evidence)
    assert "/home/" not in serialized_evidence and ".benchmark/" not in serialized_evidence
