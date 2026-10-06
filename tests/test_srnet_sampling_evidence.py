"""Portable schedule evidence must not become fitted-model accuracy claims."""

import hashlib
import json
from pathlib import Path


def test_real_schedule_evidence_is_bound_complete_and_not_accuracy():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks/srnet-sampling-20261006.json"
    record = json.loads(path.read_bytes())
    preparation = json.loads((root / "benchmarks/float256-preparation-20261006.json").read_bytes())
    assert (
        record["protocol_sha256"]
        == hashlib.sha256((root / "docs/SRNET_SAMPLING_PROTOCOL.md").read_bytes()).hexdigest()
    )
    assert record["manifest_sha256"] == preparation["manifest_sha256"]
    assert record["implementation_protocol_commit"] == "d2e3132"
    assert record["passed"] and record["epochs_audited"] == 30
    assert record["train_rows"] == 2985 and record["validation_rows_excluded"] == 765
    assert record["real_training"] == record["accuracy_metrics"] == "unavailable"
    assert not record["deployed"] and not record["primary_detection_changed"]
    assert len(record["scopes"]) == 3
    assert "/home/" not in path.read_text() and ".benchmark/" not in path.read_text()
    for scope, rows, pair_count in zip(
        record["scopes"], (2985, 2367, 618), (3160, 1578, 412), strict=True
    ):
        plan = scope["plan"]
        assert scope["passed"] and not plan["validation_used"] and not plan["deployed"]
        assert plan["real_training"] == plan["accuracy_metrics"] == "unavailable"
        assert plan["train_cache_sha256"] == preparation["splits"]["train"]["cache_sha256"]
        assert plan["train_data_sha256"] == preparation["splits"]["train"]["data_sha256"]
        assert (
            scope["plan_sha256"]
            == hashlib.sha256(
                (json.dumps(plan, indent=2, allow_nan=False) + "\n").encode()
            ).hexdigest()
        )
        assert plan["training_scope"]["rows"] == rows
        assert plan["settings"] == {"epochs": 10, "seed": 20261012}
        assert len(plan["epochs"]) == 10
        assert len({e["ordered_pair_sha256"] for e in plan["epochs"]}) == 10
        source_ids = plan["training_scope"]["source_ids"]
        for epoch, e in enumerate(plan["epochs"]):
            assert e["epoch"] == epoch and e["pairs"] == pair_count
            assert e["rows"] == 2 * pair_count
            assert sum(c["sampled_pairs"] for c in e["cells"]) == pair_count
            assert all(c["sampled_pairs"] >= c["original_pairs"] for c in e["cells"])
            assert {c["method"] for c in e["cells"]} == {"JUNIWARD", "UERD"}
            for source in source_ids:
                cells = [c for c in e["cells"] if c["source_id"] == source]
                assert sum(c["sampled_pairs"] for c in cells) == e["pairs_per_source"]
                assert len({c["sampled_pairs"] for c in cells}) == 1
