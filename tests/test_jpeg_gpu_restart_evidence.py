"""A fresh audit/startup does not recover or qualify a historical failed run."""

import hashlib
import json
from pathlib import Path


def test_gpu_restart_preserves_new_binding_old_failure_and_unchanged_limits():
    root = Path(__file__).resolve().parents[1]
    raw = (root / "benchmarks/jpeg-five-view-gpu-restart-20261013.json").read_text()
    proof = json.loads(raw)
    historical_path = root / "benchmarks/jpeg-five-view-gpu-start-20261012.json"
    historical = json.loads(historical_path.read_text())
    # Historical evidence must also validate in source archives/shallow CI clones.
    bound = proof["source_sha256"]
    assert set(bound) == {*historical["source_sha256"], "core/jpeg_study_process.py"}
    for name, checksum in bound.items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert len(checksum) == 64 and all(c in "0123456789abcdef" for c in checksum)
    digest = hashlib.sha256(json.dumps(bound, sort_keys=True).encode()).hexdigest()
    assert proof["execution_source_closure_sha256"] == digest
    assert proof["execution_source_files"] == len(bound) == 50
    assert proof["manifest_sha256"] == historical["manifest_sha256"]
    assert proof["fresh_audit_sha256"] != historical["audit_sha256"]
    assert proof["historical_audit_or_snapshot_rebound"] is False
    assert proof["historical_failed_trial_reclassified"] is False
    assert proof["historical_snapshot_resume"] is False
    assert proof["fresh_audit"]["status"] == "completed"
    assert proof["fresh_audit"]["fit_views"] + proof["fresh_audit"]["development_views"] == 6400
    assert proof["fresh_audit"]["original_identities_changed"] is False
    assert proof["fresh_audit"]["unchanged_view_pairs_retained"] is True
    assert proof["fresh_audit"]["corpus_regenerated"] is False
    capacity = {entry["rows"]: entry for entry in proof["capacity"]}
    assert capacity[8]["status"] == capacity[16]["status"] == "completed"
    assert capacity[32]["status"] == "unavailable" and not capacity[32]["retry"]
    assert capacity[32]["reason"] == "gpu_allocator_out_of_memory"
    plan = proof["trial_plan"]
    assert plan["trials"] == 8 and plan["presented_view_rows_per_trial"] == 32768
    assert plan["worker_cpu_seconds"] == [1800, 1801]
    assert plan["trial_deadline_seconds"] == 1800 and plan["controls_budget_seconds"] == 14400
    assert proof["execution"]["torch_allocator_limit_bytes"] == 4 * 1024**3
    assert proof["execution"]["host_memory_max_bytes"] == 8 * 1024**3
    assert proof["status"] == "running_observed" and proof["training_results"] == "pending"
    assert proof["observed_progress"]["completed_trials"] == 0
    assert proof["observed_progress"]["optimizer_update_count"] == "not_observed"
    assert proof["accuracy_qualification"] == "unavailable"
    assert proof["deployed"] is proof["reserved_sources_scored"] is False
    assert proof["regression"]["skipped"] == 1 and proof["regression"]["passed"] == 2635
    assert "not passed" in proof["regression"]["skip_reason"]
    for forbidden in ("/home/", "api_key", "password", "stderr"):
        assert forbidden not in raw
