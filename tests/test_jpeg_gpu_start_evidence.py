"""A dated GPU startup is not a completed learning or support qualification."""

import json
import re
from pathlib import Path


def test_gpu_start_keeps_capacity_failure_and_unmeasured_learning_explicit():
    root = Path(__file__).resolve().parents[1]
    raw = (root / "benchmarks/jpeg-five-view-gpu-start-20261012.json").read_text()
    data = json.loads(raw)
    assert data["status"] == "running_observed"
    assert len(data["source_sha256"]) == 49
    assert data["transfer"]["regular_files"] == 1285
    assert data["transfer"]["checksum_differences"] == data["transfer"]["symlinks"] == 0
    assert data["originals"] == {"fit": 512, "development": 128, "total": 640}
    execution = data["execution"]
    assert execution["device"] == "cuda:0"
    assert execution["host_memory_max_bytes"] == 8 * 1024**3
    assert execution["torch_allocator_limit_bytes"] == 4 * 1024**3
    assert execution["mixed_precision"] is execution["tf32"] is False
    capacity = {c["rows"]: c for c in data["capacity"]}
    assert capacity[8]["status"] == capacity[16]["status"] == "completed"
    assert capacity[32]["status"] == "unavailable"
    assert capacity[32]["reason"] == "gpu_allocator_out_of_memory"
    assert all(c["generated_inputs"] for c in capacity.values())
    assert capacity[8]["optimizer_updates"] == capacity[16]["optimizer_updates"] == 2
    assert data["trial_plan"]["selected_genuine_batch_rows"] == 16
    assert data["trial_plan"]["presented_view_rows_per_trial"] == 32768
    assert data["trial_plan"]["trial_deadline_seconds"] == 1800
    assert data["trial_plan"]["controls_budget_seconds"] == 14400
    progress = data["observed_progress"]
    assert progress["worker"] == "fit" and progress["active_arm"] == "legacy"
    assert progress["completed_trials"] == 0
    assert progress["final_report_available"] is progress["model_completed"] is False
    assert "not_observed" in progress["optimizer_update_count"]
    assert data["training_results"] == "pending"
    assert data["accuracy_qualification"] == "unavailable"
    assert data["deployed"] is data["reserved_sources_scored"] is data["cpu_fallback"] is False
    assert data["cloud_spending"] == 0
    for field in ("manifest_sha256", "audit_sha256"):
        assert re.fullmatch("[a-f0-9]{64}", data[field])
    assert "/home/" not in raw and "/tmp/" not in raw  # noqa: S108 - privacy assertion
    assert "password" not in raw and "api_key" not in raw
