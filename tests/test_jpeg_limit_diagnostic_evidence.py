"""A valid snapshot/replay cannot qualify the historical budget-exceeded fit."""

import hashlib
import json
from pathlib import Path


def test_historical_cpu_failure_and_read_only_replay_remain_distinct():
    root = Path(__file__).resolve().parents[1]
    raw = (root / "benchmarks/jpeg-study-limit-diagnosis-20261013.json").read_text()
    proof = json.loads(raw)
    startup = json.loads((root / "benchmarks/jpeg-five-view-gpu-start-20261012.json").read_text())
    canonical = json.dumps(startup["source_sha256"], sort_keys=True).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    assert proof["historical_source_closure_sha256"] == digest
    assert proof["historical_source_files"] == len(startup["source_sha256"]) == 49
    assert proof["status"] == "diagnosed" and proof["historical_run_status"] == "failed"
    assert proof["completed_trial_reports"] == 2 and proof["planned_trials"] == 8
    assert proof["failed_arm"] == "convolution-l2" and proof["failed_seed"] == 20261012
    assert proof["root_cause"] == "worker_cpu_time_limit"
    kernel = proof["kernel_evidence"]
    assert kernel["signal"] == 24 and kernel["signal_name"] == "SIGXCPU"
    assert kernel["worker_rlimit_cpu_seconds"] == [1800, 1801]
    assert kernel["historical_worker_returncode"] == "not_retained"
    assert proof["checkpoint"]["presented_rows"] == 32768
    assert proof["checkpoint"]["optimizer_updates"] == 8192
    replay = proof["read_only_replay"]
    assert replay["status"] == "completed" and replay["device"] == "cuda:0"
    assert replay["development_image_rows"] == 256 and replay["batch_view_comparisons"] == 320
    assert replay["weights_unchanged"] and replay["source_closure_unchanged"]
    assert replay["optimizer_updates_performed"] == 0
    assert replay["parity_atol"] == replay["parity_rtol"] == 1e-4
    assert proof["accuracy_qualification"] == "unavailable"
    assert proof["historical_failed_trial_reclassified"] is proof["deployed"] is False
    assert proof["training_restarted"] is proof["reserved_sources_scored"] is False
    assert proof["cloud_spending"] == 0
    for forbidden in ("/home/", "password", "api_key", "stderr", "command"):
        assert forbidden not in raw
