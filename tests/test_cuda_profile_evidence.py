"""Immutable physical timing evidence, never a live GPU or learning test."""

import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_protocol_and_exact_physical_report():
    protocol = (ROOT / "docs/CUDA_THROUGHPUT_PROTOCOL.md").read_bytes()
    assert hashlib.sha256(protocol).hexdigest() == (
        "1a2b108cc5f1c24a2a067fa034684b0105bc3c2fc5b049c5c88de630dc6a45a8"
    )
    raw = (ROOT / "benchmarks/cuda-wsl-generated-profile-20261009.json").read_bytes()
    assert len(raw) < 16384
    assert hashlib.sha256(raw).hexdigest() == (
        "3446a184cf126d4a617b44b896883f9387d7edff5a323dafd824d34f4c24f29b"
    )
    assert b"/home/" not in raw and b"password" not in raw and b".ssh" not in raw
    report = json.loads(raw)
    assert report["schema_version"] == "srnet-cuda-generated-profile-v1"
    assert report["status"] == "completed"
    assert report["generated_optimizer_updates"] == 64
    assert report["discarded_initial_intervals"] == 16
    values = sorted(report["steady_intervals_seconds"])
    assert len(values) == 47 and all(math.isfinite(v) and v > 0 for v in values)
    # Independent linear percentile, no call into the production budget helper.
    p95 = values[43] + 0.7 * (values[44] - values[43])
    budget = report["fixed_fit_budget"]
    assert math.isclose(budget["steady_interval_p95_seconds"], p95, abs_tol=1e-15)
    assert budget["planned_epochs"] == 5 and budget["planned_updates"] == 16440
    assert budget["fixed_overhead_seconds"] == 120
    assert budget["steady_interval_safety_factor"] == 2
    estimate = 120 + 2 * p95 * 16440
    assert math.isclose(budget["estimated_fit_seconds"], estimate, abs_tol=1e-10)
    assert estimate > budget["fit_max_seconds"] == 1800
    assert not budget["eligible_to_attempt_fixed_fit"] and budget["estimate_only"]
    execution = report["execution"]
    assert execution["device"] == "cuda:0" and execution["compute_capability"] == [12, 0]
    assert execution["torch_allocator_limit_bytes"] == 4 * 1024**3
    assert execution["host_memory_max_bytes"] == 8 * 1024**3
    assert execution["precision"] == "float32-ieee" and execution["deterministic_algorithms"]
    assert not any(execution[k] for k in ("tf32", "mixed_precision", "cpu_fallback"))
    assert (
        0
        < report["process_peak_gpu_allocated_bytes"]
        <= report["process_peak_gpu_reserved_bytes"]
        < execution["torch_allocator_limit_bytes"]
    )
    assert not any(report[k] for k in ("real_data_used", "real_model_trained", "deployed"))
    assert report["accuracy_qualification"] == "unavailable"


def test_physical_profile_sources_match_historical_snapshot_not_future_engine():
    report = json.loads((ROOT / "benchmarks/cuda-wsl-generated-profile-20261009.json").read_bytes())
    probe = json.loads((ROOT / "benchmarks/cuda-wsl-generated-probe-20261009.json").read_bytes())
    # The fourteen execution dependencies are unchanged from the independently
    # pinned earlier report. Bind the two added sources explicitly; no Git
    # history or current-source equality requirement for future engine changes.
    expected = {
        **probe["source_sha256"],
        "core/srnet_profile.py": "3fbc6460e49a33dbe43cdb38f5a01a958c0ae025f0e986aff825524717058375",
        "steganography/research_cuda_profile.py": (
            "d740411931fb466f4549e933b1cb727b7a64f577c37e9ab3787d0e30f41e6fbb"
        ),
    }
    assert len(expected) == 16 and report["source_sha256"] == expected


@pytest.mark.parametrize(
    "name,checksum,protocol,protocol_checksum",
    [
        (
            "cuda-wsl-gradient-profile-20261009.json",
            "cb1c999ffc5d8b8b3ba1fcecf230ce2edf27d3d5be1edc345492881250ea66ae",
            "CUDA_GRADIENT_TIMING_PROTOCOL.md",
            "bf015399391f2a3c23f9cc3edd8bc25ac3ef630bb2b342f5b51c5bbeacd4eb78",
        ),
        (
            "cuda-wsl-state-profile-20261009.json",
            "58c21efd8a93c60f8a83550ee29b7e635cc6320376e03cb5955dc981b1394bfa",
            "CUDA_STATE_TIMING_PROTOCOL.md",
            "78798edde84b0aeb39c0d7e0e281cce9241ab74bd44b043768224dd226ac8727",
        ),
    ],
)
def test_optimized_reports_keep_fixed_budget_and_no_accuracy(
    name,
    checksum,
    protocol,
    protocol_checksum,
):
    raw = (ROOT / "benchmarks" / name).read_bytes()
    assert len(raw) < 16384 and hashlib.sha256(raw).hexdigest() == checksum
    assert hashlib.sha256((ROOT / "docs" / protocol).read_bytes()).hexdigest() == protocol_checksum
    report = json.loads(raw)
    assert report["status"] == "completed"
    assert report["schema_version"] == "srnet-cuda-generated-profile-v1"
    assert len(report["source_sha256"]) == 16
    assert report["generated_optimizer_updates"] == 64
    assert report["discarded_initial_intervals"] == 16
    values = sorted(report["steady_intervals_seconds"])
    assert len(values) == 47 and all(math.isfinite(v) and v > 0 for v in values)
    p95 = values[43] + 0.7 * (values[44] - values[43])
    budget = report["fixed_fit_budget"]
    assert budget["planned_epochs"] == 5 and budget["planned_updates"] == 16440
    assert budget["steady_interval_safety_factor"] == 2
    assert budget["fixed_overhead_seconds"] == 120 and budget["fit_max_seconds"] == 1800
    assert math.isclose(budget["steady_interval_p95_seconds"], p95, abs_tol=1e-15)
    assert math.isclose(budget["estimated_fit_seconds"], 120 + 2 * p95 * 16440, abs_tol=1e-10)
    assert budget["estimated_fit_seconds"] > 1800
    assert not budget["eligible_to_attempt_fixed_fit"] and budget["estimate_only"]
    assert not any(report[k] for k in ("real_data_used", "real_model_trained", "deployed"))
    assert report["accuracy_qualification"] == "unavailable"
    execution = report["execution"]
    assert execution["device"] == "cuda:0" and execution["precision"] == "float32-ieee"
    assert execution["deterministic_algorithms"]
    assert not any(execution[k] for k in ("tf32", "mixed_precision", "cpu_fallback"))
    assert execution["host_memory_max_bytes"] == 8 * 1024**3
    assert execution["torch_allocator_limit_bytes"] == 4 * 1024**3
    assert (
        0
        < report["process_peak_gpu_allocated_bytes"]
        <= report["process_peak_gpu_reserved_bytes"]
        < 4 * 1024**3
    )
    assert b"/home/" not in raw and b"password" not in raw and b".ssh" not in raw
