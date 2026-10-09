"""Immutable recorded evidence checks, not a live CUDA hardware test."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def recorded(name, checksum):
    raw = (ROOT / "benchmarks" / name).read_bytes()
    assert len(raw) < 16 * 1024
    assert hashlib.sha256(raw).hexdigest() == checksum
    return json.loads(raw)


def test_recorded_physical_generated_probe_is_not_accuracy():
    report = recorded(
        "cuda-wsl-generated-probe-20261009.json",
        "27d89db7a1f8ab4aea8fff80c820f133cfc3029d5fb3134e14ff803f7c979d11",
    )
    assert report["schema_version"] == "srnet-cuda-generated-probe-v1"
    assert report["status"] == "completed"
    assert report["generated_optimizer_updates"] == 1
    assert len(report["source_sha256"]) == 14
    assert report["cpu_gpu_logit_max_difference"] == 9.1552734375e-05
    assert report["parity_atol"] == report["parity_rtol"] == 0.0001
    execution = report["execution"]
    assert execution["device"] == "cuda:0"
    assert execution["compute_capability"] == [12, 0]
    assert execution["host_memory_max_bytes"] == 8 * 1024**3
    assert execution["torch_allocator_limit_bytes"] == 4 * 1024**3
    assert execution["precision"] == "float32-ieee"
    assert execution["deterministic_algorithms"]
    assert not execution["tf32"] and not execution["mixed_precision"]
    assert not execution["cpu_fallback"]
    assert report["process_peak_gpu_allocated_bytes"] <= report["process_peak_gpu_reserved_bytes"]
    assert report["process_peak_gpu_reserved_bytes"] < execution["torch_allocator_limit_bytes"]
    assert not report["real_data_used"] and not report["real_model_trained"]
    assert not report["independent_math_oracle"] and not report["deployed"]
    assert report["accuracy_qualification"] == "unavailable"


def test_recorded_native_check_preserves_exact_train_only_schedule():
    report = recorded(
        "wsl-stream-readiness-20261009.json",
        "934d2c7fc382342c5cd42a9a3e01689bbf0169a4916cf980e2e6e6b518405523",
    )
    assert report["schema_version"] == "srnet-stream-readiness-v1"
    assert report["status"] == "completed"
    assert not report["models_trained"] and not report["detection_measured"]
    assert not report["validation_pixels_opened"]
    assert report["accuracy_qualification"] == "unavailable"
    assert report["unique_train_rows_read"] == 7380
    assert report["planned_optimizer_updates"] == 3288
    assert report["row_presentations"] == 13152
    assert report["max_batch_tensor_bytes"] == 1024**2
    assert report["ordered_tensor_sha256"] == (
        "38a8794d44f3edde43cd38e6b577b982f06b4a021e7d61ceccb0700a011ff91a"
    )
    plan = report["plan"]
    assert plan["schema_version"] == "srnet-stream-plan-v2"
    assert plan["original_train_lineages"] == 1638
    assert plan["train_tensor_bytes"] == 1934622720
    assert plan["seed"] == 20261008 and not plan["validation_used"]
    assert len(plan["source_sha256"]) == 13
    assert plan["source_sha256"] == report["source_sha256"]
    assert plan["execution"]["device"] == "cuda:0"
    assert len(plan["epochs"]) == 1
    epoch = plan["epochs"][0]
    pairs, batches = epoch["pair_schedule"], epoch["batch_schedule"]
    assert pairs["pairs"] == batches["pairs"] == 6576
    assert sum(cell["sampled_pairs"] for cell in pairs["cells"]) == 6576
    assert len(pairs["cells"]) == 6
    assert pairs["ordered_pair_sha256"] == (
        "8f1d15c66361f87703eb1c287acf3486601705b8b4f926293df7fb2dace7ed23"
    )
    assert batches["ordered_batch_sha256"] == (
        "2819b9db1099f95b5beca4ea50b488e58a9428596a9eae2259d9cd872d24936c"
    )


def test_readiness_protocols_remain_frozen():
    for name, checksum in (
        (
            "CUDA_GENERATED_PROBE_PROTOCOL.md",
            "1dc0a480136f9a2ace68fa18e345e23762973abee5b3ea030d8b958c55f9b561",
        ),
        (
            "WSL_TRANSFER_READINESS_PROTOCOL.md",
            "783566337998c88f38f9ebffdb84c54f1fbb2bc5bbd25795d002654f3990c860",
        ),
    ):
        assert hashlib.sha256((ROOT / "docs" / name).read_bytes()).hexdigest() == checksum
