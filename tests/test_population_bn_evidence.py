"""Physical own-GPU evidence remains a failed train-only intervention."""

import json
from pathlib import Path

from core import jpeg_population_control as control
from core.jpeg_timing_probe import AUDIT_SHA, MANIFEST_SHA
from steganography.research_jpeg_population_bn import sources

ROOT = Path(__file__).resolve().parents[1]


def evidence():
    raw = (ROOT / "benchmarks/jpeg-layerwise-population-bn-20261010.json").read_bytes()
    for forbidden in (b"/home/", b".benchmark/", b"password", b"api_key", b"access_token"):
        assert forbidden not in raw
    return json.loads(raw)


def test_physical_control_keeps_prior_model_and_data_hash_binding():
    report = evidence()
    assert report["status"] == "completed"
    assert report["schema_version"] == "jpeg-layerwise-population-bn-v1"
    assert report["source_sha256"] == sources()
    assert report["model_sha256"] == control.MODEL_SHA
    assert report["manifest_sha256"] == MANIFEST_SHA
    assert report["audit_sha256"] == AUDIT_SHA
    assert report["refreshed_model_sha256"] != report["model_sha256"]
    assert report["source_model_unchanged"] and report["learned_parameters_bit_identical"]
    assert report["optimizer_updates"] == 0
    assert report["calibration_rows"] == 360 and report["probe_rows"] == 120
    assert report["calibration_originals"] == 72 and report["probe_originals"] == 24
    assert report["original_groups_disjoint"]
    assert report["probe_is_historical_training_data"] and not report["validation_used"]


def test_numerical_hardware_gates_do_not_pass_detection():
    report = evidence()
    execution = report["execution"]
    assert execution["device"] == "cuda:0"
    assert execution["gpu_name"] == "NVIDIA GeForce RTX 5060 Laptop GPU"
    assert execution["host_memory_max_bytes"] == 8 * 1024**3
    assert execution["torch_allocator_limit_bytes"] == 4 * 1024**3
    assert execution["precision"] == "float32-ieee"
    assert not execution["mixed_precision"] and not execution["tf32"]
    assert not execution["cpu_fallback"] and execution["deterministic_algorithms"]
    assert report["singleton_batch_parity"] == {
        "baseline": True,
        "refreshed": True,
        "atol": 1e-4,
        "rtol": 1e-4,
    }
    assert report["numerical_gates_passed"]
    assert len(report["calibration_layers"]) == 26
    assert all(r["activation_count_per_channel"] % 360 == 0 for r in report["calibration_layers"])
    assert 0 < report["seconds"] < 1800
    assert (
        0
        < report["process_peak_gpu_allocated_bytes"]
        <= report["process_peak_gpu_reserved_bytes"]
        < 4 * 1024**3
    )
    assert report["accuracy_qualification"] == "unavailable" and not report["deployed"]


def test_all_ten_chance_level_cells_remain_visible():
    report = evidence()
    assert len(report["baseline_cells"]) == len(report["refreshed_cells"]) == 10
    for baseline, refreshed in zip(
        report["baseline_cells"], report["refreshed_cells"], strict=True
    ):
        for name in ("source_group", "quality_factor", "method"):
            assert baseline[name] == refreshed[name]
        assert refreshed["cover_rows"] == refreshed["stego_rows"] == 8
        assert refreshed["true_positive"] + refreshed["false_negative"] == 8
        assert refreshed["true_negative"] + refreshed["false_positive"] == 8
        assert refreshed["balanced_accuracy"] == 0.5
        assert 0.693 < refreshed["cross_entropy"] < baseline["cross_entropy"]
        assert refreshed["cover_rows_shared_across_method_cells"]
