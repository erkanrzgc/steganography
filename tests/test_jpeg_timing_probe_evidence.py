"""Portable real reader, unavailable CPU and completed physical CUDA evidence."""

import hashlib
import json
from pathlib import Path

from core import jpeg_timing_probe as probe

ROOT = Path(__file__).resolve().parents[1]


def record(name):
    raw = (ROOT / "benchmarks" / name).read_bytes()
    for forbidden in (b"/home/", b".benchmark/", b"password", b"api_key", b"access_token"):
        assert forbidden not in raw
    return json.loads(raw)


def test_reader_readiness_is_real_data_but_not_optimization():
    ready = record("jpeg-real-timing-reader-readiness-20261010.json")
    assert ready["status"] == "completed" and ready["real_data_used"]
    assert ready["manifest_sha256"] == probe.MANIFEST_SHA
    assert ready["audit_sha256"] == probe.AUDIT_SHA
    assert ready["real_rows_verified"] == 480
    assert ready["schedule_batches_read"] == 66 and ready["max_fetch_bytes"] == 1024**2
    assert (
        ready["ordered_fetch_bytes_sha256"]
        == "36ac51c26668d48d92eb45a6a1319726a6d326bf5ada64bc9fc7f7ace3331672"
    )
    assert ready["optimizer_updates"] == 0 and not ready["real_model_trained"]
    assert ready["GPU_timing"] == ready["accuracy_qualification"] == "unavailable"


def test_local_actual_unavailable_and_source_hashes_are_portable():
    unavailable = record("jpeg-real-timing-local-unavailable-20261010.json")
    assert unavailable["status"] == "unavailable"
    assert not unavailable["cpu_fallback"] and not unavailable["real_data_used"]
    assert not unavailable["real_model_trained"] and not unavailable["deployed"]
    assert unavailable["environment"] == {"torch_version": "2.14.0+cpu", "cuda_available": False}
    assert "projection" not in unavailable and "steady_intervals_seconds" not in unavailable
    assert unavailable["accuracy_qualification"] == "unavailable"
    assert {
        "core/jpeg_timing_probe.py",
        "steganography/research_jpeg_timing_probe.py",
    } <= unavailable["source_sha256"].keys()
    for name in ("core/jpeg_timing_probe.py", "steganography/research_jpeg_timing_probe.py"):
        assert (
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            == unavailable["source_sha256"][name]
        )


def test_protocol_binds_frozen_workload_not_old_synthetic_timing():
    protocol = record("jpeg-real-timing-probe-protocol-20261010.json")
    assert protocol["manifest_sha256"] == probe.MANIFEST_SHA
    assert protocol["audit_sha256"] == probe.AUDIT_SHA
    assert protocol["optimizer_updates"] == probe.UPDATES == 66
    assert protocol["discarded_initial_intervals"] == probe.DISCARD == 18
    assert protocol["steady_intervals"] == probe.UPDATES - 1 - probe.DISCARD == 47
    assert protocol["limits"]["job_seconds"] == probe.MAX_SECONDS == 180
    assert not protocol["cloud_creation_authorized"] and not protocol["cpu_fallback"]


def test_completed_physical_cuda_evidence_is_timing_not_accuracy():
    report = record("jpeg-real-timing-rtx5060-20261010.json")
    assert report["schema_version"] == "srnet-cuda-real-timing-v1"
    assert report["status"] == "completed"
    assert report["manifest_sha256"] == probe.MANIFEST_SHA
    assert report["audit_sha256"] == probe.AUDIT_SHA
    assert report["real_optimizer_updates"] == probe.UPDATES == 66
    assert report["discarded_initial_intervals"] == probe.DISCARD == 18
    assert len(report["steady_intervals_seconds"]) == 47
    assert report["projection"] == probe.projection(report["steady_intervals_seconds"])
    assert report["projection"]["estimate_only"]
    assert not report["projection"]["full_corpus_reader_integrated"]
    execution = report["execution"]
    assert execution["device"] == "cuda:0"
    assert execution["gpu_name"] == "NVIDIA GeForce RTX 5060 Laptop GPU"
    assert execution["torch_version"] == "2.14.0+cu130"
    assert execution["precision"] == "float32-ieee"
    assert execution["deterministic_algorithms"]
    assert not execution["cpu_fallback"]
    assert not execution["tf32"] and not execution["mixed_precision"]
    assert execution["torch_allocator_limit_bytes"] == 4 * 1024**3
    assert execution["host_memory_max_bytes"] == 8 * 1024**3
    assert report["max_fetch_bytes"] == 1024**2
    assert 0 < report["seconds"] < probe.MAX_SECONDS
    assert (
        0 < report["process_peak_gpu_allocated_bytes"]
        <= report["process_peak_gpu_reserved_bytes"]
        <= execution["torch_allocator_limit_bytes"]
        < execution["total_gpu_bytes"]
    )
    for name, digest in report["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    assert report["real_data_used"] and report["real_model_trained"]
    assert not report["production_model_trained"] and not report["deployed"]
    assert report["accuracy_qualification"] == "unavailable"
    assert "cost" not in report and "hourly_price" not in report
