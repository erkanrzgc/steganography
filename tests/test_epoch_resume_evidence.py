"""Pinned original physical resume/timing evidence, never a live accuracy test."""

import hashlib
import json
import math
from pathlib import Path


def test_frozen_physical_resume_probe_and_separate_epoch_budget():
    root = Path(__file__).resolve().parents[1]
    protocol = (root / "docs/CUDA_EPOCH_RESUME_PROTOCOL.md").read_bytes()
    assert (
        hashlib.sha256(protocol).hexdigest()
        == "2b15db4b1322cb5ef9884a7e842525e523470e47da30a5e8d20e7ad9851696c5"
    )
    raw = (root / "benchmarks/cuda-epoch-resume-probe-20261010.json").read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == "c00264d64a8522ff275694fb6823ff1b2a55450c55d42a2071a2733518c61813"
    )
    assert not any(v in raw for v in (b"/home/", b"password", b".ssh", b"environment"))
    report = json.loads(raw)
    assert report["protocol_sha256"] == hashlib.sha256(protocol).hexdigest()
    assert report["status"] == "completed" and report["exact_model_bn_optimizer_rng_loss"] is True
    assert report["generated_optimizer_updates"] == 128 and report["uninterrupted_updates"] == 64
    assert len(report["source_sha256"]) == 16
    assert {
        "core/srnet_checkpoint.py",
        "core/srnet_resume_probe.py",
        "steganography/research_srnet_epochs.py",
    } <= report["source_sha256"].keys()
    values = sorted(report["steady_intervals_seconds"])
    assert len(values) == 47 and all(math.isfinite(v) and v > 0 for v in values)
    p95 = values[43] + 0.7 * (values[44] - values[43])
    assert math.isclose(p95, report["steady_interval_p95_seconds"], abs_tol=1e-15)
    assert 120 + 2 * p95 * 3288 < 1800 < 120 + 2 * p95 * 16440
    assert report["execution"]["device"] == "cuda:0"
    assert report["execution"]["torch_allocator_limit_bytes"] == 4 * 1024**3
    assert report["execution"]["host_memory_max_bytes"] == 8 * 1024**3
    assert (
        not report["real_model_trained"] and not report["real_data_used"] and not report["deployed"]
    )
    assert report["accuracy_qualification"] == "unavailable"
    learning = (root / "docs/JPEG_EPOCH_LEARNING_PROTOCOL.md").read_bytes()
    assert (
        hashlib.sha256(learning).hexdigest()
        == "5a371c2a2a8192835ac194f2e57e21f583cbc34b3f490772e88e63f4a05d76c2"
    )
