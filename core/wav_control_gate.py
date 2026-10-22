"""Read-only acceptance of source-bound, full-exposure generated WAV controls."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

from core.nonjpeg_data import regular_read

PROTOCOL_ID = "wav-generated-full-control-20261021-v1"
SOURCES = (
    "core/__init__.py", "core/features.py", "core/nonjpeg_compare_metrics.py",
    "core/nonjpeg_data.py", "core/nonjpeg_features.py", "core/nonjpeg_linear.py",
    "core/nonjpeg_models.py", "core/spatial_cooccurrence.py", "core/spatial_parity.py",
    "core/wav_pilot.py", "core/wav_pilot_training.py", "core/wav_pilot_v3.py",
    "core/wav_control_gate.py", "steganography/research_wav_control.py",
    "tests/test_wav_control_gate.py", "tests/test_wav_pilot_v3.py",
    "docs/WAV_FULL_CONTROL_PROTOCOL_20261021.md", "pyproject.toml",
    "requirements-research-ci.txt", "benchmarks/wav-audiomnist-preparation-20261017.json",
    "scripts/audit-wav-positive-control.py",
)
SPEC: dict[str, Any] = {
    "recipe": "audio-pcm16-tlu-parity-gn1d-v3",
    "seeds": [20261016, 20261017],
    "train_files": 400, "development_files": 200,
    "epochs": 20, "updates_per_seed": 1140, "presentations_per_seed": 72000,
    "virtual_batch": 64, "final_batch_rows": 16,
    "train_auc_min": 0.60, "development_auc_min": 0.70,
    "wall_seconds_max": 3600,
    "host_bytes_max": 8 * 1024**3, "cuda_bytes_max": 4 * 1024**3,
    "swap_bytes_max": 0,
}
CELLS = {(method, rate, plane) for method in ("sequential", "scattered")
         for rate in (5, 40) for plane in (0, 1)}
CORPUS_EVIDENCE = {
    "train_files": 400, "development_files": 200,
    "train_groups": 20, "development_groups": 10, "files_per_group": 20,
    "group_overlap": 0, "decoded_pcm_overlap": 0,
    "payload_oracles_passed": 4800,
    "length_schedule": "exact_metadata_order",
}


def read_document(path: Path) -> tuple[dict[str, Any], str]:
    raw = regular_read(path, 1024**2)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("control document must be an object")
    return value, hashlib.sha256(raw).hexdigest()


def source_hash(path: Path) -> str:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("control source symlink")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("control source must be regular")
        raw = stream.read(1024**2 + 1)
    if len(raw) > 1024**2:
        raise ValueError("control source size limit")
    return hashlib.sha256(raw).hexdigest()


def read_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol, digest = read_document(path)
    if (
        protocol.get("schema_version") != "wav-full-control-protocol-v1"
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("spec") != SPEC
        or protocol.get("real_fit_authorized") is not False
        or not isinstance(protocol.get("source_closure_sha256"), dict)
        or set(protocol["source_closure_sha256"]) != set(SOURCES)
    ):
        raise ValueError("full-control protocol contract mismatch")
    for name, expected in protocol["source_closure_sha256"].items():
        if source_hash(root / name) != expected:
            raise ValueError("full-control source binding mismatch")
    return protocol, digest


def _number(value: Any, low: float, high: float) -> bool:
    return (type(value) in (int, float) and low <= value <= high and math.isfinite(value))


def evaluate(protocol: dict[str, Any], digest: str, receipt: dict[str, Any]) -> dict[str, Any]:
    """Reject incomplete exposure/provenance; preserve a completed failed learning gate."""
    base = {
        "schema_version": "wav-full-control-decision-v1", "protocol_id": PROTOCOL_ID,
        "protocol_sha256": digest, "real_fit_authorized": False,
        "production_qualified": False,
    }

    def unavailable(reason: str) -> dict[str, Any]:
        return {**base, "status": "unavailable", "reason": reason}

    if (
        receipt.get("schema_version") != "wav-full-control-result-v1"
        or receipt.get("protocol_sha256") != digest
        or receipt.get("source_closure_sha256") != protocol["source_closure_sha256"]
        or receipt.get("recipe") != SPEC["recipe"]
        or receipt.get("generated_only") is not True
        or receipt.get("real_media_opened") is not False
        or receipt.get("reserved_media_opened") is not False
        or receipt.get("checkpoint_written") is not False
        or type(receipt.get("real_optimizer_updates")) is not int
        or receipt["real_optimizer_updates"] != 0
        or receipt.get("device") != "NVIDIA GeForce RTX 5060 Laptop GPU"
    ):
        return unavailable("provenance_or_recipe_mismatch")
    corpus = receipt.get("corpus")
    if not isinstance(corpus, dict) or set(corpus) != set(CORPUS_EVIDENCE) or any(
        type(corpus[k]) is not type(v) or corpus[k] != v for k, v in CORPUS_EVIDENCE.items()
    ):
        return unavailable("complete_disjoint_corpus_required")
    limits = {
        "wall_seconds": (0.000001, SPEC["wall_seconds_max"]),
        "host_peak_bytes": (1, SPEC["host_bytes_max"]),
        "cuda_peak_bytes": (1, SPEC["cuda_bytes_max"]),
        "swap_peak_bytes": (0, 0),
    }
    if any(not _number(receipt.get(key), low, high) for key, (low, high) in limits.items()):
        return unavailable("resource_evidence_missing_or_exceeded")
    runs = receipt.get("seeds")
    if not isinstance(runs, list) or len(runs) != 2 or any(not isinstance(r, dict) for r in runs):
        return unavailable("two_complete_seeds_required")
    if any(type(r.get("seed")) is not int for r in runs) or {
        r["seed"] for r in runs
    } != set(SPEC["seeds"]):
        return unavailable("two_complete_seeds_required")
    learning = []
    for run in runs:
        counts = {
            "epochs": SPEC["epochs"], "optimizer_updates": SPEC["updates_per_seed"],
            "row_presentations": SPEC["presentations_per_seed"],
            "train_files": SPEC["train_files"], "development_files": SPEC["development_files"],
            "virtual_batch": SPEC["virtual_batch"], "final_batch_rows": SPEC["final_batch_rows"],
        }
        if any(type(run.get(k)) is not int or run[k] != v for k, v in counts.items()):
            return unavailable("full_exposure_required")
        cells = run.get("development_cells")
        if not isinstance(cells, list) or len(cells) != 8:
            return unavailable("eight_complete_cells_required")
        found = {}
        for cell in cells:
            if (
                not isinstance(cell, dict) or type(cell.get("rate_percent")) is not int
                or type(cell.get("plane")) is not int or not isinstance(cell.get("method"), str)
                or not _number(cell.get("roc_auc"), 0, 1)
            ):
                return unavailable("invalid_metric_evidence")
            key = (cell["method"], cell["rate_percent"], cell["plane"])
            if key not in CELLS or key in found:
                return unavailable("eight_complete_cells_required")
            found[key] = cell["roc_auc"]
        if not _number(run.get("train_auc_all"), 0, 1):
            return unavailable("invalid_metric_evidence")
        learning.append({
            "seed": run["seed"],
            "passed": run["train_auc_all"] >= SPEC["train_auc_min"] and all(
                found[(method, 40, 0)] >= SPEC["development_auc_min"]
                for method in ("sequential", "scattered")
            ),
        })
    return {**base, "status": "passed_generated_only" if all(r["passed"] for r in learning)
            else "failed_learning_gate", "seeds": learning}
