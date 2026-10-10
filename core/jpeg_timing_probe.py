"""Audited train-only real-I/O CUDA timing; no model publication or scoring."""

from __future__ import annotations

import hashlib
import math
import time

import numpy as np

from core import jpeg_float256, jpeg_timing, srnet_cuda, srnet_diversity_sampling, srnet_training
from core.srnet_stream import ROW_BYTES, deadline_after, document, regular_open, signature

UPDATES = 66
DISCARD = 18
MAX_SECONDS = 180
MANIFEST_SHA = "53af09292d57f4ddba383060452a21979f00028a9938af3a86ccdf4372d136ca"
AUDIT_SHA = "9df6fed4383fe9d84717c3c1edb145160a4e0c44fbeb3114a9c2a253a3b7d7bb"


class TimingReader:
    """Own one verified 120 MiB file; fetch <=4 rows, never mmap whole corpus."""

    def __init__(self, root, audit, *, deadline):
        self.stream = None
        self.deadline = deadline
        self.max_batch_bytes = 0
        deadline()
        manifest = document(root / "manifest.json", MANIFEST_SHA)
        proof = document(audit, AUDIT_SHA)
        if (
            manifest.get("schema_version") != "jpeg-real-timing-kit-v1"
            or manifest.get("status") != "completed"
            or proof.get("schema_version") != "jpeg-real-timing-independent-audit-v1"
            or proof.get("status") != "completed"
            or proof.get("manifest_sha256") != MANIFEST_SHA
            or proof.get("protocol_sha256") != manifest.get("protocol_sha256")
            or proof.get("independent_IDCT_rows") != 480
            or proof.get("validation_pixels_read") != 0
            or manifest.get("shape") != [480, 1, 256, 256]
            or manifest.get("dtype") != "<f4"
            or manifest.get("tensor_bytes") != 480 * ROW_BYTES
        ):
            raise ValueError("timing requires complete bound independent kit audit")
        self.samples = manifest["samples"]
        if len(self.samples) != 480 or {r["source_group"] for r in self.samples} != set(
            jpeg_timing.ORIGINS
        ):
            raise ValueError("timing complete source accounting mismatch")
        batches, schedule = srnet_diversity_sampling.epoch_batches(
            self.samples, seed=jpeg_timing.SEED, epoch=0
        )
        if schedule != manifest.get("schedule") or len(batches) != 192:
            raise ValueError("timing schedule binding mismatch")
        self.batches = batches[:UPDATES]
        for row in self.samples:
            deadline()
            if row["path"] != f"jpeg/{row['sha256']}.jpg":
                raise ValueError("timing JPEG name mismatch")
            jpeg_timing.read_bytes(root / row["path"], row["sha256"], row["size"])
        self.tensor_sha = manifest["tensor_sha256"]
        self.stream = regular_open(root / "pixels.f32")
        try:
            self.initial = signature(self.stream)
            if self.initial[2] != 480 * ROW_BYTES:
                raise ValueError("timing tensor byte accounting mismatch")
            self.verify()
        except BaseException:
            self.close()
            raise

    def verify(self):
        self.deadline()
        if self.stream is None or signature(self.stream) != self.initial:
            raise ValueError("timing tensor closed or mutated")
        self.stream.seek(0)
        digest = hashlib.sha256()
        while raw := self.stream.read(1024**2):
            self.deadline()
            values = np.frombuffer(raw, dtype="<f4")
            if not np.isfinite(values).all() or np.any(
                np.abs(values) > jpeg_float256.MAX_MAGNITUDE
            ):
                raise ValueError("timing tensor invalid values")
            digest.update(raw)
        if digest.hexdigest() != self.tensor_sha or signature(self.stream) != self.initial:
            raise ValueError("timing tensor identity mismatch")

    def batch(self, indices):
        self.deadline()
        if (
            self.stream is None
            or signature(self.stream) != self.initial
            or not isinstance(indices, np.ndarray)
            or indices.dtype.kind not in "iu"
            or indices.shape != (4,)
            or np.any(indices < 0)
            or np.any(indices >= 480)
        ):
            raise ValueError("timing batch or tensor state outside bounds")
        result = np.empty((4, 1, 256, 256), dtype="<f4")
        for target, index in zip(result, indices, strict=True):
            self.deadline()
            self.stream.seek(int(index) * ROW_BYTES)
            raw = self.stream.read(ROW_BYTES)
            if len(raw) != ROW_BYTES or signature(self.stream) != self.initial:
                raise ValueError("timing row truncated or mutated")
            target[:] = np.frombuffer(raw, dtype="<f4").reshape(1, 256, 256)
        self.max_batch_bytes = max(self.max_batch_bytes, result.nbytes)
        return result

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def projection(intervals):
    """Fixed warmed p95-based per-epoch estimate; never a duration guarantee."""
    if (
        not isinstance(intervals, list)
        or len(intervals) != UPDATES - 1 - DISCARD
        or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in intervals)
    ):
        raise ValueError("real timing requires 47 positive finite warmed intervals")
    median, p95 = float(np.median(intervals)), float(np.quantile(intervals, 0.95))
    estimate = p95 * 4932 * 1.25
    if not math.isfinite(estimate):
        raise ValueError("real timing projection overflow")
    return {
        "full_corpus_updates_per_epoch": 4932,
        "median_update_seconds": median,
        "p95_update_seconds": p95,
        "safety_factor": 1.25,
        "estimated_optimizer_seconds_per_epoch": estimate,
        "estimate_only": True,
        "excluded": ["startup", "upload", "validation", "checkpoint", "disk_charges"],
        "full_corpus_reader_integrated": False,
    }


def profile(root, audit):
    # Fail before touching real data on non-CUDA hosts; never silently run CPU.
    execution = srnet_cuda.inspect()
    import torch

    deadline = deadline_after(MAX_SECONDS)
    started = time.monotonic()
    with TimingReader(root, audit, deadline=deadline) as reader:
        ready_seconds = time.monotonic() - started
        starts: list[float] = []

        def fetch(indices):
            starts.append(time.monotonic())
            return reader.batch(indices)

        _, records = srnet_training._learn(
            fetch=fetch,
            batches=lambda _: reader.batches,
            epochs=1,
            seed=jpeg_timing.SEED,
            params=srnet_training.settings({"max_seconds": MAX_SECONDS}),
            target_pairs=2,
            outer_deadline=deadline,
            device="cuda:0",
        )
        if len(starts) != UPDATES or len(records) != 1 or records[0]["updates"] != UPDATES:
            raise ValueError("real timing optimizer accounting incomplete")
        reader.verify()
        deadline()
        intervals = np.diff(starts)[DISCARD:].tolist()
        return {
            "execution": execution,
            "manifest_sha256": MANIFEST_SHA,
            "audit_sha256": AUDIT_SHA,
            "real_optimizer_updates": UPDATES,
            "discarded_initial_intervals": DISCARD,
            "steady_intervals_seconds": intervals,
            "projection": projection(intervals),
            "input_verification_seconds": ready_seconds,
            "seconds": time.monotonic() - started,
            "max_fetch_bytes": reader.max_batch_bytes,
            "process_peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "process_peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "real_data_used": True,
            "real_model_trained": True,
            "optimizer_state": "timing-only, discarded; no weights/checkpoints written",
            "production_model_trained": False,
            "io_regime": (
                "verified small cache, likely warm OS cache; full-corpus cold I/O unmeasured"
            ),
            "accuracy_qualification": "unavailable",
            "deployed": False,
        }
