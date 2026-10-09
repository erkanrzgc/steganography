"""Fixed generated GPU workload timing; never a real-data learning gate."""

from __future__ import annotations

import math
import time

import numpy as np

from core import srnet_cuda, srnet_training
from core.srnet_stream import deadline_after

UPDATES = 64
DISCARD = 16
MAX_SECONDS = 180
FIT_EPOCHS = 5
UPDATES_PER_EPOCH = 3288


def budget(intervals):
    """Preregistered conservative estimate, not a real fit duration guarantee."""
    if (
        not isinstance(intervals, list)
        or len(intervals) != UPDATES - 1 - DISCARD
        or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in intervals)
    ):
        raise ValueError("profile requires exactly 47 positive finite steady intervals")
    p95 = float(np.quantile(intervals, 0.95))
    estimate = 120 + 2 * p95 * UPDATES_PER_EPOCH * FIT_EPOCHS
    if not math.isfinite(estimate):
        raise ValueError("profile duration estimate overflow")
    return {
        "planned_epochs": FIT_EPOCHS,
        "planned_updates": UPDATES_PER_EPOCH * FIT_EPOCHS,
        "fit_max_seconds": 1800,
        "fixed_overhead_seconds": 120,
        "steady_interval_safety_factor": 2,
        "steady_interval_p95_seconds": p95,
        "estimated_fit_seconds": estimate,
        "eligible_to_attempt_fixed_fit": estimate <= 1800,
        "estimate_only": True,
    }


def profile():
    execution = srnet_cuda.inspect()
    import torch

    deadline = deadline_after(MAX_SECONDS)
    started = time.monotonic()
    yy, xx = np.indices((256, 256))
    base = (128 + 24 * np.sin(xx / 11) + 16 * np.cos(yy / 13)).astype("<f4")
    values = np.stack([base, base + 8 * ((xx + yy) % 2), base.T, base.T + 8 * ((xx + yy) % 2)])
    values = values.astype("<f4").reshape(4, 1, 256, 256)
    starts: list[float] = []

    def fetch(_):
        starts.append(time.monotonic())
        return values.copy()

    _, records = srnet_training._learn(
        fetch=fetch,
        batches=lambda _: np.tile(np.array([[0, 1, 2, 3]], dtype="<i8"), (UPDATES, 1)),
        epochs=1,
        seed=20261008,
        params=srnet_training.settings({"max_seconds": MAX_SECONDS}),
        target_pairs=2,
        outer_deadline=deadline,
        device="cuda:0",
    )
    deadline()
    if len(starts) != UPDATES or len(records) != 1 or records[0]["updates"] != UPDATES:
        raise ValueError("profile optimizer accounting incomplete")
    # The shared engine synchronizes each step through finite-gradient/loss and
    # CPU numeric-state verification before fetching the next input. Intervals
    # therefore include that production overhead, not just queued GPU kernels.
    intervals = np.diff(starts)[DISCARD:].tolist()
    return {
        "execution": execution,
        "input": "generated-wave-checker-4x1x256x256-f32-v1",
        "generated_optimizer_updates": UPDATES,
        "discarded_initial_intervals": DISCARD,
        "steady_intervals_seconds": intervals,
        "fixed_fit_budget": budget(intervals),
        "process_peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "process_peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "seconds": time.monotonic() - started,
        "real_data_used": False,
        "real_model_trained": False,
        "accuracy_qualification": "unavailable",
        "deployed": False,
    }
