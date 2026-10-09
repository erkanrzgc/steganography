"""Generated exact epoch-resume parity and timing, never detection accuracy."""

import time
from typing import Any

import numpy as np

from core import srnet_checkpoint, srnet_cuda, srnet_training
from core.srnet_stream import deadline_after


def probe(folder, *, binding, device):
    """Caller provides hard OS limits; GPU execution never falls back to CPU."""
    execution = srnet_cuda.inspect() if device == "cuda:0" else {"device": "cpu"}
    deadline = deadline_after(180)
    started = time.monotonic()
    yy, xx = np.indices((256, 256))
    base = (128 + 24 * np.sin(xx / 11) + 16 * np.cos(yy / 13)).astype("<f4")
    values = np.stack([base, base + 8 * ((xx + yy) % 2), base.T, base.T + 8 * ((xx + yy) % 2)])
    values = values.astype("<f4").reshape(4, 1, 256, 256)
    starts = []

    def fetch(_):
        starts.append(time.monotonic())
        return values.copy()

    def learn(stop, resume=None):
        outputs: list[dict[str, Any]] = []
        _, records = srnet_training._learn(
            fetch=fetch,
            batches=lambda _: np.tile(np.array([[0, 1, 2, 3]], dtype="<i8"), (32, 1)),
            epochs=2,
            seed=20261008,
            params=srnet_training.settings({"threads": 2, "max_seconds": 180}),
            target_pairs=2,
            device=device,
            outer_deadline=deadline,
            segment={
                "binding": binding,
                "stop_epoch": stop,
                "resume": resume,
                "sink": outputs.append,
            },
        )
        return outputs[0], records

    complete, full_records = learn(2)
    intervals = np.diff(starts)[16:].tolist()
    first, _ = learn(1)
    checksum = srnet_checkpoint.save(first, folder / "generated-first.npz")
    loaded = srnet_checkpoint.load(folder / "generated-first.npz", checksum=checksum)
    resumed, records = learn(2, loaded)
    if (
        full_records != records
        or complete["metadata"] != resumed["metadata"]
        or any(not np.array_equal(v, resumed["arrays"][k]) for k, v in complete["arrays"].items())
    ):
        raise ValueError("generated whole-epoch resume is not exact")
    deadline()
    if len(intervals) != 47 or not np.isfinite(intervals).all() or min(intervals) <= 0:
        raise ValueError("generated timing accounting invalid")
    report = {
        "execution": execution,
        "generated_optimizer_updates": 128,
        "uninterrupted_updates": 64,
        "exact_model_bn_optimizer_rng_loss": True,
        "steady_intervals_seconds": intervals,
        "steady_interval_p95_seconds": float(np.quantile(intervals, 0.95)),
        "seconds": time.monotonic() - started,
        "real_data_used": False,
        "real_model_trained": False,
        "accuracy_qualification": "unavailable",
        "deployed": False,
    }
    if device == "cuda:0":
        import torch

        report.update(
            process_peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(0),
            process_peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(0),
        )
    return report
