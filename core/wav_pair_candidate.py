"""Untrained PCM pair-energy candidate; no fitting, file IO or deployment hook."""

from __future__ import annotations

import numpy as np

from core.nonjpeg_features import pcm16_samples

DOMAIN = "audio-pcm16-pair-energy-mlp-draft-v1"
NAMES = tuple(
    f"{window}.pair_equalization.p{plane}"
    for window in ("global", "q0", "q1", "q2", "q3")
    for plane in (0, 1)
)


def features(data: bytes) -> tuple[float, ...]:
    """Ten [0,1] statistics, retaining the full native signed-PCM16 alphabet.

    Pair bins differ only in the selected bit. For counts a,b, return
    1 - sum((a-b)^2/(a+b))/N, ignoring empty pairs. No clipping of sample
    amplitudes, resampling, padding or fitted normalization is performed.
    """
    samples = pcm16_samples(data)
    count = len(samples)
    bounds = [count * i // 4 for i in range(5)]
    windows = [(0, count), *zip(bounds[:-1], bounds[1:], strict=True)]
    result = []
    for start, end in windows:
        histogram = np.bincount(samples[start:end] + 32768, minlength=65536).astype(
            np.float64
        )
        for plane in (0, 1):
            width = 1 << plane
            pairs = histogram.reshape(-1, 2 * width)
            left, right = pairs[:, :width], pairs[:, width:]
            total = left + right
            energy = np.divide(
                (left - right) ** 2,
                total,
                out=np.zeros_like(total),
                where=total > 0,
            ).sum() / (end - start)
            # Only floating-point round-off can leave the mathematical range.
            result.append(float(np.clip(1 - energy, 0, 1)))
    return tuple(result)


def inputs(data: bytes) -> np.ndarray:
    """Explicit FP32 input for the untrained head; no metadata enters the vector."""
    return np.asarray(features(data), dtype=np.float32)[None, :]


def network():
    """Optional 418-parameter head; constructing it never fits or fetches weights."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PCM pair candidate unavailable: optional Torch missing") from exc

    class PairMLP(torch.nn.Sequential):
        def __init__(self):
            super().__init__(
                torch.nn.Linear(len(NAMES), 32, dtype=torch.float32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 2, dtype=torch.float32),
            )
            self.architecture = DOMAIN

    return PairMLP()
