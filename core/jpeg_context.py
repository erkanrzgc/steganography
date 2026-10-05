"""Versioned compression/content interactions from measured JPEG DCT summaries."""

from __future__ import annotations

import numpy as np

from core import jpeg_features as jpeg

FEATURE_VERSION = "jpeg-context-summary-v1"
FEATURE_NAMES = (
    jpeg.FEATURE_NAMES
    + tuple(f"context_magnitude_q_{u}_{v}" for u, v in jpeg.AC)
    + tuple(f"context_entropy_q_{u}_{v}" for u, v in jpeg.AC)
    + (
        "context_quantization_strength",
        "context_ac_nonzero",
        "context_ac_entropy",
        "context_ac_magnitude",
    )
)


def summary_context_features(summary: list[float]) -> list[float]:
    """No filenames, labels, source IDs, quality-factor guesses or reference covers."""
    values = np.asarray(summary, dtype=np.float64)
    if values.shape != (len(jpeg.FEATURE_NAMES),) or (
        not np.isfinite(values).all() or (values < 0).any() or (values > 1).any()
    ):
        raise ValueError("invalid JPEG context input summary")
    histogram = values[: len(jpeg.AC) * 8].reshape(-1, 8)
    if not np.allclose(histogram.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("JPEG context magnitude histograms must sum to one")
    quantization = values[-64:].reshape(8, 8) * 65535
    if (quantization < 1 - 1e-6).any():
        raise ValueError("JPEG context requires positive measured quantization")
    # Bounded monotonic physical scale, not a camera/software or JPEG QF estimate.
    strength = quantization / (quantization + 32)
    ac_strength = np.asarray([strength[u, v] for u, v in jpeg.AC])
    magnitude = histogram @ (np.arange(8) / 7)
    entropy = -np.sum(histogram * np.log2(np.maximum(histogram, 1e-300)), axis=1) / 3
    context = np.concatenate(
        (
            magnitude * ac_strength,
            entropy * (1 - ac_strength),
            [strength.mean(), np.mean(1 - histogram[:, 0]), entropy.mean(), magnitude.mean()],
        )
    )
    # New contract only: stable, bounded compact cache representation.
    return np.round(np.concatenate((values, context)), 8).tolist()


def jpeg_context_features(data: bytes) -> list[float]:
    return summary_context_features(jpeg.jpeg_features(data))
