"""Weighted Stego (WS) payload estimation for LSB replacement in lossless images.

Implements the local-predictor WS estimator of Fridrich and Goljan (2004) with
the local-variance weights of Ker and Böhme (2008). LSB replacement moves even
samples only upward and odd samples only downward, so the correlation between
a sample's parity and its prediction residual estimates the replaced fraction
independently of image content. A heavily textured or noisy cover makes the
predictor unreliable; such estimates are reported but never scored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WEIGHT_OFFSET = 5.0
MAX_PREDICTION_RMS = 24.0
BLOCK_ROWS = 256
SCORE_FLOOR = 0.10
SCORE_SPAN = 0.25
MAX_SCORE = 70


@dataclass(frozen=True)
class WsEstimate:
    """Estimated fraction of samples carrying replaced LSBs (about twice the changed fraction)."""

    payload: float
    samples: int
    prediction_rms: float
    reliable: bool


def _channels(pixels: object) -> np.ndarray:
    if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8:
        raise ValueError("uint8 sample array required")
    if pixels.ndim == 2:
        pixels = pixels[:, :, None]
    if pixels.ndim != 3 or not 1 <= pixels.shape[2] <= 4:
        raise ValueError("2-D or HxWxC (C<=4) sample array required")
    if pixels.shape[0] < 3 or pixels.shape[1] < 3:
        raise ValueError("at least 3x3 samples required")
    return pixels


def _channel_sums(channel: np.ndarray) -> tuple[float, float, float, int]:
    rows = channel.shape[0]
    numerator = denominator = squared = 0.0
    count = 0
    for start in range(1, rows - 1, BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, rows - 1)
        slab = channel[start - 1 : stop + 1].astype(np.float64)
        centre = slab[1:-1, 1:-1]
        neighbours = np.stack(
            (slab[:-2, 1:-1], slab[2:, 1:-1], slab[1:-1, :-2], slab[1:-1, 2:])
        )
        residual = centre - neighbours.mean(axis=0)
        weight = 1.0 / (WEIGHT_OFFSET + neighbours.var(axis=0))
        sign = np.where(channel[start:stop, 1:-1] & 1, 1.0, -1.0)
        numerator += float(np.sum(weight * sign * residual))
        denominator += float(np.sum(weight))
        squared += float(np.sum(residual * residual))
        count += residual.size
    return numerator, denominator, squared, count


def weighted_stego(pixels: np.ndarray) -> WsEstimate:
    """Pool the WS estimate over every channel's interior samples."""
    samples = _channels(pixels)
    numerator = denominator = squared = 0.0
    count = 0
    for index in range(samples.shape[2]):
        n, d, s, c = _channel_sums(samples[:, :, index])
        numerator, denominator, squared, count = (
            numerator + n,
            denominator + d,
            squared + s,
            count + c,
        )
    rms = float(np.sqrt(squared / count))
    return WsEstimate(
        payload=float(2.0 * numerator / denominator),
        samples=count,
        prediction_rms=rms,
        reliable=rms <= MAX_PREDICTION_RMS,
    )


def score(estimate: WsEstimate) -> int:
    """Map a reliable estimate onto the analyzer's 0..70 heuristic scale."""
    if not estimate.reliable:
        return 0
    fraction = (estimate.payload - SCORE_FLOOR) / SCORE_SPAN
    return int(round(min(1.0, max(0.0, fraction)) * MAX_SCORE))
