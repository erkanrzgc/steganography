"""CPU Fisher discriminants with signed power maps of cached rich-model features.

This is an explicit research classifier, with no image decoding, GPU use,
automatic model deployment or model loading. Fit statistics come only from
the paired matrices supplied to ``fit_grid``. Threshold calibration is a separate
operation on caller-selected calibration scores.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from research import rich_color_detector as rich

MAX_ROWS = 20_000
MAX_FEATURES = 16_384
MAX_ELEMENTS = 200_000_000
MAX_CALIBRATION_ROWS = 1_000_000


def _matrix(values: np.ndarray) -> None:
    if not isinstance(values, np.ndarray) or values.ndim != 2:
        raise ValueError("two-dimensional numeric feature matrix required")
    if (values.dtype.kind not in "fiu" or values.dtype.itemsize > 8 or len(values) > MAX_ROWS
            or not 1 <= values.shape[1] <= MAX_FEATURES or values.size > MAX_ELEMENTS):
        raise ValueError("feature matrix dtype or bounds invalid")
    for start in range(0, len(values), 256):
        block = values[start:start + 256]
        if (not np.all(np.isfinite(block)) or np.any(block > 1e12)
                or np.any(block < -1e12)):
            raise ValueError("finite features required")


def _power(value: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError("finite power in (0, 1] required")
    return float(value)


def transform(values: np.ndarray, power: float) -> np.ndarray:
    """Apply sign(x)*abs(x)**power without modifying the input matrix."""
    _matrix(values)
    power = _power(power)
    x = values if values.dtype.kind == "f" else values.astype(np.float64)
    if power == 1:
        return x
    return np.copysign(np.power(np.abs(x), power), x)


@dataclass(frozen=True)
class Detector:
    """Power-map Fisher model; scores are not probabilities or primary verdicts."""

    power: float
    fisher: rich.Detector

    def score(self, values: np.ndarray, *, batch_rows: int = 256) -> np.ndarray:
        _matrix(values)
        if values.shape[1] != self.fisher.weights.size:
            raise ValueError("feature matrix width mismatch")
        if type(batch_rows) is not int or not 1 <= batch_rows <= 4096:
            raise ValueError("batch_rows must be an integer from 1 to 4096")
        output = np.empty(len(values), dtype=np.float64)
        for start in range(0, len(values), batch_rows):
            block = values[start:start + batch_rows]
            output[start:start + len(block)] = self.fisher.score(transform(block, self.power))
        return output


def fit_grid(
    covers: np.ndarray,
    stegos: np.ndarray,
    *,
    power: float = 0.5,
    ridges: Sequence[float] = (0.03,),
) -> tuple[Detector, ...]:
    """Fit a predeclared ridge grid, sharing transformed moments and scatter.

    Cover and stego row i must belong to the same source. This function never
    selects hyperparameters or a threshold from evaluation data.
    """
    _matrix(covers)
    _matrix(stegos)
    power = _power(power)
    if covers.shape != stegos.shape or len(covers) < 2:
        raise ValueError("at least two paired feature rows required")
    if not 1 <= len(ridges) <= 16 or any(
        type(ridge) not in (int, float) or not math.isfinite(ridge) or ridge <= 0
        for ridge in ridges
    ):
        raise ValueError("one to sixteen finite positive ridges required")
    return tuple(
        Detector(power, model)
        for model in rich._fits(transform(covers, power), transform(stegos, power), tuple(ridges))
    )


def calibration_threshold(covers: np.ndarray, stegos: np.ndarray) -> float:
    """Choose balanced-accuracy threshold on explicit calibration scores only.

    Score >= threshold means stego. Tied optima choose the threshold closest
    to zero, then the smaller value. This helper does not fit a classifier.
    """
    for values in (covers, stegos):
        if (not isinstance(values, np.ndarray) or values.ndim != 1
                or values.dtype.kind not in "fiu" or not 1 <= len(values) <= MAX_CALIBRATION_ROWS
                or not np.all(np.isfinite(values))):
            raise ValueError("bounded finite one-dimensional calibration scores required")
    candidates = np.unique(np.concatenate([covers, stegos, [0.0]]))
    true_negative = np.searchsorted(np.sort(covers), candidates, side="left")
    true_positive = len(stegos) - np.searchsorted(np.sort(stegos), candidates, side="left")
    # Common-denominator integer counts preserve exact BA ties, including
    # unequal class sizes. The row bounds keep each product below 10**12.
    numerator = true_negative * len(stegos) + true_positive * len(covers)
    best = candidates[numerator == numerator.max()]
    return float(best[np.lexsort((best, np.abs(best)))[0]])
