"""Independent bounded Gibbs-entropy refinement, not a payload extractor."""

from __future__ import annotations

import numpy as np


def refine(costs, target):
    if (not isinstance(costs, tuple) or len(costs) != 2 or type(target) is not int
        or not 1 <= target <= 512 * 512):
        raise ValueError("study entropy target/cost container invalid")
    first, second = costs
    if any(not isinstance(c, np.ndarray) or c.ndim != 2 or not 1 <= c.size <= 512**2
           or c.dtype != np.float64 or not np.isfinite(c).all()
           or np.any(c < 0) or np.any(c > 1e14) for c in costs) or first.shape != second.shape:
        raise ValueError("study entropy costs outside numeric/shape bounds")

    def evaluate(value):
        positive, negative = np.exp(-value * first), np.exp(-value * second)
        denominator = 1 + positive + negative
        probabilities = (positive / denominator, negative / denominator, 1 / denominator)
        entropy = sum(float(np.sum(-p[p > 0] * np.log2(p[p > 0]))) for p in probabilities)
        return probabilities[:2], entropy

    low, high = 0.0, 1.0
    if evaluate(low)[1] < target:
        raise ValueError("study entropy target exceeds capacity")
    for _ in range(64):
        if evaluate(high)[1] <= target:
            break
        high *= 2
    else:
        raise ValueError("study entropy target cannot be bracketed")
    for _ in range(80):
        middle = (low + high) / 2
        probabilities, entropy = evaluate(middle)
        if abs(entropy - target) <= target * 1e-6:
            return probabilities
        if entropy > target:
            low = middle
        else:
            high = middle
    raise ValueError("study entropy refinement did not converge")
