"""16-bit PCM LSB steganalysis features (CPU, numpy only).

- Symmetrised co-occurrences of first- to fourth-order sample differences,
  computed on the whole signal and on its quietest quarter of 1,024-sample
  frames. Embedding changes are most visible where the signal is quietest.
- Small-amplitude histograms of the quiet frames, and counts of isolated
  values inside exact digital silence (``0, v, 0`` windows).
- Weighted-stego payload estimates (Ker & Bohme weights) with three symmetric
  polynomial predictors. These estimate the fraction of replaced LSBs even in
  noisy recordings.

``embed`` simulates LSB replacement and +/-1 matching at a given rate.
"""

from __future__ import annotations

from functools import cache

import numpy as np

FRAME = 1024
MIN_SAMPLES = FRAME
MAX_SAMPLES = 50_000_000
KINDS = ("replacement", "matching")
RESIDUALS = ((1, (1, 2)), (2, (1, 2, 4)), (3, (1, 3, 6)), (4, (2, 6)))
PREDICTORS = (
    (np.array([1, 1]) / 2, 1),
    (np.array([-1, 4, 4, -1]) / 6, 2),
    (np.array([1, -6, 15, 15, -6, 1]) / 20, 3),
)


def embed(samples: np.ndarray, kind: str, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Embed ``rate * N`` random bits at random positions of an int16 signal."""
    if not isinstance(samples, np.ndarray) or samples.dtype != np.int16 or samples.ndim != 1:
        raise ValueError("1-D int16 signal required")
    if kind not in KINDS or not 0 < rate <= 1:
        raise ValueError("known kind and rate in (0, 1] required")
    y = samples.astype(np.int32)
    n = int(round(rate * len(y)))
    pos = rng.choice(len(y), n, replace=False)
    bits = rng.integers(0, 2, n)
    if kind == "replacement":
        y[pos] = (y[pos] & ~1) | bits
    else:
        step = rng.choice(np.array([-1, 1]), n)
        step = np.where(y[pos] >= 32767, -1, np.where(y[pos] <= -32768, 1, step))
        y[pos] += np.where((y[pos] & 1) != bits, step, 0)
    return y.astype(np.int16)


@cache
def _classes(t: int, k: int) -> np.ndarray:
    base = 2 * t + 1
    codes = np.arange(base**k)
    digits = np.stack([(codes // base ** (k - 1 - j)) % base - t for j in range(k)], 1)

    def encode(d: np.ndarray) -> np.ndarray:
        out = np.zeros(len(d), np.int64)
        for j in range(k):
            out += (d[:, j] + t) * base ** (k - 1 - j)
        return out

    canon = np.min([encode(digits), encode(-digits), encode(digits[:, ::-1]),
                    encode(-digits[:, ::-1])], axis=0)
    classes: np.ndarray = np.unique(canon, return_inverse=True)[1]
    return classes


def _cooccurrence(residual: np.ndarray, q: float, t: int, k: int) -> np.ndarray:
    base = 2 * t + 1
    z = np.clip(np.round(residual / q), -t, t).astype(np.int64) + t
    n = len(z) - k + 1
    code = z[:n] * base ** (k - 1)
    for j in range(1, k):
        code += z[j : j + n] * base ** (k - 1 - j)
    counts = np.bincount(code, minlength=base**k)
    hist = np.bincount(_classes(t, k), weights=counts)
    return hist / max(hist.sum(), 1.0)


def _difference(x: np.ndarray, order: int) -> np.ndarray:
    out = x
    for _ in range(order):
        out = np.diff(out)
    return out


def _residual_block(x: np.ndarray) -> list[np.ndarray]:
    parts = []
    for order, steps in RESIDUALS:
        r = _difference(x, order)
        for q in steps:
            parts += [_cooccurrence(r, q, 3, 3), _cooccurrence(r, q, 2, 4)]
    return parts


def weighted_stego(x: np.ndarray) -> np.ndarray:
    """WS payload estimate and prediction RMS for each predictor (6 values)."""
    out = []
    for kernel, half in PREDICTORS:
        centre = x[half:-half]
        offsets = list(range(-half, 0)) + list(range(1, half + 1))
        neighbours = np.stack([x[half + o : len(x) - half + o] for o in offsets])
        residual = centre - kernel @ neighbours
        weight = 1.0 / (5.0 + neighbours.var(0))
        sign = np.where(centre.astype(np.int64) & 1, 1.0, -1.0)
        out += [2.0 * float(np.sum(weight * sign * residual) / np.sum(weight)),
                float(np.sqrt(np.mean(residual**2)))]
    return np.array(out)


def features(samples: np.ndarray) -> np.ndarray:
    """Feature vector of a 1-D int16 PCM signal."""
    if not isinstance(samples, np.ndarray) or samples.dtype != np.int16 or samples.ndim != 1:
        raise ValueError("1-D int16 signal required")
    if not MIN_SAMPLES <= len(samples) <= MAX_SAMPLES:
        raise ValueError("signal length out of bounds")
    x = samples.astype(np.float64)
    parts = _residual_block(x)
    frames = x[: len(x) // FRAME * FRAME].reshape(-1, FRAME)
    energy = np.abs(np.diff(frames, axis=1)).mean(1)
    quiet = frames[np.argsort(energy, kind="stable")[: max(1, len(frames) // 4)]].ravel()
    parts += _residual_block(quiet)
    v = quiet.astype(np.int64)
    parts.append(np.bincount(np.minimum(np.abs(v), 15), minlength=16) / v.size)
    parts.append(np.bincount(np.clip(v, -8, 8) + 8, minlength=17) / v.size)
    middle, silent = v[1:-1], (v[:-2] == 0) & (v[2:] == 0)
    parts.append(np.array([silent.mean(), ((middle == 0) & silent).mean(),
                           ((np.abs(middle) == 1) & silent).mean(),
                           ((np.abs(middle) > 1) & silent).mean()]))
    parts.append(np.array([energy.min(), np.median(energy)]))
    parts.append(weighted_stego(x))
    parts.append(low_bit_profile(samples))
    return np.concatenate(parts).astype(np.float32)


def low_bit_profile(samples: np.ndarray) -> np.ndarray:
    """Fraction of samples whose lowest k bits are zero, k = 1..8.

    Natural 16-bit recordings give about 2^-k; audio up-scaled from 8 bits, or
    with an unused LSB plane, gives 1. Any LSB embedding breaks that structure.
    """
    v = samples.astype(np.int64)
    return np.array([np.mean(v % (1 << k) == 0) for k in range(1, 9)])


DIMS = int(features(np.zeros(MIN_SAMPLES, np.int16)).size)
