"""Symmetrised SRM-style co-occurrence features for gray images (CPU, numpy only).

A compact subset of the spatial rich model (Fridrich & Kodovsky 2012):

- directional first/second/third-order residuals at several quantisation steps,
  scanned both parallel and perpendicular to the residual direction;
- min/max of first- and second-order residuals over two or four directions;
- 3x3 (KB), 5x5 (KV) and 3x3 edge (E3) kernel residuals.

Every block is a fourth-order co-occurrence of residuals clipped to T = 2. Sign
and reversal symmetry merge the 625 patterns into 169 classes, and horizontal
and vertical scans are pooled, so the features are invariant to transposition.

``weighted_gray_features`` is the selection-channel-aware variant (in the spirit
of maxSRM, Denemark et al. 2014): each co-occurrence adds the largest change
probability of the four pixels its residuals are located at, instead of one.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from research import rich_color_detector as rich

T = 2
ORDER = 4
MIN_SIDE = 16
MAX_PIXELS = 16_777_216
DIRECTIONAL = (("r1", (1, 2)), ("r2", (1, 2, 3)), ("r3", (1, 3, 5)))
# Integer kernels; the residual is the exact integer correlation divided by the
# divisor, so rounding is identical for an image and its transpose.
KB = (np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]), 4)
KV = (np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]), 12)
E3 = (np.array([[-1, 2, -1], [2, -4, 2], [0, 0, 0]]), 4)


@cache
def symmetry_classes(t: int = T, k: int = ORDER) -> np.ndarray:
    """Class index of each of the (2t+1)^k patterns under sign and reversal symmetry."""
    base = 2 * t + 1
    codes = np.arange(base**k)
    digits = np.stack([(codes // base ** (k - 1 - j)) % base - t for j in range(k)], 1)

    def encode(d: np.ndarray) -> np.ndarray:
        return np.sum([(d[:, j] + t) * base ** (k - 1 - j) for j in range(k)], axis=0)

    canon = np.min([encode(digits), encode(-digits), encode(digits[:, ::-1]),
                    encode(-digits[:, ::-1])], axis=0)
    classes: np.ndarray = np.unique(canon, return_inverse=True)[1]
    classes.flags.writeable = False
    return classes


BLOCK_DIMS = int(symmetry_classes().max()) + 1


def cooccurrence(residual: np.ndarray, q: float, weight: np.ndarray | None = None) -> np.ndarray:
    """Symmetrised counts of ``ORDER`` consecutive quantised residuals along rows.

    With ``weight`` (one value per residual), each window adds the largest weight
    of its ``ORDER`` residuals instead of one.
    """
    base = 2 * T + 1
    r = np.clip(np.round(residual / q), -T, T).astype(np.int64) + T
    n = r.shape[1] - ORDER + 1
    code = r[:, :n] * base ** (ORDER - 1)
    for j in range(1, ORDER):  # in-place accumulation; stacking the slices is ~4x slower
        code += r[:, j : j + n] * base ** (ORDER - 1 - j)
    if weight is None:
        counts = np.bincount(code.ravel(), minlength=base**ORDER)
    else:
        if weight.shape != residual.shape:
            raise ValueError("one weight per residual required")
        window = weight[:, :n].copy()
        for j in range(1, ORDER):
            np.maximum(window, weight[:, j : j + n], out=window)
        counts = np.bincount(code.ravel(), weights=window.ravel(), minlength=base**ORDER)
    return np.bincount(symmetry_classes(), weights=counts, minlength=BLOCK_DIMS)


def _correlate(x: np.ndarray, spec: tuple[np.ndarray, int]) -> np.ndarray:
    kernel, divisor = spec
    kh, kw = kernel.shape
    h, w = x.shape[0] - kh + 1, x.shape[1] - kw + 1
    out = np.zeros((h, w))
    for i in range(kh):
        for j in range(kw):
            if kernel[i, j]:
                out += kernel[i, j] * x[i : i + h, j : j + w]
    return out / divisor


def _directional(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (z[:, 1:] - z[:, :-1],
            z[:, :-2] - 2 * z[:, 1:-1] + z[:, 2:],
            -z[:, :-3] + 3 * z[:, 1:-2] - 3 * z[:, 2:-1] + z[:, 3:])


def _locations(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weights at the pixel each ``_directional`` residual is located at (the predicted one)."""
    return w[:, :-1], w[:, 1:-1], w[:, 1:-2]


def _both(z: np.ndarray, q: float, w: np.ndarray | None) -> np.ndarray:
    return cooccurrence(z, q, w) + cooccurrence(z.T, q, None if w is None else w.T)


def _blocks(x: np.ndarray, w: np.ndarray | None = None) -> dict[str, np.ndarray]:
    h, v = _directional(x), _directional(x.T)
    hw: tuple[np.ndarray | None, ...] = (None, None, None)
    vw: tuple[np.ndarray | None, ...] = (None, None, None)
    if w is not None:
        hw, vw = _locations(w), _locations(w.T)

    def t(a: np.ndarray | None) -> np.ndarray | None:
        return None if a is None else a.T

    out = {}
    for i, (name, steps) in enumerate(DIRECTIONAL):
        for q in steps:
            out[f"{name}q{q}par"] = cooccurrence(h[i], q, hw[i]) + cooccurrence(v[i], q, vw[i])
            out[f"{name}q{q}perp"] = (cooccurrence(h[i].T, q, t(hw[i]))
                                      + cooccurrence(v[i].T, q, t(vw[i])))
    centre = w2 = centre5 = None
    if w is not None:  # residual locations: (i, j) below, (i, j+1) and (i+1, j) for order two
        w2 = np.maximum(w[:-2, 1:-1], w[1:-1, :-2])
        centre, centre5 = w[1:-1, 1:-1], w[2:-2, 2:-2]
    w1 = None if w is None else w[:-1, :-1]
    right, down = h[0][:-1, :], v[0].T[:, :-1]
    out["min1"] = _both(np.minimum(right, down), 1, w1)
    out["max1"] = _both(np.maximum(right, down), 1, w1)
    right2, down2 = h[1][:-2, :], v[1].T[:, :-2]
    out["min2"] = _both(np.minimum(right2, down2), 2, w2)
    out["max2"] = _both(np.maximum(right2, down2), 2, w2)
    c = x[1:-1, 1:-1]
    around = [x[1:-1, 2:] - c, x[1:-1, :-2] - c, x[2:, 1:-1] - c, x[:-2, 1:-1] - c]
    out["min4"] = _both(np.min(around, 0), 1, centre)
    out["max4"] = _both(np.max(around, 0), 1, centre)
    kb, kv = _correlate(x, KB), _correlate(x, KV)
    flipped = (E3[0][::-1], E3[1])
    edges = ((_correlate(x, E3), centre), (_correlate(x, flipped), centre),
             (_correlate(x.T, E3), t(centre)), (_correlate(x.T, flipped), t(centre)))
    for q in (1, 2):
        out[f"kbq{q}"] = _both(kb, q, centre)
        out[f"kvq{q}"] = _both(kv, q, centre5)
        out[f"e3q{q}"] = np.sum([_both(e, q, ew) for e, ew in edges], axis=0)
    return out


BLOCKS = tuple(sorted(_blocks(np.zeros((MIN_SIDE, MIN_SIDE))).keys()))
DIMS = len(BLOCKS) * BLOCK_DIMS
COLOR_DIMS = rich.DIMS - rich.GRAY_DIMS


def gray_features(image: np.ndarray) -> np.ndarray:
    """Normalised SRM-style features of a uint8 HxW or HxWx1 gray image (``DIMS`` floats)."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("uint8 gray image required")
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim != 2 or min(image.shape) < MIN_SIDE or image.size > MAX_PIXELS:
        raise ValueError("HxW or HxWx1 gray image with bounded size required")
    blocks = _blocks(image.astype(np.float64))
    return np.concatenate([blocks[b] / blocks[b].sum() for b in BLOCKS]).astype(np.float32)


def weighted_gray_features(image: np.ndarray, change: np.ndarray) -> np.ndarray:
    """Selection-channel-aware features from a per-pixel change-probability map.

    ``change`` has the image's HxW shape with finite values in [0, 1]; each block
    is normalised by its total weight (an all-zero block stays zero).
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("uint8 gray image required")
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    if image.ndim != 2 or min(image.shape) < MIN_SIDE or image.size > MAX_PIXELS:
        raise ValueError("HxW or HxWx1 gray image with bounded size required")
    change = np.asarray(change, dtype=np.float64)
    if change.shape != image.shape or not np.all(np.isfinite(change)):
        raise ValueError("finite change map with the image shape required")
    if change.min() < 0 or change.max() > 1:
        raise ValueError("change probabilities must lie in [0, 1]")
    blocks = _blocks(image.astype(np.float64), change)
    parts = [blocks[b] / max(blocks[b].sum(), 1e-300) for b in BLOCKS]
    return np.concatenate(parts).astype(np.float32)


def color_features(image: np.ndarray) -> np.ndarray:
    """SRM-style blocks pooled over R, G and B, plus the rich-model colour co-occurrences.

    Returns ``DIMS + COLOR_DIMS`` floats for a uint8 HxWx3 image.
    """
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3:
        raise ValueError("uint8 HxWx3 image required")
    if image.shape[2] != 3 or min(image.shape[:2]) < MIN_SIDE or image.size > 3 * MAX_PIXELS:
        raise ValueError("RGB image with bounded size required")
    pooled = [_blocks(image[:, :, c].astype(np.float64)) for c in range(3)]
    gray = [np.sum([p[b] for p in pooled], axis=0) for b in BLOCKS]
    within = np.concatenate([g / g.sum() for g in gray])
    return np.concatenate([within, rich.features(image)[rich.GRAY_DIMS :]]).astype(np.float32)

