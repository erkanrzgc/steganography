"""Experimental block-DCT residual/parity descriptor; not DCTR, JRM or SRNet."""

from __future__ import annotations

import numpy as np

from core import jpeg_features as jpeg
from core.jpeg_context import FEATURE_NAMES as CONTEXT_NAMES
from core.jpeg_context import summary_context_features

FEATURE_VERSION = "jpeg-dct-residual-parity-v1"
MODES = tuple(
    sorted(((u, v) for u in range(8) for v in range(8)), key=lambda mode: (sum(mode), *mode))[:32]
)
FEATURE_NAMES = (
    CONTEXT_NAMES
    + tuple(
        f"dct_{u}_{v}_{direction}_residual_{b}"
        for u, v in MODES
        for direction in ("h", "v")
        for b in range(-2, 3)
    )
    + tuple(
        f"dct_{u}_{v}_{direction}_parity_{p}_residual_{b}"
        for u, v in MODES
        for direction in ("h", "v")
        for p in range(2)
        for b in range(-2, 3)
    )
    + tuple(f"dc_abs_{b}" for b in range(8))
)


def coefficient_features(coefficients: np.ndarray, quantization: np.ndarray) -> list[float]:
    # Reuse the signed-16-bit/dimension/budget guards before subtraction or parity.
    prefix = summary_context_features(jpeg.coefficient_features(coefficients, quantization))
    values = coefficients.astype(np.int64)
    residuals = []
    joints = []
    for u, v in MODES:
        plane = values[:, :, u, v]
        for first, second in ((plane[:, :-1], plane[:, 1:]), (plane[:-1, :], plane[1:, :])):
            residual = np.clip(second - first, -2, 2) + 2
            counts = np.bincount(residual.ravel(), minlength=5)
            residuals.extend((counts / counts.sum()).tolist())
            counts = np.bincount(((first & 1) * 5 + residual).ravel(), minlength=10)
            joints.extend((counts / counts.sum()).tolist())
    counts = np.bincount(np.minimum(np.abs(values[:, :, 0, 0]), 7).ravel(), minlength=8)
    return np.round([*prefix, *residuals, *joints, *(counts / counts.sum())], 8).tolist()


def jpeg_residual_features(data: bytes) -> list[float]:
    return jpeg.run_feature_worker(data, "core.jpeg_residual", FEATURE_NAMES)


if __name__ == "__main__":
    raise SystemExit(jpeg.main(coefficient_features))
