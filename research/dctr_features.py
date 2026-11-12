"""DCTR JPEG steganalysis features (Holub & Fridrich 2015), luminance only, numpy.

The luminance is decompressed without rounding, filtered with the 64 8x8 DCT
basis patterns, quantised as min(round(|u| / q), T) and histogrammed per basis
pattern and per 8x8 phase, with phases merged by symmetry into 25 groups:
64 x 25 x (T + 1) = 8,000 features.
"""

from __future__ import annotations

from pathlib import Path

import jpeglib
import numpy as np

T = 4
GROUPS = 25
DIMS = 64 * GROUPS * (T + 1)
MAX_BLOCKS = 262_144  # 4096 x 4096 pixels
_U = np.arange(8)
BASIS = np.array([[np.sqrt((1 if u == 0 else 2) / 8) * np.cos(np.pi * u * (2 * x + 1) / 16)
                   for x in range(8)] for u in range(8)])
FOLD = np.minimum(_U, 8 - _U) % 8
STANDARD_LUMA = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])


def ijg_table(quality: int) -> np.ndarray:
    """libjpeg (IJG) luminance table for ``quality`` in 1..100."""
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError("quality 1..100 required")
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    return np.clip((STANDARD_LUMA * scale + 50) // 100, 1, 255)


def quality_from_table(table: np.ndarray) -> int:
    """Closest IJG quality for an 8x8 luminance quantisation table."""
    table = np.asarray(table)
    if table.shape != (8, 8):
        raise ValueError("8x8 quantisation table required")
    errors = [np.abs(ijg_table(q) - table).sum() for q in range(1, 101)]
    return int(np.argmin(errors)) + 1


def step(quality: int) -> float:
    """DCTR quantisation step: 8 (2 - QF / 50) above QF 50, continued as 8 * 50 / QF below."""
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError("quality 1..100 required")
    return 8 * (2 - quality / 50) if quality >= 50 else 8 * 50 / quality


def decompress(coefficients: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Non-rounded spatial plane (minus 128) from (rows, cols, 8, 8) DCT blocks."""
    if coefficients.ndim != 4 or coefficients.shape[2:] != (8, 8):
        raise ValueError("(rows, cols, 8, 8) coefficient blocks required")
    blocks = coefficients.astype(np.float64) * table
    pixels = np.einsum("ui,hwuv,vj->hwij", BASIS, blocks, BASIS)
    h, w = coefficients.shape[:2]
    return pixels.transpose(0, 2, 1, 3).reshape(8 * h, 8 * w)


def _taps(weights: np.ndarray, shifted: list[np.ndarray]) -> np.ndarray:
    """Weighted sum of shifted views, accumulated in place (stacking is ~5x slower)."""
    out = weights[0] * shifted[0]
    for weight, view in zip(weights[1:], shifted[1:], strict=True):
        out += weight * view
    return out


def from_coefficients(coefficients: np.ndarray, table: np.ndarray) -> np.ndarray:
    """DCTR features of luminance DCT blocks quantised with ``table``."""
    if coefficients.ndim != 4 or coefficients.shape[0] * coefficients.shape[1] > MAX_BLOCKS:
        raise ValueError("bounded (rows, cols, 8, 8) coefficient blocks required")
    if min(coefficients.shape[:2]) < 2:
        raise ValueError("at least 2x2 blocks required")
    x = decompress(coefficients, table)
    q = step(quality_from_table(table))
    h, w = x.shape
    rows = np.stack([_taps(BASIS[n], [x[:, t : w - 7 + t] for t in range(8)])
                     for n in range(8)])
    group = FOLD[np.arange(h - 7) % 8][:, None] * 5 + FOLD[np.arange(w - 7) % 8][None, :]
    out = np.zeros((8, 8, GROUPS, T + 1))
    for k in range(8):
        filtered = _taps(BASIS[k], [rows[:, t : h - 7 + t, :] for t in range(8)])
        values = np.minimum(np.round(np.abs(filtered) / q), T).astype(np.int64)
        for n in range(8):
            out[k, n] = np.bincount((group * (T + 1) + values[n]).ravel(),
                                    minlength=GROUPS * (T + 1)).reshape(GROUPS, T + 1)
    out /= out.sum(-1, keepdims=True)
    return out.ravel().astype(np.float32)


def features(path: Path | str) -> np.ndarray:
    """DCTR features of a JPEG file's luminance component."""
    jpeg = jpeglib.read_dct(str(path))
    return from_coefficients(np.asarray(jpeg.Y), np.asarray(jpeg.qt[jpeg.quant_tbl_no[0]]))


CALIBRATION_CROP = 4


def reference(coefficients: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Calibration reference: decompress, crop 4x4 pixels, recompress with the same table."""
    import tempfile

    from PIL import Image

    plane = np.clip(np.round(decompress(coefficients, table) + 128), 0, 255).astype(np.uint8)
    cropped = Image.fromarray(plane[CALIBRATION_CROP:, CALIBRATION_CROP:])
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "reference.jpg"
        cropped.save(path, format="JPEG", qtables=[np.asarray(table).ravel().tolist()])
        return np.asarray(jpeglib.read_dct(str(path)).Y)


def calibrated_features(path: Path | str) -> np.ndarray:
    """DCTR features followed by their difference from the calibration reference (16,000)."""
    jpeg = jpeglib.read_dct(str(path))
    coefficients = np.asarray(jpeg.Y)
    table = np.asarray(jpeg.qt[jpeg.quant_tbl_no[0]])
    own = from_coefficients(coefficients, table)
    return np.concatenate([own, own - from_coefficients(reference(coefficients, table), table)])
