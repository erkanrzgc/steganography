"""Weighted Stego (WS) LSB-replacement payload estimation."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from modules import lsb_replacement as ws
from modules.image_bitplane import ImageBitplaneAnalyzer


def smooth_cover(shape: tuple[int, ...], seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows, cols = shape[:2]
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    base = 128 + 60 * np.sin(x / 23.0) * np.cos(y / 31.0) + 0.1 * (x - y)
    if len(shape) == 3:
        base = base[:, :, None] + np.arange(shape[2])[None, None, :] * 7.0
    noisy = base + rng.normal(0, 2.0, base.shape)
    return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)


def replace_lsb(cover: np.ndarray, rate: float, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flat = cover.reshape(-1).copy()
    chosen = rng.permutation(flat.size)[: int(round(flat.size * rate))]
    flat[chosen] = (flat[chosen] & 254) | rng.integers(0, 2, chosen.size, dtype=np.uint8)
    return flat.reshape(cover.shape)


def reference_payload(pixels: np.ndarray) -> float:
    """Independent unblocked WS formula used only to check the bounded implementation."""
    s = pixels.astype(np.float64)
    if s.ndim == 2:
        s = s[:, :, None]
    c = s[1:-1, 1:-1]
    up, down, left, right = s[:-2, 1:-1], s[2:, 1:-1], s[1:-1, :-2], s[1:-1, 2:]
    neighbours = np.stack([up, down, left, right])
    prediction = neighbours.mean(axis=0)
    weight = 1.0 / (5.0 + neighbours.var(axis=0))
    flipped_sign = np.where(c.astype(np.int64) % 2 == 1, 1.0, -1.0)
    return float(2.0 * np.sum(weight * flipped_sign * (c - prediction)) / np.sum(weight))


def test_clean_smooth_cover_estimates_near_zero_and_is_reliable():
    estimate = ws.weighted_stego(smooth_cover((96, 96, 3)))
    assert estimate.reliable
    assert abs(estimate.payload) < 0.08


def test_forty_percent_replacement_is_estimated_near_its_rate():
    estimate = ws.weighted_stego(replace_lsb(smooth_cover((96, 96, 3)), 0.4))
    assert estimate.reliable
    assert 0.30 <= estimate.payload <= 0.50


def test_estimates_increase_with_replacement_rate():
    cover = smooth_cover((80, 80, 3))
    values = [ws.weighted_stego(replace_lsb(cover, rate)).payload for rate in (0.0, 0.1, 0.4, 1.0)]
    assert values == sorted(values)
    assert values[-1] > 0.85


def test_matches_independent_unblocked_formula_across_blocks():
    image = replace_lsb(smooth_cover((ws.BLOCK_ROWS * 2 + 37, 41, 3)), 0.25)
    assert ws.weighted_stego(image).payload == pytest.approx(reference_payload(image), abs=1e-9)


def test_grayscale_and_sample_count():
    gray = smooth_cover((50, 60))
    rgb = smooth_cover((50, 60, 3))
    assert ws.weighted_stego(gray).samples == 48 * 58
    assert ws.weighted_stego(rgb).samples == 48 * 58 * 3
    assert ws.weighted_stego(gray).payload == pytest.approx(reference_payload(gray), abs=1e-9)


def test_unpredictable_noise_is_marked_unreliable():
    noise = np.random.default_rng(505).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    estimate = ws.weighted_stego(noise)
    assert not estimate.reliable
    assert estimate.prediction_rms > ws.MAX_PREDICTION_RMS


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((10, 10), dtype=np.uint16),
        np.zeros((2, 10), dtype=np.uint8),
        np.zeros((10, 2, 3), dtype=np.uint8),
        np.zeros((10,), dtype=np.uint8),
        np.zeros((10, 10, 5), dtype=np.uint8),
        [[1, 2, 3]] * 3,
    ],
)
def test_rejects_invalid_inputs(value):
    with pytest.raises(ValueError):
        ws.weighted_stego(value)


def test_estimate_is_immutable():
    estimate = ws.weighted_stego(smooth_cover((20, 20)))
    with pytest.raises(FrozenInstanceError):
        estimate.payload = 1.0  # type: ignore[misc]


def test_score_mapping_bounds():
    assert ws.score(ws.WsEstimate(0.0, 100, 2.0, True)) == 0
    assert ws.score(ws.WsEstimate(0.45, 100, 2.0, True)) == 70
    assert ws.score(ws.WsEstimate(0.9, 100, 99.0, False)) == 0
    assert 0 < ws.score(ws.WsEstimate(0.2, 100, 2.0, True)) < 45


def _signal(tmp_path: Path, rgb: np.ndarray):
    path = tmp_path / "image.png"
    Image.fromarray(rgb).save(path)
    result = ImageBitplaneAnalyzer().analyze(path)
    (signal,) = [s for s in result.signals if s.name == "weighted_stego_payload"]
    return signal, result


def test_analyzer_flags_replacement_as_heuristic_only(tmp_path: Path):
    signal, result = _signal(tmp_path, replace_lsb(smooth_cover((96, 96, 3)), 0.4))
    assert signal.evidence == "heuristic"
    assert signal.score >= 45
    assert signal.category == "image_weighted_stego"
    assert result.suspicion >= signal.score


def test_analyzer_keeps_clean_cover_and_noise_quiet(tmp_path: Path):
    clean, _ = _signal(tmp_path, smooth_cover((96, 96, 3)))
    assert clean.evidence == "informational" and clean.score < 45
    noise = np.random.default_rng(3).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    noisy, _ = _signal(tmp_path, noise)
    assert noisy.evidence == "informational" and noisy.score == 0
    assert "unreliable" in noisy.detail
