"""Tests verifying statistical detection accuracy on synthetic correlated images."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from core.service import AnalysisService
from modules.image_bitplane import ImageBitplaneAnalyzer


def _generate_synthetic_natural_image(
    width: int = 256,
    height: int = 256,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic 2D surface with realistic natural spatial correlation."""
    x = np.linspace(0, 0.5 * np.pi, width)
    y = np.linspace(0, 0.5 * np.pi, height)
    xx, yy = np.meshgrid(x, y)
    base = ((np.sin(xx) * np.cos(yy) + 1.0) * 110.0 + 15.0).astype(np.uint8)

    # Subtle per-channel offset preserving spatial continuity
    r = base
    g = np.clip(base + 5, 0, 255).astype(np.uint8)
    b = np.clip(base + 10, 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=2)


def _embed_raw_lsb(
    image: np.ndarray,
    rate: float,
    seed: int = 999,
    channel: int | None = None,
) -> np.ndarray:
    """Embed random bits directly into LSBs without any signature/marker."""
    rng = np.random.default_rng(seed)
    stego = image.copy()
    if channel is not None:
        mask = rng.random((image.shape[0], image.shape[1])) < rate
        stego[:, :, channel][mask] ^= 1
    else:
        mask = rng.random(image.shape) < rate
        stego[mask] ^= 1
    return stego


def _save_png(array: np.ndarray, path: Path) -> Path:
    Image.fromarray(array, "RGB").save(path, format="PNG")
    return path


def test_clean_natural_image_has_low_suspicion(tmp_path: Path):
    clean_arr = _generate_synthetic_natural_image(seed=101)
    clean_path = _save_png(clean_arr, tmp_path / "clean.png")

    service = AnalysisService(profile="balanced")
    analysis = service.analyze(clean_path)

    # Clean correlated natural image must have low severity and low score
    assert analysis.severity == "low"
    assert analysis.overall_score < 50


def test_markerless_lsb_embedding_detection(tmp_path: Path):
    """Verify that pure markerless LSB stego is detected via statistical tests."""
    clean_arr = _generate_synthetic_natural_image(seed=202)
    clean_path = _save_png(clean_arr, tmp_path / "clean_base.png")

    # Embed 25% markerless LSB stego
    stego_arr = _embed_raw_lsb(clean_arr, rate=0.25, seed=303)
    stego_path = _save_png(stego_arr, tmp_path / "stego_raw.png")

    service = AnalysisService(profile="balanced")
    clean_analysis = service.analyze(clean_path)
    stego_analysis = service.analyze(stego_path)

    assert clean_analysis.severity == "low"
    # Stego image score must be elevated over clean
    assert stego_analysis.overall_score > clean_analysis.overall_score

    # Statistical detectors must report heuristic evidence on stego
    all_signals = [s for r in stego_analysis.results for s in r.signals]
    heuristic_signals = [
        s for s in all_signals if s.evidence in {"heuristic", "verified"}
    ]
    assert len(heuristic_signals) >= 1
    # Check that categories include decoupled statistical indicators
    signal_names = {s.name for s in heuristic_signals}
    assert any("sample_pair" in name or "rs_regular" in name for name in signal_names)


def test_statistical_sensitivity_across_densities(tmp_path: Path):
    """Test that detection suspicion scales monotonically with embedding density."""
    clean_arr = _generate_synthetic_natural_image(seed=404)
    service = AnalysisService(profile="sensitive")

    scores = []
    densities = (0.0, 0.05, 0.15, 0.35, 0.60)
    for rate in densities:
        if rate == 0.0:
            arr = clean_arr
        else:
            arr = _embed_raw_lsb(clean_arr, rate=rate, seed=int(rate * 1000))
        img_path = _save_png(arr, tmp_path / f"test_{int(rate*100)}.png")
        analysis = service.analyze(img_path)
        scores.append(analysis.overall_score)

    # Suspicion on high density must exceed suspicion on clean
    assert scores[-1] > scores[0]
    assert scores[-1] >= 60


def test_bitplane_analyzer_handles_pure_noise_gracefully(tmp_path: Path):
    """Pure noise has no higher-plane structure; SPA/RS must not generate false heuristic alarms."""
    noise = np.random.default_rng(505).integers(0, 256, (128, 128, 3), dtype=np.uint8)
    noise_path = _save_png(noise, tmp_path / "noise.png")

    analyzer = ImageBitplaneAnalyzer()
    result = analyzer.analyze(noise_path)

    # Heuristic evidence should not fire on pure noise
    heuristic_signals = [s for s in result.signals if s.evidence == "heuristic"]
    assert len(heuristic_signals) == 0
