"""WAV LSB features: generated CPU QA only."""

import numpy as np
import pytest

from research import rich_color_detector as rich
from research import wav_lsb_features as wav


def signal(seed, n=8192, noise=0.6, silence=True):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = 3000 * np.sin(2 * np.pi * t / (40 + seed % 7)) + rng.normal(0, noise, n)
    if silence:
        x[: n // 4] = 0  # digital silence, as in padded clips
    return np.clip(np.round(x), -32768, 32767).astype(np.int16)


@pytest.mark.parametrize("kind", wav.KINDS)
def test_embed_changes_only_lsb_scale_and_is_seeded(kind):
    x = signal(1)
    y = wav.embed(x, kind, 0.5, np.random.default_rng(2))
    assert y.dtype == np.int16 and y.shape == x.shape
    assert np.abs(y.astype(int) - x).max() == 1
    assert 0.15 < (y != x).mean() < 0.35  # about half of the touched samples change
    assert np.array_equal(y, wav.embed(x, kind, 0.5, np.random.default_rng(2)))
    edge = np.array([32767, -32768] * 600, np.int16)
    assert np.abs(wav.embed(edge, kind, 1.0, np.random.default_rng(3)).astype(int)).max() <= 32768


@pytest.mark.parametrize("args", [(np.zeros(10, np.int32), "matching", 0.1),
                                  (np.zeros((4, 4), np.int16), "matching", 0.1),
                                  (np.zeros(10, np.int16), "other", 0.1),
                                  (np.zeros(10, np.int16), "matching", 0.0)])
def test_embed_rejects_bad_input(args):
    with pytest.raises(ValueError):
        wav.embed(*args, np.random.default_rng(0))


def test_weighted_stego_estimates_replacement_rate():
    rng = np.random.default_rng(4)
    x = np.clip(np.round(2000 * np.sin(np.arange(40000) / 300) + rng.normal(0, 3, 40000)),
                -32768, 32767).astype(np.int16)
    clean = wav.weighted_stego(x.astype(float))[4]
    stego = wav.weighted_stego(wav.embed(x, "replacement", 0.4, rng).astype(float))[4]
    assert abs(clean) < 0.08 and abs(stego - 0.4) < 0.08


def test_features_shape_bounds_and_silence_cues():
    f = wav.features(signal(5))
    assert f.shape == (wav.DIMS,) and f.dtype == np.float32 and np.isfinite(f).all()
    stego = wav.features(wav.embed(signal(5), "replacement", 0.2, np.random.default_rng(1)))
    assert not np.array_equal(f, stego)
    assert wav.features(signal(6, n=wav.MIN_SAMPLES)).shape == (wav.DIMS,)
    for bad in (np.zeros(wav.MIN_SAMPLES - 1, np.int16), np.zeros(wav.MIN_SAMPLES, np.int32),
                np.zeros((wav.MIN_SAMPLES, 2), np.int16)):
        with pytest.raises(ValueError):
            wav.features(bad)


def test_features_separate_matching_in_quiet_signals():
    covers = [signal(i, noise=0.4) for i in range(24)]
    cf = np.stack([wav.features(c) for c in covers])
    sf = np.stack([wav.features(wav.embed(c, "matching", 0.4, np.random.default_rng(i)))
                   for i, c in enumerate(covers)])
    detector = rich.fit(cf[:16], sf[:16], 1.0)
    assert rich.auc(detector.score(cf[16:]), detector.score(sf[16:])) > 0.9


def test_low_bit_profile_exposes_upscaled_8_bit_audio():
    upscaled = (signal(8) // 256 * 256).astype(np.int16)
    assert np.allclose(wav.low_bit_profile(upscaled), 1.0)
    stego = wav.embed(upscaled, "matching", 0.1, np.random.default_rng(0))
    assert wav.low_bit_profile(stego)[0] < 0.98
    natural = wav.low_bit_profile(signal(9, noise=40, silence=False))
    assert abs(natural[0] - 0.5) < 0.05 and natural[-1] < 0.05
    assert wav.features(upscaled)[-8:].tolist() == [1.0] * 8
