"""SRM-style gray features: generated CPU QA only."""

import numpy as np
import pytest

from research import rich_color_detector as rich
from research import spatial_matching_probe as matching
from research import srm_features as srm


def texture(seed, size=48):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    base = 128 + 40 * np.sin(x / (4.0 + seed % 5)) * np.cos(y / 7.0)
    return np.clip(base + rng.normal(0, 1.5, (size, size)), 0, 255).round().astype(np.uint8)


def test_symmetry_classes_merge_sign_and_reversal():
    classes = srm.symmetry_classes()
    assert classes.shape == (625,) and srm.BLOCK_DIMS == 169 == classes.max() + 1
    base = 5

    def code(d):
        return sum((v + 2) * base ** (3 - j) for j, v in enumerate(d))

    d = (2, -1, 0, 1)
    same = [d, tuple(-v for v in d), d[::-1], tuple(-v for v in d[::-1])]
    assert len({classes[code(p)] for p in same}) == 1
    assert classes[code((1, 0, 0, 0))] != classes[code((2, 0, 0, 0))]
    with pytest.raises(ValueError):
        classes[0] = 1


def test_cooccurrence_counts_every_window():
    r = np.array([[0, 1, -1, 3, -3, 0]], float)
    counts = srm.cooccurrence(r, 1)
    assert counts.sum() == 3 and counts.shape == (srm.BLOCK_DIMS,)
    assert np.array_equal(srm.cooccurrence(2 * r, 2), counts)


def test_gray_features_are_normalised_and_transpose_invariant():
    img = texture(1)
    f = srm.gray_features(img)
    assert f.shape == (srm.DIMS,) and f.dtype == np.float32
    assert srm.DIMS == len(srm.BLOCKS) * srm.BLOCK_DIMS == 28 * 169
    sums = f.reshape(len(srm.BLOCKS), srm.BLOCK_DIMS).sum(1)
    assert np.allclose(sums, 1.0, atol=1e-5)
    assert np.array_equal(f, srm.gray_features(np.ascontiguousarray(img.T)))
    assert np.array_equal(f, srm.gray_features(img[:, :, None]))


@pytest.mark.parametrize(
    "image",
    [np.zeros((16, 16), np.int16), np.zeros((16, 16, 3), np.uint8), np.zeros((8, 64), np.uint8),
     np.zeros((16, 16, 2), np.uint8), [[0]]],
)
def test_gray_features_reject_bad_input(image):
    with pytest.raises(ValueError):
        srm.gray_features(image)


def test_gray_features_bound_pixel_count(monkeypatch):
    monkeypatch.setattr(srm, "MAX_PIXELS", 100)
    with pytest.raises(ValueError):
        srm.gray_features(np.zeros((16, 16), np.uint8))


def test_features_separate_strong_matching():
    covers = [texture(i)[:, :, None] for i in range(30)]
    stegos = [matching.embed(c, "matching", 1.0, np.random.default_rng(i))
              for i, c in enumerate(covers)]
    cf = np.stack([srm.gray_features(c) for c in covers])
    sf = np.stack([srm.gray_features(s) for s in stegos])
    detector = rich.fit(cf[:20], sf[:20], 1.0)
    assert rich.auc(detector.score(cf[20:]), detector.score(sf[20:])) > 0.9


def test_color_features_pool_channels_and_append_colour_block():
    gray = texture(7)
    rgb = np.stack([gray, np.roll(gray, 1, 0), np.roll(gray, 1, 1)], axis=2)
    f = srm.color_features(rgb)
    assert f.shape == (srm.DIMS + srm.COLOR_DIMS,) and f.dtype == np.float32
    assert np.allclose(f[: srm.DIMS].reshape(len(srm.BLOCKS), -1).sum(1), 1.0, atol=1e-5)
    assert np.array_equal(f[srm.DIMS :], rich.features(rgb)[rich.GRAY_DIMS :])
    same = srm.color_features(np.repeat(gray[:, :, None], 3, axis=2))
    assert np.allclose(same[: srm.DIMS], srm.gray_features(gray), atol=1e-6)
    for bad in (rgb[:, :, :1], rgb.astype(np.int16), rgb[:8], rgb[:, :, 0]):
        with pytest.raises(ValueError):
            srm.color_features(bad)
