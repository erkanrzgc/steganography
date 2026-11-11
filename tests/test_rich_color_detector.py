"""Rich-model RGB detector: generated CPU QA only."""

import json

import numpy as np
import pytest

from research import rich_color_detector as rich
from research import spatial_matching_probe as matching
from tests.test_external_rgb_training import banks, photos


def smooth(seed, size=48):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    base = 128 + 50 * np.sin(x / (5.0 + seed % 7)) * np.cos(y / 9.0)
    shared = rng.normal(0, 0.6, (size, size, 1))
    return np.clip(base[:, :, None] + shared + [0, 3, -3], 0, 255).round().astype(np.uint8)


def test_features_are_normalised_blocks_of_the_declared_width():
    f = rich.features(smooth(1))
    assert f.shape == (rich.DIMS,) and f.dtype == np.float32
    edges = np.cumsum([0, 625, 625, 125] + [343] * 4)
    assert all(np.isclose(f[a:b].sum(), 1.0, atol=1e-5) for a, b in zip(edges, edges[1:],
                                                                         strict=False))
    assert np.array_equal(f, rich.features(smooth(1)))


@pytest.mark.parametrize(
    "image",
    [np.zeros((16, 16, 3), np.int16), np.zeros((16, 16), np.uint8), np.zeros((16, 16, 4), np.uint8),
     np.zeros((16, 16, 2), np.uint8),
     np.zeros((4, 16, 3), np.uint8), [[0]]],
)
def test_features_reject_non_rgb_or_tiny_images(image):
    with pytest.raises(ValueError):
        rich.features(image)


def test_gray_features_are_the_within_channel_block():
    gray = smooth(3)[:, :, 1:2]
    f = rich.features(gray)
    assert f.shape == (rich.GRAY_DIMS,)
    assert np.isclose(f[:625].sum(), 1.0, atol=1e-5)
    stego = matching.embed(gray, "matching", 0.4, np.random.default_rng(1))
    assert not np.array_equal(f, rich.features(stego))


def test_features_bound_pixel_count(monkeypatch):
    monkeypatch.setattr(rich, "MAX_PIXELS", 100)
    with pytest.raises(ValueError):
        rich.features(np.zeros((16, 16, 3), np.uint8))


def test_auc_handles_ties_and_rejects_empty_classes():
    assert rich.auc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 1.0
    assert rich.auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
    assert rich.auc(np.array([0.0, 2.0]), np.array([1.0, 2.0])) == pytest.approx(0.625)
    with pytest.raises(ValueError):
        rich.auc(np.array([]), np.array([1.0]))


def test_pairs_are_deterministic_and_bounded_changes():
    covers = [smooth(i) for i in range(3)]
    first = rich.pair_features(covers, salt=4)
    again = rich.pair_features(covers, salt=4, workers=2)
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
    other = rich.pair_features(covers, salt=5, augment=True)
    assert not np.array_equal(first[1], other[1])
    for bad in ({"salt": 1.0}, {"salt": 1, "workers": -1}):
        with pytest.raises(ValueError):
            rich.pair_features(covers, **bad)
    with pytest.raises(ValueError):
        rich.pair_features([], salt=1)
    rng = np.random.default_rng([rich.TRAIN_SEED, 4, 0])
    kind = matching.KINDS[int(rng.integers(0, 2))]
    rate = rich.RATES[int(rng.integers(0, 4))]
    expected = rich.features(matching.embed(covers[0], kind, rate, rng))
    assert np.array_equal(first[1][0], expected)
    assert np.array_equal(first[0][0], rich.features(covers[0]))


def test_detector_separates_strong_matching_and_round_trips(tmp_path):
    covers = [smooth(i) for i in range(24)]
    cf = np.stack([rich.features(c) for c in covers])
    sf = np.stack([rich.features(matching.embed(c, "matching", 1.0, np.random.default_rng(i)))
                   for i, c in enumerate(covers)])
    detector = rich.fit(cf[:16], sf[:16], 1.0)
    assert rich.auc(detector.score(cf[16:]), detector.score(sf[16:])) > 0.9
    detector.save(tmp_path / "d.npz")
    loaded = rich.Detector.load(tmp_path / "d.npz")
    assert np.allclose(loaded.score(cf), detector.score(cf)) and loaded.ridge == 1.0
    with pytest.raises(FileExistsError):
        detector.save(tmp_path / "d.npz")
    with pytest.raises(ValueError, match="width"):
        detector.score(cf[:, :5])
    np.savez(tmp_path / "bad.npz", schema=np.array("other"))
    with pytest.raises(ValueError, match="schema"):
        rich.Detector.load(tmp_path / "bad.npz")
    for bad in ((cf[:1], sf[:1], 1.0), (cf, sf[:3], 1.0), (cf, sf, 0.0), (cf[0], sf[0], 1.0)):
        with pytest.raises(ValueError):
            rich.fit(*bad)


def test_ridge_selection_uses_group_holdout():
    covers = [smooth(i) for i in range(20)]
    cf, sf = rich.pair_features(covers, salt=1)
    ridge, scores = rich.select_ridge(cf, sf, np.arange(20))
    assert ridge in rich.LAMBDAS and set(scores) == {str(r) for r in rich.LAMBDAS}
    with pytest.raises(ValueError):
        rich.select_ridge(cf, sf, np.zeros(20, int))
    with pytest.raises(ValueError):
        rich.select_ridge(cf, sf, np.ones(20, int))


def test_evaluate_reports_auc_accuracy_and_ordering():
    covers = [smooth(i) for i in range(8)]
    cf, sf = rich.pair_features(covers, salt=2)
    detector = rich.fit(cf, sf, 0.3)
    stegos = {"m": [matching.embed(c, "matching", 1.0, np.random.default_rng(i))
                    for i, c in enumerate(covers)]}
    out = rich.evaluate(detector, covers, stegos)["m"]
    assert set(out) == {"auc", "accuracy", "ordered"}
    assert all(0 <= v <= 1 for v in out.values())


def test_cli_trains_saves_and_scores_validation(tmp_path, capsys):
    bank, vbank, *_ = banks(tmp_path, crops=4)
    assert rich.main([str(bank), str(tmp_path / "o"), "--valid", str(vbank)]) == 0
    result = json.loads((tmp_path / "o" / "result.json").read_text())
    assert result["pairs"] == 12 and result["dims"] == rich.DIMS
    assert set(result["validation"]) == {f"{k}-{r}" for k, r in rich.ext.CONDITIONS}
    assert (tmp_path / "o" / "detector.npz").exists()
    assert '"schema": "rich-color-detector-v1"' in capsys.readouterr().out
    limited = rich.train(bank, tmp_path / "p", limit=10)
    assert limited["pairs"] == 10 and "validation" not in limited
    with pytest.raises(FileExistsError):
        rich.train(bank, tmp_path / "p")
    assert photos  # shared generated-photo helper stays importable
