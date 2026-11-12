"""DCTR JPEG features and the Fisher ensemble: generated CPU QA only."""

import jpeglib
import numpy as np
import pytest
from PIL import Image

from research import dctr_features as dctr
from research import fld_ensemble as ens
from research import rich_color_detector as rich


def gray_jpeg(tmp_path, seed, quality=75, size=64):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    img = 128 + 50 * np.sin(x / (3.0 + seed % 4)) * np.cos(y / 6.0) + rng.normal(0, 4, (size, size))
    path = tmp_path / f"g{seed}.jpg"
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(path, quality=quality)
    return path


def test_quality_tables_and_steps():
    for q in (5, 50, 75, 90, 100):
        assert dctr.quality_from_table(dctr.ijg_table(q)) == q
    assert dctr.step(75) == 4.0 and dctr.step(50) == 8.0 and dctr.step(25) == 16.0
    for bad in (0, 101, 75.0):
        with pytest.raises(ValueError):
            dctr.ijg_table(bad)
        with pytest.raises(ValueError):
            dctr.step(bad)
    with pytest.raises(ValueError):
        dctr.quality_from_table(np.ones((4, 4)))


def test_decompress_matches_libjpeg_within_rounding(tmp_path):
    path = gray_jpeg(tmp_path, 1)
    jpeg = jpeglib.read_dct(str(path))
    plane = dctr.decompress(jpeg.Y, jpeg.qt[0])
    decoded = np.asarray(Image.open(path), float)
    assert np.abs(np.clip(np.round(plane + 128), 0, 255) - decoded).max() <= 1
    with pytest.raises(ValueError):
        dctr.decompress(np.zeros((2, 2, 4, 4)), jpeg.qt[0])


def test_features_shape_normalisation_and_sensitivity(tmp_path, monkeypatch):
    path = gray_jpeg(tmp_path, 2)
    f = dctr.features(path)
    assert f.shape == (dctr.DIMS,) == (8000,) and f.dtype == np.float32
    assert np.allclose(f.reshape(-1, dctr.T + 1).sum(1), 1.0, atol=1e-5)
    jpeg = jpeglib.read_dct(str(path))
    changed = np.array(jpeg.Y)
    changed[1, 1, 0, 1] += 1
    assert not np.array_equal(f, dctr.from_coefficients(changed, jpeg.qt[0]))
    for bad in (np.zeros((1, 4, 8, 8)), np.zeros((4, 4, 8))):
        with pytest.raises(ValueError):
            dctr.from_coefficients(bad, jpeg.qt[0])
    monkeypatch.setattr(dctr, "MAX_BLOCKS", 63)
    with pytest.raises(ValueError):
        dctr.from_coefficients(changed, jpeg.qt[0])


def test_ensemble_separates_shifted_features_and_round_trips(tmp_path):
    rng = np.random.default_rng(0)
    covers = rng.normal(0, 1, (120, 30))
    stegos = covers + rng.normal(0, 0.3, (120, 30)) + np.r_[np.full(5, 1.5), np.zeros(25)]
    model = ens.fit(covers[:80], stegos[:80], d_sub=10, learners=9, seed=3)
    scores = model.score(covers[80:]), model.score(stegos[80:])
    assert rich.auc(*scores) > 0.9 and np.all(np.abs(scores[0]) <= 1)
    model.save(tmp_path / "e.npz")
    loaded = ens.Ensemble.load(tmp_path / "e.npz")
    assert np.allclose(loaded.score(covers), model.score(covers))
    with pytest.raises(FileExistsError):
        model.save(tmp_path / "e.npz")
    with pytest.raises(ValueError, match="width"):
        model.score(covers[:, :5])
    np.savez(tmp_path / "bad.npz", schema=np.array("other"))
    with pytest.raises(ValueError, match="schema"):
        ens.Ensemble.load(tmp_path / "bad.npz")
    same = ens.fit(covers[:80], stegos[:80], d_sub=10, learners=9, seed=3)
    assert np.array_equal(same.score(covers), model.score(covers))
    one = ens.fit(covers[:80], stegos[:80], d_sub=1, learners=2)
    assert one.columns.shape == (2, 1)


@pytest.mark.parametrize("kwargs", [{"d_sub": 0}, {"learners": 0}, {"ridge": 0.0}])
def test_ensemble_rejects_bad_parameters(kwargs):
    x = np.zeros((4, 3))
    with pytest.raises(ValueError):
        ens.fit(x, x, **kwargs)
    with pytest.raises(ValueError):
        ens.fit(x, x[:3])



def test_calibrated_features_append_reference_difference(tmp_path):
    path = gray_jpeg(tmp_path, 5, size=72)
    jpeg = jpeglib.read_dct(str(path))
    ref = dctr.reference(jpeg.Y, jpeg.qt[0])
    assert ref.shape == (9, 9, 8, 8)  # 68 cropped pixels pad to nine blocks
    f = dctr.calibrated_features(path)
    own = dctr.features(path)
    assert f.shape == (2 * dctr.DIMS,) and np.array_equal(f[: dctr.DIMS], own)
    assert np.allclose(f[dctr.DIMS :], own - dctr.from_coefficients(ref, jpeg.qt[0]))
    assert np.abs(f[dctr.DIMS :]).sum() > 0
