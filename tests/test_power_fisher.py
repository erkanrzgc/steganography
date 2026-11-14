"""Generated numeric fixtures for power-mapped CPU rich-model classifiers."""

import numpy as np
import pytest

from research import power_fisher as power


def test_signed_power_is_exact_and_preserves_input():
    values = np.array([[-16., -1., 0., 1., 16.]], dtype=np.float32)
    original = values.copy()
    assert np.array_equal(power.transform(values, .5), [[-4, -1, 0, 1, 4]])
    assert np.array_equal(power.transform(values, .25), [[-2, -1, 0, 1, 2]])
    assert np.array_equal(power.transform(values, 1), values)
    assert np.array_equal(values, original)
    integers = np.array([[-1_000_000_000_000, 4]], dtype=np.int64)
    assert np.all(np.isfinite(power.transform(integers, .5)))


@pytest.mark.parametrize('exponent', [0, -1, 1.1, float('inf'), float('nan'), True, '0.5'])
def test_invalid_power_rejected(exponent):
    with pytest.raises(ValueError):
        power.transform(np.ones((2, 3)), exponent)


@pytest.mark.parametrize('values', [
    [[1, 2]], np.zeros(3), np.zeros((2, 0)), np.array([[object()]], dtype=object),
    np.array([[float('nan')]]), np.array([[float('inf')]]), np.array([[1e20]]),
    np.array([[np.iinfo(np.int64).min]], dtype=np.int64),
    np.broadcast_to(0., (power.MAX_ROWS + 1, 1)),
    np.broadcast_to(0., (1, power.MAX_FEATURES + 1)),
    np.broadcast_to(0., (power.MAX_ROWS, power.MAX_FEATURES)),
])
def test_feature_validation_precedes_work(values):
    with pytest.raises(ValueError):
        power.transform(values, .5)


def test_fisher_matches_independent_closed_form_and_keeps_pairs_private():
    rng = np.random.default_rng(20261114)
    covers = rng.uniform(.01, 1, (15, 4))
    stegos = covers + rng.uniform(.01, .1, covers.shape)
    originals = covers.copy(), stegos.copy()
    models = power.fit_grid(covers, stegos, power=.5, ridges=(.03, .3))
    c, s = np.sqrt(covers), np.sqrt(stegos)
    both = np.concatenate([c, s])
    mean, scale = both.mean(0), both.std(0) + 1e-12
    cz, sz = (c - mean) / scale, (s - mean) / scale
    cc, ss = cz.mean(0), sz.mean(0)
    scatter = ((cz - cc).T @ (cz - cc) + (sz - ss).T @ (sz - ss)) / (len(both) - 1)
    for ridge, model in zip((.03, .3), models, strict=True):
        weights = np.linalg.solve(scatter + ridge * np.eye(4), ss - cc)
        expected = cz @ weights - .5 * (cc + ss) @ weights
        assert np.allclose(model.score(covers, batch_rows=2), expected, atol=1e-12)
        assert model.fisher.ridge == ridge
    assert np.array_equal(covers, originals[0]) and np.array_equal(stegos, originals[1])
    assert models[0].score(np.empty((0, 4))).shape == (0,)
    with pytest.raises(ValueError, match='width'):
        models[0].score(np.zeros((1, 3)))
    for batch in (0, -1, 4097, True, 1.5):
        with pytest.raises(ValueError, match='batch_rows'):
            models[0].score(covers, batch_rows=batch)


@pytest.mark.parametrize('ridges', [(), (-1,), (0,), (float('nan'),), (float('inf'),),
                                    (True,), tuple([1.] * 17)])
def test_fit_rejects_invalid_grids_and_pair_shapes(ridges):
    with pytest.raises(ValueError):
        power.fit_grid(np.ones((3, 2)), np.ones((3, 2)), ridges=ridges)


def test_fit_requires_two_complete_pairs():
    for c, s in [(np.zeros((1, 2)), np.zeros((1, 2))),
                 (np.zeros((3, 2)), np.zeros((2, 2)))]:
        with pytest.raises(ValueError, match='paired'):
            power.fit_grid(c, s)


def test_threshold_agrees_with_independent_scalar_search_including_ties():
    covers = np.array([-2., -1., 0., 0., 3.])
    stegos = np.array([-1., 1., 1., 2.])
    candidates = sorted(set(covers) | set(stegos) | {0.})
    def scalar(t):
        return .5 * (sum(float(v < t) for v in covers) / len(covers)
                     + sum(float(v >= t) for v in stegos) / len(stegos))
    best = sorted(candidates, key=lambda t: (-scalar(t), abs(t), t))[0]
    assert power.calibration_threshold(covers, stegos) == best
    assert power.calibration_threshold(np.ones(3), np.ones(3)) == 0


def test_exact_count_tie_cannot_be_broken_by_float_roundoff():
    covers = np.array([3., 3., -4.])
    stegos = np.array([2., 4., 0.])
    # Zero and four both classify four of six correctly. Float subtraction
    # can make four appear better by one ulp; the declared tie rule picks zero.
    assert power.calibration_threshold(covers, stegos) == 0


@pytest.mark.parametrize('bad', [np.array([]), np.ones((2, 1)), np.array([float('nan')]),
                                  np.array([object()]), [1, 2],
                                  np.broadcast_to(0., (power.MAX_CALIBRATION_ROWS + 1,))])
def test_threshold_rejects_invalid_score_sets(bad):
    with pytest.raises(ValueError):
        power.calibration_threshold(bad, np.ones(2))
    with pytest.raises(ValueError):
        power.calibration_threshold(np.ones(2), bad)
