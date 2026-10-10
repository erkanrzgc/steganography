"""Portable scalar replay: discrete decisions exact, float roundoff bounded."""

import numpy as np


def assert_metrics_replay(actual, expected):
    assert set(actual) == set(expected)
    for key in actual:
        if key in {"scores", "cross_entropy"}:
            np.testing.assert_allclose(
                actual[key], expected[key], rtol=0, atol=1e-14, equal_nan=False
            )
        else:
            assert actual[key] == expected[key]
