"""Independent convolution/edge pooling oracles and full fractional evaluation."""

import numpy as np
import pytest

from core import srnet
from core import srnet_reference as reference


def test_convolution_and_edge_average_scalar_oracles():
    image = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
    weights = np.ones((1, 1, 3, 3))
    actual = reference.convolution(image, weights, np.array([2.0]))
    expected = np.empty_like(image)
    for y in range(4):
        for x in range(4):
            expected[0, 0, y, x] = (
                image[0, 0, max(0, y - 1) : min(4, y + 2), max(0, x - 1) : min(4, x + 2)].sum() + 2
            )
    np.testing.assert_array_equal(actual, expected)
    pooled = reference.average_reduce(image)
    assert pooled.shape == (1, 1, 2, 2)
    np.testing.assert_array_equal(pooled, [[[[2.5, 4], [8.5, 10]]]])
    projected = reference.convolution(image, np.ones((1, 1, 1, 1)), np.zeros(1), stride=2)
    np.testing.assert_array_equal(projected, image[:, :, ::2, ::2])


def test_full_independent_float64_forward_with_nondefault_bn():
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261014)
            model = srnet.network().eval()
        for name, value in model.named_buffers():
            if name.endswith("running_mean"):
                value.fill_(0.125)
            if name.endswith("running_var"):
                value.fill_(1.25)
        pixels = np.random.default_rng(51).normal(0.125, 0.25, (1, 1, 256, 256)).astype("<f4")
        arrays = {k: v.detach().numpy() for k, v in model.state_dict().items()}
        native = srnet.float_logits(model, pixels)
        independent = reference.reference_logits(arrays, pixels)
        audit = reference.compare(native, independent)
        assert audit["passed"] and audit["decisions_equal"]
        for invalid in (
            pixels.astype("f8"),
            pixels[:, :, :128, :128],
            np.full_like(pixels, float("nan")),
        ):
            with pytest.raises(ValueError):
                reference.reference_logits(arrays, invalid)
    finally:
        torch.set_num_threads(previous)


def test_comparison_rejects_nonfinite_and_decision_changes():
    for invalid in (np.zeros((2, 2)), np.array([[float("nan"), 1]])):
        with pytest.raises(ValueError):
            reference.compare(invalid, np.zeros((1, 2)))
    assert reference.compare(np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]]))["passed"]
    assert not reference.compare(np.array([[2.0, 1.0]]), np.array([[1.0, 2.0]]))["passed"]
    assert not reference.compare(np.zeros((1, 2)), np.array([[1e-6, 0.0]]))["passed"]
    # A large common logit offset must not hide a changed probability margin.
    assert not reference.compare(np.array([[1e8, 1e8 + 1]]), np.array([[1e8, 1e8 + 2]]))["passed"]


def test_finite_input_can_still_produce_invalid_reference():
    pytest.importorskip("torch")
    model = srnet.network().eval()
    arrays = {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}
    # Finite extreme parameters are valid storage, but arithmetic can overflow.
    for name, value in arrays.items():
        if name.endswith(".0.weight"):
            value.fill(np.finfo(np.float32).max)
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="nonfinite logits"),
    ):
        reference.reference_logits(arrays, np.ones((1, 1, 256, 256), dtype="<f4"))
