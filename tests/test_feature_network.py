"""Nonlinear research math/limits, not accuracy evidence."""

import copy
import json

import numpy as np
import pytest

from core.feature_model import RESIDUAL_MLP, feature_model, feature_network
from steganography.research import export_onnx, train_model
from steganography.research_features import model_domain, read_feature_checkpoint
from tests.test_research_training import training_fixture

torch = pytest.importorskip("torch")


def checkpoint():
    torch.manual_seed(17)
    return {
        "features": 3,
        "state_dict": feature_network(3, RESIDUAL_MLP).state_dict(),
        "preprocessing": {
            "architecture": RESIDUAL_MLP,
            "inference_arithmetic": "float64",
            "normalization": {
                "method": "train-only-standardization",
                "mean": [0.1, 0.2, 0.3],
                "scale": [0.5, 0.25, 1],
            },
        },
    }


def test_independent_residual_math_and_export(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    c = checkpoint()
    c.update(domain="jpeg-dct-residual-parity-mlp64-v1", input_shape=[3])
    x = np.array([[1, -2, 0.5], [0.1, 0.2, 0.3], [-1, 2, 0]], dtype=np.float32)
    z = (x.astype(np.float64) - [0.1, 0.2, 0.3]) / [0.5, 0.25, 1]
    w = {k: v.numpy().astype(np.float64) for k, v in c["state_dict"].items()}
    expected = (
        z @ w["skip.weight"].T
        + w["skip.bias"]
        + np.maximum(0, z @ w["hidden.weight"].T + w["hidden.bias"]) @ w["output.weight"].T
        + w["output.bias"]
    ).astype(np.float32)
    model = feature_model(c).eval()
    np.testing.assert_allclose(
        model(torch.from_numpy(x)).detach().numpy(), expected, atol=1e-6, rtol=0
    )
    path = tmp_path / "model.pt"
    torch.save(c, path)
    card = export_onnx(path, tmp_path / "model.onnx")
    assert card["preprocessing"]["architecture"] == RESIDUAL_MLP
    session = ort.InferenceSession(str(tmp_path / "model.onnx"))
    for batch in (x, x[:1], np.tile(x, (7, 1))):
        actual = session.run(None, {"input": batch})[0]
        np.testing.assert_allclose(
            actual, model(torch.from_numpy(batch)).detach().numpy(), atol=1e-6, rtol=0
        )


@pytest.mark.parametrize(
    "dimension,architecture",
    [(0, RESIDUAL_MLP), (4097, RESIDUAL_MLP), (True, RESIDUAL_MLP), (3, "unknown"), (3, [])],
)
def test_architecture_bounds(dimension, architecture):
    with pytest.raises(ValueError):
        feature_network(dimension, architecture)


def test_bad_weights_and_contracts():
    c = checkpoint()
    bad = copy.deepcopy(c)
    bad["state_dict"]["hidden.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        feature_model(bad)
    bad = copy.deepcopy(c)
    bad["preprocessing"]["normalization"]["mean"] = [0]
    with pytest.raises(ValueError, match="normalization"):
        feature_model(bad)
    with pytest.raises(ValueError, match="unsupported"):
        model_domain("spatial-summary-v1", RESIDUAL_MLP)
    assert model_domain("jpeg-dct-residual-parity-v1", RESIDUAL_MLP).endswith("mlp64-v1")


@pytest.mark.parametrize("state", [None, {}, {"weight": "not a tensor"}])
def test_invalid_checkpoint_parameters(tmp_path, state):
    c = checkpoint()
    c["state_dict"] = state
    path = tmp_path / "invalid.pt"
    torch.save(c, path)
    with pytest.raises(ValueError, match="parameters"):
        read_feature_checkpoint(path)


@pytest.mark.parametrize(
    "setting,value",
    [
        ("architecture", RESIDUAL_MLP),
        ("weight_decay", -1),
        ("weight_decay", float("nan")),
        ("weight_decay", True),
        ("weight_decay", []),
        ("weight_decay", 2),
    ],
)
def test_training_rejects_invalid_contracts(tmp_path, setting, value):
    path, config = training_fixture(tmp_path)
    config[setting] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        train_model(path, tmp_path / "bad.pt")
    assert not (tmp_path / "bad.pt").exists()
