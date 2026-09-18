"""Real optional CPU training/export smoke; never detector accuracy evidence."""

import json

import numpy as np
import pytest

from steganography.research import export_onnx, train_model
from steganography.research_features import extract_features, training_inputs
from tests.test_research_features import config_for, setup_features

torch = pytest.importorskip("torch", reason="optional research dependency unavailable")


def training_fixture(tmp_path):
    _, _, manifest_path = setup_features(tmp_path)
    artifact = tmp_path / "features.json"
    extract_features(manifest_path, artifact)
    config = {**config_for(manifest_path, artifact), "epochs": 3, "seed": 42}
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps(config))
    return config_path, config


def test_real_cpu_training_and_onnx_parity(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    config_path, config = training_fixture(tmp_path)
    output = tmp_path / "baseline.pt"
    report = train_model(config_path, output)
    assert report["samples"] == 2 and np.isfinite(report["loss"])
    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    assert checkpoint["domain"] == "spatial-summary-linear-v1"
    assert checkpoint["training_provenance"]["features_sha256"] == config["features_sha256"]
    replay = tmp_path / "repeat.pt"
    train_model(config_path, replay)
    second = torch.load(replay, map_location="cpu", weights_only=True)
    for key in checkpoint["state_dict"]:
        assert torch.equal(checkpoint["state_dict"][key], second["state_dict"][key])
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        train_model(config_path, output)
    assert output.read_bytes() == before
    model = torch.nn.Linear(12, 1)
    model.load_state_dict(checkpoint["state_dict"])
    x, _, _ = training_inputs(config)
    expected = model(torch.from_numpy(x)).detach().numpy()
    model_path = tmp_path / "baseline.onnx"
    contract = export_onnx(output, model_path)
    onnx.checker.check_model(onnx.load(model_path))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    for batch in (x, x[:1], np.concatenate([x, x])):
        actual = session.run(["logit"], {"input": batch})[0]
        np.testing.assert_allclose(
            actual, model(torch.from_numpy(batch)).detach().numpy(), atol=1e-6
        )
    assert expected.shape == (2, 1)
    assert contract["training_provenance"] == checkpoint["training_provenance"]
    assert not contract["calibrated"]
    with pytest.raises(FileExistsError):
        export_onnx(output, model_path)


@pytest.mark.parametrize(
    "setting,value",
    [("learning_rate", 0), ("learning_rate", float("nan")), ("epochs", 0), ("epochs", 100001)],
)
def test_training_rejects_invalid_settings(tmp_path, setting, value):
    config_path, config = training_fixture(tmp_path)
    config[setting] = value
    config_path.write_text(json.dumps(config))
    out = tmp_path / "invalid.pt"
    with pytest.raises(ValueError):
        train_model(config_path, out)
    assert not out.exists()
