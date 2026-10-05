import hashlib
import io
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from core import jpeg_features as jf
from core import jpeg_residual as jr
from core.jpeg_context import jpeg_context_features
from steganography.research import ResearchManifestError, export_onnx, train_model
from steganography.research_features import feature_contract, training_inputs
from steganography.research_weighting import training_weights
from tests.test_jpeg_context import multi_source  # noqa: F401


def test_independent_signed_residual_parity_oracle_and_dc():
    y = np.random.default_rng(61).integers(-8, 9, (3, 4, 8, 8), dtype=np.int16)
    y[0, 0, 0, 0] = -32768
    y[0, 1, 0, 0] = 32767
    values = jr.coefficient_features(y, np.full((8, 8), 17))
    modes = sorted([(u, v) for u in range(8) for v in range(8)], key=lambda p: (sum(p), *p))[:32]
    residuals, joints = [], []
    for u, v in modes:
        for direction in ("h", "v"):
            counts = [0] * 5
            joint = [0] * 10
            for row in range(3 if direction == "h" else 2):
                for col in range(3 if direction == "h" else 4):
                    first = int(y[row, col, u, v])
                    second = int(y[row + (direction == "v"), col + (direction == "h"), u, v])
                    b = max(-2, min(2, second - first)) + 2
                    counts[b] += 1
                    joint[(first % 2) * 5 + b] += 1
            residuals.extend(round(c / sum(counts), 8) for c in counts)
            joints.extend(round(c / sum(joint), 8) for c in joint)
    dc = [0] * 8
    for row in range(3):
        for col in range(4):
            dc[min(abs(int(y[row, col, 0, 0])), 7)] += 1
    expected = residuals + joints + [round(c / 12, 8) for c in dc]
    assert len(values) == len(jr.FEATURE_NAMES) == 2066
    assert values[1098:] == expected
    assert min(values) >= 0 and max(values) <= 1
    modified = y.copy()
    modified[1, 1, 0, 0] += 1
    changed = jr.coefficient_features(modified, np.full((8, 8), 17))
    assert values[:1098] == changed[:1098] and values[1098:] != changed[1098:]
    assert feature_contract(jr.FEATURE_VERSION)[0] == jr.FEATURE_NAMES


@pytest.mark.parametrize("fault", ["shape", "float", "coefficient-range", "zero-q", "nan-q"])
def test_reuses_native_array_and_overflow_guards(fault):
    y = np.zeros((2, 2, 8, 8), dtype=np.int64)
    q = np.ones((8, 8))
    if fault == "shape":
        y = y[0]
    elif fault == "float":
        y = y.astype(float)
    elif fault == "coefficient-range":
        y[0, 0, 0, 0] = 32768
    elif fault == "zero-q":
        q[0, 0] = 0
    else:
        q[0, 0] = np.nan
    with pytest.raises(ValueError):
        jr.coefficient_features(y, q)


def test_bounded_real_worker_preserves_context_prefix_and_limits(monkeypatch):
    pytest.importorskip("jpeglib")
    stream = io.BytesIO()
    Image.new("RGB", (32, 32), (10, 40, 70)).save(stream, format="JPEG")
    data = stream.getvalue()
    expected = jf.worker(data, jr.coefficient_features)
    assert jr.jpeg_residual_features(data) == expected
    assert expected[:1098] == jpeg_context_features(data)
    with pytest.raises(ValueError, match="unknown"):
        jf.run_feature_worker(data, "untrusted.module", jr.FEATURE_NAMES)
    monkeypatch.setattr("resource.setrlimit", lambda *args: None)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(data)))
    assert jf.main(jr.coefficient_features) == 0


def test_residual_worker_rejects_malformed_output_and_deadlines(monkeypatch):
    data = io.BytesIO()
    Image.new("L", (32, 32)).save(data, format="JPEG")

    def fake_run(command, *, stdout, **kwargs):
        assert command[-1] == "core.jpeg_residual"
        stdout.write(json.dumps([0] * 2065).encode())
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(jf.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="output"):
        jr.jpeg_residual_features(data.getvalue())

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("fixed", 15)

    monkeypatch.setattr(jf.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="15 seconds"):
        jr.jpeg_residual_features(data.getvalue())


def test_residual_training_requires_multi_origin_and_exports_contract(multi_source, tmp_path):  # noqa: F811
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    config, _ = multi_source
    path = tmp_path / "context.json"
    artifact = json.loads(path.read_bytes())
    artifact.update(feature_version=jr.FEATURE_VERSION, feature_names=list(jr.FEATURE_NAMES))
    vector = jr.coefficient_features(np.zeros((2, 2, 8, 8), dtype=np.int16), np.ones((8, 8)))
    for row in artifact["rows"]:
        row["values"] = vector
    path.write_text(json.dumps(artifact))
    config["features_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _, _, provenance = training_inputs(config)
    with pytest.raises(ResearchManifestError, match="explicit multi-source"):
        training_weights({k: v for k, v in config.items() if k != "sample_weighting"}, provenance)
    settings = tmp_path / "train.json"
    settings.write_text(json.dumps(config))
    checkpoint = tmp_path / "residual.pt"
    train_model(settings, checkpoint)
    assert (
        torch.load(checkpoint, weights_only=True)["domain"] == "jpeg-dct-residual-parity-linear-v1"
    )
    card = export_onnx(checkpoint, tmp_path / "residual.onnx")
    assert card["preprocessing"]["feature_version"] == jr.FEATURE_VERSION
    assert card["training_provenance"]["source_balance"]["declared_sources"] == 2
