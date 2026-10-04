import io
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from core import jpeg_features as jf
from steganography import research_features as rf
from steganography.research import ResearchManifestError
from tests.test_research_features import config_for, setup_features


def jpeg_bytes(size=(32, 32), fmt="JPEG"):
    data = io.BytesIO()
    Image.new("RGB", size, (10, 40, 70)).save(data, format=fmt)
    return data.getvalue()


def test_hand_computed_dct_feature_order_and_signs():
    coefficients = np.zeros((2, 2, 8, 8), dtype=np.int16)
    coefficients[:, :, 0, 1] = [[-2, -1], [0, 8]]
    q = np.ones((8, 8), dtype=np.uint16)
    result = jf.coefficient_features(coefficients, q)
    assert len(result) == len(jf.FEATURE_NAMES) == 968
    assert result[:8] == [0.25, 0.25, 0.25, 0, 0, 0, 0, 0.25]
    assert result[8:16] == [1, 0, 0, 0, 0, 0, 0, 0]
    # First horizontal signed pairs: (-2,-1) and (0,>=2).
    pair = result[504:529]
    assert pair[1] == pair[14] == 0.5 and sum(pair) == 1
    # Vertical: (-2,0) and (-1,>=2).
    pair = result[529:554]
    assert pair[2] == pair[9] == 0.5 and sum(pair) == 1
    assert result[-64:] == [1 / 65535] * 64
    coefficients[:, :, 0, 0] = -32768
    assert jf.coefficient_features(coefficients, q) == result  # DC intentionally omitted.


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((8, 8), np.int16),
        ((1, 2, 8, 8), np.int16),
        ((2, 2, 8, 7), np.int16),
        ((2, 2, 8, 8), np.float32),
    ],
)
def test_coefficient_dimensions_and_types(shape, dtype):
    with pytest.raises(ValueError, match="arrays"):
        jf.coefficient_features(np.zeros(shape, dtype=dtype), np.ones((8, 8)))


def test_coefficient_ranges_and_limits(monkeypatch):
    data = np.zeros((2, 2, 8, 8), dtype=np.int64)
    for q in (np.ones((2, 2)), np.zeros((8, 8)), np.full((8, 8), np.nan), np.full((8, 8), 65536)):
        with pytest.raises(ValueError):
            jf.coefficient_features(data, q)
    data[0, 0, 0, 1] = -32769
    with pytest.raises(ValueError, match="16-bit"):
        jf.coefficient_features(data, np.ones((8, 8)))
    monkeypatch.setattr(jf, "MAX_PIXELS", -64000)
    with pytest.raises(ValueError):
        jf.coefficient_features(data, np.ones((8, 8)))


def test_jpeg_preflight_and_optional_dependency(monkeypatch):
    for data in (b"", b"x" * (jf.MAX_IMAGE_BYTES + 1)):
        with pytest.raises(ValueError, match="byte"):
            jf.jpeg_features(data)
    for data in (jpeg_bytes((8, 8)), jpeg_bytes(fmt="PNG")):
        with pytest.raises(ValueError, match="dimensions"):
            jf.jpeg_features(data)
    monkeypatch.setattr(jf.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="dct extra"):
        jf.jpeg_features(jpeg_bytes())
    monkeypatch.setattr(jf, "MAX_PIXELS", 100)
    with pytest.raises(ValueError, match="dimensions"):
        jf.validate_jpeg(jpeg_bytes())


def test_worker_rejects_timeout_failure_and_malformed_output(monkeypatch):
    monkeypatch.setattr(jf.importlib.util, "find_spec", lambda name: True)
    data = jpeg_bytes()

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 15
        assert "SECRET" not in kwargs["env"]
        raise subprocess.TimeoutExpired("fixture", 15)

    monkeypatch.setenv("SECRET", "do-not-forward")
    monkeypatch.setattr(jf.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="15 seconds"):
        jf.jpeg_features(data)
    for output, code, error in (
        (b"", 1, RuntimeError),
        (b"x" * (jf.MAX_OUTPUT_BYTES + 1), 0, RuntimeError),
        (b"invalid", 0, ValueError),
        (b"{}", 0, ValueError),
        (b"[]", 0, ValueError),
        (json.dumps([float("nan")] * 968).encode(), 0, ValueError),
        (json.dumps([2] * 968).encode(), 0, ValueError),
    ):

        def fake_run(*args, output=output, code=code, **kwargs):
            kwargs["stdout"].write(output)
            return SimpleNamespace(returncode=code)

        monkeypatch.setattr(jf.subprocess, "run", fake_run)
        with pytest.raises(error):
            jf.jpeg_features(data)


def test_worker_exact_real_jpeg_and_limits_entrypoint(monkeypatch):
    jpeglib = pytest.importorskip("jpeglib")
    data = jpeg_bytes()
    expected = jf.worker(data)
    assert jf.jpeg_features(data) == expected
    calls = []
    monkeypatch.setattr("resource.setrlimit", lambda *args: calls.append(args))
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(data)))
    assert jf.main() == 0 and len(calls) == 4
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"bad")))
    assert jf.main() == 1
    monkeypatch.setattr(
        jpeglib, "read_dct", lambda path: SimpleNamespace(load=lambda: None, num_components=4)
    )
    with pytest.raises(ValueError, match="colorspace"):
        jf.worker(data)


def test_feature_contract_and_document_size_guards(tmp_path, monkeypatch):
    root, _, manifest = setup_features(tmp_path)
    out = tmp_path / "features.json"
    with pytest.raises(ResearchManifestError, match="contract"):
        rf.extract_features(manifest, out, feature_version="unknown")
    with pytest.raises(ResearchManifestError, match="workers"):
        rf.extract_features(manifest, out, workers=0)
    rf.extract_features(manifest, out)
    artifact = json.loads(out.read_text())
    artifact.pop("feature_version")
    out.write_text(json.dumps(artifact))
    with pytest.raises(ResearchManifestError, match="contract"):
        rf.training_inputs(config_for(manifest, out))
    original_read = rf.read_document
    monkeypatch.setattr(rf, "MAX_DOCUMENT_BYTES", 1)
    monkeypatch.setattr(rf, "read_document", lambda path: (json.loads(path.read_text()), "fixture"))
    with pytest.raises(ResearchManifestError, match="artifact exceeds"):
        rf.extract_features(manifest, tmp_path / "too-large")
    monkeypatch.setattr(rf, "read_document", original_read)
