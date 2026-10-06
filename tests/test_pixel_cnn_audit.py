"""Independent arithmetic oracle and honest numeric gate reporting."""

import runpy
from pathlib import Path

import numpy as np
import pytest

AUDIT = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit-pixel-cnn.py"))


@pytest.mark.parametrize("scaled", [False, True])
def test_numpy_forward_matches_native(scaled):
    torch = pytest.importorskip("torch")
    from core.jpeg_cnn import ARCHITECTURE, PIXEL_ARCHITECTURE, network, pixel_logits

    architecture = PIXEL_ARCHITECTURE if scaled else ARCHITECTURE

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(71)
        model = network(architecture).eval()
    raw = np.random.default_rng(71).integers(0, 256, (2, 1, 128, 128), dtype=np.uint8)
    weights = {k: v.numpy().astype(np.float64) for k, v in model.state_dict().items()}
    np.testing.assert_allclose(
        AUDIT["numpy_logits"](raw, weights, architecture),
        pixel_logits(model, raw),
        atol=1e-6,
        rtol=0,
    )


@pytest.mark.parametrize(
    ("actual", "saved", "passed", "decisions"),
    [
        ([0.3], [0.3], True, True),
        ([0.30001], [0.3], False, True),
        ([0.49999999], [0.5], False, False),
    ],
)
def test_numeric_gate(actual, saved, passed, decisions):
    result = AUDIT["comparison"](np.array(actual), np.array(saved))
    assert result["passed"] is passed
    assert result["decisions_equal"] is decisions


def test_optional_version_absence():
    assert AUDIT["version"]("nonexistent-pixel-cnn-audit-dependency") == "unavailable"


def test_sigmoid_extremes():
    scores = AUDIT["probability"](np.array([-1e9, 0, 1e9]))
    assert np.isfinite(scores).all()
    assert scores[0] < 1e-300 and scores[1] == 0.5 and scores[2] == 1
