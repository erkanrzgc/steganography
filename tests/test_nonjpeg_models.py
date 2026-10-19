import builtins
import io

import numpy as np
import pytest
from PIL import Image

from core import nonjpeg_models as models
from tests.test_nonjpeg_features import image_bytes, wav_bytes


def test_native_input_transform_preserves_single_lsb_and_int32_differences():
    samples = np.resize(np.array([-32768, 32767, 0, -1, 1], dtype="<i2"), 160)
    result = models.inputs(wav_bytes(samples), models.AUDIO_DOMAIN)
    assert result.shape == (1, 3, 160) and result.dtype == np.float32
    assert result[0, 1, 1] == 65535 / 32768
    assert result[0, 2, 2] == -98302 / 32768
    changed = samples.copy()
    changed[5] ^= 1
    other = models.inputs(wav_bytes(changed), models.AUDIO_DOMAIN)
    assert abs(float(result[0, 0, 5] - other[0, 0, 5])) == 1 / 32768
    for mode, domain, channels in (("L", models.GRAY_DOMAIN, 1), ("RGB", models.RGB_DOMAIN, 3)):
        out = io.BytesIO()
        Image.new(mode, (256, 256), color=(1, 1, 1) if mode == "RGB" else 1).save(out, format="BMP")
        x = models.inputs(out.getvalue(), domain)
        assert x.shape == (1, channels, 256, 256) and x.dtype == np.float32
        assert (x == 1).all()
    for domain in ("jpeg", "unknown"):
        with pytest.raises(ValueError, match="domain"):
            models.inputs(b"", domain)
        with pytest.raises(ValueError, match="domain"):
            models.network(domain)
    with pytest.raises(ValueError, match="grayscale"):
        models.inputs(image_bytes(), models.GRAY_DOMAIN)


def test_generated_backward_only_and_preserve_rng_threads():
    torch = pytest.importorskip("torch")
    old_rng, old_threads = torch.get_rng_state(), torch.get_num_threads()
    result = models.generated_smoke()
    assert result["status"] == "ready" and result["optimizer_updates"] == 0
    assert result["checkpoint_written"] is False
    assert set(result["models"]) == set(models.DOMAINS)
    assert all(
        r["finite_gradients"] and r["parameters_unchanged"] for r in result["models"].values()
    )
    assert torch.equal(torch.get_rng_state(), old_rng) and torch.get_num_threads() == old_threads
    with torch.random.fork_rng(devices=[]):
        gray, rgb = models.network(models.GRAY_DOMAIN), models.network(models.RGB_DOMAIN)
    assert gray.architecture != rgb.architecture
    assert gray.front[0][0].in_channels == 1 and rgb.front[0][0].in_channels == 3
    assert list(gray.state_dict()) == list(rgb.state_dict())
    for key, value in gray.state_dict().items():
        if key != "front.0.0.weight":
            assert value.shape == rgb.state_dict()[key].shape


def test_torch_absence_is_unavailable(monkeypatch):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("missing research dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert models.generated_smoke()["status"] == "unavailable"
    with pytest.raises(RuntimeError, match="unavailable"):
        models.network(models.AUDIO_DOMAIN)


def test_nonfinite_smoke_fails_and_restores_policy(monkeypatch):
    torch = pytest.importorskip("torch")
    old_threads = torch.get_num_threads()
    monkeypatch.setattr(torch.nn.functional, "cross_entropy", lambda x, y: x.sum() * float("nan"))
    with pytest.raises(ValueError, match="forward/backward"):
        models.generated_smoke()
    assert torch.get_num_threads() == old_threads
