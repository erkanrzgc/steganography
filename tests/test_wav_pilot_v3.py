"""AudioMNIST v3 CNN recipe: inputs, batch-independent normalization, optimizer."""

import io
import wave

import numpy as np
import pytest

from core import nonjpeg_models, wav_pilot_training, wav_pilot_v3


def wav_bytes(samples):
    # The native decoder requires at least 160 frames.
    samples = np.pad(np.asarray(samples), (0, max(0, 160 - len(samples))))
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(48000)
        audio.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return output.getvalue()


def test_inputs_truncate_residuals_in_sample_units_and_expose_bit_planes():
    samples = [0, 1, -1, -2, 3, 40, -32768, 32767]
    value = wav_pilot_v3.inputs(wav_bytes(samples))
    assert value.shape == (1, 4, 160) and value.dtype == np.float32
    raw = np.pad(np.asarray(samples, dtype=np.int64), (0, 152))
    first = np.diff(raw, prepend=raw[0])
    second = np.diff(first, prepend=first[0])
    assert np.array_equal(value[0, 0], np.clip(first, -8, 8) / 8)
    assert np.array_equal(value[0, 1], np.clip(second, -8, 8) / 8)
    # Two's-complement planes: -1 -> ...1111, -2 -> ...1110, -32768 -> 1000...0000.
    assert value[0, 2, :8].tolist() == [-1, 1, 1, -1, 1, -1, -1, 1]
    assert value[0, 3, :8].tolist() == [-1, -1, 1, 1, 1, -1, -1, 1]
    assert np.abs(value).max() == 1


def test_one_lsb_change_reaches_the_model_at_unit_scale():
    cover = np.zeros(160, dtype=np.int16)
    stego = cover.copy()
    stego[10] = 1
    delta = wav_pilot_v3.inputs(wav_bytes(stego)) - wav_pilot_v3.inputs(wav_bytes(cover))
    v1 = nonjpeg_models.inputs(wav_bytes(stego), nonjpeg_models.AUDIO_DOMAIN)
    v1 -= nonjpeg_models.inputs(wav_bytes(cover), nonjpeg_models.AUDIO_DOMAIN)
    assert np.abs(delta).max() == 2 and np.abs(v1).max() < 1e-4


def test_network_is_batch_independent_and_has_no_batchnorm():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    model = wav_pilot_v3.network()
    assert model.architecture == wav_pilot_v3.AUDIO_V3_DOMAIN
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    assert any(isinstance(m, torch.nn.GroupNorm) for m in model.modules())
    values = torch.from_numpy(wav_pilot_v3.inputs(wav_bytes(np.arange(-300, 300) * 7)))
    model.train()
    trained = model(values)
    model.eval()
    assert trained.shape == (1, 2) and torch.equal(trained, model(values))


def test_optimizer_is_adamw_with_weight_only_decay():
    torch = pytest.importorskip("torch")
    model = wav_pilot_v3.network()
    optimizer = wav_pilot_v3.optimizer(torch, model)
    assert isinstance(optimizer, torch.optim.AdamW)
    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == 0.0001 and no_decay["weight_decay"] == 0.0
    assert decay["lr"] == no_decay["lr"] == 0.001
    assert all(p.ndim > 1 for p in decay["params"]) and all(p.ndim == 1 for p in no_decay["params"])
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(model.parameters()))


def test_v3_recipe_binds_v3_parts_and_v1_default_is_unchanged():
    recipe = wav_pilot_v3.V3_RECIPE
    assert recipe.domain == wav_pilot_v3.AUDIO_V3_DOMAIN
    assert (recipe.inputs, recipe.network, recipe.optimizer) == (
        wav_pilot_v3.inputs,
        wav_pilot_v3.network,
        wav_pilot_v3.optimizer,
    )
    v1 = wav_pilot_training.V1_RECIPE
    data = wav_bytes(np.arange(160))
    assert v1.domain == nonjpeg_models.AUDIO_DOMAIN
    assert np.array_equal(v1.inputs(data), nonjpeg_models.inputs(data, nonjpeg_models.AUDIO_DOMAIN))


def test_v1_recipe_optimizer_keeps_plain_sgd():
    torch = pytest.importorskip("torch")
    model = wav_pilot_training.V1_RECIPE.network()
    optimizer = wav_pilot_training.V1_RECIPE.optimizer(torch, model)
    assert type(optimizer) is torch.optim.SGD
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.0001, 0.0]
    assert all(group["momentum"] == 0 for group in optimizer.param_groups)


def test_network_reports_missing_torch(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("blocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="optional Torch missing"):
        wav_pilot_v3.network()
