"""Independent scalar pair oracle and generated-only, zero-update model checks."""

import builtins
import io
import wave
from pathlib import Path

import numpy as np
import pytest

from core import wav_pair_candidate as candidate


def wav(samples, *, channels=1, width=2, rate=48000):
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(width)
        stream.setframerate(rate)
        stream.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return output.getvalue()


def scalar(samples):
    result = []
    size = len(samples)
    windows = [(0, size)] + [(i * size // 4, (i + 1) * size // 4) for i in range(4)]
    for start, stop in windows:
        for bit in (0, 1):
            counts = {}
            for value in samples[start:stop]:
                value = int(value) + 32768
                pair = counts.setdefault(value & ~(1 << bit), [0, 0])
                pair[(value >> bit) & 1] += 1
            energy = sum((a - b) ** 2 / (a + b) for a, b in counts.values())
            result.append(1 - energy / (stop - start))
    return result


@pytest.mark.parametrize("size", [160, 161, 643, 220500])
def test_vector_matches_independent_scalar_over_full_signed_range(size):
    samples = np.random.default_rng(size).integers(-32768, 32768, size=size)
    actual = candidate.features(wav(samples))
    assert isinstance(actual, tuple) and len(actual) == 10
    assert np.isfinite(actual).all() and min(actual) >= 0 and max(actual) <= 1
    assert actual == pytest.approx(scalar(samples), abs=1e-12)


@pytest.mark.parametrize("plane", [0, 1])
@pytest.mark.parametrize("base", [-32768, -4, 0, 32764])
def test_exact_equalization_and_unmodified_other_plane(plane, base):
    original = np.full(160, base)
    changed = original.copy()
    changed[::2] ^= 1 << plane
    assert candidate.features(wav(original)) == (0.0,) * 10
    actual = candidate.features(wav(changed))
    assert actual[plane::2] == (1.0,) * 5
    assert actual[1 - plane::2] == (0.0,) * 5


def test_quarters_keep_uneven_tail_and_localize_changes():
    samples = np.zeros(161, dtype=np.int32)
    samples[120:140] = 1
    actual = candidate.features(wav(samples))
    assert actual[2:8] == (0.0,) * 6
    assert actual[8] == pytest.approx(1 - 1 / 41**2)
    assert actual == pytest.approx(scalar(samples))


def test_bit_pairs_do_not_merge_different_upper_bits_or_sign_boundaries():
    samples = np.tile([-32768, 32767, -1, 0, 1, 2, 3, 4], 20)
    assert candidate.features(wav(samples)) == pytest.approx(scalar(samples))


def test_single_lsb_reaches_fp32_head_input():
    samples = np.zeros(160, dtype=np.int32)
    original = candidate.inputs(wav(samples))
    samples[0] = 1
    changed = candidate.inputs(wav(samples))
    assert original.shape == changed.shape == (1, 10)
    assert original.dtype == changed.dtype == np.float32
    assert changed[0, 0] == pytest.approx(4 * 159 / 160**2)
    assert changed[0, 2] > 0 and np.array_equal(changed[0, 4:], original[0, 4:])


@pytest.mark.parametrize("samples", [np.zeros(159), np.zeros(220501)])
def test_native_frame_limits(samples):
    with pytest.raises(ValueError, match="frame/rate/duration"):
        candidate.features(wav(samples))


@pytest.mark.parametrize("options", [{"channels": 2}, {"width": 1}, {"rate": 4000}])
def test_rejects_unsupported_native_domain(options):
    with pytest.raises(ValueError, match="frame/rate/duration"):
        candidate.features(wav(np.zeros(320), **options))


@pytest.mark.parametrize("data", [b"bad", wav(np.zeros(160))[:-1]])
def test_corrupt_container_is_rejected(data):
    with pytest.raises(ValueError, match="RIFF"):
        candidate.inputs(data)


def test_byte_only_service_does_not_open_carrier_paths():
    with pytest.raises(TypeError):
        candidate.features(Path("not-opened.wav"))


def test_names_are_unique_and_fixed_order():
    assert len(set(candidate.NAMES)) == 10
    assert candidate.NAMES[:2] == ("global.pair_equalization.p0", "global.pair_equalization.p1")
    assert candidate.NAMES[-1] == "q3.pair_equalization.p1"


def test_generated_forward_backward_keeps_parameters_unchanged():
    torch = pytest.importorskip("torch")
    with torch.random.fork_rng():
        torch.manual_seed(20261022)
        model = candidate.network()
    assert model.architecture == candidate.DOMAIN
    assert sum(p.numel() for p in model.parameters()) == 418
    assert all(p.dtype == torch.float32 for p in model.parameters())
    before = [p.detach().clone() for p in model.parameters()]
    values = torch.from_numpy(candidate.inputs(wav(np.arange(160)))).requires_grad_(True)
    model.train()
    logits = model(values)
    model.eval()
    assert logits.shape == (1, 2) and torch.equal(logits.detach(), model(values).detach())
    torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
    assert values.grad is not None and bool(torch.isfinite(values.grad).all())
    assert all(
        p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()
    )
    assert any(bool((p.grad != 0).any()) for p in model.parameters())
    assert all(torch.equal(p, saved) for p, saved in zip(model.parameters(), before, strict=True))


def test_missing_torch_is_explicitly_unavailable(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="unavailable.*optional Torch missing"):
        candidate.network()
