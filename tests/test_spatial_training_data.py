"""Independent payload, grouping and bounds for generated spatial preparation."""

import dataclasses

import numpy as np
import pytest

from core import spatial_training_data as d


@pytest.mark.parametrize("mode,channels", [("L", 1), ("RGB", 3)])
def test_scattered_payload_scalar_oracle_and_single_lsb(mode, channels):
    original = (
        np.arange(256 * 256 * channels, dtype=np.uint32).astype("u1").reshape(256, 256, channels)
    )
    before = original.copy()
    payload = bytes(i % 251 for i in range((original.size * 2 // 5) // 8))
    control, selected = d.embed(original, payload, 19)
    assert np.array_equal(before, original)
    assert np.array_equal(control & 254, original & 254)
    assert np.max(np.abs(control.astype(int) - original.astype(int))) == 1
    untouched = np.ones(original.size, dtype=bool)
    untouched[selected] = False
    assert np.array_equal(control.ravel()[untouched], original.ravel()[untouched])
    assert d.recover(control, selected) == payload
    flat = control.ravel()
    for i in range(32):
        assert int(flat[selected[i]]) & 1 == (payload[i // 8] >> (7 - i % 8)) & 1
    other, other_positions = d.embed(original, payload, 20)
    assert not np.array_equal(selected, other_positions)
    assert d.recover(other, other_positions) == payload


@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_generated_sources_reproducible_disjoint_and_fp32(mode):
    train = d.GeneratedSpec(mode, 2026102801, "train", 2)
    dev = d.GeneratedSpec(mode, 2026102801, "development", 2)
    first, second = d.manifest(train), d.manifest(dev)
    assert first == d.manifest(train) and first["generated_only"]
    assert not {r["group"] for r in first["sources"]} & {r["group"] for r in second["sources"]}
    assert first["sources"][0] == d.manifest(dataclasses.replace(train, originals=1))["sources"][0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        train.seed = 4
    x, labels = d.batch(train, (1, 0))
    assert x.shape == (4, 1 if mode == "L" else 3, 256, 256)
    assert x.dtype == np.dtype("<f4") and labels.tolist() == [0, 1, 0, 1]
    assert np.array_equal(x.astype("u1").astype("<f4"), x)
    cover, control, record = d.pair(train, 1)
    assert not (cover & 1).any() and (control & 1).any()
    assert record["payload_bytes"] == (cover.size * 2 // 5) // 8
    assert train.domain != dataclasses.replace(train, mode="RGB" if mode == "L" else "L").domain


@pytest.mark.parametrize("value", [-1, 2**32, True, 1.5, "7"])
def test_seed_bounds(value):
    with pytest.raises(ValueError):
        d.seed(value)


@pytest.mark.parametrize(
    "edit",
    [
        {"mode": "RGBA"},
        {"split": "test"},
        {"split": "unassigned"},
        {"originals": 0},
        {"originals": 129},
        {"originals": True},
    ],
)
def test_spec_limits(edit):
    with pytest.raises(ValueError):
        d.GeneratedSpec(**{"mode": "L", "seed": 8, "split": "train", "originals": 1, **edit})


@pytest.mark.parametrize(
    "value",
    [
        None,
        np.zeros((256, 256), dtype="u1"),
        np.zeros((255, 256, 1), dtype="u1"),
        np.zeros((256, 256, 4), dtype="u1"),
        np.zeros((256, 256, 1), dtype="uint16"),
    ],
)
def test_native_depth_channel_size_not_silently_converted(value):
    with pytest.raises(ValueError):
        d.pixels(value)


def test_invalid_payload_positions_and_sources():
    x = np.zeros((256, 256, 1), dtype="u1")
    selected = d.positions(x.size, 0)
    for value in (None, bytearray(len(selected) // 8), b"", bytes(len(selected) // 8 + 1)):
        with pytest.raises(ValueError):
            d.embed(x, value, 0)
    for value in (
        None,
        selected.astype("int32"),
        selected[:-1],
        np.zeros_like(selected),
        np.full_like(selected, -1),
        np.full_like(selected, x.size),
    ):
        with pytest.raises(ValueError):
            d.recover(x, value)
    for size in (0, 65537, True):
        with pytest.raises(ValueError):
            d.positions(size, 0)
    spec = d.GeneratedSpec("L", 1, "train", 2)
    for value in (None, {}, "native-path"):
        with pytest.raises(ValueError):
            d.manifest(value)
        with pytest.raises(ValueError):
            d.pair(value, 0)
        with pytest.raises(ValueError):
            d.batch(value, (0,))
    for index in (-1, 2, True):
        with pytest.raises(ValueError):
            d.pair(spec, index)
    for indices in ([], (), (0, 0), (2,), (-1,), (True,), (0, 1, 0, 1, 0)):
        with pytest.raises(ValueError):
            d.batch(spec, indices)


def test_failed_independent_recovery_cannot_make_generated_pair(monkeypatch):
    monkeypatch.setattr(d, "recover", lambda *args: b"wrong")
    with pytest.raises(ValueError, match="recovery"):
        d.pair(d.GeneratedSpec("L", 1, "train", 1), 0)


def test_fp32_adapter_rejects_fractional_samples(monkeypatch):
    fractional = np.full((256, 256, 1), 128.125, dtype=np.float64)
    monkeypatch.setattr(d, "pair", lambda *_: (fractional, fractional, {}))
    with pytest.raises(ValueError, match="FP32"):
        d.batch(d.GeneratedSpec("L", 1, "train", 1), (0,))
