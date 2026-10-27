"""Generated numeric state: all RNGs, order, optimizer slots and hostile extents."""

import copy
import random
import struct

import numpy as np
import pytest

from core import colab_artifacts as a
from core import spatial_colab as p
from core import spatial_colab_state as s

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def snapshot():
    with torch.random.fork_rng(devices=[]):
        model = p.models.network(p.DOMAINS[0])
        optimizer = torch.optim.AdamW(model.parameters(), foreach=False, fused=False)
        meta, arrays = s.capture(
            model,
            optimizer,
            source_sha256="a" * 64,
            data_sha256="b" * 64,
            order=np.array([2, 1, 0], dtype="<i8"),
            consumption={
                "epoch": 0,
                "cursor": 1,
                "optimizer_updates": 0,
                "presentations": 1,
                "elapsed_seconds": 12.5,
                "output_bytes": 7,
            },
        )
    return meta, arrays


def load(path, meta, checksum):
    return s.load(
        path,
        checksum=checksum,
        environment=meta["environment"],
        source_sha256=meta["source_sha256"],
        data_sha256=meta["data_sha256"],
    )


@pytest.mark.parametrize("domain", p.DOMAINS)
def test_complete_generated_roundtrip(domain, tmp_path):
    py_rng, np_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            model = p.models.network(domain).train()
            optimizer = torch.optim.AdamW(model.parameters(), foreach=False, fused=False)
            # Numeric synthetic slots exercise nonempty state without optimizer.step().
            for parameter in model.parameters():
                optimizer.state[parameter] = {
                    "step": torch.tensor(3.0),
                    "exp_avg": torch.full_like(parameter, 0.125),
                    "exp_avg_sq": torch.full_like(parameter, 0.25),
                }
            meta, arrays = s.capture(
                model,
                optimizer,
                source_sha256="a" * 64,
                data_sha256="b" * 64,
                order=np.array([1, 0], dtype="<i8"),
                consumption={
                    "epoch": 1,
                    "cursor": 1,
                    "optimizer_updates": 3,
                    "presentations": 6,
                    "elapsed_seconds": 7,
                    "output_bytes": 8,
                },
            )
            expected = random.random(), np.random.random(), torch.rand(2)  # noqa: S311
            path = tmp_path / "state.numeric"
            checksum = s.save(path, meta, arrays)
            restored_meta, restored_arrays = load(path, meta, checksum)
            for key in arrays:
                assert np.array_equal(arrays[key], restored_arrays[key])
            with torch.no_grad():
                next(model.parameters()).zero_()
            optimizer.state.clear()
            result = s.restore(model, optimizer, restored_meta, restored_arrays)
            assert random.random() == expected[0] and np.random.random() == expected[1]  # noqa: S311
            assert torch.equal(torch.rand(2), expected[2])
            assert result["consumption"] == meta["consumption"]
            assert np.array_equal(result["order"], [1, 0])
            for name, parameter in model.named_parameters():
                assert np.array_equal(parameter.detach().numpy(), arrays["model/" + name])
                assert float(optimizer.state[parameter]["step"]) == 3
                assert bool((optimizer.state[parameter]["exp_avg_sq"] == 0.25).all())
            with pytest.raises(FileExistsError):
                s.save(path, meta, arrays)
    finally:
        random.setstate(py_rng)
        np.random.set_state(np_rng)


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema", "old"),
        ("domain", "jpeg"),
        ("source_sha256", "invalid"),
        ("environment", {}),
        ("consumption", {}),
        ("rows", 0),
        ("rng_bytes", {}),
        ("numpy_rng", []),
        ("python_gauss", float("nan")),
        ("optimizer", {}),
        ("initialized", ["missing"]),
        ("training", 1),
    ],
)
def test_metadata_rejects(snapshot, key, value):
    meta, _ = snapshot
    bad = copy.deepcopy(meta)
    bad[key] = value
    with pytest.raises(ValueError):
        s.specs(bad)


@pytest.mark.parametrize(
    "fault",
    [
        "fields",
        "env-value",
        "count",
        "time",
        "cursor",
        "rng-size",
        "device-rng",
        "numpy-position",
        "boolean-option",
        "optimizer-nan",
        "betas",
        "empty-updates",
        "array-keys",
        "array-dtype",
        "nonfinite",
        "variance",
        "order",
        "python-index",
    ],
)
def test_state_adversaries(snapshot, fault):
    meta, original = snapshot
    meta, arrays = copy.deepcopy(meta), dict(original)
    if fault == "fields":
        meta["unexpected"] = "reject extra fields"
    elif fault == "env-value":
        meta["environment"]["python"] = []
    elif fault == "count":
        meta["consumption"]["presentations"] = -1
    elif fault == "time":
        meta["consumption"]["elapsed_seconds"] = float("nan")
    elif fault == "cursor":
        meta["consumption"]["cursor"] = 4
    elif fault == "rng-size":
        meta["rng_bytes"]["cpu"] = 2**32
    elif fault == "device-rng":
        meta["rng_bytes"]["cuda"] = 16
    elif fault == "numpy-position":
        meta["numpy_rng"][0] = 625
    elif fault == "boolean-option":
        meta["optimizer"]["fused"] = True
    elif fault == "optimizer-nan":
        meta["optimizer"]["lr"] = float("nan")
    elif fault == "betas":
        meta["optimizer"]["betas"] = [1, 0]
    elif fault == "empty-updates":
        meta["consumption"]["optimizer_updates"] = 1
    elif fault == "array-keys":
        arrays.pop("rng/cpu")
    elif fault == "array-dtype":
        arrays["data/order"] = arrays["data/order"].astype("f8")
    elif fault == "nonfinite":
        arrays["model/classifier.weight"] = np.full((2, 512), float("nan"), dtype="<f4")
    elif fault == "variance":
        arrays["model/front.0.1.running_var"] = np.full((64,), -1, dtype="<f4")
    elif fault == "order":
        arrays["data/order"] = np.array([0, 0, 2], dtype="<i8")
    elif fault == "python-index":
        arrays["rng/python"] = arrays["rng/python"].copy()
        arrays["rng/python"][-1] = 625
    with pytest.raises(ValueError):
        s.validate(meta, arrays)


@pytest.mark.parametrize(
    "fault", ["hash", "magic", "short", "extent", "metadata", "env", "source", "data", "trailing"]
)
def test_load_rejects_before_numeric_allocation(snapshot, tmp_path, monkeypatch, fault):
    meta, arrays = snapshot
    path = tmp_path / "good"
    checksum = s.save(path, meta, arrays)
    raw = path.read_bytes()
    if fault == "magic":
        raw = b"BADMAGIC" + raw[8:]
    elif fault == "short":
        raw = s.MAGIC + b"xx"
    elif fault == "extent":
        raw = raw[:-1]
    elif fault == "metadata":
        raw = s.MAGIC + struct.pack("<I", s.MAX_META + 1)
    elif fault == "trailing":
        raw += b"x"
    target = tmp_path / "bad"
    target.write_bytes(raw)
    environment = dict(meta["environment"])
    if fault == "env":
        environment["torch"] = "wrong"
    monkeypatch.setattr(
        s.np, "frombuffer", lambda *args, **kwargs: pytest.fail("unsafe allocation")
    )
    with pytest.raises(ValueError):
        s.load(
            target,
            checksum="0" * 64 if fault == "hash" else a.digest(raw),
            environment=environment,
            source_sha256="0" * 64 if fault == "source" else "a" * 64,
            data_sha256="0" * 64 if fault == "data" else "b" * 64,
        )
    assert checksum == a.digest(path.read_bytes())


def test_empty_optimizer_restore_and_target_gates(snapshot, tmp_path, monkeypatch):
    meta, arrays = snapshot
    model = p.models.network(p.DOMAINS[0])
    opt = torch.optim.AdamW(model.parameters(), foreach=False, fused=False)
    s.restore(model, opt, meta, arrays)
    assert not opt.state
    with pytest.raises(ValueError, match="one AdamW"):
        s.optimizer_options(model, torch.optim.SGD(model.parameters(), lr=0.1))
    reverse = torch.optim.AdamW(list(model.parameters())[::-1], foreach=False, fused=False)
    with pytest.raises(ValueError, match="order"):
        s.optimizer_options(model, reverse)
    opt.param_groups[0]["lr"] = 0.123
    with pytest.raises(ValueError, match="target"):
        s.restore(model, opt, meta, arrays)
    opt.param_groups[0]["lr"] = meta["optimizer"]["lr"]
    with pytest.raises(ValueError, match="model tensors"):
        s.restore(model.half(), opt, meta, arrays)
    model.float()
    altered = copy.deepcopy(meta)
    altered["rng_bytes"]["cpu"] += 1
    altered_arrays = {**arrays, "rng/cpu": np.zeros(altered["rng_bytes"]["cpu"], dtype="u1")}
    with pytest.raises(ValueError, match="RNG size"):
        s.restore(model, opt, altered, altered_arrays)
    monkeypatch.setattr(s, "MAX_META", 4)
    with pytest.raises(ValueError, match="header size"):
        s.save(tmp_path / "large", meta, arrays)


def test_generated_probe_zero_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **k: pytest.fail("optimizer called"))
    report = s.generated_probe(tmp_path / "probe")
    assert report["status"] == "ready" and report["optimizer_updates"] == 0
    assert len(report["states"]) == 2 and report["generated_only"]


def test_capture_rejects_unknown_slot_and_rng(monkeypatch):
    model = p.models.network(p.DOMAINS[0])
    opt = torch.optim.AdamW(model.parameters(), foreach=False, fused=False)
    kwargs = dict(
        source_sha256="a" * 64,
        data_sha256="b" * 64,
        order=np.array([0], dtype="<i8"),
        consumption={},
    )
    model.front.eval()
    with pytest.raises(ValueError, match="uniform"):
        s.capture(model, opt, **kwargs)
    model.train()
    opt.state[next(model.parameters())] = {"unknown": torch.tensor(0)}
    with pytest.raises(ValueError, match="slot"):
        s.capture(model, opt, **kwargs)
    monkeypatch.setattr(np.random, "get_state", lambda: {"bit_generator": "unknown"})
    with pytest.raises(ValueError, match="MT19937"):
        s.capture(model, opt, **kwargs)
