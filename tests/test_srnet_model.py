"""Bounded SRNet numeric persistence, BN preservation and hostile headers."""

import io
import zipfile

import numpy as np
import pytest

from core import srnet
from core import srnet_model as storage
from tests.test_jrm_reference import sha

torch = pytest.importorskip("torch")


@pytest.fixture
def model():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(31)
        net = srnet.network()
    # BN values deliberately differ from defaults to prove persisted inference.
    for name, value in net.named_buffers():
        if name.endswith("running_mean"):
            value.fill_(0.25)
        if name.endswith("running_var"):
            value.fill_(1.5)
        if name.endswith("num_batches_tracked"):
            value.fill_(9)
    yield net.eval()
    torch.set_num_threads(previous)


def test_numeric_bn_roundtrip_and_rng_restore(model, tmp_path):
    assert set(storage.SHAPES) == set(model.state_dict()) and len(storage.SHAPES) == 183
    path = tmp_path / "model.npz"
    checksum = storage.save_model(model, path)
    rng = torch.get_rng_state().clone()
    loaded = storage.load_model(path, checksum=checksum)
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])
    raw = np.full((1, 1, 256, 256), 128.125, dtype="<f4")
    np.testing.assert_array_equal(srnet.float_logits(model, raw), srnet.float_logits(loaded, raw))
    with pytest.raises(FileExistsError):
        storage.save_model(model, path)
    with pytest.raises(ValueError, match="checksum"):
        storage.load_model(path, checksum="0" * 64)
    link = tmp_path / "link.npz"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="non-symlink"):
        storage.load_model(link, checksum=checksum)
    with pytest.raises(FileExistsError):
        storage.save_model(model, link)


@pytest.mark.parametrize(
    "fault", ["keys", "shape", "dtype", "nan", "variance", "counter", "negative-counter"]
)
def test_numeric_state_validation(model, fault):
    arrays = {k: v.numpy().copy() for k, v in model.state_dict().items()}
    if fault == "keys":
        arrays.pop("classifier.weight")
    if fault == "shape":
        arrays["classifier.weight"] = np.ones((1, 512), dtype="<f4")
    if fault == "dtype":
        arrays["classifier.weight"] = arrays["classifier.weight"].astype("f8")
    if fault == "nan":
        arrays["classifier.weight"][0, 0] = float("nan")
    if fault == "variance":
        arrays["front.0.1.running_var"][0] = -1
    if fault == "counter":
        arrays["front.0.1.num_batches_tracked"] = np.array(10_000_001, dtype="<i8")
    if fault == "negative-counter":
        arrays["front.0.1.num_batches_tracked"] = np.array(-1, dtype="<i8")
    with pytest.raises(ValueError):
        storage.validate(arrays)


def test_save_stage_architecture_and_limits(model, tmp_path, monkeypatch):
    model.front[0][1].train()
    with pytest.raises(ValueError, match="eval"):
        storage.save_model(model, tmp_path / "train.npz")
    model.eval()
    model.architecture = "unknown"
    with pytest.raises(ValueError, match="architecture"):
        storage.save_model(model, tmp_path / "unknown.npz")
    model.architecture = srnet.ARCHITECTURE
    monkeypatch.setattr(storage, "MAX_BYTES", 128)
    with pytest.raises(ValueError, match="byte limits"):
        storage.save_model(model, tmp_path / "large.npz")
    assert not (tmp_path / "large.npz").exists()
    path = tmp_path / "oversized.npz"
    path.write_bytes(bytes(129))
    with pytest.raises(ValueError, match="bounded"):
        storage.load_model(path, checksum=sha(path))


@pytest.mark.parametrize(
    "fault", ["archive", "giant", "object", "truncated", "version", "expanded"]
)
def test_hostile_array_preflight(model, tmp_path, monkeypatch, fault):
    valid = tmp_path / "valid.npz"
    storage.save_model(model, valid)
    target = tmp_path / "invalid.npz"
    with (
        zipfile.ZipFile(valid) as source,
        zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as out,
    ):
        for member in source.infolist():
            raw = source.read(member)
            if member.filename == "classifier.weight.npy":
                if fault == "archive":
                    continue
                if fault in {"giant", "object"}:
                    header = io.BytesIO()
                    np.lib.format.write_array_header_1_0(
                        header,
                        {
                            "descr": "|O" if fault == "object" else "<f4",
                            "fortran_order": False,
                            "shape": (2, 512) if fault == "object" else (2**40, 512),
                        },
                    )
                    raw = header.getvalue()
                if fault == "truncated":
                    raw = raw[:-1]
                if fault == "version":
                    raw = raw[:6] + b"\x03\x00" + raw[8:]
                if fault == "expanded":
                    raw = bytes(storage.MAX_BYTES + 1)
            out.writestr(member.filename, raw)
    monkeypatch.setattr(
        storage.np, "load", lambda *a, **k: pytest.fail("hostile header reached NumPy loading")
    )
    with pytest.raises(ValueError):
        storage.load_model(target, checksum=sha(target))


def test_npy_v2_supported(model, tmp_path):
    path = tmp_path / "v2.npz"
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in model.state_dict().items():
            data = value.numpy()
            buffer = io.BytesIO()
            np.lib.format.write_array_header_2_0(
                buffer, {"descr": data.dtype.str, "fortran_order": False, "shape": data.shape}
            )
            buffer.write(data.tobytes())
            archive.writestr(name + ".npy", buffer.getvalue())
    loaded = storage.load_model(path, checksum=sha(path))
    assert torch.equal(loaded.front[0][1].running_var, model.front[0][1].running_var)
