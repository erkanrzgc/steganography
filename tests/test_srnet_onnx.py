"""Optional CPU export/replay, eval guards and hostile input bounds."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from core import srnet, srnet_onnx, srnet_reference


def test_dynamic_batch_export_and_independent_replay(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261014)
            model = srnet.network().eval()
        out = tmp_path / "model.onnx"
        checksum = srnet_onnx.export_model(model, out)
        raw = np.random.default_rng(91).normal(0.125, 0.25, (4, 1, 256, 256)).astype("<f4")
        for batch in (1, 2, 4):
            native = srnet.float_logits(model, raw[:batch])
            exported = srnet_onnx.onnx_logits(out, raw[:batch], checksum=checksum)
            assert all(
                srnet_reference.compare(native[i : i + 1], exported[i : i + 1])["passed"]
                for i in range(batch)
            )
        with pytest.raises(FileExistsError):
            srnet_onnx.export_model(model, out)
        for invalid in (
            raw.astype("f8"),
            raw[:, :, :128, :128],
            np.full_like(raw, float("nan")),
            np.concatenate([raw, raw]),
        ):
            with pytest.raises(ValueError):
                srnet_onnx.onnx_logits(out, invalid, checksum=checksum)
        with pytest.raises(ValueError, match="checksum"):
            srnet_onnx.onnx_logits(out, raw[:1], checksum="0" * 64)
        link = tmp_path / "link.onnx"
        link.symlink_to(out)
        with pytest.raises(ValueError):
            srnet_onnx.onnx_logits(link, raw[:1], checksum=checksum)
        with pytest.raises(FileExistsError):
            srnet_onnx.export_model(model, link)
        model.front[0][1].train()
        with pytest.raises(ValueError, match="eval"):
            srnet_onnx.export_model(model, tmp_path / "train.onnx")
        model.eval()
        monkeypatch.setattr(srnet_onnx.srnet_model, "MAX_BYTES", 16)
        with pytest.raises(ValueError, match="byte limit"):
            srnet_onnx.export_model(model, tmp_path / "large.onnx")
        with pytest.raises(ValueError, match="bounded"):
            srnet_onnx.onnx_logits(out, raw[:1], checksum=checksum)
    finally:
        torch.set_num_threads(previous)


def test_optional_absence_and_invalid_runtime_output(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    model = srnet.network().eval()
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "onnx", None)
        with pytest.raises(RuntimeError, match="optional research"):
            srnet_onnx.export_model(model, tmp_path / "absent.onnx")
    path = tmp_path / "fake.onnx"
    path.write_bytes(b"bounded")
    checksum = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    raw = np.zeros((1, 1, 256, 256), dtype="<f4")
    monkeypatch.setattr(srnet_onnx, "preflight", lambda _: None)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "onnxruntime", None)
        with pytest.raises(RuntimeError, match="optional ml"):
            srnet_onnx.onnx_logits(path, raw, checksum=checksum)
    fake = SimpleNamespace(
        SessionOptions=lambda: SimpleNamespace(),
        InferenceSession=lambda *a, **k: SimpleNamespace(
            run=lambda *a: [np.full((1, 2), float("nan"))]
        ),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    with pytest.raises(ValueError, match="invalid ONNX logits"):
        srnet_onnx.onnx_logits(path, raw, checksum=checksum)
