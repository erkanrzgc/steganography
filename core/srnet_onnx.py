"""Explicit bounded research ONNX export; no model installation."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np

from core import srnet, srnet_model
from core.srnet_onnx_guard import preflight


def export_model(model, out: Path):
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("SRNet export output exists or uses a symlink")
    if getattr(model, "architecture", None) != srnet.ARCHITECTURE or any(
        m.training for m in model.modules()
    ):
        raise ValueError("SRNet export requires correct architecture and all-stage eval")
    srnet_model.validate({k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    try:
        import onnx
        import torch
    except ImportError as exc:
        raise RuntimeError("SRNet export requires optional research dependencies") from exc
    buffer = io.BytesIO()
    torch.onnx.export(
        model,
        (torch.zeros((1, 1, 256, 256), dtype=torch.float32, device="cpu"),),
        buffer,  # type: ignore[arg-type]  # Legacy dynamo=False accepts binary file objects
        input_names=["pixels"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"pixels": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    data = buffer.getvalue()
    if len(data) > srnet_model.MAX_BYTES:
        raise ValueError("SRNet ONNX export exceeds byte limit")
    onnx.checker.check_model(onnx.load_model_from_string(data))
    with out.open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def onnx_logits(path: Path, pixels, *, checksum: str):
    if (
        not path.is_file()
        or any(p.is_symlink() for p in (path, *path.parents))
        or path.stat().st_size > srnet_model.MAX_BYTES
    ):
        raise ValueError("SRNet ONNX must be bounded regular non-symlink")
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.dtype("<f4")
        or pixels.ndim != 4
        or pixels.shape[1:] != (1, 256, 256)
        or not 1 <= len(pixels) <= 4
        or not np.isfinite(pixels).all()
        or np.any(np.abs(pixels) > 2**36)
    ):
        raise ValueError("invalid bounded ONNX float256 input")
    with path.open("rb") as stream:
        data = stream.read(srnet_model.MAX_BYTES + 1)
    if len(data) > srnet_model.MAX_BYTES or hashlib.sha256(data).hexdigest() != checksum:
        raise ValueError("SRNet ONNX checksum/byte limit failure")
    preflight(data)
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("SRNet replay requires optional ml dependency") from exc
    settings = ort.SessionOptions()
    settings.intra_op_num_threads, settings.inter_op_num_threads = 2, 1
    session = ort.InferenceSession(data, sess_options=settings, providers=["CPUExecutionProvider"])
    result = session.run(["logits"], {"pixels": pixels})[0]
    if result.shape != (len(pixels), 2) or not np.isfinite(result).all():
        raise ValueError("invalid ONNX logits")
    return result
