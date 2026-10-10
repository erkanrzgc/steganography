"""Explicit experimental eight-group SRNet variant; legacy BN stays unchanged."""

from __future__ import annotations

import hashlib
import io

import numpy as np

from core import srnet, srnet_model

ARCHITECTURE = "srnet-gray12-groupnorm8-research-v1"
SHAPES = {
    k: v
    for k, v in srnet_model.SHAPES.items()
    if not k.endswith(("running_mean", "running_var", "num_batches_tracked"))
}


def network():
    import torch

    model = srnet.network()
    count = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.BatchNorm2d):
                if child.num_features % 8 or child.weight is None or child.bias is None:
                    raise ValueError("GN requires tracked affine divisible channel layers")
                norm = torch.nn.GroupNorm(
                    8,
                    child.num_features,
                    eps=child.eps,
                    affine=True,
                    dtype=torch.float32,
                    device="cpu",
                )
                norm.load_state_dict({"weight": child.weight.detach(), "bias": child.bias.detach()})
                setattr(parent, name, norm)
                count += 1
    if count != 26:
        raise ValueError("GN requires the exact 26-layer architecture")
    model.architecture = ARCHITECTURE
    validate(model.state_dict())
    return model


def validate(state):
    import torch

    if set(state) != set(SHAPES):
        raise ValueError("GN numeric keys mismatch")
    floats, device = [], None
    for name, shape in SHAPES.items():
        value = state[name]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != torch.float32
            or value.layout != torch.strided
            or value.device.type not in {"cpu", "cuda"}
            or value.device.index not in {None, 0}
            or (device is not None and device != value.device)
        ):
            raise ValueError("GN numeric shape/dtype/device mismatch")
        device = value.device
        floats.append(value.detach().reshape(-1))
    with torch.no_grad():
        if not bool(torch.isfinite(torch.cat(floats)).all()):
            raise ValueError("GN numeric state nonfinite")


def parameter_sha(model):
    """Bind the common learned tensors, excluding architecture-specific buffers."""
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        digest.update(name.encode() + b"\0")
        digest.update(value.detach().cpu().numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()


def save_model(model, path):
    if path.exists() or any(p.is_symlink() for p in (path, *path.parents)):
        raise FileExistsError("GN snapshot exists or uses symlink")
    if getattr(model, "architecture", None) != ARCHITECTURE or any(
        m.training for m in model.modules()
    ):
        raise ValueError("GN snapshot requires the distinct eval architecture")
    validate(model.state_dict())
    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    raw = buffer.getvalue()
    if len(raw) > srnet_model.MAX_BYTES:
        raise ValueError("GN snapshot exceeds bound")
    # Round-trip only these just-produced, bounded bytes, never an untrusted archive.
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        if any(not np.array_equal(arrays[k], archive[k]) for k in SHAPES):
            raise ValueError("GN snapshot round-trip mismatch")
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()
