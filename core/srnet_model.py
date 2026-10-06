"""Exact bounded numeric SRNet weights/BN state; never pickle or code."""

from __future__ import annotations

import hashlib
import io
import math
import zipfile
from pathlib import Path

import numpy as np

from core import srnet

MAX_BYTES = 32 * 1024**2


def shapes():
    result: dict[str, tuple[int, ...]] = {}

    def block(prefix, before, after, kernel=3):
        result[prefix + ".0.weight"] = (after, before, kernel, kernel)
        result[prefix + ".0.bias"] = (after,)
        for suffix in ("weight", "bias", "running_mean", "running_var"):
            result[prefix + ".1." + suffix] = (after,)
        result[prefix + ".1.num_batches_tracked"] = ()

    block("front.0", 1, 64)
    block("front.1", 64, 16)
    for i in range(2, 7):
        for branch in (0, 1):
            block(f"front.{i}.branch.{branch}", 16, 16)
    for i, (before, after) in enumerate(((16, 16), (16, 64), (64, 128), (128, 256))):
        block(f"middle.{i}.branch.0", before, after)
        block(f"middle.{i}.branch.1", after, after)
        block(f"middle.{i}.skip", before, after, kernel=1)
    block("tail.0", 256, 512)
    block("tail.1", 512, 512)
    result["classifier.weight"] = (2, 512)
    return result


SHAPES = shapes()


def dtype(name):
    return np.dtype("<i8" if name.endswith("num_batches_tracked") else "<f4")


def validate(arrays):
    if set(arrays) != set(SHAPES):
        raise ValueError("SRNet numeric state keys mismatch")
    for name, shape in SHAPES.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype(name) or not np.isfinite(value).all():
            raise ValueError("SRNet numeric state shape/dtype/finite mismatch")
        if name.endswith("running_var") and np.any(value < 0):
            raise ValueError("SRNet BN running variance is negative")
        if name.endswith("num_batches_tracked") and not 0 <= int(value) <= 10_000_000:
            raise ValueError("SRNet BN counter outside limits")


def save_model(model, path: Path):
    if path.exists() or any(p.is_symlink() for p in (path, *path.parents)):
        raise FileExistsError("SRNet output exists or uses a symlink")
    if getattr(model, "architecture", None) != srnet.ARCHITECTURE or any(
        m.training for m in model.modules()
    ):
        raise ValueError("SRNet snapshot requires correct architecture and all-stage eval")
    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    validate(arrays)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    data = buffer.getvalue()
    if len(data) > MAX_BYTES:
        raise ValueError("SRNet numeric state exceeds byte limits")
    with path.open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def load_model(path: Path, *, checksum: str):
    if (
        not path.is_file()
        or any(p.is_symlink() for p in (path, *path.parents))
        or path.stat().st_size > MAX_BYTES
    ):
        raise ValueError("SRNet model must be bounded regular non-symlink")
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != checksum:
        raise ValueError("SRNet model checksum/byte limit failure")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if (
            len(members) != len(SHAPES)
            or {m.filename for m in members} != {k + ".npy" for k in SHAPES}
            or sum(m.file_size for m in members) > MAX_BYTES
        ):
            raise ValueError("SRNet archive contract mismatch")
        for member in members:
            name = member.filename[:-4]
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, kind = np.lib.format.read_array_header_1_0(
                        stream, max_header_size=4096
                    )
                elif version == (2, 0):
                    shape, _, kind = np.lib.format.read_array_header_2_0(
                        stream, max_header_size=4096
                    )
                else:
                    raise ValueError("unsupported SRNet array header")
                if (
                    shape != SHAPES[name]
                    or kind != dtype(name)
                    or stream.tell() + math.prod(shape) * kind.itemsize != member.file_size
                ):
                    raise ValueError("SRNet array header/size mismatch")
    with np.load(io.BytesIO(data), allow_pickle=False, max_header_size=4096) as archive:
        arrays = {k: archive[k] for k in SHAPES}
    validate(arrays)
    import torch

    with torch.random.fork_rng(devices=[]):
        model = srnet.network()
    model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in arrays.items()}, strict=True)
    return model.eval()
