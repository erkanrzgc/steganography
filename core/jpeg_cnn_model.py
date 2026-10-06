"""Bounded numeric-only serialization for the fixed pixel CNN; no pickle."""

from __future__ import annotations

import hashlib
import io
import math
import zipfile
from pathlib import Path

import numpy as np

from core import jpeg_cnn as cnn

MAX_BYTES = 1024**2
SHAPES = {
    "filters": (3, 1, 3, 3),
    "layers.0.weight": (8, 3, 3, 3),
    "layers.0.bias": (8,),
    "layers.3.weight": (16, 8, 3, 3),
    "layers.3.bias": (16,),
    "layers.6.weight": (16, 16, 3, 3),
    "layers.6.bias": (16,),
    "layers.10.weight": (1, 16),
    "layers.10.bias": (1,),
}


def validate(arrays):
    if set(arrays) != set(SHAPES) or any(
        arrays[k].shape != shape
        or arrays[k].dtype != np.dtype("<f4")
        or not np.isfinite(arrays[k]).all()
        for k, shape in SHAPES.items()
    ):
        raise ValueError("pixel model dimensions/dtype/parameters mismatch")
    if not np.array_equal(arrays["filters"], np.array(cnn.FILTERS, dtype="<f4")[:, None]):
        raise ValueError("pixel model fixed filters changed")


def save_model(model, path: Path) -> str:
    if path.exists() or any(p.is_symlink() for p in (path, *path.parents)):
        raise FileExistsError("pixel model output exists or uses a symlink")
    arrays = {k: v.detach().cpu().numpy().astype("<f4") for k, v in model.state_dict().items()}
    validate(arrays)
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    data = buffer.getvalue()
    if len(data) > MAX_BYTES:
        raise ValueError("pixel model exceeds byte limit")
    with path.open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def load_model(path: Path, *, checksum: str):
    if (
        not path.is_file()
        or any(p.is_symlink() for p in (path, *path.parents))
        or path.stat().st_size > MAX_BYTES
    ):
        raise ValueError("pixel model must be bounded, regular and non-symlink")
    with path.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != checksum:
        raise ValueError("pixel model checksum/byte limit failure")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if (
            len(members) != len(SHAPES)
            or {m.filename for m in members} != {k + ".npy" for k in SHAPES}
            or sum(m.file_size for m in members) > MAX_BYTES
        ):
            raise ValueError("pixel model archive contract mismatch")
        for member in members:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(
                        stream, max_header_size=4096
                    )
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(
                        stream, max_header_size=4096
                    )
                else:
                    raise ValueError("unsupported pixel model array header")
                if (
                    shape != SHAPES[member.filename[:-4]]
                    or dtype != np.dtype("<f4")
                    or stream.tell() + math.prod(shape) * dtype.itemsize != member.file_size
                ):
                    raise ValueError("pixel model array header/size mismatch")
    with np.load(io.BytesIO(data), allow_pickle=False, max_header_size=4096) as archive:
        arrays = {k: archive[k] for k in SHAPES}
    validate(arrays)
    import torch

    model = cnn.network()
    model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in arrays.items()}, strict=True)
    return model.eval()
