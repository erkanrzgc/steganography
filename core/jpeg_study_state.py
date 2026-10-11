"""New recipe-bound numeric snapshots; never reinterpret legacy checkpoints."""

from __future__ import annotations

import hashlib
import io
import json
import math
import stat
import zipfile

import numpy as np

from core import srnet_model
from core.jpeg_scale import identity
from core.srnet_checkpoint import MAX_BYTES, MAX_META, PARAMETERS, header
from core.srnet_stream import regular_open


def capture(model, optimizer, binding, presented_rows):
    arrays = {"model/" + k: v.detach().cpu().numpy().copy()
              for k, v in model.state_dict().items()}
    for name, parameter in model.named_parameters():
        for slot in ("exp_avg", "exp_inf", "step"):
            arrays[f"optimizer/{name}/{slot}"] = (
                optimizer.state[parameter][slot].detach().cpu().numpy().astype("<f4", copy=True)
            )
    result = {"metadata": {"schema_version": "jpeg-study-snapshot-v1",
                           "binding": binding, "presented_rows": presented_rows},
              "arrays": arrays}
    validate(result)
    return result


def specs():
    result = {"model/" + k: (s, srnet_model.dtype(k)) for k, s in srnet_model.SHAPES.items()}
    for name, shape in PARAMETERS.items():
        for slot in ("exp_avg", "exp_inf", "step"):
            result[f"optimizer/{name}/{slot}"] = (
                () if slot == "step" else shape, np.dtype("<f4")
            )
    return result


def validate(snapshot):
    if not isinstance(snapshot, dict) or set(snapshot) != {"metadata", "arrays"}:
        raise ValueError("study snapshot container invalid")
    meta = snapshot["metadata"]
    if (not isinstance(meta, dict)
        or set(meta) != {"schema_version", "binding", "presented_rows"}
        or meta["schema_version"] != "jpeg-study-snapshot-v1"
        or type(meta["presented_rows"]) is not int
        or meta["presented_rows"] not in {16384, 32768}):
        raise ValueError("study snapshot metadata invalid")
    identity(meta["binding"])
    arrays = snapshot["arrays"]
    expected = specs()
    if not isinstance(arrays, dict) or set(arrays) != set(expected):
        raise ValueError("study snapshot numeric keys invalid")
    steps = set()
    for name, (shape, dtype) in expected.items():
        value = arrays[name]
        if (not isinstance(value, np.ndarray) or value.shape != shape
            or value.dtype != dtype or not np.isfinite(value).all()):
            raise ValueError("study snapshot numeric shape/dtype/finite invalid")
        if name.endswith("/exp_inf") and np.any(value < 0):
            raise ValueError("study snapshot negative infinity norm")
        if name.endswith("/step") or name.endswith("num_batches_tracked"):
            steps.add(float(value))
    if len(steps) != 1 or next(iter(steps)) not in {
        meta["presented_rows"] // n for n in (4, 8, 16, 32)
    }:
        raise ValueError("study snapshot inconsistent update counts")
    srnet_model.validate({k: arrays["model/" + k] for k in srnet_model.SHAPES})


def save(snapshot, path):
    from core.jpeg_study_data import fresh

    validate(snapshot)
    fresh(path)
    raw_meta = json.dumps(snapshot["metadata"], sort_keys=True, allow_nan=False).encode()
    buffer = io.BytesIO()
    np.savez(buffer, **snapshot["arrays"], __metadata__=np.frombuffer(raw_meta, dtype="u1"))
    raw = buffer.getvalue()
    if len(raw) > MAX_BYTES:
        raise ValueError("study snapshot exceeds byte limit")
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def load(path, checksum, binding):
    identity(checksum)
    identity(binding)
    with regular_open(path) as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES or hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("study snapshot checksum/byte limit mismatch")
    expected = specs()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if (len(members) != len(expected) + 1
            or {m.filename for m in members} != {k + ".npy" for k in expected} | {
                "__metadata__.npy"
            } or sum(m.file_size for m in members) > MAX_BYTES
            or any(m.compress_type != zipfile.ZIP_STORED or m.flag_bits & 1
                   or stat.S_ISLNK(m.external_attr >> 16) for m in members)):
            raise ValueError("study snapshot archive encoding/member/expansion bounds")
        for member in members:
            with archive.open(member) as stream:
                shape, _, dtype = header(stream)
                if member.filename == "__metadata__.npy":
                    if len(shape) != 1 or not 1 <= shape[0] <= MAX_META or dtype != np.dtype("u1"):
                        raise ValueError("study snapshot metadata header invalid")
                elif (shape, dtype) != expected[member.filename[:-4]]:
                    raise ValueError("study snapshot array header invalid")
                if stream.tell() + math.prod(shape) * dtype.itemsize != member.file_size:
                    raise ValueError("study snapshot array length mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=4096) as archive:
        meta = json.loads(archive["__metadata__"].tobytes())
        if not isinstance(meta, dict) or meta.get("binding") != binding:
            raise ValueError("study snapshot recipe/source binding mismatch")
        result = {"metadata": meta, "arrays": {k: archive[k].copy() for k in expected}}
    validate(result)
    return result


def restore(snapshot, model, optimizer, binding, rows):
    import torch

    validate(snapshot)
    meta, arrays = snapshot["metadata"], snapshot["arrays"]
    if meta["binding"] != binding or type(rows) is not int or rows not in (4, 8, 16, 32):
        raise ValueError("study resume binding/batch mismatch")
    updates = meta["presented_rows"] // rows
    if any(float(arrays[f"optimizer/{name}/step"]) != updates for name in PARAMETERS):
        raise ValueError("study resume update/batch mismatch")
    model.load_state_dict({k: torch.from_numpy(arrays["model/" + k].copy())
                          for k in srnet_model.SHAPES}, strict=True)
    for name, parameter in model.named_parameters():
        optimizer.state[parameter] = {
            slot: torch.from_numpy(arrays[f"optimizer/{name}/{slot}"].copy()).to(parameter.device)
            for slot in ("exp_avg", "exp_inf", "step")
        }
    return meta["presented_rows"]
