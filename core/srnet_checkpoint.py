"""Bounded numeric epoch checkpoints: no pickle, code, downloads or deployment."""

from __future__ import annotations

import hashlib
import io
import json
import math
import stat
import zipfile

import numpy as np

from core import srnet, srnet_model
from core.jpeg_scale import identity
from core.srnet_stream import regular_open

MAX_BYTES = 96 * 1024**2
MAX_META = 65536
PARAMETERS = {k: v for k, v in srnet_model.SHAPES.items() if k.endswith(("weight", "bias"))}


def schedule(batches, epochs, *, deadline=None):
    if type(epochs) is not int or not 1 <= epochs <= 50:
        raise ValueError("checkpoint epoch count outside bounds")
    counts, digest = [], hashlib.sha256()
    for epoch in range(epochs):
        if deadline is not None:
            deadline()
        values = batches(epoch)
        if (
            not isinstance(values, np.ndarray)
            or values.dtype.kind not in "iu"
            or values.ndim != 2
            or values.shape[1] != 4
            or not 1 <= len(values) <= 100000
            or np.any(values < 0)
            or np.any(values >= 12000)
        ):
            raise ValueError("checkpoint requires bounded deterministic four-row schedules")
        counts.append(len(values))
        digest.update(np.asarray(values, dtype="<i8").tobytes())
    return counts, digest.hexdigest()


def metadata(meta):
    from core.srnet_training import settings

    keys = {
        "schema_version",
        "architecture",
        "binding",
        "seed",
        "settings",
        "device",
        "torch_version",
        "epochs",
        "next_epoch",
        "epoch_training",
        "scheduled_updates",
        "schedule_sha256",
        "cpu_rng_bytes",
        "cuda_rng_bytes",
        "accuracy_qualification",
    }
    if not isinstance(meta, dict) or set(meta) != keys:
        raise ValueError("checkpoint metadata contract mismatch")
    identity(meta["binding"])
    identity(meta["schedule_sha256"])
    if (
        meta["schema_version"] != "srnet-epoch-checkpoint-v1"
        or meta["architecture"] != srnet.ARCHITECTURE
        or not isinstance(meta["device"], str)
        or meta["device"] not in {"cpu", "cuda:0"}
        or type(meta["seed"]) is not int
        or not 0 <= meta["seed"] < 2**32
        or type(meta["epochs"]) is not int
        or not 1 <= meta["epochs"] <= 50
        or type(meta["next_epoch"]) is not int
        or not 1 <= meta["next_epoch"] <= meta["epochs"]
        or not isinstance(meta["torch_version"], str)
        or not 1 <= len(meta["torch_version"]) <= 128
        or any(
            c not in "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.+_-"
            for c in meta["torch_version"]
        )
        or meta["accuracy_qualification"] != "unavailable"
        or not isinstance(meta["settings"], dict)
        or set(meta["settings"]) != {"threads", "max_seconds", "learning_rate", "weight_decay"}
        or settings(meta["settings"]) != meta["settings"]
        or type(meta["cpu_rng_bytes"]) is not int
        or not 16 <= meta["cpu_rng_bytes"] <= 16384
        or type(meta["cuda_rng_bytes"]) is not int
        or (
            meta["cuda_rng_bytes"] != 0
            if meta["device"] == "cpu"
            else not 16 <= meta["cuda_rng_bytes"] <= 16384
        )
        or not isinstance(meta["scheduled_updates"], list)
        or len(meta["scheduled_updates"]) != meta["epochs"]
        or any(type(v) is not int or not 1 <= v <= 100000 for v in meta["scheduled_updates"])
        or not isinstance(meta["epoch_training"], list)
        or len(meta["epoch_training"]) != meta["next_epoch"]
    ):
        raise ValueError("checkpoint metadata values invalid")
    for epoch, record in enumerate(meta["epoch_training"]):
        if (
            not isinstance(record, dict)
            or set(record) != {"epoch", "updates", "mean_pair_loss"}
            or type(record["epoch"]) is not int
            or record["epoch"] != epoch
            or type(record["updates"]) is not int
            or record["updates"] != meta["scheduled_updates"][epoch]
            or type(record["mean_pair_loss"]) not in (int, float)
            or not math.isfinite(record["mean_pair_loss"])
            or record["mean_pair_loss"] < 0
        ):
            raise ValueError("checkpoint complete-epoch accounting invalid")


def specs(meta):
    result = {"model/" + k: (s, srnet_model.dtype(k)) for k, s in srnet_model.SHAPES.items()}
    for name, shape in PARAMETERS.items():
        for slot in ("exp_avg", "exp_inf", "step"):
            result[f"optimizer/{name}/{slot}"] = (() if slot == "step" else shape, np.dtype("<f4"))
    result["rng/cpu"] = ((meta["cpu_rng_bytes"],), np.dtype("u1"))
    if meta["device"] == "cuda:0":
        result["rng/cuda"] = ((meta["cuda_rng_bytes"],), np.dtype("u1"))
    return result


def validate(snapshot):
    if not isinstance(snapshot, dict) or set(snapshot) != {"metadata", "arrays"}:
        raise ValueError("checkpoint container contract mismatch")
    meta, arrays = snapshot["metadata"], snapshot["arrays"]
    metadata(meta)
    expected = specs(meta)
    if not isinstance(arrays, dict) or set(arrays) != set(expected):
        raise ValueError("checkpoint numeric keys mismatch")
    updates = sum(meta["scheduled_updates"][: meta["next_epoch"]])
    for name, (shape, dtype) in expected.items():
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != dtype
            or not np.isfinite(value).all()
        ):
            raise ValueError("checkpoint numeric shape/dtype/finite mismatch")
        if name.endswith("/exp_inf") and np.any(value < 0):
            raise ValueError("checkpoint negative Adamax infinity norm")
        if (name.endswith("/step") or name.endswith("num_batches_tracked")) and float(
            value
        ) != updates:
            raise ValueError("checkpoint optimizer/BN update count mismatch")
    srnet_model.validate({k: arrays["model/" + k] for k in srnet_model.SHAPES})


def capture(model, optimizer, records, context):
    import torch

    arrays = {"model/" + k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
    for name, parameter in model.named_parameters():
        for slot in ("exp_avg", "exp_inf", "step"):
            arrays[f"optimizer/{name}/{slot}"] = (
                optimizer.state[parameter][slot].detach().cpu().numpy().copy()
            )
    arrays["rng/cpu"] = torch.get_rng_state().numpy().copy()
    if context["device"] == "cuda:0":
        arrays["rng/cuda"] = torch.cuda.get_rng_state(0).numpy().copy()
    meta = {
        **json.loads(json.dumps(context, allow_nan=False)),
        "schema_version": "srnet-epoch-checkpoint-v1",
        "architecture": srnet.ARCHITECTURE,
        "next_epoch": len(records),
        "epoch_training": [dict(r) for r in records],
        "torch_version": str(torch.__version__),
        "cpu_rng_bytes": len(arrays["rng/cpu"]),
        "cuda_rng_bytes": len(arrays.get("rng/cuda", [])),
        "accuracy_qualification": "unavailable",
    }
    result = {"metadata": meta, "arrays": arrays}
    validate(result)
    return result


def restore(snapshot, model, optimizer, context):
    import torch

    validate(snapshot)
    meta, arrays = snapshot["metadata"], snapshot["arrays"]
    if any(meta[k] != v for k, v in context.items()) or meta["torch_version"] != str(
        torch.__version__
    ):
        raise ValueError("checkpoint binding/settings/schedule/backend mismatch")
    if meta["cpu_rng_bytes"] != len(torch.get_rng_state()) or (
        meta["device"] == "cuda:0" and meta["cuda_rng_bytes"] != len(torch.cuda.get_rng_state(0))
    ):
        raise ValueError("checkpoint RNG implementation mismatch")
    model.load_state_dict(
        {k: torch.from_numpy(arrays["model/" + k].copy()) for k in srnet_model.SHAPES}, strict=True
    )
    for name, parameter in model.named_parameters():
        optimizer.state[parameter] = {
            slot: torch.from_numpy(arrays[f"optimizer/{name}/{slot}"].copy()).to(
                "cpu" if slot == "step" else context["device"]
            )
            for slot in ("exp_avg", "exp_inf", "step")
        }
    torch.set_rng_state(torch.from_numpy(arrays["rng/cpu"].copy()))
    if context["device"] == "cuda:0":
        torch.cuda.set_rng_state(torch.from_numpy(arrays["rng/cuda"].copy()), 0)
    return [dict(r) for r in meta["epoch_training"]]


def save(snapshot, path):
    validate(snapshot)
    if path.exists() or any(p.is_symlink() for p in (path, *path.parents)):
        raise FileExistsError("checkpoint output exists or uses a symlink")
    raw_meta = json.dumps(snapshot["metadata"], sort_keys=True, allow_nan=False).encode()
    if len(raw_meta) > MAX_META:
        raise ValueError("checkpoint metadata size exceeded")
    buffer = io.BytesIO()
    np.savez(buffer, **snapshot["arrays"], __metadata__=np.frombuffer(raw_meta, dtype="u1"))
    raw = buffer.getvalue()
    if len(raw) > MAX_BYTES:
        raise ValueError("checkpoint size exceeded")
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream, max_header_size=4096)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(stream, max_header_size=4096)
    raise ValueError("checkpoint array header version invalid")


def load(path, *, checksum):
    identity(checksum)
    with regular_open(path) as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES or hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("checkpoint checksum/size mismatch")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if (
            not 1 <= len(members) <= 512
            or len({m.filename for m in members}) != len(members)
            or "__metadata__.npy" not in {m.filename for m in members}
            or sum(m.file_size for m in members) > MAX_BYTES
            or any(
                m.compress_type != zipfile.ZIP_STORED
                or m.flag_bits & 1
                or stat.S_ISLNK(m.external_attr >> 16)
                for m in members
            )
        ):
            raise ValueError("checkpoint archive limits/encoding invalid")
        with archive.open("__metadata__.npy") as stream:
            shape, _, dtype = header(stream)
            if len(shape) != 1 or not 1 <= shape[0] <= MAX_META or dtype != np.dtype("u1"):
                raise ValueError("checkpoint metadata header invalid")
            info = archive.getinfo("__metadata__.npy")
            if stream.tell() + shape[0] != info.file_size:
                raise ValueError("checkpoint metadata length invalid")
            meta = json.loads(stream.read(shape[0]))
        metadata(meta)
        expected = {**specs(meta), "__metadata__": (shape, np.dtype("u1"))}
        if {m.filename for m in members} != {k + ".npy" for k in expected}:
            raise ValueError("checkpoint archive members invalid")
        for member in members:
            with archive.open(member) as stream:
                actual_shape, _, dtype = header(stream)
                wanted_shape, wanted_dtype = expected[member.filename[:-4]]
                if (
                    actual_shape != wanted_shape
                    or dtype != wanted_dtype
                    or stream.tell() + math.prod(actual_shape) * dtype.itemsize != member.file_size
                ):
                    raise ValueError("checkpoint array shape/dtype/length invalid")
    with np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=4096) as archive:
        result = {"metadata": meta, "arrays": {k: archive[k].copy() for k in specs(meta)}}
    validate(result)
    return result
