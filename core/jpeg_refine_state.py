"""Numeric refinement snapshots with explicit branch ancestry; legacy v1 is rejected."""

from __future__ import annotations

import hashlib
import io
import json
import math
import stat
import zipfile

import numpy as np

from core import jpeg_refine_schedule as schedule
from core import srnet_model
from core.jpeg_scale import identity
from core.jpeg_study import SEEDS
from core.srnet_checkpoint import MAX_BYTES, MAX_META, PARAMETERS, header
from core.srnet_stream import regular_open


def capture(model, optimizer, metadata):
    arrays = {"model/" + k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
    for name, parameter in model.named_parameters():
        for slot in ("exp_avg", "exp_inf", "step"):
            arrays[f"optimizer/{name}/{slot}"] = (
                optimizer.state[parameter][slot].detach().cpu().numpy().astype("<f4", copy=True)
            )
    result = {"metadata": dict(metadata), "arrays": arrays}
    validate(result)
    return result


def metadata(binding, arm, seed, stop, schedule_sha256, parent=None, warmup=None):
    return {
        "schema_version": "jpeg-refine-snapshot-v2",
        "binding": binding,
        "arm": arm,
        "seed": seed,
        "total_updates": stop,
        "new_updates": schedule.INTERVAL,
        "next_update": stop,
        "schedule_sha256": schedule_sha256,
        "parent_sha256": parent,
        "parent_binding": binding if parent else None,
        "warmup_sha256": warmup,
    }


def validate_metadata(meta):
    if (
        not isinstance(meta, dict)
        or set(meta)
        != {
            "schema_version",
            "binding",
            "arm",
            "seed",
            "total_updates",
            "new_updates",
            "next_update",
            "schedule_sha256",
            "parent_sha256",
            "parent_binding",
            "warmup_sha256",
        }
        or meta["schema_version"] != "jpeg-refine-snapshot-v2"
        or meta["arm"] not in ("warmup", *schedule.ARMS)
        or type(meta["seed"]) is not int
        or meta["seed"] not in SEEDS
        or type(meta["total_updates"]) is not int
        or type(meta["new_updates"]) is not int
        or meta["new_updates"] != schedule.INTERVAL
        or type(meta["next_update"]) is not int
        or meta["next_update"] != meta["total_updates"]
    ):
        raise ValueError("refine snapshot metadata invalid")
    stop = meta["total_updates"]
    if (
        stop % schedule.INTERVAL
        or not schedule.INTERVAL <= stop <= schedule.FINAL
        or ((stop <= schedule.WARMUP) != (meta["arm"] == "warmup"))
    ):
        raise ValueError("refine snapshot exposure/arm mismatch")
    for key in ("binding", "schedule_sha256"):
        identity(meta[key])
    if stop == schedule.INTERVAL:
        if meta["parent_sha256"] is not None or meta["parent_binding"] is not None:
            raise ValueError("refine initial snapshot cannot have a parent")
    else:
        identity(meta["parent_sha256"])
        if meta["parent_binding"] != meta["binding"]:
            raise ValueError("refine parent binding mismatch")
    if meta["arm"] == "warmup":
        if meta["warmup_sha256"] is not None:
            raise ValueError("refine warmup cannot import a historical model")
    else:
        identity(meta["warmup_sha256"])
        if stop == schedule.WARMUP + schedule.INTERVAL and (
            meta["warmup_sha256"] != meta["parent_sha256"]
        ):
            raise ValueError("refine branch must inherit its shared warmup")


def specs():
    result = {"model/" + k: (s, srnet_model.dtype(k)) for k, s in srnet_model.SHAPES.items()}
    for name, shape in PARAMETERS.items():
        for slot in ("exp_avg", "exp_inf", "step"):
            result[f"optimizer/{name}/{slot}"] = (() if slot == "step" else shape, np.dtype("<f4"))
    return result


def validate(snapshot):
    if not isinstance(snapshot, dict) or set(snapshot) != {"metadata", "arrays"}:
        raise ValueError("refine snapshot container invalid")
    meta = snapshot["metadata"]
    validate_metadata(meta)
    arrays = snapshot["arrays"]
    expected = specs()
    if not isinstance(arrays, dict) or set(arrays) != set(expected):
        raise ValueError("refine snapshot numeric keys invalid")
    steps = set()
    for name, (shape, dtype) in expected.items():
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != dtype
            or not np.isfinite(value).all()
        ):
            raise ValueError("refine snapshot numeric shape/dtype/finite invalid")
        if name.endswith("/exp_inf") and np.any(value < 0):
            raise ValueError("refine snapshot negative infinity norm")
        if name.endswith("/step") or name.endswith("num_batches_tracked"):
            steps.add(float(value))
    if steps != {float(meta["total_updates"])}:
        raise ValueError("refine snapshot inconsistent model/BN/optimizer update counts")
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
        raise ValueError("refine snapshot exceeds byte limit")
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def load(path, checksum, binding):
    try:
        return _load(path, checksum, binding)
    except (zipfile.BadZipFile, EOFError, OverflowError, RecursionError) as exc:
        raise ValueError("refine malformed numeric archive") from exc


def _load(path, checksum, binding):
    identity(checksum)
    identity(binding)
    with regular_open(path) as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES or hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("refine snapshot checksum/byte limit mismatch")
    expected = specs()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if (
            len(members) != len(expected) + 1
            or {m.filename for m in members}
            != {k + ".npy" for k in expected} | {"__metadata__.npy"}
            or sum(m.file_size for m in members) > MAX_BYTES
            or any(
                m.compress_type != zipfile.ZIP_STORED
                or m.flag_bits & 1
                or stat.S_ISLNK(m.external_attr >> 16)
                for m in members
            )
        ):
            raise ValueError("refine snapshot archive encoding/member/expansion bounds")
        for member in members:
            with archive.open(member) as stream:
                shape, _, dtype = header(stream)
                if member.filename == "__metadata__.npy":
                    if len(shape) != 1 or not 1 <= shape[0] <= MAX_META or dtype != np.dtype("u1"):
                        raise ValueError("refine snapshot metadata header invalid")
                elif (shape, dtype) != expected[member.filename[:-4]]:
                    raise ValueError("refine snapshot array header invalid")
                if stream.tell() + math.prod(shape) * dtype.itemsize != member.file_size:
                    raise ValueError("refine snapshot array length mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=4096) as archive:
        meta = json.loads(archive["__metadata__"].tobytes())
        if not isinstance(meta, dict) or meta.get("binding") != binding:
            raise ValueError("refine snapshot recipe/source binding mismatch")
        result = {"metadata": meta, "arrays": {k: archive[k].copy() for k in expected}}
    validate(result)
    return result


def restore(
    snapshot,
    model,
    optimizer,
    binding,
    *,
    arm,
    seed,
    start,
    parent_sha256,
    schedule_sha256,
    warmup_sha256=None,
):
    import torch

    validate(snapshot)
    meta, arrays = snapshot["metadata"], snapshot["arrays"]
    expected_arm = "warmup" if start <= schedule.WARMUP else arm
    if (
        meta["binding"] != binding
        or meta["arm"] != expected_arm
        or meta["seed"] != seed
        or meta["total_updates"] != start
        or meta["schedule_sha256"] != schedule_sha256
        or (
            arm != "warmup"
            and (
                (start == schedule.WARMUP and warmup_sha256 != parent_sha256)
                or (start > schedule.WARMUP and meta["warmup_sha256"] != warmup_sha256)
            )
        )
    ):
        raise ValueError("refine resume source/parent/schedule/arm mismatch")
    identity(parent_sha256)
    model.load_state_dict(
        {k: torch.from_numpy(arrays["model/" + k].copy()) for k in srnet_model.SHAPES}, strict=True
    )
    for name, parameter in model.named_parameters():
        optimizer.state[parameter] = {
            slot: torch.from_numpy(arrays[f"optimizer/{name}/{slot}"].copy()).to(parameter.device)
            for slot in ("exp_avg", "exp_inf", "step")
        }
    return start
