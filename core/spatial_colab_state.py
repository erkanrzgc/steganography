"""Numeric AdamW resume format; exercised only with generated states here."""

from __future__ import annotations

import json
import math
import platform
import random
import struct
from pathlib import Path

import numpy as np

from core import colab_artifacts as artifacts
from core import spatial_colab as profile
from core import srnet_model

MAGIC = b"SPCOLAB1"
MAX_BYTES = 128 * artifacts.MIB
MAX_META = 65536


def specs(meta: dict) -> dict:
    if not isinstance(meta, dict) or set(meta) != {
        "schema",
        "domain",
        "environment",
        "source_sha256",
        "data_sha256",
        "order_sha256",
        "optimizer",
        "initialized",
        "rng_bytes",
        "numpy_rng",
        "python_gauss",
        "consumption",
        "rows",
        "training",
    }:
        raise ValueError("checkpoint metadata fields")
    if meta["schema"] != "spatial-colab-state-v1" or meta["domain"] not in profile.DOMAINS:
        raise ValueError("checkpoint domain")
    if type(meta["training"]) is not bool:
        raise ValueError("checkpoint model mode")
    for key in ("source_sha256", "data_sha256", "order_sha256"):
        artifacts.identity(meta[key])
    env = meta["environment"]
    if not isinstance(env, dict) or set(env) != {
        "python",
        "numpy",
        "pillow",
        "torch",
        "cuda",
        "cudnn",
        "gpu",
    }:
        raise ValueError("checkpoint environment")
    if any(
        v is not None and (not isinstance(v, (str, int)) or len(str(v)) > 128) for v in env.values()
    ):
        raise ValueError("checkpoint environment values")
    count = meta["consumption"]
    if not isinstance(count, dict) or set(count) != {
        "epoch",
        "cursor",
        "optimizer_updates",
        "presentations",
        "elapsed_seconds",
        "output_bytes",
    }:
        raise ValueError("checkpoint consumption fields")
    if any(
        type(count[k]) is not int or not 0 <= count[k] <= 10**12
        for k in count
        if k != "elapsed_seconds"
    ):
        raise ValueError("checkpoint consumption bounds")
    if (
        type(count["elapsed_seconds"]) not in (int, float)
        or not math.isfinite(count["elapsed_seconds"])
        or count["elapsed_seconds"] < 0
    ):
        raise ValueError("checkpoint time debit")
    rows = meta["rows"]
    if type(rows) is not int or not 1 <= rows <= 100_000 or count["cursor"] > rows:
        raise ValueError("checkpoint order bounds")
    if not isinstance(meta["rng_bytes"], dict) or set(meta["rng_bytes"]) != {"cpu", "cuda"}:
        raise ValueError("checkpoint RNG fields")
    for key, size in meta["rng_bytes"].items():
        if type(size) is not int or not (16 <= size <= 16384 or key == "cuda" and size == 0):
            raise ValueError("checkpoint RNG size")
    if (env["gpu"] is None) != (meta["rng_bytes"]["cuda"] == 0):
        raise ValueError("checkpoint device/RNG mismatch")
    nrng = meta["numpy_rng"]
    if (
        not isinstance(nrng, list)
        or len(nrng) != 3
        or type(nrng[0]) is not int
        or not 0 <= nrng[0] <= 624
        or nrng[1] not in (0, 1)
        or not isinstance(nrng[2], (int, float))
        or not math.isfinite(nrng[2])
    ):
        raise ValueError("checkpoint NumPy RNG metadata")
    if meta["python_gauss"] is not None and (
        type(meta["python_gauss"]) not in (int, float) or not math.isfinite(meta["python_gauss"])
    ):
        raise ValueError("checkpoint Python RNG metadata")
    optimizer: dict = meta["optimizer"]
    if not isinstance(optimizer, dict) or set(optimizer) != {
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
    }:
        raise ValueError("checkpoint AdamW options")
    if (
        optimizer["amsgrad"] is not False
        or optimizer["maximize"] is not False
        or optimizer["capturable"] is not False
        or optimizer["differentiable"] is not False
        or optimizer["foreach"] is not False
        or optimizer["fused"] is not False
    ):
        raise ValueError("checkpoint requires explicit scalar FP32 AdamW")
    for key in ("lr", "eps", "weight_decay"):
        if (
            type(optimizer[key]) not in (int, float)
            or not math.isfinite(optimizer[key])
            or optimizer[key] < 0
        ):
            raise ValueError("checkpoint optimizer finite bounds")
    if (
        not isinstance(optimizer["betas"], list)
        or len(optimizer["betas"]) != 2
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v < 1
            for v in optimizer["betas"]
        )
    ):
        raise ValueError("checkpoint optimizer betas")
    shapes = dict(srnet_model.SHAPES)
    if meta["domain"] == profile.DOMAINS[1]:
        shapes["front.0.0.weight"] = (64, 3, 3, 3)
    params = [k for k in shapes if k.endswith(("weight", "bias"))]
    if meta["initialized"] not in ([], params) or (
        count["optimizer_updates"] > 0 and not meta["initialized"]
    ):
        raise ValueError("checkpoint initialized optimizer states")
    result = {"model/" + k: (shape, srnet_model.dtype(k)) for k, shape in shapes.items()}
    for key in meta["initialized"]:
        for slot in ("step", "exp_avg", "exp_avg_sq"):
            result[f"optimizer/{key}/{slot}"] = (
                () if slot == "step" else shapes[key],
                np.dtype("<f4"),
            )
    result.update(
        {
            "rng/cpu": ((meta["rng_bytes"]["cpu"],), np.dtype("u1")),
            "rng/python": ((625,), np.dtype("<u4")),
            "rng/numpy": ((624,), np.dtype("<u4")),
            "data/order": ((rows,), np.dtype("<i8")),
        }
    )
    if meta["rng_bytes"]["cuda"]:
        result["rng/cuda"] = ((meta["rng_bytes"]["cuda"],), np.dtype("u1"))
    return result


def validate(meta: dict, arrays: dict) -> None:
    expected = specs(meta)
    if set(arrays) != set(expected):
        raise ValueError("checkpoint array keys")
    for key, (shape, dtype) in expected.items():
        value = arrays[key]
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != dtype
            or not np.isfinite(value).all()
        ):
            raise ValueError("checkpoint numeric shape/dtype/finite")
        if key.endswith(("exp_avg_sq", "running_var")) and np.any(value < 0):
            raise ValueError("checkpoint negative variance")
        if key.endswith("/step") and float(value) != meta["consumption"]["optimizer_updates"]:
            raise ValueError("checkpoint optimizer debit mismatch")
    order = arrays["data/order"]
    if (
        not np.array_equal(np.sort(order), np.arange(meta["rows"]))
        or artifacts.digest(order.tobytes()) != meta["order_sha256"]
    ):
        raise ValueError("checkpoint data order mismatch")
    if int(arrays["rng/python"][-1]) > 624:
        raise ValueError("checkpoint Python RNG index")


def optimizer_options(model, optimizer) -> dict:
    import torch

    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise ValueError("checkpoint supports one AdamW group")
    group = optimizer.param_groups[0]
    if [id(p) for p in group["params"]] != [id(p) for p in model.parameters()]:
        raise ValueError("checkpoint optimizer parameter order")
    keys = (
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
    )
    return {k: list(group[k]) if k == "betas" else group[k] for k in keys}


def capture(
    model, optimizer, *, source_sha256: str, data_sha256: str, order: np.ndarray, consumption: dict
) -> tuple[dict, dict]:
    import torch

    if any(module.training != model.training for module in model.modules()):
        raise ValueError("checkpoint requires uniform train/eval mode")
    cpu = torch.get_rng_state().numpy().copy()
    cuda = (
        torch.cuda.get_rng_state(0).cpu().numpy().copy()
        if torch.cuda.is_available()
        else np.empty(0, dtype="u1")
    )
    python = random.getstate()
    numpy = np.random.get_state()
    if not isinstance(numpy, tuple) or numpy[0] != "MT19937":
        raise ValueError("checkpoint requires NumPy MT19937 state")
    arrays = {"model/" + k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
    arrays.update(
        {
            "rng/cpu": cpu,
            "rng/python": np.array(python[1], dtype="<u4"),
            "rng/numpy": numpy[1].astype("<u4"),
            "data/order": order.copy(),
        }
    )
    if len(cuda):
        arrays["rng/cuda"] = cuda
    initialized = []
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter, {})
        if state:
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("checkpoint optimizer slot keys")
            initialized.append(name)
            for slot, value in state.items():
                arrays[f"optimizer/{name}/{slot}"] = value.detach().cpu().numpy().copy()
    meta = {
        "schema": "spatial-colab-state-v1",
        "domain": model.architecture,
        "environment": profile.environment(),
        "source_sha256": source_sha256,
        "data_sha256": data_sha256,
        "order_sha256": artifacts.digest(order.tobytes()),
        "optimizer": optimizer_options(model, optimizer),
        "initialized": initialized,
        "rng_bytes": {"cpu": len(cpu), "cuda": len(cuda)},
        "python_gauss": python[2],
        "numpy_rng": [numpy[2], numpy[3], numpy[4]],
        "consumption": dict(consumption),
        "rows": len(order),
        "training": model.training,
    }
    validate(meta, arrays)
    return meta, arrays


def save(path: Path, meta: dict, arrays: dict) -> str:
    validate(meta, arrays)
    header = artifacts.encode(meta)
    if len(header) > MAX_META:
        raise ValueError("checkpoint header size")
    parts = [MAGIC, struct.pack("<I", len(header)), header]
    for key in sorted(arrays):
        parts.append(arrays[key].tobytes(order="C"))
    return artifacts.put(path, b"".join(parts), MAX_BYTES)


def load(
    path: Path, *, checksum: str, environment: dict, source_sha256: str, data_sha256: str
) -> tuple[dict, dict]:
    raw = artifacts.read(path, MAX_BYTES)
    if artifacts.digest(raw) != artifacts.identity(checksum) or raw[:8] != MAGIC or len(raw) < 12:
        raise ValueError("checkpoint checksum/header")
    size = struct.unpack("<I", raw[8:12])[0]
    if not 1 <= size <= MAX_META or len(raw) < 12 + size:
        raise ValueError("checkpoint metadata extent")
    meta = json.loads(raw[12 : 12 + size])
    expected = specs(meta)
    if (
        meta["environment"] != environment
        or meta["source_sha256"] != source_sha256
        or meta["data_sha256"] != data_sha256
    ):
        raise ValueError("checkpoint incompatible environment/source/data")
    offset = 12 + size
    total = offset + sum(math.prod(shape) * dtype.itemsize for shape, dtype in expected.values())
    if len(raw) != total:
        raise ValueError("checkpoint truncated/extra numeric extent")
    arrays = {}
    for key in sorted(expected):
        shape, dtype = expected[key]
        count = math.prod(shape)
        arrays[key] = (
            np.frombuffer(raw, dtype=dtype, count=count, offset=offset).reshape(shape).copy()
        )
        offset += count * dtype.itemsize
    validate(meta, arrays)
    return meta, arrays


def restore(model, optimizer, meta: dict, arrays: dict) -> dict:
    import torch

    validate(meta, arrays)
    if (
        meta["environment"] != profile.environment()
        or meta["domain"] != model.architecture
        or meta["optimizer"] != optimizer_options(model, optimizer)
    ):
        raise ValueError("checkpoint incompatible restore target")
    if len(arrays["rng/cpu"]) != len(torch.get_rng_state()) or (
        "rng/cuda" in arrays and len(arrays["rng/cuda"]) != len(torch.cuda.get_rng_state(0))
    ):
        raise ValueError("checkpoint incompatible RNG size")
    model_arrays = {
        k.removeprefix("model/"): value for k, value in arrays.items() if k.startswith("model/")
    }
    target = model.state_dict()
    if set(target) != set(model_arrays) or any(
        tuple(target[k].shape) != value.shape or target[k].dtype != torch.from_numpy(value).dtype
        for k, value in model_arrays.items()
    ):
        raise ValueError("checkpoint incompatible model tensors")
    # Validate RNG encodings before touching the target or global generators.
    torch.Generator().set_state(torch.from_numpy(arrays["rng/cpu"].copy()))
    if "rng/cuda" in arrays:
        torch.Generator(device="cuda:0").set_state(torch.from_numpy(arrays["rng/cuda"].copy()))
    state = {
        k.removeprefix("model/"): torch.from_numpy(v.copy())
        for k, v in arrays.items()
        if k.startswith("model/")
    }
    model.load_state_dict(state, strict=True)
    model.train(meta["training"])
    model.zero_grad(set_to_none=True)
    optimizer.state.clear()
    for name, parameter in model.named_parameters():
        if name in meta["initialized"]:
            optimizer.state[parameter] = {
                slot: torch.from_numpy(arrays[f"optimizer/{name}/{slot}"].copy()).to(
                    "cpu" if slot == "step" else parameter.device
                )
                for slot in ("step", "exp_avg", "exp_avg_sq")
            }
    random.setstate((3, tuple(int(v) for v in arrays["rng/python"]), meta["python_gauss"]))
    np.random.set_state(("MT19937", arrays["rng/numpy"], *meta["numpy_rng"]))
    torch.set_rng_state(torch.from_numpy(arrays["rng/cpu"].copy()))
    if "rng/cuda" in arrays:
        torch.cuda.set_rng_state(torch.from_numpy(arrays["rng/cuda"].copy()), 0)
    return {"consumption": dict(meta["consumption"]), "order": arrays["data/order"].copy()}


def generated_probe(out: Path) -> dict:
    """One generated numeric round trip per domain; zero optimizer calls."""
    import torch

    artifacts.directory(out)
    records = []
    for domain in profile.DOMAINS:
        with torch.random.fork_rng(devices=[]):
            model = profile.models.network(domain)
            optimizer = torch.optim.AdamW(model.parameters(), foreach=False, fused=False)
            meta, arrays = capture(
                model,
                optimizer,
                source_sha256="0" * 64,
                data_sha256="1" * 64,
                order=np.array([2, 0, 1], dtype="<i8"),
                consumption={
                    "epoch": 0,
                    "cursor": 0,
                    "optimizer_updates": 0,
                    "presentations": 0,
                    "elapsed_seconds": 0,
                    "output_bytes": 0,
                },
            )
            path = out / ("gray.numeric" if domain == profile.DOMAINS[0] else "rgb.numeric")
            checksum = save(path, meta, arrays)
            restored_meta, restored_arrays = load(
                path,
                checksum=checksum,
                environment=meta["environment"],
                source_sha256="0" * 64,
                data_sha256="1" * 64,
            )
            restore(model, optimizer, restored_meta, restored_arrays)
            if any(
                not np.array_equal(value, restored_arrays[key]) for key, value in arrays.items()
            ):
                raise ValueError("generated roundtrip mismatch")
            records.append({"domain": domain, "sha256": checksum, "bytes": path.stat().st_size})
    return {
        "status": "ready",
        "generated_only": True,
        "optimizer_updates": 0,
        "states": records,
        "python": platform.python_version(),
    }
