"""Source-bound, generated-only SRNet capacity measurement; no optimizer."""

from __future__ import annotations

import contextlib
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import PIL

from core import colab_artifacts as artifacts
from core import nonjpeg_models as models

BASE_COMMIT = "2be950e8af0984bc012bc89d3ca072cea01c24f3"
MAX_SECONDS = 600
SEEDS = (2026102301, 2026102302)
BATCHES = (2, 4, 8)
DOMAINS = (models.GRAY_DOMAIN, models.RGB_DOMAIN)
ENVIRONMENT = {
    "runtime": "2026.07",
    "python": "3.12.13",
    "numpy": "2.0.2",
    "pillow": "11.3.0",
    "torch": "2.11.0+cu128",
    "cuda": "12.8",
}
SOURCES = (
    "core/__init__.py",
    "core/colab_artifacts.py",
    "core/spatial_colab.py",
    "core/spatial_colab_data.py",
    "core/spatial_colab_state.py",
    "core/spatial_colab_bundle.py",
    "core/nonjpeg_models.py",
    "core/nonjpeg_features.py",
    "core/features.py",
    "core/spatial_parity.py",
    "core/spatial_cooccurrence.py",
    "core/srnet.py",
    "core/srnet_model.py",
    "scripts/spatial-colab.py",
    "requirements-colab-gpu.txt",
    "docs/PNG_BMP_COLAB_PROTOCOL_20261023.md",
)


def bindings(root: Path) -> dict:
    result = {}
    for name in SOURCES:
        # The package initializer is deliberately empty.
        path = root / name
        artifacts.safe(path)
        result[name] = artifacts.digest(
            b"" if path.stat().st_size == 0 else artifacts.read(path, artifacts.MIB)
        )
    return result


def freeze(root: Path, path: Path) -> dict:
    value = {
        "schema": "spatial-colab-profile-v1",
        "base_commit": BASE_COMMIT,
        "source_sha256": bindings(root),
        "environment": ENVIRONMENT,
        "max_seconds": MAX_SECONDS,
        "seeds": list(SEEDS),
        "batches": list(BATCHES),
        "domains": list(DOMAINS),
        "shape": [256, 256],
        "dtype": "float32",
        "warmup_per_cell": 1,
        "measured_per_cell": 3,
        "optimizer_updates": 0,
        "real_media": False,
        "amp": False,
        "tf32": False,
        "retry": False,
        "max_receipt_bytes": artifacts.MIB,
    }
    artifacts.put(path, artifacts.encode(value))
    return value


def verify(root: Path, path: Path) -> dict:
    value = artifacts.document(path)
    expected = {
        "schema": "spatial-colab-profile-v1",
        "base_commit": BASE_COMMIT,
        "source_sha256": bindings(root),
        "environment": ENVIRONMENT,
        "max_seconds": MAX_SECONDS,
        "seeds": list(SEEDS),
        "batches": list(BATCHES),
        "domains": list(DOMAINS),
        "shape": [256, 256],
        "dtype": "float32",
        "warmup_per_cell": 1,
        "measured_per_cell": 3,
        "optimizer_updates": 0,
        "real_media": False,
        "amp": False,
        "tf32": False,
        "retry": False,
        "max_receipt_bytes": artifacts.MIB,
    }
    if artifacts.encode(value) != artifacts.encode(expected):
        raise ValueError("profile source/protocol binding mismatch")
    return value


def environment() -> dict:
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@contextlib.contextmanager
def fp32_policy():
    import torch

    previous = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    finally:
        (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic,
        ) = previous


def cell(domain: str, seed: int, batch: int, *, device: str = "cuda:0") -> dict:
    """No optimizer is constructed. CPU is available only for generated QA."""
    import torch

    if (
        domain not in DOMAINS
        or seed not in SEEDS
        or batch not in BATCHES
        or device not in {"cpu", "cuda:0"}
    ):
        raise ValueError("profile cell contract")
    started = time.monotonic()
    cuda = device == "cuda:0"
    with torch.random.fork_rng(devices=[0] if cuda else []), fp32_policy():
        torch.manual_seed(seed)
        model = models.network(domain).to(device).train()
        parameters = [p.detach().clone() for p in model.parameters()]
        channels = 1 if domain == DOMAINS[0] else 3
        pixels = torch.randint(
            0, 256, (batch, channels, 256, 256), dtype=torch.uint8, device=device
        )
        values = pixels.to(torch.float32).requires_grad_()
        lsb_preserved = torch.equal(values.detach().to(torch.uint8), pixels)
        if not lsb_preserved:
            raise ValueError("one-LSB input preservation failed")
        labels = torch.arange(batch, device=device) % 2
        if cuda:
            torch.cuda.synchronize(0)
            torch.cuda.reset_peak_memory_stats(0)
        durations: list[float] = []
        for repeat in range(4):
            model.zero_grad(set_to_none=True)
            values.grad = None
            if cuda:
                torch.cuda.synchronize(0)
            tick = time.monotonic()
            logits = model(values)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            if cuda:
                torch.cuda.synchronize(0)
            elapsed = time.monotonic() - tick
            if (
                not bool(torch.isfinite(loss))
                or values.grad is None
                or not bool(torch.isfinite(values.grad).all())
                or not bool((values.grad != 0).any())
                or any(
                    p.grad is None or not bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()
                )
            ):
                raise ValueError("nonfinite/absent generated gradients")
            if repeat:
                durations.append(elapsed)
        if any(
            not torch.equal(p, old) for p, old in zip(model.parameters(), parameters, strict=True)
        ):
            raise ValueError("profile changed parameters")
        return {
            "domain": domain,
            "seed": seed,
            "batch": batch,
            "status": "ready",
            "forward_backward_seconds": durations,
            "cell_seconds": time.monotonic() - started,
            "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(0) if cuda else None,
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(0) if cuda else None,
            "process_peak_ram_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "parameters": sum(p.numel() for p in model.parameters()),
            "finite_gradients": True,
            "parameters_unchanged": True,
            "lsb_preserved": True,
            "optimizer_updates": 0,
        }


def worker(root: Path, protocol: Path, out: Path) -> dict:
    verify(root, protocol)
    import torch

    if not torch.cuda.is_available():
        return {"status": "unavailable", "reason": "gpu_missing", "environment": environment()}
    actual = environment()
    if any(actual[k] != v for k, v in ENVIRONMENT.items() if k != "runtime"):
        return {"status": "unavailable", "reason": "environment_mismatch", "environment": actual}
    artifacts.put(out / "execution.json", artifacts.encode({"environment": actual}))
    torch.set_num_threads(2)
    for index, (domain, seed, batch) in enumerate(
        (d, s, b) for d in DOMAINS for s in SEEDS for b in BATCHES
    ):
        try:
            record = cell(domain, seed, batch)
        except (RuntimeError, ValueError):
            return {
                "status": "unavailable",
                "reason": "gpu_oom_or_numerics",
                "environment": actual,
                "failed_cell": {"domain": domain, "seed": seed, "batch": batch},
            }
        artifacts.put(out / f"cell-{index:02d}.json", artifacts.encode(record))
        torch.cuda.empty_cache()
    return {"status": "ready", "environment": actual}


def costs(cells: list[dict]) -> list[dict]:
    table = []
    for domain in DOMAINS:
        for seed in SEEDS:
            for batch in BATCHES:
                found = [
                    c
                    for c in cells
                    if (c["domain"], c["seed"], c["batch"]) == (domain, seed, batch)
                ]
                times = found[0]["forward_backward_seconds"] if found else []
                if times and (
                    len(times) != 3 or any(not math.isfinite(v) or v <= 0 for v in times)
                ):
                    raise ValueError("invalid measured times")
                table.append(
                    {
                        "domain": domain,
                        "seed": seed,
                        "batch": batch,
                        "status": "ready" if times else "unavailable",
                        "median_forward_backward_seconds": float(np.median(times))
                        if times
                        else None,
                        "full_training_seconds": None,
                        "optimizer_cost": "unmeasured",
                        "evaluation_cost": "unmeasured",
                        "data_io_cost": "unmeasured",
                    }
                )
    return table


def supervise(root: Path, protocol: Path, out: Path) -> dict:
    started = time.monotonic()
    verify(root, protocol)
    if "drive" in (p.lower() for p in out.absolute().parts):
        raise ValueError("measurement requires local working disk")
    protocol_sha = artifacts.digest(artifacts.read(protocol, artifacts.MIB))
    marker = out.parent / ("spatial-profile-" + protocol_sha + ".attempt")
    artifacts.put(marker, artifacts.encode({"protocol_sha256": protocol_sha, "retry": False}))
    artifacts.directory(out)
    result: dict = {"status": "unavailable", "reason": "worker_failed"}
    with (out / "worker.log").open("xb") as log:
        try:
            process = subprocess.run(  # noqa: S603 — fixed source-bound adapter, no shell.
                [
                    sys.executable,
                    str(root / "scripts/spatial-colab.py"),
                    "_worker",
                    "--protocol",
                    str(protocol.absolute()),
                    "--out",
                    str(out.absolute()),
                ],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                timeout=max(0.01, MAX_SECONDS - (time.monotonic() - started) - 2),
                check=False,
                env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
            )
            if process.returncode == 0:
                result = artifacts.document(out / "worker.json")
        except subprocess.TimeoutExpired:
            result = {"status": "unavailable", "reason": "deadline"}
    cells = [
        artifacts.document(out / f"cell-{i:02d}.json")
        for i in range(12)
        if (out / f"cell-{i:02d}.json").exists()
    ]
    if len(cells) != 12 and result["status"] == "ready":
        result = {"status": "unavailable", "reason": "incomplete_cells"}
    if time.monotonic() - started >= MAX_SECONDS:
        result = {"status": "unavailable", "reason": "deadline"}
    if (out / "execution.json").exists():
        result["environment"] = artifacts.document(out / "execution.json")["environment"]
    result.update(
        {
            "schema": "spatial-colab-measurement-v1",
            "protocol_sha256": protocol_sha,
            "elapsed_seconds": time.monotonic() - started,
            "max_seconds": MAX_SECONDS,
            "optimizer_updates": 0,
            "real_media_opened": False,
            "retry": False,
            "cells": cells,
            "cost_table": costs(cells),
            "full_training_budget": "missing",
            "output_bytes_before_receipt": sum(p.stat().st_size for p in out.iterdir()),
        }
    )
    result["output_bytes"] = 0
    for _ in range(4):
        result["output_bytes"] = result["output_bytes_before_receipt"] + len(
            artifacts.encode(result)
        )
    if result["output_bytes"] > artifacts.MIB:
        raise ValueError("measurement total output limit")
    artifacts.put(out / "measurement.json", artifacts.encode(result))
    return result
