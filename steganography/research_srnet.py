"""Explicit isolated synthetic readiness check; no data/weights/downloads."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from core import srnet
from steganography.research_jpeg import write_json
from steganography.research_pixel_cnn import fresh


def smoke():
    try:
        import torch
    except ImportError:
        return {"status": "unavailable", "reason": "optional research dependency absent"}
    started = time.monotonic()
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261013)
            model = srnet.network()
            initial = model.front[0][0].weight.detach().clone()
            raw = np.random.default_rng(20261013).integers(0, 256, (2, 1, 128, 128), dtype=np.uint8)
            optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)
            values = torch.from_numpy(raw.astype(np.float32))
            loss = torch.nn.functional.cross_entropy(
                model(values), torch.tensor([0, 1], device="cpu")
            )
            loss.backward()
            gradients = all(
                p.grad is not None and bool(torch.isfinite(p.grad).all())
                for p in model.parameters()
            )
            if not bool(torch.isfinite(loss)) or not gradients:
                return {"status": "failed", "reason": "nonfinite learning smoke"}
            optimizer.step()
            parameter_update = not torch.equal(initial, model.front[0][0].weight)
            model.eval()
            together = srnet.logits(model, raw)
            alone = np.concatenate([srnet.logits(model, raw[i : i + 1]) for i in range(2)])
            difference = float(np.max(np.abs(together - alone)))
            decisions = bool(np.array_equal(together.argmax(1), alone.argmax(1)))
            return {
                "status": "completed"
                if difference <= 1e-5 and decisions and parameter_update
                else "failed",
                "architecture": srnet.ARCHITECTURE,
                "learned_parameters": sum(p.numel() for p in model.parameters()),
                "finite_gradients": gradients,
                "first_layer_updated": parameter_update,
                "singleton_batch_max_logit_difference": difference,
                "singleton_batch_decisions_equal": decisions,
                "absolute_tolerance": 1e-5,
                "seconds": time.monotonic() - started,
                "torch_version": str(torch.__version__),
            }
    finally:
        torch.set_num_threads(previous_threads)


def run_preflight(out: Path):
    fresh(out)
    report: dict[str, Any]
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
    try:
        with tempfile.TemporaryFile() as output:
            result = subprocess.run(  # noqa: S603 - fixed module, no shell/user arguments
                [sys.executable, "-m", "steganography.research_srnet"],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.DEVNULL,
                env=environment,
                timeout=90,
                check=False,
            )
            output.seek(0)
            data = output.read(65537)
        if result.returncode != 0 or len(data) > 65536:
            report = {"status": "failed", "reason": "bounded worker failed"}
        else:
            report = json.loads(data)
            if not isinstance(report, dict) or report.get("status") not in {
                "completed",
                "failed",
                "unavailable",
            }:
                raise ValueError("invalid readiness report")
    except subprocess.TimeoutExpired:
        report = {"status": "failed", "reason": "readiness worker deadline exceeded"}
    except (OSError, ValueError):
        report = {"status": "failed", "reason": "readiness worker unavailable/invalid"}
    report.update(
        {
            "schema_version": "srnet-readiness-v1",
            "synthetic_only": True,
            "qualification": "unavailable",
            "deployed": False,
            "primary_detection_changed": False,
        }
    )
    write_json(out, report)
    return report


def main():
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (60, 61))
        resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        print(json.dumps({"status": "unavailable", "reason": "worker resource limits unavailable"}))
        return 0
    try:
        report = smoke()
    except Exception:
        report = {"status": "failed", "reason": "learning smoke failed"}
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
