"""Isolated generated CUDA warmup/parity probe; never a real-data accuracy test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from core import srnet, srnet_cuda, srnet_training
from core.srnet_stream import deadline_after, document, regular_open
from steganography.research_jpeg import write_json
from steganography.research_srnet_stream import fresh, snapshot


def sources():
    result = snapshot()
    with regular_open(Path(__file__)) as stream:
        result["steganography/research_cuda_probe.py"] = hashlib.sha256(
            stream.read(1024**2)
        ).hexdigest()
    return result


def probe(out):
    fresh(out)
    started = time.monotonic()
    deadline = deadline_after(60)
    original = sources()
    execution = srnet_cuda.inspect()
    import torch

    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        yy, xx = np.indices((256, 256))
        base = (128 + 24 * np.sin(xx / 11) + 16 * np.cos(yy / 13)).astype("<f4")
        values = np.stack([base, base + 8 * ((xx + yy) % 2), base.T, base.T + 8 * ((xx + yy) % 2)])
        values = values.astype("<f4").reshape(4, 1, 256, 256)
        model, records = srnet_training._learn(
            fetch=lambda _: values.copy(),
            batches=lambda _: np.array([[0, 1, 2, 3]]),
            epochs=1,
            seed=20261008,
            params=srnet_training.settings({"max_seconds": 60}),
            target_pairs=2,
            outer_deadline=deadline,
            device="cuda:0",
        )
        deadline()
        cpu_logits = srnet.float_logits(model, values[:1])
        with srnet_cuda.policy(), torch.no_grad():
            model.to("cuda:0")
            gpu_logits = model(torch.from_numpy(values[:1]).to("cuda:0")).detach().cpu().numpy()
            torch.cuda.synchronize(0)
            model.cpu()
        if not np.isfinite(gpu_logits).all() or not np.allclose(
            cpu_logits, gpu_logits, atol=1e-4, rtol=1e-4
        ):
            raise ValueError("generated CUDA/CPU inference parity failed")
        deadline()
        if sources() != original:
            raise ValueError("CUDA probe sources changed")
        report = {
            "schema_version": "srnet-cuda-generated-probe-v1",
            "status": "completed",
            "execution": execution,
            "source_sha256": original,
            "input": "generated-wave-checker-4x1x256x256-f32-v1",
            "generated_optimizer_updates": records[0]["updates"],
            "generated_mean_loss": records[0]["mean_pair_loss"],
            "input_tensor_bytes": values.nbytes,
            "cpu_gpu_logit_max_difference": float(np.max(np.abs(cpu_logits - gpu_logits))),
            "parity_atol": 1e-4,
            "parity_rtol": 1e-4,
            "process_peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "process_peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "seconds": time.monotonic() - started,
            "real_data_used": False,
            "real_model_trained": False,
            "independent_math_oracle": False,
            "accuracy_qualification": "unavailable",
            "deployed": False,
        }
        write_json(out, report)
        return report
    finally:
        torch.set_num_threads(previous)


def run_job(out):
    fresh(out)
    source_hashes = sources()
    root = Path(__file__).resolve().parents[1]
    try:
        with tempfile.TemporaryFile() as response:
            result = subprocess.run(  # noqa: S603 — fixed trusted probe, no shell/child workers
                [
                    sys.executable,
                    "-m",
                    "steganography.research_cuda_probe",
                    "--worker",
                    "--out",
                    str(out.absolute()),
                ],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=response,
                stderr=subprocess.DEVNULL,
                timeout=90,
                check=False,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": str(root),
                    "OMP_NUM_THREADS": "2",
                    "OPENBLAS_NUM_THREADS": "1",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                },
            )
            response.seek(0)
            raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError("CUDA worker output exceeded limits")
        result_data = json.loads(raw)
        if not isinstance(result_data, dict):
            raise ValueError("CUDA worker response must be an object")
        if result_data.get("status") == "unavailable":
            report = {
                "schema_version": "srnet-cuda-generated-probe-v1",
                "status": "unavailable",
                "reason": "bounded_cuda_prerequisites_unavailable",
                "real_data_used": False,
                "real_model_trained": False,
                "accuracy_qualification": "unavailable",
                "hardware_probe_completed": False,
                "cpu_fallback": False,
                "source_sha256": source_hashes,
            }
            write_json(out, report)
            return report
        if result.returncode != 0 or result_data.get("status") != "completed":
            raise ValueError("CUDA worker incomplete")
        return document(out, result_data["report_sha256"])
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("CUDA probe failed; partial evidence unusable") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            import resource

            srnet_cuda.host_bound()  # RAM bound before CUDA virtual-address reservations.
            resource.setrlimit(resource.RLIMIT_CPU, (120, 121))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            probe(args.out)
            with regular_open(args.out) as stream:
                checksum = hashlib.sha256(stream.read(1024**2 + 1)).hexdigest()
            print(json.dumps({"status": "completed", "report_sha256": checksum}))
        else:
            report = run_job(args.out)
            print(json.dumps({"status": report["status"]}))
            return 0 if report["status"] == "completed" else 2
    except srnet_cuda.CUDAUnavailable as exc:
        print(json.dumps({"status": "unavailable", "reason": exc.reason}))
        return 2
    except Exception:
        print('{"status":"failed"}')
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
