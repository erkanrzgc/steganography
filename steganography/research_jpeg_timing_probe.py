"""Explicit isolated real-data CUDA timing; never rents hardware or deploys weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from core import jpeg_timing_probe as probe
from core import srnet_cuda
from core.srnet_stream import document, regular_open
from steganography.research_cuda_probe import sources as existing_sources
from steganography.research_jpeg import write_json
from steganography.research_srnet_stream import fresh


def sources():
    hashes = existing_sources()
    root = Path(__file__).resolve().parents[1]
    for name in (
        "core/jpeg_timing.py",
        "core/jpeg_timing_probe.py",
        "core/srnet_diversity_sampling.py",
        "steganography/research_jpeg_timing_probe.py",
    ):
        with regular_open(root / name) as stream:
            raw = stream.read(1024**2 + 1)
        if len(raw) > 1024**2:
            raise ValueError("timing source exceeds bound")
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return hashes


def execute(root, audit, out):
    fresh(out)
    before = sources()
    report = probe.profile(root, audit)
    if sources() != before:
        raise ValueError("real timing sources changed")
    report.update(
        schema_version="srnet-cuda-real-timing-v1", status="completed", source_sha256=before
    )
    write_json(out, report)
    return report


def run_job(root, audit, out):
    fresh(out)
    before = sources()
    code_root = Path(__file__).resolve().parents[1]
    try:
        with tempfile.TemporaryFile() as response:
            result = subprocess.run(  # noqa: S603 - fixed isolated trusted worker, no shell
                [
                    sys.executable,
                    "-m",
                    "steganography.research_jpeg_timing_probe",
                    "--worker",
                    "--root",
                    str(root.absolute()),
                    "--audit",
                    str(audit.absolute()),
                    "--out",
                    str(out.absolute()),
                ],
                cwd=code_root,
                stdin=subprocess.DEVNULL,
                stdout=response,
                stderr=subprocess.DEVNULL,
                timeout=probe.MAX_SECONDS + 30,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": str(code_root),
                    "OMP_NUM_THREADS": "2",
                    "OPENBLAS_NUM_THREADS": "1",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                },
            )
            response.seek(0)
            raw = response.read(65537)
        if len(raw) > 65536:
            raise ValueError("timing worker response exceeds bound")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("timing worker response must be an object")
        if sources() != before:
            raise ValueError("timing sources mutated")
        if data.get("status") == "unavailable" and result.returncode == 2:
            report = {
                "schema_version": "srnet-cuda-real-timing-v1",
                "status": "unavailable",
                "reason": "bounded_cuda_prerequisites_unavailable",
                "source_sha256": before,
                "manifest_sha256": probe.MANIFEST_SHA,
                "audit_sha256": probe.AUDIT_SHA,
                "cpu_fallback": False,
                "real_data_used": False,
                "real_model_trained": False,
                "accuracy_qualification": "unavailable",
                "deployed": False,
            }
            write_json(out, report)
            return report
        if result.returncode or data.get("status") != "completed":
            raise ValueError("timing worker incomplete")
        report = document(out, data["report_sha256"])
        if (
            report.get("schema_version") != "srnet-cuda-real-timing-v1"
            or report.get("status") != "completed"
            or report.get("source_sha256") != before
            or report.get("manifest_sha256") != probe.MANIFEST_SHA
            or report.get("audit_sha256") != probe.AUDIT_SHA
            or report.get("real_optimizer_updates") != probe.UPDATES
            or report.get("execution", {}).get("device") != "cuda:0"
            or report.get("real_data_used") is not True
            or report.get("real_model_trained") is not True
            or report.get("production_model_trained") is not False
            or report.get("deployed") is not False
            or report.get("accuracy_qualification") != "unavailable"
            or report.get("projection") != probe.projection(report.get("steady_intervals_seconds"))
        ):
            raise ValueError("timing parent/worker proof mismatch")
        return report
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("real CUDA timing failed; partial evidence unusable") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "audit", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            import resource

            srnet_cuda.host_bound()
            resource.setrlimit(resource.RLIMIT_CPU, (360, 361))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            execute(args.root, args.audit, args.out)
            with regular_open(args.out) as stream:
                checksum = hashlib.sha256(stream.read(1024**2 + 1)).hexdigest()
            print(json.dumps({"status": "completed", "report_sha256": checksum}))
            return 0
        report = run_job(args.root, args.audit, args.out)
        print(json.dumps({"status": report["status"]}))
        return 0 if report["status"] == "completed" else 2
    except (srnet_cuda.CUDAUnavailable, ImportError):
        print('{"status":"unavailable"}')
    except Exception:
        print('{"status":"failed"}')
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
