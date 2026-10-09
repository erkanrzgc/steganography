"""Explicit isolated generated CUDA timing, no dataset access or model output."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from core import srnet_cuda, srnet_profile
from core.srnet_stream import document, regular_open
from steganography.research_cuda_probe import sources as probe_sources
from steganography.research_jpeg import write_json
from steganography.research_srnet_stream import fresh


def sources():
    result = probe_sources()
    root = Path(__file__).resolve().parents[1]
    for name in ("core/srnet_profile.py", "steganography/research_cuda_profile.py"):
        with regular_open(root / name) as stream:
            raw = stream.read(1024**2 + 1)
        if len(raw) > 1024**2:
            raise ValueError("profile source size exceeded")
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def execute(out):
    fresh(out)
    before = sources()
    report = srnet_profile.profile()
    if sources() != before:
        raise ValueError("profile sources changed")
    report.update(
        schema_version="srnet-cuda-generated-profile-v1",
        status="completed",
        source_sha256=before,
    )
    write_json(out, report)
    return report


def run_job(out):
    fresh(out)
    original = sources()
    root = Path(__file__).resolve().parents[1]
    try:
        with tempfile.TemporaryFile() as response:
            result = subprocess.run(  # noqa: S603 — fixed trusted worker, no shell
                [
                    sys.executable,
                    "-m",
                    "steganography.research_cuda_profile",
                    "--worker",
                    "--out",
                    str(out.absolute()),
                ],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=response,
                stderr=subprocess.DEVNULL,
                timeout=srnet_profile.MAX_SECONDS + 30,
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
            raise ValueError("profile worker response oversized")
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("profile worker response must be an object")
        if data.get("status") == "unavailable":
            report = {
                "schema_version": "srnet-cuda-generated-profile-v1",
                "status": "unavailable",
                "reason": "bounded_cuda_prerequisites_unavailable",
                "profile_completed": False,
                "cpu_fallback": False,
                "real_data_used": False,
                "real_model_trained": False,
                "accuracy_qualification": "unavailable",
                "source_sha256": original,
            }
            write_json(out, report)
            return report
        if result.returncode != 0 or data.get("status") != "completed":
            raise ValueError("profile worker incomplete")
        report = document(out, data["report_sha256"])
        if report.get("source_sha256") != original or sources() != original:
            raise ValueError("profile parent/worker sources changed")
        return report
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("CUDA profile failed; partial evidence unusable") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker:
            import resource

            srnet_cuda.host_bound()
            resource.setrlimit(resource.RLIMIT_CPU, (360, 361))
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            execute(args.out)
            with regular_open(args.out) as stream:
                checksum = hashlib.sha256(stream.read(1024**2 + 1)).hexdigest()
            print(json.dumps({"status": "completed", "report_sha256": checksum}))
        else:
            report = run_job(args.out)
            print(json.dumps({"status": report["status"]}))
            return 0 if report["status"] == "completed" else 2
    except (srnet_cuda.CUDAUnavailable, ImportError) as exc:
        reason = (
            exc.reason if isinstance(exc, srnet_cuda.CUDAUnavailable) else "torch_not_installed"
        )
        print(json.dumps({"status": "unavailable", "reason": reason}))
        return 2
    except Exception:
        print('{"status":"failed"}')
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
