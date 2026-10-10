"""Explicit isolated own-GPU batch capacity check; no datasets or model exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

from core import srnet_batch_probe, srnet_cuda
from core.srnet_stream import regular_open
from steganography.research_jpeg import write_json
from steganography.research_jpeg_curriculum import sources as previous_sources
from steganography.research_srnet_stream import fresh


def sources():
    result = previous_sources()
    root = Path(__file__).resolve().parents[1]
    for name in (
        "core/srnet_batch_probe.py",
        "steganography/research_batch_probe.py",
        "docs/JPEG_BATCH_PROBE_PROTOCOL.md",
    ):
        with regular_open(root / name) as stream:
            raw = stream.read(1024**2 + 1)
        if len(raw) > 1024**2:
            raise ValueError("batch probe source exceeds bound")
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def worker(rows):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (160, 165))
    resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
    before = sources()
    try:
        result = {"status": "completed", **srnet_batch_probe.probe(rows)}
    except srnet_cuda.CUDAUnavailable as exc:
        result = {"status": "unavailable", "reason": exc.reason}
    except ImportError:
        result = {"status": "unavailable", "reason": "optional_dependency_unavailable"}
    except Exception as exc:
        import torch

        if not isinstance(exc, torch.cuda.OutOfMemoryError):
            raise
        result = {"status": "unavailable", "reason": "gpu_allocator_out_of_memory"}
    if sources() != before:
        raise ValueError("batch probe sources changed")
    return {**result, "source_sha256": before}


def run(rows, out):
    srnet_batch_probe.generated(rows)  # Validate before spawning or creating output.
    fresh(out)
    before = sources()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryFile() as output:
        try:
            process = subprocess.run(  # noqa: S603 - fixed module, bounded child, no shell
                [sys.executable, "-m", "steganography.research_batch_probe",
                 "--worker", "--rows", str(rows)],
                cwd=root, stdin=subprocess.DEVNULL, stdout=output,
                stderr=subprocess.DEVNULL, timeout=180, check=False,
                env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                     "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            )
        except subprocess.TimeoutExpired:
            result = {"status": "unavailable", "reason": "worker_timeout"}
        else:
            output.seek(0)
            raw = output.read(65537)
            if process.returncode or not 0 < len(raw) <= 65536:
                raise ValueError("batch worker failed or output exceeded bounds")
            result = json.loads(raw)
            if (
                not isinstance(result, dict)
                or result.get("status") not in {"completed", "unavailable"}
                or result.get("source_sha256") != before
                or (result.get("status") == "completed" and result.get("rows") != rows)
            ):
                raise ValueError("batch worker response invalid")
    if sources() != before:
        raise ValueError("batch probe sources changed")
    report = {
        **result, "schema_version": "srnet-real-batch-generated-probe-v1",
        "rows": rows, "source_sha256": before, "real_data_used": False,
        "real_model_trained": False, "accuracy_qualification": "unavailable",
        "cpu_fallback": False, "deployed": False,
    }
    write_json(out, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, type=int, choices=srnet_batch_probe.ROWS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.worker and args.out is None:
        parser.error("--out is required")
    if args.worker:
        print(json.dumps(worker(args.rows), allow_nan=False))
        return 0
    result = run(args.rows, args.out)
    print(json.dumps({"status": result["status"], "rows": args.rows}))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
