"""Explicit CPU-only non-JPEG preparation/audit with isolated bounded workers."""

from __future__ import annotations

import argparse
import contextlib
import os
import resource
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from core.nonjpeg_data import config, document, prepare, regular_read, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "core/nonjpeg_data.py",
    "core/nonjpeg_audit.py",
    "steganography/research_nonjpeg.py",
    "steganography/research.py",
    "core/dataset.py",
    "docs/NONJPEG_PREPARATION_PROTOCOL.md",
)


def fingerprints():
    return {name: sha(regular_read(ROOT / name, 8 * 1024**2)) for name in FILES}


def cancel_requested(signum, frame):
    raise KeyboardInterrupt


def worker(module: str, arguments: list[str]) -> None:
    if module not in ("steganography.research_nonjpeg", "core.nonjpeg_audit"):
        raise ValueError("fixed worker module required")
    process = subprocess.Popen(  # noqa: S603 - fixed owned Python modules, no shell
        [sys.executable, "-m", module, *arguments],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
    )
    try:
        if process.wait(timeout=1800) != 0:
            raise ValueError("bounded worker failed")
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def research_manifests(out: Path, audit_hash: str):
    from steganography.research import verify_dataset_manifest

    corpus, manifest_hash = document(out / "manifest.json")
    exports = {}
    for family, formats in (("spatial", ("PNG", "BMP")), ("audio", ("WAV",))):
        rows = [
            {**row, "payload_rate": row.get("embedded_bits", 0) / row.get("eligible_units", 1)}
            for row in corpus["samples"]
            if row["format"] in formats
        ]
        if not rows:
            continue
        result = {
            "schema_version": "1.0",
            "seed": 20261012,
            "samples": rows,
            "sample_count": len(rows),
            "development_only": True,
            "preparation_manifest_sha256": manifest_hash,
            "independent_audit_sha256": audit_hash,
            "partition": {"policy": "identity-camera-device-development-v1", "test_sources": []},
            "accuracy_qualification": "unavailable",
            "model_trained": False,
        }
        verify_dataset_manifest(result, source=out, verify_files=True)
        filename = f"{family}-research.json"
        write_json(out / filename, result)
        exports[filename] = sha(regular_read(out / filename, 8 * 1024**2))
    return exports


def run(config_path: Path, out: Path):
    _, digest = config(config_path)
    sources = fingerprints()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("fresh nonsymlink output required")
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    write_json(out / "source-bindings.json", sources)
    started = time.monotonic()
    result = {
        "schema_version": "nonjpeg-preparation-report-v1",
        "status": "incomplete",
        "config_sha256": digest,
        "source_sha256": sources,
        "optimizer_updates": 0,
        "model_trained": False,
        "detector_scored": False,
        "accuracy_qualification": "unavailable",
        "deployed": False,
        "gpu_used": False,
        "cloud_spending": 0,
        "stage": "preparation",
    }
    main_thread = threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGTERM)
    if main_thread:
        signal.signal(signal.SIGTERM, cancel_requested)
    try:
        worker(
            "steganography.research_nonjpeg",
            ["--config", str(config_path.absolute()), "--out", str(out.absolute()), "--worker"],
        )
        if config(config_path)[1] != digest or fingerprints() != sources:
            raise ValueError("input/source changed")
        result["stage"] = "independent_audit"
        worker(
            "core.nonjpeg_audit",
            [
                "--config",
                str(config_path.absolute()),
                "--root",
                str(out.absolute()),
                "--bindings",
                str((out / "source-bindings.json").absolute()),
            ],
        )
        audit, audit_hash = document(out / "independent-audit.json")
        if (
            audit["status"] != "completed"
            or audit["config_sha256"] != digest
            or audit["source_sha256"] != sources
            or config(config_path)[1] != digest
            or fingerprints() != sources
        ):
            raise ValueError("independent audit/source binding")
        result.update(
            status="completed",
            stage="complete",
            audit_sha256=audit_hash,
            manifest_sha256=audit["manifest_sha256"],
            original_counts=audit["original_counts"],
            artifact_count=audit["artifact_count"],
            artifact_bytes=audit["artifact_bytes"],
            independent_scalar_recoveries=audit["independent_scalar_recoveries"],
            research_manifests=research_manifests(out, audit_hash),
        )
    except KeyboardInterrupt:
        result.update(status="cancelled", reason="cancelled")
    except (ValueError, OSError, subprocess.TimeoutExpired):
        result["reason"] = "preparation_or_independent_audit_incomplete"
    finally:
        if main_thread:
            signal.signal(signal.SIGTERM, previous)
    result["seconds"] = time.monotonic() - started
    write_json(out / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        cfg, digest = config(args.config)
        prepare(cfg, args.out, config_sha256=digest, source_sha256=fingerprints())
        return 0
    result = run(args.config, args.out)
    print(result["status"])
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
