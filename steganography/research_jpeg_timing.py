"""Explicit local timing-kit preparation and resource-limited BOWS worker."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from core import jpeg_timing
from core.srnet_stream import deadline_after


def generate(data, out, lineage):
    if importlib.metadata.version("conseal") != "2025.11":
        raise ValueError("timing kit requires pinned conseal 2025.11")
    with tempfile.TemporaryFile() as output:
        process = subprocess.run(  # noqa: S603 - fixed bounded simulation worker
            [
                sys.executable,
                "-m",
                "steganography.research_jpeg_timing",
                "worker",
                str(out),
                lineage,
            ],
            input=data,
            stdout=output,
            stderr=subprocess.DEVNULL,
            timeout=90,
            env={
                **os.environ,
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
        output.seek(0)
        raw = output.read(256 * 1024 + 1)
    if process.returncode or len(raw) > 256 * 1024:
        raise ValueError("timing simulation worker failed or exceeded output limit")
    return json.loads(raw)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "worker":
        import resource

        from steganography.research_jpeg_corpus import generate_lineage

        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (80, 85))
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024, 256 * 1024))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        try:
            if len(args) != 3:
                return 1
            rows = generate_lineage(
                sys.stdin.buffer.read(1024**2 + 1),
                Path(args[1]),
                args[2],
                "train",
                source_group="BOWS2",
            )
            print(json.dumps(rows, separators=(",", ":"), allow_nan=False))
            return 0
        except Exception:
            return 1
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("root", "bows", "out", "protocol", "audit", "bows-audit"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    for option in (
        "protocol-sha256",
        "index-sha256",
        "audit-sha256",
        "bows-sha256",
        "bows-audit-sha256",
    ):
        parser.add_argument(f"--{option}", required=True)
    parsed = vars(parser.parse_args(args))
    try:
        result = jpeg_timing.prepare(**parsed, deadline=deadline_after(600), generator=generate)
        print(json.dumps(result, allow_nan=False))
        return 0
    except Exception as exc:
        # No absolute paths or source data in the portable status.
        print(json.dumps({"status": "failed", "reason": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
