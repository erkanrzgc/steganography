"""Explicit local CUDA population-BN control; no rentals or deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
from pathlib import Path

from core import jpeg_population_control, srnet_model
from core.srnet_stream import regular_open
from steganography.research_jpeg import write_json
from steganography.research_jpeg_timing_probe import sources as timing_sources
from steganography.research_srnet_stream import fresh


def sources():
    result = timing_sources()
    root = Path(__file__).resolve().parents[1]
    for name in (
        "core/srnet_population_bn.py",
        "core/jpeg_population_control.py",
        "steganography/research_jpeg_population_bn.py",
        "docs/JPEG_POPULATION_BN_PROTOCOL.md",
    ):
        with regular_open(root / name) as stream:
            raw = stream.read(1024**2 + 1)
        if len(raw) > 1024**2:
            raise ValueError("population source exceeds bound")
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def execute(root, audit, model, out):
    fresh(out)
    before = sources()
    clone, report = jpeg_population_control.control(root, audit, model)
    if sources() != before:
        raise ValueError("population control sources changed")
    fresh(out)
    out.mkdir(parents=False)
    report["refreshed_model_sha256"] = srnet_model.save_model(clone, out / "model.npz")
    report["source_sha256"] = before
    write_json(out / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "audit", "model", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3601))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    report = execute(args.root, args.audit, args.model, args.out)
    print(
        json.dumps(
            {
                "status": report["status"],
                "numerical_gates_passed": report["numerical_gates_passed"],
            }
        )
    )
    return 0 if report["numerical_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
