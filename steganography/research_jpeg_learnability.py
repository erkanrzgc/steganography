"""Explicit own-GPU real-pair control; report only, no deployment/downloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
from pathlib import Path

from core import jpeg_learnability
from core.srnet_stream import regular_open
from steganography.research_jpeg import write_json
from steganography.research_jpeg_norm_learning import sources as norm_sources
from steganography.research_srnet_stream import fresh


def sources():
    result = norm_sources()
    root = Path(__file__).resolve().parents[1]
    for name in (
        "core/srnet_samepool.py",
        "core/jpeg_learnability.py",
        "steganography/research_jpeg_learnability.py",
        "docs/JPEG_LEARNABILITY_PROTOCOL.md",
    ):
        with regular_open(root / name) as stream:
            raw = stream.read(1024**2 + 1)
        if len(raw) > 1024**2:
            raise ValueError("learnability source exceeds bound")
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def execute(root, audit, arm, out):
    fresh(out)
    before = sources()
    report = jpeg_learnability.fit_arm(root, audit, arm)
    if sources() != before:
        raise ValueError("learnability sources changed")
    fresh(out)
    out.mkdir(parents=False)
    report["source_sha256"] = before
    write_json(out / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "audit", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=("legacy", "reference"), required=True)
    args = parser.parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3601))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024**2, 2 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    report = execute(args.root, args.audit, args.arm, args.out)
    print(
        json.dumps(
            {
                "status": report["status"],
                "arm": args.arm,
                "learnability_criterion_passed": report["final"]["learnability_criterion_passed"],
            }
        )
    )
    return 0 if all(report["singleton_batch_parity"][k] for k in ("initial", "final")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
