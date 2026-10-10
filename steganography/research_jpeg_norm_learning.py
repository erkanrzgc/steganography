"""Explicit fresh local BN/GN learning pilot; no downloads, rentals or deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
from pathlib import Path

from core import jpeg_norm_learning, srnet_groupnorm, srnet_model
from core.srnet_stream import regular_open
from steganography.research_jpeg import write_json
from steganography.research_jpeg_population_bn import sources as population_sources
from steganography.research_srnet_stream import fresh


def sources():
    result = population_sources()
    root = Path(__file__).resolve().parents[1]
    for name in (
        "core/srnet_groupnorm.py",
        "core/srnet_norm_training.py",
        "core/jpeg_norm_learning.py",
        "steganography/research_jpeg_norm_learning.py",
        "docs/JPEG_NORM_LEARNING_PROTOCOL.md",
    ):
        with regular_open(root / name) as stream:
            raw = stream.read(1024**2 + 1)
        if len(raw) > 1024**2:
            raise ValueError("normalization learning source exceeds bound")
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def execute(root, audit, arm, out):
    fresh(out)
    before = sources()
    model, report = jpeg_norm_learning.fit_arm(root, audit, arm)
    if sources() != before:
        raise ValueError("normalization learning sources changed")
    fresh(out)
    out.mkdir(parents=False)
    writer = srnet_model.save_model if arm == "bn" else srnet_groupnorm.save_model
    report["model_sha256"] = writer(model, out / "model.npz")
    report["source_sha256"] = before
    write_json(out / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "audit", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--arm", choices=("bn", "gn"), required=True)
    args = parser.parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3601))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    report = execute(args.root, args.audit, args.arm, args.out)
    print(
        json.dumps(
            {
                "status": report["status"],
                "arm": args.arm,
                "optimizer_updates": report["optimizer_updates"],
            }
        )
    )
    return 0 if report["singleton_batch_parity"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
