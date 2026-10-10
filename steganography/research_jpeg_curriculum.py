"""Explicit local preparation and own-GPU curriculum, never a deployment job."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

from core import jpeg_curriculum
from core.srnet_stream import document, regular_open
from steganography.research_jpeg import write_json
from steganography.research_jpeg_learnability import sources as previous_sources
from steganography.research_srnet_stream import fresh


def sources():
    result = previous_sources()
    root = Path(__file__).resolve().parents[1]
    for name in (
        "core/jpeg_curriculum.py",
        "steganography/research_jpeg_curriculum.py",
        "docs/JPEG_CURRICULUM_PROTOCOL.md",
    ):
        with regular_open(root / name) as stream:
            raw = stream.read(1024**2 + 1)
        if len(raw) > 1024**2:
            raise ValueError("curriculum source exceeds bound")
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def generate(data, lineage):
    if importlib.metadata.version("conseal") != "2025.11":
        raise ValueError("curriculum requires pinned conseal 2025.11")
    with tempfile.TemporaryFile() as output:
        process = subprocess.run(  # noqa: S603 - fixed bounded module, no shell
            [sys.executable, "-m", "steganography.research_jpeg_curriculum", "worker", lineage],
            input=data,
            stdout=output,
            stderr=subprocess.DEVNULL,
            timeout=90,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        )
        output.seek(0)
        raw = output.read(256 * 1024 + 1)
    if process.returncode or not 0 < len(raw) <= 256 * 1024:
        raise ValueError("curriculum worker failed or exceeded bound")
    return raw


def worker(lineage):
    import conseal
    import jpeglib

    from core.jpeg_float256 import region
    from core.jpeg_scale import identity

    identity(lineage)
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (80, 85))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024, 256 * 1024))
    data = sys.stdin.buffer.read(256 * 1024 + 1)
    if not 0 < len(data) <= 256 * 1024 or region(data)["image_size"] != [512, 512]:
        raise ValueError("curriculum worker input bound")
    with tempfile.TemporaryDirectory(prefix="jpeg-curriculum-") as folder:
        cover, stego = Path(folder) / "cover.jpg", Path(folder) / "stego.jpg"
        with cover.open("xb") as stream:
            stream.write(data)
        jpeglib.version.set("6b")
        jpeg = jpeglib.read_dct(str(cover))
        jpeg.load()
        if jpeg.jpeg_color_space.name != "JCS_GRAYSCALE":
            raise ValueError("curriculum requires grayscale")
        pixels = jpeglib.read_spatial(str(cover), jpeglib.JCS_GRAYSCALE).spatial[..., 0]
        seed = int.from_bytes(
            hashlib.sha256(f"curriculum:{jpeg_curriculum.SEED}:{lineage}".encode()).digest()[:4],
            "big",
        )
        jpeg.Y = conseal.juniward.simulate_single_channel(
            pixels, jpeg.Y, jpeg.get_component_qt(0), 0.4, seed=seed
        )
        jpeg.write_dct(str(stego))
        with stego.open("rb") as stream:
            raw = stream.read(256 * 1024 + 1)
        if not 0 < len(raw) <= 256 * 1024:
            raise ValueError("curriculum output bound")
        sys.stdout.buffer.write(raw)


def execute(args):
    fresh(args.out)
    before = sources()
    if args.action == "prepare":
        report = jpeg_curriculum.prepare(args.root, args.audit, args.out, generate)
    else:
        if args.high is None or args.high_sha256 is None:
            raise ValueError("explicit high cache/checksum required")
        if document(args.high / "manifest.json", args.high_sha256).get("source_sha256") != before:
            raise ValueError("curriculum preparation source binding mismatch")
        report = jpeg_curriculum.fit(args.root, args.audit, args.high, args.high_sha256)
        fresh(args.out)
        args.out.mkdir(parents=False)
    if sources() != before:
        raise ValueError("curriculum sources changed")
    report["source_sha256"] = before
    if args.action == "prepare":
        write_json(args.out / "manifest.json", report)
    write_json(args.out / "report.json", report)
    return report


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if args and args[0] == "worker":
        if len(args) != 2:
            return 1
        worker(args[1])
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train"))
    for name in ("root", "audit", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--high", type=Path)
    parser.add_argument("--high-sha256")
    parsed = parser.parse_args(args)
    resource.setrlimit(resource.RLIMIT_CPU, (4800, 4801))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
    if parsed.action == "prepare":
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    report = execute(parsed)
    print(json.dumps({"status": report["status"], "action": parsed.action}))
    return 0 if parsed.action == "prepare" or report["singleton_batch_parity"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
