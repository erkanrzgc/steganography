#!/usr/bin/env python3
"""Audit frozen raw pixel caches; independent bounded crop oracle, no model training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.jpeg_features import MAX_IMAGE_BYTES, validate_jpeg  # noqa: E402
from steganography.research_features import read_document  # noqa: E402
from steganography.research_jpeg import write_json  # noqa: E402
from steganography.research_pixels import load_pixels, regular  # noqa: E402

MANIFEST_SHA = "0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14"


def sha(path):
    regular(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024**2):
            digest.update(data)
    return digest.hexdigest()


def oracle_main():
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (15, 16))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_IMAGE_BYTES, MAX_IMAGE_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        data = sys.stdin.buffer.read(MAX_IMAGE_BYTES + 1)
        validate_jpeg(data)
        with Image.open(io.BytesIO(data)) as image:
            if image.mode not in {"L", "RGB"} or min(image.size) < 128:
                raise ValueError("invalid oracle geometry/colorspace")
            width, height = image.size
            full = np.asarray(image.convert("L"), dtype="u1")
        # Independent full-raster indexing, not the production crop method.
        x, y = (width - 128) // 2, (height - 128) // 2
        raw = full[y : y + 128, x : x + 128].tobytes()
        if len(raw) != 16384:
            raise ValueError("oracle shape mismatch")
        sys.stdout.buffer.write(raw)
        return 0
    except Exception:
        return 1


def independent_crop(data):
    if not 0 < len(data) <= MAX_IMAGE_BYTES:
        raise ValueError("independent pixel oracle input exceeds limits")
    with tempfile.TemporaryFile() as output:
        result = subprocess.run(  # noqa: S603 - trusted script, fixed oracle, bounded bytes
            [sys.executable, str(ROOT / "scripts/audit-pixel-preparation.py"), "--oracle"],
            input=data,
            stdout=output,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
            cwd=ROOT,
            env={
                "PATH": os.environ.get("PATH", ""),
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
        output.seek(0)
        raw = output.read(16385)
    if result.returncode != 0 or len(raw) != 16384:
        raise ValueError("independent pixel oracle failed")
    return np.frombuffer(raw, dtype="u1").reshape(128, 128)


def audit(manifest_path, corpus, experiment):
    manifest, digest = read_document(manifest_path)
    if digest != MANIFEST_SHA:
        raise ValueError("frozen pixel preparation manifest mismatch")
    count = 0
    for sample in manifest["samples"]:
        path = corpus / sample["path"]
        if sha(path) != sample["sha256"] or path.stat().st_size != sample["size"]:
            raise ValueError("pixel corpus integrity failure")
        count += 1
    caches, artifacts, examples = {}, {}, []
    for split in ("train", "validation"):
        cache = experiment / split / "cache.json"
        values, samples, descriptor = load_pixels(
            manifest_path, cache, checksum=sha(cache), split=split
        )
        seen = set()
        for i, sample in enumerate(samples):
            key = (sample["source_group"], sample.get("quality_factor"), sample["method"])
            if key in seen:
                continue
            seen.add(key)
            with (corpus / sample["path"]).open("rb") as stream:
                data = stream.read(MAX_IMAGE_BYTES + 1)
            expected = independent_crop(data)
            np.testing.assert_array_equal(expected, values[i, 0])
            examples.append(
                {
                    "split": split,
                    "sha256": sample["sha256"],
                    "method": sample["method"],
                    "quality_factor": sample.get("quality_factor"),
                }
            )
        caches[split] = {
            "rows": len(samples),
            "shape": descriptor["shape"],
            "decoder": descriptor["decoder"],
            "bytes": values.nbytes,
            "descriptor_sha256": sha(cache),
            "data_sha256": descriptor["data_sha256"],
        }
        for name in ("cache.json", "pixels.u8"):
            artifacts[f"{split}/{name}"] = sha(experiment / split / name)
    execution, execution_hash = read_document(experiment / "extraction-summary.json")
    return {
        "schema_version": "pixel-residual-preparation-v1",
        "manifest_sha256": digest,
        "protocol_sha256": sha(ROOT / "docs/PIXEL_RESIDUAL_PREPARATION_PROTOCOL.md"),
        "caches": caches,
        "artifact_sha256": {**artifacts, "extraction-summary.json": execution_hash},
        "execution": execution,
        "versions": {k: importlib.metadata.version(k) for k in ("numpy", "pillow")},
        "independent_audit": {
            "passed": True,
            "files_rehashed": count,
            "crop_examples": examples,
            "crop_examples_checked": len(examples),
            "independent_jpeg_decoder": False,
        },
        "trained": False,
        "deployed": False,
        "calibrated": False,
        "primary_detection_changed": False,
        "support_status": "experimental",
        "qualification": "unavailable",
        "accuracy": "unavailable",
    }


def main():
    if sys.argv[1:] == ["--oracle"]:
        return oracle_main()
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "corpus", "experiment", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    write_json(args.out, audit(args.manifest, args.corpus, args.experiment))
    print(
        "Complete: corpus/cache integrity and bounded full-raster crop parity; no accuracy claim."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
