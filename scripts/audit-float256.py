#!/usr/bin/env python3
"""Independent full-raster SciPy IDCT audit of fixed local float preparation."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import jpeg_float256 as fp  # noqa: E402
from steganography.research_features import read_document  # noqa: E402
from steganography.research_jpeg import write_json  # noqa: E402
from steganography.research_pixels import load_pixels, regular  # noqa: E402

MANIFEST_SHA = "0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def selection(samples):
    indices = []
    for quality in (None, 75, 95):
        for method in (None, "JUNIWARD", "UERD"):
            matches = [
                i
                for i, s in enumerate(samples)
                if s["quality_factor"] == quality and s["method"] == method
            ]
            if not matches:
                raise ValueError("missing fixed audit context")
            indices.append(matches[0])
    return indices


def reference(path, width, height):
    import jpeglib
    from scipy.fft import idctn

    jpeglib.version.set("6b")
    jpeg = jpeglib.read_dct(str(path))
    coefficients = jpeg.Y.astype(np.float64) * jpeg.get_component_qt(0)
    full = idctn(coefficients, axes=(-2, -1), norm="ortho") + 128
    full = full.transpose(0, 2, 1, 3).reshape(coefficients.shape[0] * 8, coefficients.shape[1] * 8)
    # Recompute the declared phase-aligned crop, do not call the implementation.
    left, top = 8 * ((width - 256) // 16), 8 * ((height - 256) // 16)
    return full[top : top + 256, left : left + 256].astype("<f4")


def audit(manifest_path, source, cache_root):
    manifest, digest = read_document(manifest_path)
    if digest != MANIFEST_SHA:
        raise ValueError("frozen corpus checksum mismatch")
    for sample in manifest["samples"]:
        path = source / sample["path"]
        regular(path)
        if path.stat().st_size != sample["size"] or sha(path) != sample["sha256"]:
            raise ValueError("original corpus changed")
    examples, splits = [], {}
    for split in ("train", "validation"):
        cache = cache_root / split / "cache.json"
        values, samples, descriptor = load_pixels(
            manifest_path, cache, checksum=sha(cache), split=split, _float=True
        )
        splits[split] = {
            "rows": len(samples),
            "cache_sha256": sha(cache),
            "data_sha256": descriptor["data_sha256"],
            "bytes": values.nbytes,
        }
        for i in selection(samples):
            sample = samples[i]
            width, height = descriptor["rows"][i]["image_size"]
            expected = reference(source / sample["path"], width, height)
            difference = float(np.max(np.abs(expected - values[i, 0])))
            examples.append(
                {
                    "split": split,
                    "row": i,
                    "sha256": sample["sha256"],
                    "method": sample["method"],
                    "quality_factor": sample["quality_factor"],
                    "maximum_pixel_difference": difference,
                    "passed": difference <= 1e-4,
                }
            )
        del values
    return {
        "schema_version": "float256-preparation-v1",
        "manifest_sha256": digest,
        "protocol_sha256": sha(ROOT / "docs/JPEG_FLOAT256_PROTOCOL.md"),
        "feature_version": fp.FEATURE_VERSION,
        "decoder": fp.decoder_contract(),
        "files_rehashed": len(manifest["samples"]),
        "splits": splits,
        "examples": examples,
        "absolute_tolerance": 1e-4,
        "relative_tolerance": 0,
        "mathematical_audit_passed": all(e["passed"] for e in examples),
        "native_parser_independently_verified": False,
        "deployed": False,
        "qualification": "unavailable",
        "real_training": "unavailable",
        "accuracy_metrics": "unavailable",
        "primary_detection_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "source", "cache-root", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.manifest, args.source, args.cache_root)
    write_json(args.out, report)
    print("Complete: integrity and mathematical reconstruction audited, not accuracy.")
    return 0 if report["mathematical_audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
