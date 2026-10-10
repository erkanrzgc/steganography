#!/usr/bin/env python3
"""Independent whole-kit byte/lineage and full-raster SciPy IDCT audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import time
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import jpeglib
import numpy as np
from scipy.fft import idctn

PROTOCOL_SHA = "a96560070d08a3e9de34ccac1a8d9e5124d71fd48f1337b255af91ee396bcf96"
ROW_BYTES = 256 * 256 * 4


def read(path, limit):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("audit does not follow symlinks")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("audit requires regular files")
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("audit size limit")
    return raw


def sha(data):
    return hashlib.sha256(data).hexdigest()


def doc(path, checksum=None):
    raw = read(path, 16 * 1024**2)
    if checksum is not None and sha(raw) != checksum:
        raise ValueError("audit metadata binding mismatch")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("audit metadata object required")
    return value, sha(raw)


def name(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("audit unsafe path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("audit unsafe path")
    return value


def audit(root, bows, old_root, protocol, manifest_sha256):
    started = time.monotonic()
    frozen, _ = doc(protocol, PROTOCOL_SHA)
    manifest, digest = doc(root / "manifest.json", manifest_sha256)
    binding = frozen["inputs"]
    if (
        manifest.get("schema_version") != "jpeg-real-timing-kit-v1"
        or manifest.get("status") != "completed"
        or manifest.get("protocol_sha256") != PROTOCOL_SHA
        or manifest.get("inputs") != binding
    ):
        raise ValueError("audit timing contract mismatch")
    source, _ = doc(bows / "source.json", binding["bows_sha256"])
    index, _ = doc(old_root / "index.json", binding["index_sha256"])
    prior = []
    for block in index["blocks"]:
        value, _ = doc(old_root / name(block["path"]), block["manifest_sha256"])
        prior.extend(value["samples"])
    originals = manifest["bows_originals"]
    expected_originals = sorted(
        (r for r in source["samples"] if r["split"] == "train"),
        key=lambda r: sha(f"timing-kit:20261010:BOWS2:{r['lineage']}".encode()),
    )[:32]
    if originals != expected_originals:
        raise ValueError("audit BOWS train-only deterministic selection mismatch")
    for row in originals:
        raw = read(bows / name(row["path"]), 1024**2)
        if len(raw) != row["bytes"] or sha(raw) != row["lineage"] or sha(raw) != row["sha256"]:
            raise ValueError("audit BOWS original bytes mismatch")
    rows = manifest["samples"]
    if len(rows) != 480 or len({r["sha256"] for r in rows}) != 480:
        raise ValueError("audit unique complete row count")
    for origin in ("ALASKA2", "BOSSbase-1.01"):
        available = {
            r["lineage"] for r in prior if r["source_group"] == origin and r["split"] == "train"
        }
        chosen = set(
            sorted(available, key=lambda key: sha(f"timing-kit:20261010:{origin}:{key}".encode()))[
                :32
            ]
        )
        expected = [r for r in prior if r["source_group"] == origin and r["lineage"] in chosen]
        actual = [r for r in rows if r["source_group"] == origin]
        if {r["sha256"] for r in expected} != {r["sha256"] for r in actual}:
            raise ValueError("audit prior complete selection mismatch")
        by_hash = {r["sha256"]: r for r in expected}
        for row in actual:
            if {k: v for k, v in row.items() if k != "path"} != {
                k: v for k, v in by_hash[row["sha256"]].items() if k != "path"
            }:
                raise ValueError("audit prior metadata changed")
    owners = {}
    groups = defaultdict(list)
    for row in rows:
        if (
            row["split"] != "train"
            or owners.setdefault(row["lineage"], row["source_group"]) != row["source_group"]
        ):
            raise ValueError("audit role or cross-origin original overlap")
        groups[(row["source_group"], row["lineage"], row["quality_factor"])].append(row)
    if len(owners) != 96:
        raise ValueError("audit original count")
    for key, family in groups.items():
        if len(family) != 3 or {(r["label"], r["method"]) for r in family} != {
            ("cover", None),
            ("stego", "JUNIWARD"),
            ("stego", "UERD"),
        }:
            raise ValueError("audit incomplete method family")
        if key[0] == "BOWS2" and (
            key[1] not in {r["lineage"] for r in originals} or key[2] not in (75, 95)
        ):
            raise ValueError("audit BOWS context mismatch")
    tensor = read(root / "pixels.f32", 480 * ROW_BYTES)
    if len(tensor) != 480 * ROW_BYTES or sha(tensor) != manifest["tensor_sha256"]:
        raise ValueError("audit tensor identity mismatch")
    jpeglib.version.set("6b")
    maximum = 0.0
    for i, row in enumerate(rows):
        if time.monotonic() - started > 300:
            raise ValueError("audit deadline exceeded")
        path = root / name(row["path"])
        # Native ALASKA JPEGs retain the original shared 2 MiB carrier bound.
        raw = read(path, 2 * 1024 * 1024)
        if len(raw) != row["size"] or sha(raw) != row["sha256"]:
            raise ValueError("audit JPEG identity mismatch")
        jpeg = jpeglib.read_dct(str(path))
        width, height = int(jpeg.width), int(jpeg.height)
        if min(width, height) < 256 or width * height > 4_000_000:
            raise ValueError("audit geometry bounds")
        geometry = {
            "image_size": [width, height],
            "region": [8 * ((width - 256) // 16), 8 * ((height - 256) // 16), 256, 256],
        }
        if geometry != manifest["geometry"][i]:
            raise ValueError("audit crop geometry mismatch")
        coefficients = jpeg.Y.astype(np.float64) * jpeg.get_component_qt(0)
        full = idctn(coefficients, axes=(-2, -1), norm="ortho") + 128
        full = full.transpose(0, 2, 1, 3).reshape(
            coefficients.shape[0] * 8, coefficients.shape[1] * 8
        )
        left, top, _, _ = geometry["region"]
        expected = full[top : top + 256, left : left + 256].astype("<f4")
        observed = np.frombuffer(tensor, dtype="<f4", count=256**2, offset=i * ROW_BYTES).reshape(
            256, 256
        )
        difference = float(np.max(np.abs(expected - observed)))
        if not np.isfinite(observed).all() or difference > 1e-4:
            raise ValueError("audit independent IDCT mismatch")
        maximum = max(maximum, difference)
        if row["source_group"] == "BOWS2" and row["label"] == "stego":
            family = groups[("BOWS2", row["lineage"], row["quality_factor"])]
            cover = next(r for r in family if r["label"] == "cover")
            reference = jpeglib.read_dct(str(root / name(cover["path"])))
            delta = jpeg.Y.astype(np.int64) - reference.Y.astype(np.int64)
            if (
                np.count_nonzero(delta) != row["coefficient_changes"]
                or not np.count_nonzero(delta)
                or np.abs(delta).max() > 1
                or not np.array_equal(jpeg.qt, reference.qt)
            ):
                raise ValueError("audit embedding changes mismatch")
    return {
        "schema_version": "jpeg-real-timing-independent-audit-v1",
        "status": "completed",
        "manifest_sha256": digest,
        "protocol_sha256": PROTOCOL_SHA,
        "originals": 96,
        "jpeg_rows": 480,
        "tensor_bytes": len(tensor),
        "rows_per_source": dict(Counter(r["source_group"] for r in rows)),
        "independent_IDCT_rows": 480,
        "maximum_pixel_difference": maximum,
        "absolute_tolerance": 1e-4,
        "relative_tolerance": 0,
        "BOWS_coefficient_pairs_checked": 128,
        "validation_pixels_read": 0,
        "native_parser_independently_verified": False,
        "camera_independence": "unverified",
        "model_trained": False,
        "accuracy_qualification": "unavailable",
        "GPU_timing": "unavailable",
        "cloud_resources_created": 0,
        "audit_script_sha256": sha(read(Path(__file__), 1024**2)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "bows", "old-root", "protocol", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = vars(parser.parse_args())
    out = args.pop("out")
    result = audit(**args)
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("audit output symlink")
    with out.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print("Complete: 480 independent IDCT checks; no accuracy or GPU timing claim.")


if __name__ == "__main__":
    main()
