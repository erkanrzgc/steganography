"""Explicit train/validation pixel caches, isolated from summary-feature/model contracts."""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from core import jpeg_pixels as pixels
from core.jpeg_features import MAX_IMAGE_BYTES
from steganography.research import ResearchManifestError
from steganography.research_features import read_document, selected_samples
from steganography.research_jpeg import write_json

MAX_ROWS = 4000
MAX_CACHE_BYTES = 64 * 1024**2
MAX_SECONDS = 1800


def regular(path):
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ResearchManifestError("pixel input must be regular and non-symlink")


def extract_pixels(manifest_path: Path, out: Path, *, source: Path, split: str, workers: int = 1):
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("pixel output exists or uses a symlink")
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ResearchManifestError("pixel workers must be 1..4")
    manifest, digest = read_document(manifest_path)
    samples = selected_samples(manifest, split)
    if len(samples) > MAX_ROWS or any(s["format"] != "JPEG" for s in samples):
        raise ResearchManifestError("pixel cache requires at most 4000 JPEG rows")

    def extract(batch):
        files, rows = [], []
        for sample in batch:
            path = source / sample["path"]
            regular(path)
            with path.open("rb") as stream:
                data = stream.read(MAX_IMAGE_BYTES + 1)
            if len(data) != sample["size"] or hashlib.sha256(data).hexdigest() != sample["sha256"]:
                raise ResearchManifestError("pixel JPEG integrity failure")
            files.append(data)
            rows.append(
                {**{k: sample[k] for k in ("sha256", "lineage", "label")}, **pixels.region(data)}
            )
        return pixels.pixel_batch(files).tobytes(), rows

    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    data_hash, rows = hashlib.sha256(), []
    batches = [samples[i : i + pixels.MAX_BATCH] for i in range(0, len(samples), pixels.MAX_BATCH)]
    with (
        (out / "pixels.u8").open("xb") as stream,
        ThreadPoolExecutor(max_workers=workers) as executor,
    ):
        for first in range(0, len(batches), workers):
            if time.monotonic() - started > MAX_SECONDS:
                raise ResearchManifestError("pixel extraction deadline exceeded")
            for raw, batch_rows in executor.map(extract, batches[first : first + workers]):
                if len(raw) != len(batch_rows) * pixels.PIXELS:
                    raise ResearchManifestError("pixel worker byte count mismatch")
                stream.write(raw)
                data_hash.update(raw)
                rows.extend(batch_rows)
        if time.monotonic() - started > MAX_SECONDS:
            raise ResearchManifestError("pixel extraction deadline exceeded")
    descriptor = {
        "schema_version": "research-pixel-cache-v1",
        "feature_version": pixels.FEATURE_VERSION,
        "manifest_sha256": digest,
        "split": split,
        "dtype": "uint8",
        "shape": [len(rows), 1, pixels.SIDE, pixels.SIDE],
        "decoder": pixels.decoder_contract(),
        "data_sha256": data_hash.hexdigest(),
        "rows": rows,
        "support_status": "experimental",
        "deployed": False,
    }
    write_json(out / "cache.json", descriptor)
    return descriptor


def load_pixels(manifest_path: Path, cache: Path, *, checksum: str, split: str):
    manifest, digest = read_document(manifest_path)
    samples = selected_samples(manifest, split)
    descriptor, actual = read_document(cache)
    rows = descriptor.get("rows")
    if (
        actual != checksum
        or descriptor.get("schema_version") != "research-pixel-cache-v1"
        or descriptor.get("feature_version") != pixels.FEATURE_VERSION
        or descriptor.get("manifest_sha256") != digest
        or descriptor.get("split") != split
        or descriptor.get("dtype") != "uint8"
        or descriptor.get("decoder") != pixels.decoder_contract()
        or descriptor.get("shape") != [len(samples), 1, pixels.SIDE, pixels.SIDE]
        or len(samples) > MAX_ROWS
        or any(s["format"] != "JPEG" for s in samples)
        or not isinstance(rows, list)
        or len(rows) != len(samples)
    ):
        raise ResearchManifestError("pixel cache/manifest contract mismatch")
    for row, sample in zip(rows, samples, strict=True):
        if not isinstance(row, dict) or any(
            row.get(k) != sample[k] for k in ("sha256", "lineage", "label")
        ):
            raise ResearchManifestError("pixel row identity mismatch")
        size, region = row.get("image_size"), row.get("region")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(type(n) is not int or n < pixels.SIDE for n in size)
            or size[0] * size[1] > 4_000_000
            or not isinstance(region, list)
            or len(region) != 4
            or any(type(n) is not int for n in region)
            or region
            != [
                (size[0] - pixels.SIDE) // 2,
                (size[1] - pixels.SIDE) // 2,
                pixels.SIDE,
                pixels.SIDE,
            ]
        ):
            raise ResearchManifestError("pixel region contract mismatch")
    path = cache.parent / "pixels.u8"
    regular(path)
    size = len(samples) * pixels.PIXELS
    if size > MAX_CACHE_BYTES or path.stat().st_size != size:
        raise ResearchManifestError("pixel cache byte limits exceeded")
    with path.open("rb") as stream:
        data = stream.read(size + 1)
    if len(data) != size or hashlib.sha256(data).hexdigest() != descriptor.get("data_sha256"):
        raise ResearchManifestError("pixel cache checksum mismatch")
    return (
        np.frombuffer(data, dtype="u1").reshape(len(samples), 1, pixels.SIDE, pixels.SIDE),
        samples,
        descriptor,
    )
