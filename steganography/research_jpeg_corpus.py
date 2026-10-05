"""Explicit local-only BOSSbase JPEG simulation; originals are never modified."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from core.dataset import identity_keys
from core.jpeg_context import FEATURE_NAMES, FEATURE_VERSION, summary_context_features
from core.jpeg_features import MAX_IMAGE_BYTES, coefficient_features
from steganography.research import ResearchManifestError, verify_dataset_manifest
from steganography.research_features import read_document
from steganography.research_jpeg import write_json

METHODS = ("JUNIWARD", "UERD")
QUALITIES = (75, 95)
ALPHA = 0.2
SEED = 20261006
MAX_WORKER_OUTPUT = 256 * 1024


def generate_lineage(data: bytes, out: Path, lineage: str, split: str):
    """Trusted worker routine: fixed recipes, coefficient round-trip validation."""
    import conseal
    import jpeglib

    if not 0 < len(data) <= 1024 * 1024:
        raise ResearchManifestError("bounded PGM input required")
    if hashlib.sha256(data).hexdigest() != lineage or split not in {"train", "validation"}:
        raise ResearchManifestError("original lineage identity mismatch")
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("lineage output exists or uses a symlink")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "PPM" or image.mode != "L" or image.size != (512, 512):
            raise ResearchManifestError("BOSSbase worker requires 512x512 grayscale PGM")
        image.load()
        out.mkdir(parents=True, exist_ok=False)
        rows = []
        for quality in QUALITIES:
            cover = out / f"q{quality}-cover.jpg"
            with cover.open("xb") as stream:
                image.save(stream, format="JPEG", quality=quality, optimize=False)
            original = jpeglib.read_dct(str(cover))
            original.load()
            coefficients = original.Y.copy()
            table = original.qt[int(original.quant_tbl_no[0])].copy()
            pixels = jpeglib.read_spatial(str(cover), jpeglib.JCS_GRAYSCALE).spatial[..., 0]
            for method in (None, *METHODS):
                path = cover if method is None else out / f"q{quality}-{method}.jpg"
                changes = 0
                if method is not None:
                    seed = int.from_bytes(
                        hashlib.sha256(f"{SEED}:{lineage}:{quality}:{method}".encode()).digest()[
                            :4
                        ],
                        "big",
                    )
                    if method == "JUNIWARD":
                        modified = conseal.juniward.simulate_single_channel(
                            pixels, coefficients, table, ALPHA, seed=seed
                        )
                    else:
                        modified = conseal.uerd.simulate_single_channel(
                            coefficients, table, ALPHA, payload_mode="bpnzAC", seed=seed
                        )
                    difference = modified.astype(np.int64) - coefficients.astype(np.int64)
                    changes = int(np.count_nonzero(difference))
                    if not changes or np.abs(difference).max() > 1:
                        raise ResearchManifestError("simulation has invalid coefficient changes")
                    stego = jpeglib.read_dct(str(cover))
                    stego.load()
                    stego.Y = modified
                    # No user-controlled name, no existing file, owned nonsymlink directory.
                    if path.exists() or path.is_symlink():
                        raise FileExistsError("simulation output exists")
                    stego.write_dct(str(path))
                decoded = jpeglib.read_dct(str(path))
                decoded.load()
                if not np.array_equal(
                    decoded.Y, coefficients if method is None else modified
                ) or not np.array_equal(decoded.qt[int(decoded.quant_tbl_no[0])], table):
                    raise ResearchManifestError("JPEG coefficient/quantization round-trip failed")
                with path.open("rb") as stream:
                    encoded = stream.read(MAX_IMAGE_BYTES + 1)
                if not 0 < len(encoded) <= MAX_IMAGE_BYTES:
                    raise ResearchManifestError("generated JPEG exceeds limits")
                rows.append(
                    {
                        "path": f"{lineage}/{path.name}",
                        "sha256": hashlib.sha256(encoded).hexdigest(),
                        "size": len(encoded),
                        "lineage": lineage,
                        "split": split,
                        "label": "cover" if method is None else "stego",
                        "method": method,
                        "source_group": "BOSSbase-1.01",
                        "format": "JPEG",
                        "quality_factor": quality,
                        "camera": None,
                        "device": None,
                        "app": None,
                        "payload_rate": None if method is None else ALPHA,
                        "payload_unit": "bpnzAC",
                        "rate_percent": None,
                        "coefficient_changes": changes,
                        "ground_truth": "upstream embedding simulation",
                        "values": summary_context_features(coefficient_features(decoded.Y, table)),
                    }
                )
    return rows


def generate_boss(
    root: Path,
    out: Path,
    *,
    source_sha256: str,
    reserved_manifests: list[Path],
    count: int = 128,
):
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("JPEG corpus output exists or uses a symlink")
    if type(count) is not int or not 2 <= count <= 128:
        raise ResearchManifestError("JPEG development count must be between 2 and 128")
    source, checksum = read_document(root / "source.json")
    if checksum != source_sha256:
        raise ResearchManifestError("BOSSbase source checksum mismatch")
    if source.get("schema_version") != "boss-development-acquisition-v1" or (
        source.get("purpose") != "development covers only"
    ):
        raise ResearchManifestError("only reserved-safe BOSSbase development originals are allowed")
    if not reserved_manifests:
        raise ResearchManifestError("frozen reserved manifests are required")
    reserved = set()
    reserved_members = set()
    reserved_hashes = []
    for path in reserved_manifests:
        document, digest = read_document(path)
        if not document.get("samples"):
            raise ResearchManifestError("reserved manifest is empty")
        reserved_hashes.append(digest)
        for row in document["samples"]:
            if not isinstance(row, dict) or not identity_keys(row):
                raise ResearchManifestError("reserved identity is invalid")
            reserved.update(identity_keys(row))
            if row.get("upstream_member"):
                reserved_members.add(row["upstream_member"])
    originals = source.get("samples", [])
    if not isinstance(originals, list) or not count <= len(originals) <= 1000:
        raise ResearchManifestError("original sample count outside bounds")
    if any(
        not isinstance(s, dict)
        or s.get("label") != "cover"
        or s.get("split") not in ("train", "validation")
        or s.get("lineage") != s.get("sha256")
        or identity_keys(s) & reserved
        or s.get("upstream_member") in reserved_members
        or type(s.get("size")) is not int
        or not 0 < s["size"] <= 1024 * 1024
        for s in originals
    ):
        raise ResearchManifestError("original roles/lineages overlap reserved data")
    if sorted(reserved_hashes) != sorted(source.get("reserved_manifest_sha256", [])):
        raise ResearchManifestError("reserved provenance mismatch")
    selection, selection_hash = read_document(root / "selection.json")
    if source.get("selection_sha256") != selection_hash or sorted(
        s.get("upstream_member", "") for s in originals
    ) != sorted(s.get("upstream_member", "") for s in selection.get("members", [])):
        raise ResearchManifestError("original selection membership/checksum mismatch")
    verify_dataset_manifest({"schema_version": "1.0", "samples": originals}, verify_files=False)
    for sample in originals:
        path = root / sample["path"]
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
            raise ResearchManifestError("original must be a regular nonsymlink file")
        with path.open("rb") as stream:
            data = stream.read(1024 * 1024 + 1)
        if len(data) != sample["size"] or hashlib.sha256(data).hexdigest() != sample["sha256"]:
            raise ResearchManifestError("original size/hash integrity mismatch")
    selected = sorted(
        originals,
        key=lambda s: hashlib.sha256(f"jpeg-context:{SEED}:{s['sha256']}".encode()).digest(),
    )[:count]
    if {s["split"] for s in selected} != {"train", "validation"}:
        raise ResearchManifestError("selected originals require both development splits")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("conseal", "jpeglib", "numpy", "pillow", "scipy", "numba")
    }
    if versions["conseal"] != "2025.11":
        raise ResearchManifestError("this protocol requires conseal 2025.11")
    out.mkdir(parents=True, exist_ok=False)

    def run(sample):
        path = root / sample["path"]
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ResearchManifestError("original uses a symlink")
        with path.open("rb") as stream:
            data = stream.read(1024 * 1024 + 1)
        if len(data) != sample["size"] or hashlib.sha256(data).hexdigest() != sample["sha256"]:
            raise ResearchManifestError("original changed after verification")
        env = {
            "PATH": os.environ.get("PATH", ""),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        }
        import tempfile

        with tempfile.TemporaryFile() as output:
            try:
                result = subprocess.run(  # noqa: S603 - fixed module, generated path and SHA
                    [
                        sys.executable,
                        "-m",
                        "steganography.research_jpeg_corpus",
                        "worker",
                        str(out / sample["sha256"]),
                        sample["sha256"],
                        sample["split"],
                    ],
                    input=data,
                    stdout=output,
                    stderr=subprocess.DEVNULL,
                    env=env,
                    cwd=Path(__file__).resolve().parents[1],
                    timeout=90,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ResearchManifestError("JPEG simulation worker timed out") from exc
            output.seek(0)
            raw = output.read(MAX_WORKER_OUTPUT + 1)
        if result.returncode or len(raw) > MAX_WORKER_OUTPUT:
            raise ResearchManifestError("JPEG simulation worker failed or exceeded output bounds")
        rows = json.loads(raw)
        if not isinstance(rows, list) or len(rows) != 6:
            raise ResearchManifestError("incomplete JPEG simulated lineage")
        expected = {(q, m) for q in QUALITIES for m in (None, *METHODS)}
        if (
            any(
                not isinstance(r, dict)
                or type(r.get("quality_factor")) is not int
                or (r.get("method") is not None and not isinstance(r.get("method"), str))
                for r in rows
            )
            or {(r.get("quality_factor"), r.get("method")) for r in rows} != expected
        ):
            raise ResearchManifestError("worker JPEG recipes mismatch")
        for row in rows:
            if (
                row.get("lineage") != sample["sha256"]
                or row.get("split") != sample["split"]
                or row.get("path")
                != f"{sample['sha256']}/q{row['quality_factor']}-{row['method'] or 'cover'}.jpg"
                or row.get("source_group") != "BOSSbase-1.01"
                or row.get("label") != ("cover" if row["method"] is None else "stego")
                or not isinstance(row.get("values"), list)
                or len(row["values"]) != len(FEATURE_NAMES)
                or any(
                    type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1
                    for v in row["values"]
                )
            ):
                raise ResearchManifestError("worker JPEG identity/feature mismatch")
        return rows

    with ThreadPoolExecutor(max_workers=2) as executor:
        groups = list(executor.map(run, selected))
    rows = [r for group in groups for r in group]
    features = [{k: row[k] for k in ("sha256", "lineage", "label", "values")} for row in rows]
    manifest = {
        "schema_version": "1.0",
        "source": ".",
        "samples": [{k: v for k, v in row.items() if k != "values"} for row in rows],
        "source_sha256": checksum,
        "partition": {"policy": "identity-camera-device-development-v1", "test_sources": []},
        "catalog": {
            "source_records": {
                "BOSSbase-1.01": {
                    "origin_manifest_sha256": checksum,
                    "license": source["license"],
                    "source_url": source["source_url"],
                }
            }
        },
        "generation": {
            "versions": versions,
            "seed": SEED,
            "qualities": list(QUALITIES),
            "alpha": ALPHA,
            "methods": list(METHODS),
            "reserved_manifest_sha256": reserved_hashes,
            "feature_version": FEATURE_VERSION,
            "limitations": "simulated embedding, no encoded payload/exact recovery claim",
        },
    }
    verify_dataset_manifest(manifest, source=out)
    write_json(out / "feature-rows.json", {"rows": features})
    manifest["feature_rows_sha256"] = hashlib.sha256(
        (out / "feature-rows.json").read_bytes()
    ).hexdigest()
    write_json(out / "manifest.json", manifest)
    return {
        "original_lineages": count,
        "jpeg_files": len(rows),
        "versions": versions,
        "manifest_sha256": hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest(),
    }


def main():
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (80, 85))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_IMAGE_BYTES, MAX_IMAGE_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        if len(sys.argv) != 5 or sys.argv[1] != "worker":
            return 1
        rows = generate_lineage(
            sys.stdin.buffer.read(1024 * 1024 + 1), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
        )
        print(json.dumps(rows, separators=(",", ":"), allow_nan=False))
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
