"""Reserved-safe, controlled spatial development; never installed inference."""

from __future__ import annotations

import argparse
import hashlib
import io
import random
import re
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.dataset import identity_keys
from core.features import MAX_IMAGE_BYTES, MAX_PIXELS
from steganography.benchmarking.metrics import classification_metrics
from steganography.research import (
    ResearchManifestError,
    _expected_calibration_error,
    export_onnx,
    train_model,
    verify_dataset_manifest,
)
from steganography.research_features import (
    extract_features,
    feature_inputs,
    predict_validation,
    read_document,
)
from steganography.research_jpeg import write_json

SEED = 20261005
RATES = (5, 20, 40)
METHODS = ("sequential", "scattered")
MAX_CORPUS_BYTES = 2 * 1024**3


def replacement(pixels: np.ndarray, lineage: str, method: str, rate: int):
    """Independent grayscale recipe plus scalar extraction oracle (no Carrier)."""
    if (
        method not in METHODS
        or rate not in RATES
        or pixels.dtype != np.uint8
        or pixels.ndim != 2
        or min(pixels.shape) < 8
        or pixels.size > MAX_PIXELS
    ):
        raise ResearchManifestError("invalid replacement recipe")
    count = pixels.size * rate // 100 // 8 * 8
    context = f"{SEED}:{lineage}:{method}:{rate}".encode()
    payload = hashlib.shake_256(context).digest(count // 8)
    generator = random.Random(int.from_bytes(hashlib.sha256(context).digest(), "big"))  # noqa: S311 - reproducible recipe
    positions = (
        np.arange(count)
        if method == "sequential"
        else np.asarray(generator.sample(range(pixels.size), count))
    )
    flat = pixels.reshape(-1).copy()
    flat[positions] = (flat[positions] & 254) | np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8)
    )
    # Deliberately separate scalar byte assembly from the vectorized embedding.
    recovered = bytearray(count // 8)
    raw = flat.tobytes()
    for index, position in enumerate(positions):
        recovered[index // 8] |= (raw[int(position)] & 1) << (7 - index % 8)
    selected = np.zeros(pixels.size, dtype=bool)
    selected[positions] = True
    original = pixels.reshape(-1)
    if (
        bytes(recovered) != payload
        or np.max(np.abs(flat.astype(np.int16) - original)) > 1
        or not np.array_equal(flat[~selected], original[~selected])
    ):
        raise ResearchManifestError("independent replacement oracle failed")
    return flat.reshape(pixels.shape), hashlib.sha256(payload).hexdigest(), count


def generate_corpus(
    root: Path, out: Path, *, source_sha256: str, reserved_manifests: list[Path]
) -> dict[str, Any]:
    source, digest = read_document(root / "source.json")
    selection, selection_hash = read_document(root / "selection.json")
    if digest != source_sha256 or source.get("selection_sha256") != selection_hash:
        raise ResearchManifestError("source/selection checksum mismatch")
    if (
        source.get("schema_version") != "boss-development-acquisition-v1"
        or source.get("purpose") != "development covers only"
    ):
        raise ResearchManifestError("explicit development covers required")
    reserved: set[tuple[str, ...]] = set()
    reserved_members: set[str] = set()
    hashes = []
    for path in reserved_manifests:
        document, reserved_hash = read_document(path)
        if not document.get("samples"):
            raise ResearchManifestError("empty reserved manifest")
        for row in document["samples"]:
            if not identity_keys(row):
                raise ResearchManifestError("invalid reserved identity")
            reserved.update(identity_keys(row))
            if row.get("upstream_member"):
                reserved_members.add(row["upstream_member"])
        hashes.append(reserved_hash)
    if not hashes or sorted(hashes) != sorted(source.get("reserved_manifest_sha256", [])):
        raise ResearchManifestError("reserved provenance mismatch")
    rows = source.get("samples", [])
    if not 2 <= len(rows) <= 1000 or {r.get("split") for r in rows} != {"train", "validation"}:
        raise ResearchManifestError("development count/splits invalid")
    if sorted(r["upstream_member"] for r in rows) != sorted(
        r["upstream_member"] for r in selection["members"]
    ):
        raise ResearchManifestError("selection membership mismatch")
    seen = set()
    # Validate every input before creating any derivative.
    for row in rows:
        sha = row.get("sha256", "")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", sha)
            or sha in seen
            or row.get("lineage") != sha
            or row.get("source_group") != "BOSSbase-1.01"
            or row.get("label") != "cover"
            or row.get("method") is not None
            or not re.fullmatch(r"[0-9]{4}\.pgm", row["path"])
            or row["split"] != ("train" if int(sha[:16], 16) / 2**64 < 0.8 else "validation")
            or identity_keys(row) & reserved
            or row.get("upstream_member") in reserved_members
        ):
            raise ResearchManifestError("invalid or overlapping development identity")
        seen.add(sha)
        path = root / row["path"]
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
            raise ResearchManifestError("input must be regular nonsymlink file")
        with path.open("rb") as stream:
            data = stream.read(MAX_IMAGE_BYTES + 1)
        if (
            len(data) > MAX_IMAGE_BYTES
            or len(data) != row["size"]
            or hashlib.sha256(data).hexdigest() != sha
        ):
            raise ResearchManifestError("input integrity failed")
        member = next(
            m for m in selection["members"] if m["upstream_member"] == row["upstream_member"]
        )
        if member["size"] != len(data) or member["crc32"] != zlib.crc32(data):
            raise ResearchManifestError("selection member integrity failed")
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "PPM" or image.mode != "L" or image.size != (512, 512):
                raise ResearchManifestError("expected unchanged grayscale 512x512 PGM")
            image.load()
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ResearchManifestError("symlink corpus output forbidden")
    out.mkdir(parents=True, exist_ok=False)
    artifacts = []
    total = 0
    for index, row in enumerate(rows):
        path = root / row["path"]
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
            raise ResearchManifestError("generation input changed to nonregular file")
        with path.open("rb") as stream:
            original_data = stream.read(MAX_IMAGE_BYTES + 1)
        if (
            len(original_data) != row["size"]
            or hashlib.sha256(original_data).hexdigest() != row["sha256"]
        ):
            raise ResearchManifestError("generation input changed after preflight")
        with Image.open(io.BytesIO(original_data)) as image:
            pixels = np.asarray(image, dtype=np.uint8)
        if pixels.size > MAX_PIXELS:
            raise ResearchManifestError("pixel limit exceeded")
        fmt = "PNG" if int(row["sha256"][-1], 16) % 2 == 0 else "BMP"
        recipes = [(None, 0)] + [(method, rate) for method in METHODS for rate in RATES]
        for method, rate in recipes:
            modified, payload_hash, bits = (
                (pixels, None, 0)
                if method is None
                else replacement(pixels, row["sha256"], method, rate)
            )
            buffer = io.BytesIO()
            Image.fromarray(modified).save(buffer, format=fmt)
            data = buffer.getvalue()
            total += len(data)
            sha = hashlib.sha256(data).hexdigest()
            if total > MAX_CORPUS_BYTES or sha in seen or ("hash", sha) in reserved:
                raise ResearchManifestError("output budget/identity violation")
            with Image.open(io.BytesIO(data)) as decoded:
                if not np.array_equal(np.asarray(decoded), modified):
                    raise ResearchManifestError("lossless output verification failed")
            seen.add(sha)
            name = f"{index:04d}-{method or 'cover'}-{rate}.{fmt.lower()}"
            with (out / name).open("xb") as stream:
                stream.write(data)
            artifacts.append(
                {
                    **row,
                    "path": name,
                    "sha256": sha,
                    "size": len(data),
                    "format": fmt,
                    "label": "cover" if method is None else "stego",
                    "method": method,
                    "rate_percent": rate,
                    "payload_sha256": payload_hash,
                    "payload_bits": bits,
                }
            )
        if (index + 1) % 100 == 0:
            print(f"verified {index + 1} spatial lineages", flush=True)
    manifest = {
        "schema_version": "1.0",
        "source": ".",
        "samples": artifacts,
        "source_sha256": digest,
        "selection_sha256": selection_hash,
        "catalog": {"source_url": source["source_url"], "license": source["license"]},
        "partition": {
            "policy": "identity-camera-device-development-v1",
            "test_sources": [],
            "reserved_manifest_sha256": sorted(hashes),
            "seed": SEED,
            "camera_device_metadata_complete": False,
        },
        "total_bytes": total,
        "support_status": "experimental",
    }
    verify_dataset_manifest(manifest, source=out)
    write_json(out / "manifest.json", manifest)
    return manifest


def cell_metrics(
    rows: list[dict[str, Any]], *, threshold: int | float = 50, score_scale: float = 100
) -> dict[str, Any]:
    metrics = classification_metrics(
        [(r["label"] == "stego", r["score"] * score_scale) for r in rows],
        threshold=threshold,
        recommend_threshold=False,
    )
    metrics.pop("recommended_threshold")
    metrics.pop("average_precision")
    metrics["expected_calibration_error"] = _expected_calibration_error(
        [(r["label"] == "stego", r["score"]) for r in rows]
    )
    return metrics


def paired_intervals(
    rows: list[dict[str, Any]], *, threshold: int | float = 50, score_scale: float = 100
) -> dict[str, list[float]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["lineage"], []).append(row)
    generator = random.Random(SEED)  # noqa: S311 - statistical resampling
    lineages = sorted(groups)
    values: dict[str, list[float]] = {
        k: []
        for k in (
            "roc_auc",
            "balanced_accuracy",
            "recall",
            "false_positive_rate",
            "expected_calibration_error",
        )
    }
    for _ in range(200):
        sample = [r for _ in lineages for r in groups[generator.choice(lineages)]]
        metrics = cell_metrics(sample, threshold=threshold, score_scale=score_scale)
        for key in values:
            values[key].append(metrics[key])
    return {
        key: [float(v) for v in np.quantile(distribution, [0.025, 0.975])]
        for key, distribution in values.items()
    }


def run_development(root: Path, out: Path, *, source_sha256: str, reserved_manifests: list[Path]):
    started = time.monotonic()
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ResearchManifestError("symlink experiment output forbidden")
    out.mkdir(parents=True, exist_ok=False)
    corpus = out / "corpus"
    manifest = generate_corpus(
        root, corpus, source_sha256=source_sha256, reserved_manifests=reserved_manifests
    )
    experiments = {}
    for version in ("spatial-summary-v1", "spatial-cooccurrence-v1"):
        directory = out / version
        directory.mkdir()
        for split in ("train", "validation"):
            print(f"extracting {version} {split}", flush=True)
            feature_path = directory / f"{split}-features.json"
            summary = extract_features(
                corpus / "manifest.json",
                feature_path,
                source=corpus,
                split=split,
                feature_version=version,
                workers=4,
            )
            write_json(
                directory / f"{split}-config.json",
                {
                    "manifest": str(corpus / "manifest.json"),
                    "features": str(feature_path),
                    "features_sha256": summary["sha256"],
                    "epochs": 300,
                    "learning_rate": 0.01,
                    "seed": SEED,
                    "standardize": True,
                    "class_balanced": True,
                },
            )
        checkpoint = directory / "baseline.pt"
        training = train_model(directory / "train-config.json", checkpoint)
        predictions = predict_validation(
            directory / "validation-config.json", checkpoint, directory / "predictions.json"
        )
        card = export_onnx(checkpoint, directory / "baseline.onnx")
        parity = onnx_parity(
            directory / "validation-config.json", checkpoint, directory / "baseline.onnx"
        )
        cells = {}
        for method in METHODS:
            for rate in RATES:
                rows = [
                    r
                    for r in predictions["predictions"]
                    if r["label"] == "cover"
                    or (r["method"] == method and r["rate_percent"] == rate)
                ]
                cells[f"{method}-{rate}"] = {
                    "metrics": cell_metrics(rows),
                    "bootstrap_95_percent": paired_intervals(rows),
                }
        experiments[version] = {
            "training": training,
            "by_method_rate": cells,
            "checkpoint_sha256": predictions["checkpoint_sha256"],
            "prediction_sha256": hashlib.sha256(
                (directory / "predictions.json").read_bytes()
            ).hexdigest(),
            "onnx_sha256": card["onnx_sha256"],
            "onnx_parity": parity,
        }
        write_json(directory / "results.json", experiments[version])
    report = {
        "schema_version": "spatial-development-run-v1",
        "source_sha256": source_sha256,
        "manifest_sha256": hashlib.sha256((corpus / "manifest.json").read_bytes()).hexdigest(),
        "counts": verify_dataset_manifest(manifest, verify_files=False),
        "experiments": experiments,
        "deployed": False,
        "calibrated": False,
        "cross_source_gate": "unavailable",
        "support_status": "experimental",
        "scope": "same-source validation; controlled LSB on real grayscale covers",
        "old_test_images_accessed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(out / "report.json", report)
    return report


def onnx_parity(config_path: Path, checkpoint: Path, onnx: Path):
    import onnxruntime as ort
    import torch

    from core.feature_model import feature_model

    config, _ = read_document(config_path)
    x, _, _ = feature_inputs(config, split="validation")
    model = feature_model(torch.load(checkpoint, weights_only=True)).eval()
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: x})[0]
    logits = float(np.max(np.abs(expected - actual)))
    probabilities = float(
        np.max(
            np.abs(
                torch.sigmoid(torch.from_numpy(expected)).numpy()
                - torch.sigmoid(torch.from_numpy(actual)).numpy()
            )
        )
    )
    return {
        "samples": len(x),
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 0,
        "max_logit_difference": logits,
        "max_probability_difference": probabilities,
        "passed": logits <= 1e-6 and probabilities <= 1e-6,
        "threshold_decisions_equal": bool(np.array_equal(expected >= 0, actual >= 0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--reserved-manifest", type=Path, action="append", required=True)
    args = parser.parse_args()
    run_development(
        args.root,
        args.out,
        source_sha256=args.source_sha256,
        reserved_manifests=args.reserved_manifest,
    )


if __name__ == "__main__":
    main()
