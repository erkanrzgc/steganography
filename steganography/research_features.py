"""Manifest-bound features and train-only input validation."""

from __future__ import annotations

import hashlib
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from core import jpeg_context, jpeg_residual
from core import jpeg_features as jpeg
from core import spatial_cooccurrence as cooccurrence
from core import spatial_parity as parity
from core.feature_model import feature_model
from core.features import FEATURE_NAMES, FEATURE_VERSION, MAX_IMAGE_BYTES, spatial_features
from steganography.research import ResearchManifestError, verify_dataset_manifest

MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
MAX_ROWS = 10_000
MODEL_DOMAINS = {
    FEATURE_VERSION: "spatial-summary-linear-v1",
    jpeg.FEATURE_VERSION: "jpeg-dct-summary-linear-v1",
    jpeg_context.FEATURE_VERSION: "jpeg-context-summary-linear-v1",
    jpeg_residual.FEATURE_VERSION: "jpeg-dct-residual-parity-linear-v1",
    cooccurrence.FEATURE_VERSION: "spatial-cooccurrence-linear-v1",
    parity.FEATURE_VERSION: "spatial-parity-residual-linear-v1",
}


def read_feature_checkpoint(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Bound file/archive input, then validate loaded feature dimensions."""
    import torch

    limit = 16 * 1024 * 1024
    if (
        any(p.is_symlink() for p in (path, *path.parents))
        or not path.is_file()
        or path.stat().st_size > limit
    ):
        raise ResearchManifestError("invalid bounded checkpoint")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ResearchManifestError("invalid bounded checkpoint")
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ResearchManifestError("checkpoint checksum mismatch")
    import io

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            if not 1 <= len(members) <= 256 or sum(m.file_size for m in members) > limit:
                raise ResearchManifestError("checkpoint archive exceeds limits")
    except zipfile.BadZipFile as exc:
        raise ResearchManifestError("checkpoint must use bounded ZIP serialization") from exc
    checkpoint = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or type(checkpoint.get("features")) is not int
        or not 1 <= checkpoint["features"] <= 4096
    ):
        raise ResearchManifestError("invalid checkpoint feature dimensions")
    return checkpoint


def read_document(path: Path) -> tuple[dict[str, Any], str]:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ResearchManifestError("symlink documents are not accepted")
    if not path.is_file():
        raise ResearchManifestError("research document must be a regular file")
    with path.open("rb") as stream:
        data = stream.read(MAX_DOCUMENT_BYTES + 1)
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ResearchManifestError("research document exceeds byte limit")
    document = json.loads(data)
    if not isinstance(document, dict):
        raise ResearchManifestError("research document must be an object")
    return document, hashlib.sha256(data).hexdigest()


def selected_samples(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    if split not in {"train", "validation"}:
        raise ResearchManifestError("features may only access train or validation")
    if not isinstance(manifest.get("partition"), dict):
        raise ResearchManifestError("features require a partitioned manifest")
    verify_dataset_manifest(manifest, verify_files=False)
    selected = [s for s in manifest["samples"] if s["split"] == split]
    if not 1 <= len(selected) <= MAX_ROWS:
        raise ResearchManifestError("feature row count outside limits")
    if {s["label"] for s in selected} != {"cover", "stego"}:
        raise ResearchManifestError("selected split needs both labels")
    if any(not s.get("lineage") for s in selected):
        raise ResearchManifestError("feature samples require lineage")
    return selected


def extract_features(
    manifest_path: Path,
    out: Path,
    *,
    split: str = "train",
    source: Path | None = None,
    feature_version: str = FEATURE_VERSION,
    workers: int = 1,
) -> dict[str, Any]:
    if not 1 <= workers <= 4:
        raise ResearchManifestError("feature workers must be 1..4")
    names, extractor = feature_contract(feature_version)
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("feature output exists or uses a symlink")
    manifest, manifest_hash = read_document(manifest_path)
    selected = selected_samples(manifest, split)
    root = Path(source or manifest.get("source", ""))

    def extract_row(sample):
        path = root / sample["path"]
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ResearchManifestError("symlink feature images are not accepted")
        if not path.is_file():
            raise ResearchManifestError("feature image must be a regular file")
        with path.open("rb") as stream:
            data = stream.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise ResearchManifestError("feature image exceeds byte limit")
        if len(data) != sample["size"] or hashlib.sha256(data).hexdigest() != sample["sha256"]:
            raise ResearchManifestError("feature image integrity failed")
        return {
            "sha256": sample["sha256"],
            "lineage": sample["lineage"],
            "label": sample["label"],
            "values": extractor(data),
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(extract_row, selected))
    artifact = {
        "schema_version": "research-features-1",
        "feature_version": feature_version,
        "feature_names": list(names),
        "manifest_sha256": manifest_hash,
        "split": split,
        "rows": rows,
        "support_status": "experimental",
    }
    data = json.dumps(artifact, separators=(",", ":"), allow_nan=False).encode()
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ResearchManifestError("feature artifact exceeds byte limit")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        stream.write(data)
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "samples": len(rows),
        "split": split,
        "feature_version": feature_version,
    }


def training_inputs(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reject legacy unbound NPZ, validation/test rows, tampering and nonfinite data."""
    return feature_inputs(config, split="train")


def feature_contract(version: str):
    if version == FEATURE_VERSION:
        return FEATURE_NAMES, spatial_features
    if version == jpeg.FEATURE_VERSION:
        return jpeg.FEATURE_NAMES, jpeg.jpeg_features
    if version == jpeg_context.FEATURE_VERSION:
        return jpeg_context.FEATURE_NAMES, jpeg_context.jpeg_context_features
    if version == jpeg_residual.FEATURE_VERSION:
        return jpeg_residual.FEATURE_NAMES, jpeg_residual.jpeg_residual_features
    if version == cooccurrence.FEATURE_VERSION:
        return cooccurrence.FEATURE_NAMES, cooccurrence.spatial_cooccurrence_features
    if version == parity.FEATURE_VERSION:
        return parity.FEATURE_NAMES, parity.spatial_parity_features
    raise ResearchManifestError("unknown feature contract")


def feature_inputs(
    config: dict[str, Any],
    *,
    split: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not all(config.get(key) for key in ("manifest", "features", "features_sha256")):
        raise ResearchManifestError("training requires manifest, features and features_sha256")
    manifest, manifest_hash = read_document(Path(config["manifest"]))
    selected = selected_samples(manifest, split)
    artifact, artifact_hash = read_document(Path(config["features"]))
    if artifact_hash != config["features_sha256"]:
        raise ResearchManifestError("feature artifact checksum mismatch")
    version = artifact.get("feature_version")
    if not isinstance(version, str):
        raise ResearchManifestError("missing feature contract")
    names, _ = feature_contract(version)
    if (
        artifact.get("schema_version") != "research-features-1"
        or artifact.get("feature_names") != list(names)
        or artifact.get("manifest_sha256") != manifest_hash
        or artifact.get("split") != split
    ):
        raise ResearchManifestError("feature contract or training split mismatch")
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) != len(selected):
        raise ResearchManifestError("feature rows must match the complete training split")
    vectors = []
    for row, sample in zip(rows, selected, strict=True):
        if not isinstance(row, dict) or any(
            row.get(k) != sample[k] for k in ("sha256", "lineage", "label")
        ):
            raise ResearchManifestError("feature row provenance mismatch")
        values = row.get("values")
        if (
            not isinstance(values, list)
            or len(values) != len(names)
            or any(type(value) not in (int, float) for value in values)
        ):
            raise ResearchManifestError("invalid feature vector")
        vector = np.asarray(values, dtype=np.float64)
        if not np.isfinite(vector).all() or (vector < 0).any() or (vector > 1).any():
            raise ResearchManifestError("feature values must be finite and in [0, 1]")
        vectors.append(vector)
    provenance = {
        "manifest_sha256": manifest_hash,
        "features_sha256": artifact_hash,
        "feature_version": version,
        "feature_names": list(names),
        "split": split,
        "samples": len(rows),
        "calibrated": False,
    }
    return (
        np.asarray(vectors, dtype=np.float32),
        np.asarray([s["label"] == "stego" for s in selected], dtype=np.float32),
        provenance,
    )


def predict_validation(
    config_path: Path,
    checkpoint_path: Path,
    out: Path,
    *,
    required_domain: str | None = None,
    schema_version: str = "research-feature-predictions-v1",
) -> dict[str, Any]:
    """Shared validation-only inference; test image access is not an option."""
    import torch

    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("prediction output exists or uses a symlink")
    config, _ = read_document(config_path)
    features, _, provenance = feature_inputs(config, split="validation")
    checkpoint = read_feature_checkpoint(checkpoint_path)
    domain = MODEL_DOMAINS[provenance["feature_version"]]
    if (
        checkpoint.get("domain") != domain
        or (required_domain is not None and domain != required_domain)
        or checkpoint["preprocessing"]["feature_version"] != provenance["feature_version"]
        or checkpoint["preprocessing"]["feature_names"] != provenance["feature_names"]
        or checkpoint["training_provenance"]["manifest_sha256"] != provenance["manifest_sha256"]
    ):
        raise ResearchManifestError("checkpoint/validation contract mismatch")
    model = feature_model(checkpoint).eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(features))).numpy().reshape(-1)
    if not np.isfinite(scores).all():
        raise ResearchManifestError("nonfinite validation predictions")
    manifest, _ = read_document(Path(config["manifest"]))
    samples = [s for s in manifest["samples"] if s["split"] == "validation"]
    report = {
        "schema_version": schema_version,
        "split": "validation",
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "provenance": provenance,
        "calibrated": False,
        "support_status": "experimental",
        "predictions": [
            {
                "sha256": s["sha256"],
                "lineage": s["lineage"],
                "label": s["label"],
                "method": s["method"],
                "rate_percent": s.get("rate_percent"),
                "format": s.get("format"),
                "score": float(score),
            }
            for s, score in zip(samples, scores, strict=True)
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report
