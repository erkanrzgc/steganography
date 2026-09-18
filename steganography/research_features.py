"""Manifest-bound features and train-only input validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from core.features import FEATURE_NAMES, FEATURE_VERSION, MAX_IMAGE_BYTES, spatial_features
from steganography.research import ResearchManifestError, verify_dataset_manifest

MAX_DOCUMENT_BYTES = 64 * 1024 * 1024
MAX_ROWS = 10_000


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
    manifest_path: Path, out: Path, *, split: str = "train", source: Path | None = None
) -> dict[str, Any]:
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("feature output exists or uses a symlink")
    manifest, manifest_hash = read_document(manifest_path)
    selected = selected_samples(manifest, split)
    root = Path(source or manifest.get("source", ""))
    rows = []
    for sample in selected:
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
        rows.append(
            {
                "sha256": sample["sha256"],
                "lineage": sample["lineage"],
                "label": sample["label"],
                "values": spatial_features(data),
            }
        )
    artifact = {
        "schema_version": "research-features-1",
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "manifest_sha256": manifest_hash,
        "split": split,
        "rows": rows,
        "support_status": "experimental",
    }
    data = json.dumps(artifact, indent=2, allow_nan=False).encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        stream.write(data)
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "samples": len(rows),
        "split": split,
        "feature_version": FEATURE_VERSION,
    }


def training_inputs(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reject legacy unbound NPZ, validation/test rows, tampering and nonfinite data."""
    if not all(config.get(key) for key in ("manifest", "features", "features_sha256")):
        raise ResearchManifestError("training requires manifest, features and features_sha256")
    manifest, manifest_hash = read_document(Path(config["manifest"]))
    selected = selected_samples(manifest, "train")
    artifact, artifact_hash = read_document(Path(config["features"]))
    if artifact_hash != config["features_sha256"]:
        raise ResearchManifestError("feature artifact checksum mismatch")
    if (
        artifact.get("schema_version") != "research-features-1"
        or artifact.get("feature_version") != FEATURE_VERSION
        or artifact.get("feature_names") != list(FEATURE_NAMES)
        or artifact.get("manifest_sha256") != manifest_hash
        or artifact.get("split") != "train"
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
            or len(values) != len(FEATURE_NAMES)
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
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "split": "train",
        "samples": len(rows),
        "calibrated": False,
    }
    return (
        np.asarray(vectors, dtype=np.float32),
        np.asarray([s["label"] == "stego" for s in selected], dtype=np.float32),
        provenance,
    )
