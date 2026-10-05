"""Explicit train-only weighting for the controlled spatial research corpus."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from steganography.research import ResearchManifestError
from steganography.research_features import read_document, selected_samples

RECIPE = "spatial-low-payload-x4-v1"


def training_weights(config: dict[str, Any], provenance: dict[str, Any]) -> np.ndarray | None:
    """Never derive weights from validation scores, features or installed models."""
    if "sample_weighting" not in config:
        return None
    if config["sample_weighting"] != RECIPE:
        raise ResearchManifestError("unknown training weighting recipe")
    if provenance["feature_version"] != "spatial-parity-residual-v1":
        raise ResearchManifestError("weighting requires the parity-residual feature contract")
    manifest, checksum = read_document(Path(config["manifest"]))
    if checksum != provenance["manifest_sha256"] or provenance["split"] != "train":
        raise ResearchManifestError("weighting training manifest mismatch")
    samples = selected_samples(manifest, "train")
    if len(samples) != provenance["samples"]:
        raise ResearchManifestError("weighting training sample count mismatch")
    expected = {(None, 0), *((m, r) for m in ("sequential", "scattered") for r in (5, 20, 40))}
    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        groups.setdefault(sample["lineage"], []).append(sample)
    if any(
        len(group) != 7
        or {(s.get("method"), s.get("rate_percent")) for s in group} != expected
        or any((s["label"] == "cover") != (s.get("method") is None) for s in group)
        for group in groups.values()
    ):
        raise ResearchManifestError("weighting requires complete controlled training lineages")
    return np.asarray(
        [4.0 if s["label"] == "stego" and s["rate_percent"] == 5 else 1.0 for s in samples],
        dtype=np.float32,
    )
