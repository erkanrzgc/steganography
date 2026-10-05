"""Explicit train-only weighting for controlled spatial and multi-origin JPEG research."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from steganography.research import ResearchManifestError
from steganography.research_features import read_document, selected_samples

RECIPE = "spatial-low-payload-x4-v1"
JPEG_RECIPE = "jpeg-source-class-balanced-v1"


def training_weights(config: dict[str, Any], provenance: dict[str, Any]) -> np.ndarray | None:
    """Never derive weights from validation scores, features or installed models."""
    context_model = provenance.get("feature_version") == "jpeg-context-summary-v1"
    if context_model and config.get("sample_weighting") != JPEG_RECIPE:
        raise ResearchManifestError(
            "JPEG context training requires explicit multi-source weighting"
        )
    if "sample_weighting" not in config:
        return None
    if config["sample_weighting"] not in (RECIPE, JPEG_RECIPE):
        raise ResearchManifestError("unknown training weighting recipe")
    if config["sample_weighting"] == JPEG_RECIPE and not context_model:
        raise ResearchManifestError("multi-source weighting requires the JPEG context contract")
    if (
        config["sample_weighting"] == RECIPE
        and provenance["feature_version"] != "spatial-parity-residual-v1"
    ):
        raise ResearchManifestError("weighting requires the parity-residual feature contract")
    manifest, checksum = read_document(Path(config["manifest"]))
    if checksum != provenance["manifest_sha256"] or provenance["split"] != "train":
        raise ResearchManifestError("weighting training manifest mismatch")
    samples = selected_samples(manifest, "train")
    if len(samples) != provenance["samples"]:
        raise ResearchManifestError("weighting training sample count mismatch")
    if context_model:
        return _jpeg_source_weights(manifest, samples, provenance)
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


def _jpeg_source_weights(manifest, samples, provenance):
    from collections import Counter

    source_values = [s.get("source_group") for s in samples]
    if any(not isinstance(s, str) or not s.strip() for s in source_values):
        raise ResearchManifestError("JPEG context training requires named source groups")
    sources = set(source_values)
    if len(sources) < 2:
        raise ResearchManifestError("JPEG context training requires at least two declared sources")
    catalog = manifest.get("catalog", {})
    records = catalog.get("source_records", {}) if isinstance(catalog, dict) else None
    if not isinstance(records, dict) or not sources <= records.keys():
        raise ResearchManifestError("multi-source origin records are required")
    origins = []
    for source in sorted(sources):
        record = records[source]
        digest = record.get("origin_manifest_sha256") if isinstance(record, dict) else None
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or any(
                not isinstance(record.get(k), str) or not record[k].strip()
                for k in ("license", "source_url")
            )
        ):
            raise ResearchManifestError("invalid multi-source origin/license record")
        origins.append(digest)
    if len(set(origins)) != len(sources):
        raise ResearchManifestError("renamed copies of one origin are not multiple sources")
    counts = Counter((s["source_group"], s["label"]) for s in samples)
    if any(not counts[(source, label)] for source in sources for label in ("cover", "stego")):
        raise ResearchManifestError("each training source requires both labels")
    method_sets = []
    for source in sorted(sources):
        methods = [
            s.get("method")
            for s in samples
            if s["source_group"] == source and s["label"] == "stego"
        ]
        if any(not isinstance(m, str) or not m.strip() for m in methods):
            raise ResearchManifestError("multi-source positive method families must be declared")
        method_sets.append(set(methods))
    if any(methods != method_sets[0] for methods in method_sets[1:]):
        raise ResearchManifestError("source-specific method families create a training shortcut")
    provenance["source_balance"] = {
        "declared_sources": len(sources),
        "common_method_families": len(method_sets[0]),
        "origin_manifest_sha256": origins,
        "independence": "declared origins, not verified camera/perceptual independence",
    }
    return np.asarray(
        [
            len(samples) / (2 * len(sources) * counts[(s["source_group"], s["label"])])
            for s in samples
        ],
        dtype=np.float32,
    )
