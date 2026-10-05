"""Bound, provenance-aware validation diagnostics; never a deployment gate."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Any

from steganography.research import ResearchManifestError
from steganography.research_features import (
    MAX_ROWS,
    MODEL_DOMAINS,
    feature_contract,
    read_document,
    selected_samples,
)
from steganography.research_jpeg import write_json
from steganography.research_spatial import cell_metrics, paired_intervals

MAX_CELLS = 256


def _source(sample: dict[str, Any]) -> str | None:
    value = sample.get("source_group")
    return value if isinstance(value, str) and value.strip() else None


def _opaque(value: Any, prefix: str) -> str:
    if value is None:
        return "unknown"
    return f"{prefix}-{hashlib.sha256(str(value).encode()).hexdigest()[:16]}"


def _quality(sample: dict[str, Any]) -> str:
    value = sample.get("quality_factor")
    if value is None:
        return "unknown"
    if type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= 100:
        raise ResearchManifestError("declared quality factor must be in [1, 100]")
    return f"qf-{value:g}"


def diagnose_validation(
    manifest_path: Path,
    predictions_path: Path,
    model_card_path: Path,
    out: Path,
    *,
    predictions_sha256: str,
    model_card_sha256: str,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Audit cached scores by source/format/declared quality; don't retune them."""
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("diagnostic output exists or uses a symlink")
    if (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ResearchManifestError("diagnostic threshold must be finite and in [0, 1]")
    manifest, manifest_hash = read_document(manifest_path)
    if not isinstance(manifest.get("samples"), list) or len(manifest["samples"]) > MAX_ROWS:
        raise ResearchManifestError("diagnostic manifest row limit exceeded")
    train = selected_samples(manifest, "train")
    validation = selected_samples(manifest, "validation")
    predictions, prediction_hash = read_document(predictions_path)
    card, card_hash = read_document(model_card_path)
    if prediction_hash != predictions_sha256 or card_hash != model_card_sha256:
        raise ResearchManifestError("diagnostic input checksum mismatch")
    provenance = predictions.get("provenance", {})
    training = card.get("training_provenance", {})
    preprocessing = card.get("preprocessing", {})
    if not all(isinstance(v, dict) for v in (provenance, training, preprocessing)):
        raise ResearchManifestError("diagnostic provenance must be an object")
    version = provenance.get("feature_version")
    if not isinstance(version, str):
        raise ResearchManifestError("diagnostic feature contract missing")
    names, _ = feature_contract(version)
    if (
        card.get("domain") != MODEL_DOMAINS[version]
        or provenance.get("manifest_sha256") != manifest_hash
        or provenance.get("split") != "validation"
        or provenance.get("samples") != len(validation)
        or provenance.get("feature_names") != list(names)
        or training.get("manifest_sha256") != manifest_hash
        or training.get("split") != "train"
        or training.get("samples") != len(train)
        or training.get("feature_version") != version
        or training.get("feature_names") != list(names)
        or preprocessing.get("feature_version") != version
        or preprocessing.get("feature_names") != list(names)
    ):
        raise ResearchManifestError("diagnostic training/validation contract mismatch")
    rows = predictions.get("predictions")
    identity = ("sha256", "lineage", "label", "method")
    optional = ("rate_percent", "format")
    legacy = predictions.get("schema_version") == "jpeg-development-predictions-v1" and (
        version == "jpeg-dct-summary-v1"
    )
    if (
        not isinstance(rows, list)
        or len(rows) != len(validation)
        or any(
            not isinstance(row, dict)
            or any(k not in row or row.get(k) != sample.get(k) for k in identity)
            or any(k in row and row[k] != sample.get(k) for k in optional)
            or (not legacy and any(k not in row for k in optional))
            or type(row.get("score")) not in (int, float)
            or not math.isfinite(row["score"])
            or not 0 <= row["score"] <= 1
            for row, sample in zip(rows, validation, strict=True)
        )
    ):
        raise ResearchManifestError("diagnostic validation identity/score mismatch")
    augmented_fields = sum(k not in row for row in rows for k in optional)
    rows = [
        {**row, **{k: sample.get(k) for k in optional if k not in row}}
        for row, sample in zip(rows, validation, strict=True)
    ]
    for row in rows:
        rate = row.get("rate_percent")
        if rate is not None and (
            type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 100
        ):
            raise ResearchManifestError("diagnostic payload rate must be in [0, 100]")
    train_sources = {_source(s) for s in train} - {None}
    validation_sources = {_source(s) for s in validation} - {None}
    metadata = {
        field: {
            "known_rows": sum(s.get(field) is not None for s in validation),
            "unknown_rows": sum(s.get(field) is None for s in validation),
        }
        for field in ("quality_factor", "camera", "device", "app")
    }
    contexts: dict[tuple[str, str], list[dict[str, Any]]] = {("pooled", "all"): rows}
    for sample, row in zip(validation, rows, strict=True):
        fmt = str(sample.get("format", "")).lower()
        fmt = "jpeg" if fmt == "jpg" else fmt
        for dimension, value in (
            ("source", _opaque(_source(sample), "source")),
            ("format", fmt if fmt in {"png", "bmp", "jpeg", "tif", "tiff"} else "unknown"),
            ("quality_factor", _quality(sample)),
        ):
            if dimension == "quality_factor" and metadata["quality_factor"]["known_rows"] == 0:
                continue
            contexts.setdefault((dimension, value), []).append(row)
    families = {
        (_opaque(r.get("method"), "method"), r.get("rate_percent"))
        for r in rows
        if r["label"] == "stego"
    }
    method_names = {
        _opaque(name, "method"): name
        for name in ("sequential", "scattered", "JMiPOD", "JUNIWARD", "UERD")
    }
    if len(contexts) * len(families) > MAX_CELLS:
        raise ResearchManifestError("diagnostic context cell limit exceeded")
    cells = []
    for (dimension, context), members in sorted(contexts.items()):
        for method, rate in sorted(families, key=lambda pair: (pair[0], str(pair[1]))):
            selected = [
                r
                for r in members
                if r["label"] == "cover"
                or (_opaque(r.get("method"), "method"), r.get("rate_percent")) == (method, rate)
            ]
            labels = Counter(r["label"] for r in selected)
            both = bool(labels["cover"] and labels["stego"])
            measurable = both and context != "unknown"
            cells.append(
                {
                    "dimension": dimension,
                    "context": context,
                    "method_id": method,
                    "method_family": method_names.get(method, "unknown"),
                    "rate_percent": rate,
                    "covers": labels["cover"],
                    "stego": labels["stego"],
                    "status": "experimental" if measurable else "unavailable",
                    "reason": None
                    if measurable
                    else (
                        "unknown context metadata"
                        if context == "unknown"
                        else "context lacks both labels"
                    ),
                    "metrics": cell_metrics(selected, threshold=threshold, score_scale=1)
                    if measurable
                    else None,
                    "bootstrap_95_percent": paired_intervals(
                        selected, threshold=threshold, score_scale=1
                    )
                    if measurable
                    else None,
                }
            )
    all_known = all(_source(s) is not None for s in train + validation)
    disjoint = bool(train_sources and validation_sources and not train_sources & validation_sources)
    report = {
        "schema_version": "research-generalization-diagnostics-v1",
        "manifest_sha256": manifest_hash,
        "predictions_sha256": prediction_hash,
        "model_card_sha256": card_hash,
        "feature_version": version,
        "threshold": threshold,
        "legacy_fields_from_bound_manifest": augmented_fields,
        "source_separation": {
            "training_source_groups": len(train_sources),
            "validation_source_groups": len(validation_sources),
            "shared_source_groups": len(train_sources & validation_sources),
            "unseen_validation_source_groups": len(validation_sources - train_sources),
            "metadata_complete": all_known,
            "independent_source_evidence": "candidate_not_qualified"
            if all_known and disjoint
            else "unavailable",
        },
        "metadata_coverage": metadata,
        "cells": cells,
        "support_status": "experimental",
        "deployed": False,
        "limitations": [
            "Cached validation diagnostics, not a blind test or support qualification.",
            "Source identity declarations cannot prove independence or camera provenance.",
            "Format and declared quality are measured context, not learned score corrections.",
            "Small/single-label cells and unknown metadata cannot demonstrate generalization.",
            "No training, calibration, threshold search, image access or detector changes.",
        ],
    }
    write_json(out, report)
    return report
