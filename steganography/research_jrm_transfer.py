"""Fixed leave-one-training-origin-out diagnostic, not blind qualification."""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

from steganography import research_jrm as rj
from steganography.research import ResearchManifestError
from steganography.research_features import read_document
from steganography.research_jpeg import write_json
from steganography.research_spatial import cell_metrics, paired_intervals
from steganography.research_weighting import _jpeg_source_weights

KEYS = (
    "roc_auc",
    "balanced_accuracy",
    "recall",
    "false_positive_rate",
    "expected_calibration_error",
)
TARGETS = (0.90, 0.85, 0.80, 0.03, 0.05)


def run_transfer(config: dict, out: Path) -> dict:
    """Reuse both complete caches; fit two fixed single-source ensembles only."""
    rj._fresh(out)
    manifest_path = Path(config["manifest"])
    manifest, digest = read_document(manifest_path)
    if digest != config["manifest_sha256"]:
        raise ResearchManifestError("transfer manifest checksum mismatch")
    train_cache, val_cache = (Path(config[k]) for k in ("train_cache", "validation_cache"))
    _, training, _ = rj.load_cache(
        manifest_path, train_cache, checksum=config["train_cache_sha256"], split="train"
    )
    _, validation, _ = rj.load_cache(
        manifest_path, val_cache, checksum=config["validation_cache_sha256"], split="validation"
    )
    _jpeg_source_weights(manifest, training, {})
    rj.paired_indices(training)
    rj.paired_indices(validation)
    _, scope = rj.training_scope(training)
    _, val_scope = rj.training_scope(validation)
    sources = scope["source_ids"]
    if len(sources) != 2 or sources != val_scope["source_ids"]:
        raise ResearchManifestError("transfer requires the same two declared development origins")
    reference, reference_hash = read_document(Path(config["reference_predictions"]))
    rows = reference.get("predictions")
    if (
        reference_hash != config["reference_predictions_sha256"]
        or reference.get("schema_version") != "jrm-reference-predictions-v1"
        or reference.get("manifest_sha256") != digest
        or reference.get("validation_cache_sha256") != config["validation_cache_sha256"]
        or not isinstance(rows, list)
        or len(rows) != len(validation)
        or any(
            not isinstance(r, dict)
            or any(r.get(k) != s[k] for k in ("sha256", "lineage", "label", "method"))
            or type(r.get("score")) not in (int, float)
            or not math.isfinite(r["score"])
            or not 0 <= r["score"] <= 1
            for r, s in zip(rows, validation, strict=True)
        )
    ):
        raise ResearchManifestError("transfer reference identity/checksum/score mismatch")
    rj.jrm.require_version()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    models, cells = {}, []
    for source in sources:
        model_dir = out / source
        card = rj.train_reference(
            manifest_path,
            train_cache,
            model_dir,
            cache_sha256=config["train_cache_sha256"],
            training_source_id=source,
        )
        card_hash = hashlib.sha256((model_dir / "model-card.json").read_bytes()).hexdigest()
        prediction = rj.predict_reference(
            manifest_path,
            val_cache,
            model_dir,
            model_dir / "predictions.json",
            cache_sha256=config["validation_cache_sha256"],
            card_sha256=card_hash,
        )
        models[source] = {
            "training": card,
            "model_card_sha256": card_hash,
            "predictions_sha256": hashlib.sha256(
                (model_dir / "predictions.json").read_bytes()
            ).hexdigest(),
        }
        for target in sources:
            for family in ("JUNIWARD", "UERD"):
                indices = [
                    i
                    for i, sample in enumerate(validation)
                    if "source-" + hashlib.sha256(sample["source_group"].encode()).hexdigest()[:16]
                    == target
                    and (sample["label"] == "cover" or sample["method"] == family)
                ]
                after = [prediction["predictions"][i] for i in indices]
                new = cell_metrics(after, threshold=0.5, score_scale=1)
                old = cell_metrics([rows[i] for i in indices], threshold=0.5, score_scale=1)
                cells.append(
                    {
                        "training_source_id": source,
                        "validation_source_id": target,
                        "training_excluded_target": source != target,
                        "method_family": family,
                        "original_lineages": len({r["lineage"] for r in after}),
                        "reference": old,
                        "new": new,
                        "new_bootstrap_95_percent": paired_intervals(
                            after, threshold=0.5, score_scale=1
                        ),
                        "delta_new_minus_reference": {k: new[k] - old[k] for k in KEYS},
                        "numeric_failures": [
                            k
                            for i, k in enumerate(KEYS)
                            if (new[k] > TARGETS[i] if i >= 3 else new[k] < TARGETS[i])
                        ],
                    }
                )
    report = {
        "schema_version": "jrm-source-transfer-development-v1",
        "manifest_sha256": digest,
        "reference_predictions_sha256": reference_hash,
        "cache_sha256": {
            "train": config["train_cache_sha256"],
            "validation": config["validation_cache_sha256"],
        },
        "models": models,
        "comparison_cells": cells,
        "support_status": "experimental",
        "deployed": False,
        "calibrated": False,
        "primary_detection_changed": False,
        "qualification": "unavailable",
        "scope": (
            "Training-origin exclusion; reused inspected development validation, "
            "not blind or new-source qualification."
        ),
        "bootstrap": {
            "resamples": 200,
            "seed": 20261005,
            "unit": "original lineage; correlated qualities together",
        },
        "seconds": time.monotonic() - started,
    }
    write_json(out / "report.json", report)
    return report
