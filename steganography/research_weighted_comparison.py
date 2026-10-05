"""One preregistered weighting comparison, reusing bound parity feature vectors."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path

from steganography.research import ResearchManifestError, export_onnx, train_model
from steganography.research_features import feature_inputs, predict_validation, read_document
from steganography.research_jpeg import write_json
from steganography.research_spatial import (
    METHODS,
    RATES,
    cell_metrics,
    onnx_parity,
    paired_intervals,
)
from steganography.research_weighting import RECIPE, training_weights

IDENTITY = ("sha256", "lineage", "label", "method", "rate_percent", "format")


def run_comparison(reference_dir: Path, out: Path, *, reference_sha256: str):
    started = time.monotonic()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("comparison output exists or uses a symlink")
    reference, reference_hash = read_document(reference_dir / "report.json")
    if reference_hash != reference_sha256:
        raise ResearchManifestError("reference report checksum mismatch")
    old, old_hash = read_document(reference_dir / "predictions.json")
    if old_hash != reference["predictions_sha256"]:
        raise ResearchManifestError("reference predictions checksum mismatch")
    configs = {}
    for split in ("train", "validation"):
        config, _ = read_document(reference_dir / f"{split}-config.json")
        _, _, provenance = feature_inputs(config, split=split)
        if (
            provenance["manifest_sha256"] != reference["manifest_sha256"]
            or provenance["feature_version"] != "spatial-parity-residual-v1"
            or provenance["features_sha256"] != reference["feature_artifacts"][split]["sha256"]
        ):
            raise ResearchManifestError("cached feature/reference contract mismatch")
        configs[split] = {
            **config,
            "epochs": 300,
            "learning_rate": 0.01,
            "seed": 20261005,
            "standardize": True,
            "class_balanced": True,
            "inference_arithmetic": "float64",
        }
        if split == "train":
            configs[split]["sample_weighting"] = RECIPE
            training_weights(configs[split], provenance)
    manifest, _ = read_document(Path(configs["train"]["manifest"]))
    expected = [s for s in manifest["samples"] if s["split"] == "validation"]
    previous = old.get("predictions", [])
    if len(previous) != len(expected) or any(
        any(row.get(k) != sample.get(k) for k in IDENTITY)
        or type(row.get("score")) not in (float, int)
        or not math.isfinite(row["score"])
        or not 0 <= row["score"] <= 1
        for row, sample in zip(previous, expected, strict=True)
    ):
        raise ResearchManifestError("reference validation identity/score mismatch")
    out.mkdir(parents=True, exist_ok=False)
    for split, config in configs.items():
        write_json(out / f"{split}-config.json", config)
    checkpoint = out / "weighted.pt"
    training = train_model(out / "train-config.json", checkpoint)
    prediction = predict_validation(
        out / "validation-config.json", checkpoint, out / "predictions.json"
    )
    rows = prediction["predictions"]
    if len(rows) != len(previous) or any(
        any(row.get(k) != old_row.get(k) for k in IDENTITY)
        for row, old_row in zip(rows, previous, strict=True)
    ):
        raise ResearchManifestError("new validation identity mismatch")
    card = export_onnx(checkpoint, out / "weighted.onnx")
    parity = onnx_parity(out / "validation-config.json", checkpoint, out / "weighted.onnx")
    cells = {}
    for method in METHODS:
        for rate in RATES:

            def select(records, method=method, rate=rate):
                return [
                    r
                    for r in records
                    if r["label"] == "cover"
                    or (r["method"] == method and r["rate_percent"] == rate)
                ]

            selected = select(rows)
            new_metrics = cell_metrics(selected)
            old_metrics = cell_metrics(select(previous))
            cells[f"{method}-{rate}"] = {
                "reference": old_metrics,
                "new": new_metrics,
                "new_bootstrap_95_percent": paired_intervals(selected),
                "delta": {
                    k: new_metrics[k] - old_metrics[k]
                    for k in (
                        "roc_auc",
                        "balanced_accuracy",
                        "recall",
                        "false_positive_rate",
                        "expected_calibration_error",
                    )
                },
            }
    report = {
        "schema_version": "spatial-weighted-development-v1",
        "manifest_sha256": reference["manifest_sha256"],
        "reference_report_sha256": reference_hash,
        "reference_predictions_sha256": old_hash,
        "feature_artifacts": reference["feature_artifacts"],
        "training": training,
        "checkpoint_sha256": prediction["checkpoint_sha256"],
        "onnx_sha256": card["onnx_sha256"],
        "predictions_sha256": hashlib.sha256((out / "predictions.json").read_bytes()).hexdigest(),
        "onnx_parity": parity,
        "by_method_rate": cells,
        "deployed": False,
        "calibrated": False,
        "cross_source_gate": "unavailable",
        "scope": "iterative same-source inspected validation; cached vectors",
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--reference-sha256", required=True)
    args = parser.parse_args()
    run_comparison(args.reference_dir, args.out, reference_sha256=args.reference_sha256)


if __name__ == "__main__":
    main()
