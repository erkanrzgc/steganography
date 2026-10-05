"""Fixed parity-residual development comparison on an existing audited corpus."""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path
from typing import Any

from core.spatial_parity import FEATURE_VERSION
from steganography.research import (
    ResearchManifestError,
    export_onnx,
    train_model,
    verify_dataset_manifest,
)
from steganography.research_features import extract_features, predict_validation, read_document
from steganography.research_jpeg import write_json
from steganography.research_spatial import (
    METHODS,
    RATES,
    cell_metrics,
    onnx_parity,
    paired_intervals,
)


def run_comparison(
    corpus: Path, out: Path, *, manifest_sha256: str, reference_report: Path, reference_sha256: str
):
    started = time.monotonic()
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("comparison output exists or uses a symlink")
    manifest, manifest_hash = read_document(corpus / "manifest.json")
    reference, reference_hash = read_document(reference_report)
    if manifest_hash != manifest_sha256 or reference_hash != reference_sha256:
        raise ResearchManifestError("comparison source checksum mismatch")
    if reference.get("manifest_sha256") != manifest_hash:
        raise ResearchManifestError("reference comparison corpus mismatch")
    old, old_hash = read_document(
        reference_report.parent / "spatial-cooccurrence-v1/predictions.json"
    )
    if reference["experiments"]["spatial-cooccurrence-v1"]["prediction_sha256"] != old_hash:
        raise ResearchManifestError("reference predictions checksum mismatch")
    counts = verify_dataset_manifest(manifest, source=corpus)
    if counts["splits"]["test"] or counts["sample_count"] > 7000:
        raise ResearchManifestError("development-only bounded corpus required")
    groups: dict[str, list[dict[str, Any]]] = {}
    for sample in manifest["samples"]:
        groups.setdefault(sample["lineage"], []).append(sample)
    recipes = {(None, 0), *((m, r) for m in METHODS for r in RATES)}
    if any(
        len(group) != 7 or {(r.get("method"), r.get("rate_percent")) for r in group} != recipes
        for group in groups.values()
    ):
        raise ResearchManifestError("complete controlled lineages required")
    expected_rows = [s for s in manifest["samples"] if s["split"] == "validation"]
    if len(old.get("predictions", [])) != len(expected_rows) or any(
        any(
            row.get(k) != sample.get(k)
            for k in ("sha256", "lineage", "label", "method", "rate_percent", "format")
        )
        or type(row.get("score")) not in (float, int)
        or not math.isfinite(row["score"])
        or not 0 <= row["score"] <= 1
        for row, sample in zip(old["predictions"], expected_rows, strict=True)
    ):
        raise ResearchManifestError("reference validation identity/score mismatch")
    out.mkdir(parents=True, exist_ok=False)
    feature_artifacts = {}
    for split in ("train", "validation"):
        print(f"extracting parity-residual {split}", flush=True)
        path = out / f"{split}-features.json"
        artifact = extract_features(
            corpus / "manifest.json",
            path,
            source=corpus,
            split=split,
            feature_version=FEATURE_VERSION,
            workers=4,
        )
        feature_artifacts[split] = {**artifact, "bytes": path.stat().st_size}
        write_json(
            out / f"{split}-config.json",
            {
                "manifest": str(corpus / "manifest.json"),
                "features": str(path),
                "features_sha256": artifact["sha256"],
                "epochs": 300,
                "learning_rate": 0.01,
                "seed": 20261005,
                "standardize": True,
                "class_balanced": True,
                "inference_arithmetic": "float64",
            },
        )
    checkpoint = out / "baseline.pt"
    training = train_model(out / "train-config.json", checkpoint)
    predictions = predict_validation(
        out / "validation-config.json", checkpoint, out / "predictions.json"
    )
    rows = predictions["predictions"]
    if len(rows) != len(old["predictions"]) or any(
        any(
            row[k] != previous[k]
            for k in ("sha256", "lineage", "label", "method", "rate_percent", "format")
        )
        for row, previous in zip(rows, old["predictions"], strict=True)
    ):
        raise ResearchManifestError("reference/new validation row identity mismatch")
    card = export_onnx(checkpoint, out / "baseline.onnx")
    parity = onnx_parity(out / "validation-config.json", checkpoint, out / "baseline.onnx")
    cells = {}
    for method in METHODS:
        for rate in RATES:
            selected = [
                r
                for r in rows
                if r["label"] == "cover" or (r["method"] == method and r["rate_percent"] == rate)
            ]
            previous = [
                r
                for r in old["predictions"]
                if r["label"] == "cover" or (r["method"] == method and r["rate_percent"] == rate)
            ]
            new_metrics = cell_metrics(selected)
            old_metrics = cell_metrics(previous)
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
        "schema_version": "spatial-parity-development-v1",
        "manifest_sha256": manifest_hash,
        "reference_report_sha256": reference_hash,
        "reference_predictions_sha256": old_hash,
        "counts": counts,
        "feature_artifacts": feature_artifacts,
        "training": training,
        "checkpoint_sha256": predictions["checkpoint_sha256"],
        "onnx_sha256": card["onnx_sha256"],
        "predictions_sha256": hashlib.sha256((out / "predictions.json").read_bytes()).hexdigest(),
        "onnx_parity": parity,
        "by_method_rate": cells,
        "deployed": False,
        "calibrated": False,
        "cross_source_gate": "unavailable",
        "scope": "iterative same-source inspected validation",
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    a = parser.parse_args()
    run_comparison(
        a.corpus,
        a.out,
        manifest_sha256=a.manifest_sha256,
        reference_report=a.reference_report,
        reference_sha256=a.reference_sha256,
    )


if __name__ == "__main__":
    main()
