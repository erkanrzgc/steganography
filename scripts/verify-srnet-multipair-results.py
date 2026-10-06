#!/usr/bin/env python3
"""Replay scalar metric arithmetic on complete local two-source predictions."""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from steganography.research_features import read_document  # noqa: E402
from steganography.research_jpeg import write_json  # noqa: E402


def verify(evidence_path, predictions_path):
    evidence, evidence_sha = read_document(evidence_path)
    predictions, predictions_sha = read_document(predictions_path)
    diagnostic = evidence["diagnostics"]
    rows = predictions["predictions"]
    if (
        diagnostic["status"] != "completed"
        or predictions["status"] != "completed"
        or predictions_sha != diagnostic["predictions_sha256"]
        or len(rows) != diagnostic["validation_rows"]
        or len(rows) != 765
        or len(diagnostic["cells"]) != 6
        or predictions["model_sha256"] != evidence["model_sha256"]
        or predictions["model_card_sha256"] != evidence["model_card_sha256"]
        or predictions["training_plan_sha256"] != evidence["training_plan_sha256"]
    ):
        raise ValueError("complete checksum-bound multipair predictions required")
    # Fixed, trusted in-repository independent arithmetic; never imported from
    # an artifact or a supplied path. No model, image or executable artifact loads.
    spec = importlib.util.spec_from_file_location(
        "scalar_metric_reference", ROOT / "scripts/audit-jrm-reference.py"
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    maximum = 0.0
    keys = set()
    for cell in diagnostic["cells"]:
        key = cell["source_id"], cell["quality_factor"], cell["method"]
        if key in keys:
            raise ValueError("duplicate metric cell")
        keys.add(key)
        selected = [
            row
            for row in rows
            if row["source_id"] == key[0]
            and row["quality_factor"] == key[1]
            and (row["label"] == "cover" or row["method"] == key[2])
        ]
        values, confusion = reference.independent_metrics(selected)
        reported = np.array([cell["metrics"][k] for k in reference.KEYS])
        if not np.isfinite(values).all() or not np.isfinite(reported).all():
            raise ValueError("nonfinite metric audit")
        difference = float(np.max(np.abs(values - reported)))
        maximum = max(maximum, difference)
        if difference > 1e-6 or confusion != cell["metrics"]["confusion"]:
            raise ValueError("independent scalar metric mismatch")
    return {
        "schema_version": "srnet-multipair-metric-audit-v1",
        "evidence_sha256": evidence_sha,
        "predictions_sha256": predictions_sha,
        "cells_audited": len(keys),
        "metric_names": list(reference.KEYS),
        "maximum_metric_difference": maximum,
        "absolute_tolerance": 1e-6,
        "confusion_counts_exact": True,
        "passed": True,
        "accuracy_qualification": "unavailable",
        "model_weights_loaded": False,
        "primary_detection_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.out, verify(args.evidence, args.predictions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
