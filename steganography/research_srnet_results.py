"""Portable context-cell diagnostics of verified SRNet validation output."""

from __future__ import annotations

import json
import math
from pathlib import Path

from core import srnet_reference
from core.srnet_sampling import source_id
from steganography.research import ResearchManifestError
from steganography.research_features import read_document, selected_samples
from steganography.research_jpeg import write_json
from steganography.research_spatial import cell_metrics, paired_intervals
from steganography.research_srnet_evaluate import checked_card, configuration, oracle_rows
from steganography.research_srnet_fit import bound_training


def summarize(config, predictions_path: Path, out: Path, *, predictions_sha256: str):
    """Never tune thresholds or convert reused development results to support."""
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("SRNet diagnostic output exists or uses a symlink")
    fit_config = configuration(config)
    values, _, _, plan, settings, _ = bound_training(fit_config)
    del values
    manifest, manifest_sha = read_document(Path(config["manifest"]))
    samples = selected_samples(manifest, "validation")
    card, card_sha = read_document(Path(config["model_dir"]) / "model-card.json")
    if card_sha != config["card_sha256"]:
        raise ResearchManifestError("SRNet diagnostic card checksum mismatch")
    checked_card(card, plan, settings, config["plan_sha256"])
    cache, cache_sha = read_document(Path(config["validation_cache"]))
    if cache_sha != config["validation_cache_sha256"] or cache.get("decoder") != plan["decoder"]:
        raise ResearchManifestError("SRNet diagnostic validation cache mismatch")
    report, checksum = read_document(predictions_path)
    expected = {
        "schema_version": "srnet-validation-predictions-v1",
        "manifest_sha256": manifest_sha,
        "train_cache_sha256": config["cache_sha256"],
        "training_plan_sha256": config["plan_sha256"],
        "validation_cache_sha256": config["validation_cache_sha256"],
        "validation_data_sha256": cache["data_sha256"],
        "model_card_sha256": card_sha,
        "model_sha256": card["model_sha256"],
        "training_scope": plan["training_scope"],
        "threshold": 0.5,
        "tie_policy": "stego",
        "calibrated": False,
        "deployed": False,
        "qualification": "unavailable",
        "validation_rows": len(samples),
        "native_rows_evaluated": len(samples),
    }
    if checksum != predictions_sha256 or any(
        json.dumps(report.get(k), sort_keys=True) != json.dumps(v, sort_keys=True)
        for k, v in expected.items()
    ):
        raise ResearchManifestError("SRNet diagnostic prediction provenance mismatch")
    audit = report.get("independent_forward_audit")
    indices = oracle_rows(samples)
    if (
        not isinstance(audit, list)
        or len(audit) != len(indices)
        or any(not isinstance(a, dict) for a in audit)
        or [a.get("sha256") for a in audit] != [samples[i]["sha256"] for i in indices]
    ):
        raise ResearchManifestError("SRNet diagnostic incomplete independent audit")
    for row in audit:
        if (
            type(row.get("passed")) is not bool
            or type(row.get("decisions_equal")) is not bool
            or row.get("absolute_tolerance") != srnet_reference.ABSOLUTE_TOLERANCE
            or row.get("relative_tolerance") != srnet_reference.RELATIVE_TOLERANCE
            or row.get("score_tolerance") != srnet_reference.SCORE_TOLERANCE
            or any(
                type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) or row[k] < 0
                for k in ("maximum_logit_difference", "maximum_score_difference")
            )
            or (
                row["passed"]
                and (
                    not row["decisions_equal"]
                    or row["maximum_score_difference"] > srnet_reference.SCORE_TOLERANCE
                )
            )
        ):
            raise ResearchManifestError("SRNet diagnostic invalid numerical audit")
    passed = all(a["passed"] for a in audit)
    expected_status = "completed" if passed else "failed_numerical_gate"
    rows = report.get("predictions")
    if report.get("status") != expected_status or not isinstance(rows, list):
        raise ResearchManifestError("SRNet diagnostic gate/status mismatch")
    if not passed and rows:
        raise ResearchManifestError("SRNet diagnostic failed gates cannot publish scores")
    cells = []
    if passed:
        if len(rows) != len(samples):
            raise ResearchManifestError("SRNet diagnostic complete validation required")
        for sample, row in zip(samples, rows, strict=True):
            metadata = {k: sample[k] for k in ("sha256", "lineage", "label", "method")}
            metadata.update(
                source_id=source_id(sample["source_group"]),
                quality_factor=sample.get("quality_factor"),
            )
            if (
                not isinstance(row, dict)
                or any(
                    json.dumps(row.get(k), sort_keys=True) != json.dumps(v, sort_keys=True)
                    for k, v in metadata.items()
                )
                or type(row.get("score")) not in (int, float)
                or not math.isfinite(row["score"])
                or not 0 <= row["score"] <= 1
            ):
                raise ResearchManifestError("SRNet diagnostic row identity/score mismatch")
        keys = sorted(
            {
                (r["source_id"], r["quality_factor"], r["method"])
                for r in rows
                if r["label"] == "stego"
            },
            key=str,
        )
        for source, quality, method in keys:
            selected = [
                r
                for r in rows
                if r["source_id"] == source
                and r["quality_factor"] == quality
                and (r["label"] == "cover" or r["method"] == method)
            ]
            covers = [r["lineage"] for r in selected if r["label"] == "cover"]
            stegos = [r["lineage"] for r in selected if r["label"] == "stego"]
            if (
                set(covers) != set(stegos)
                or len(covers) != len(set(covers))
                or len(stegos) != len(set(stegos))
            ):
                raise ResearchManifestError("SRNet diagnostic unmatched validation cell")
            metrics = cell_metrics(selected, threshold=0.5, score_scale=1)
            metric_pass = (
                metrics["roc_auc"] >= 0.90
                and metrics["balanced_accuracy"] >= 0.85
                and metrics["recall"] >= 0.80
                and metrics["false_positive_rate"] <= 0.03
                and metrics["expected_calibration_error"] <= 0.05
            )
            cells.append(
                {
                    "source_id": source,
                    "quality_factor": quality,
                    "method": method,
                    "metrics": metrics,
                    "bootstrap_95_percent": paired_intervals(
                        selected, threshold=0.5, score_scale=1
                    ),
                    "bootstrap_unit": "cover lineage; all derivatives retained together",
                    "independent_lineages": len({r["lineage"] for r in selected}),
                    "metric_gates_passed": metric_pass,
                    "sample_size_gate": metrics["negatives"] >= 1000
                    and metrics["positives"] >= 1000,
                    "support_status": "experimental",
                    "qualified": False,
                }
            )
    result = {
        "schema_version": "srnet-context-diagnostics-v1",
        "status": expected_status,
        "manifest_sha256": manifest_sha,
        "predictions_sha256": checksum,
        "model_card_sha256": card_sha,
        "model_sha256": card["model_sha256"],
        "training_plan_sha256": config["plan_sha256"],
        "training_scope": plan["training_scope"],
        "independent_forward_audit": audit,
        "validation_rows": len(samples),
        "cells": cells,
        "bootstrap_replicates": 200,
        "bootstrap_seed": 20261005,
        "threshold": 0.5,
        "calibrated": False,
        "deployed": False,
        "qualified": False,
        "validation_independence": "reused development validation; not blind or untouched",
        "source_independence": "declared origins; camera/device independence unavailable",
        "onnx_replay": "unavailable",
    }
    write_json(out, result)
    return result
