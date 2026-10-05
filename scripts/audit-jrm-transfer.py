#!/usr/bin/env python3
"""Replay source-transfer votes, paired metrics and corpus integrity without training."""

from __future__ import annotations

import argparse
import math
import runpy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from steganography import research_jrm as rj  # noqa: E402
from steganography.research_features import read_document, selected_samples  # noqa: E402
from steganography.research_jpeg import write_json  # noqa: E402

# Reuse checked-in independent arithmetic, not the production metric helpers.
HELPERS = runpy.run_path(str(ROOT / "scripts/audit-jrm-reference.py"))
sha, check = HELPERS["sha"], HELPERS["check"]
KEYS = HELPERS["KEYS"]


def scalar_votes(model, features):
    values = []
    for row in features:
        votes = 0
        for space, weights, bias in zip(model.subspaces, model.weights, model.biases, strict=True):
            margin = math.fsum(
                float(row[i]) * float(w) for i, w in zip(space, weights, strict=True)
            ) - float(bias)
            votes += (margin > 0) - (margin < 0)
        values.append((votes / len(model.biases) + 1) / 2)
    return np.array(values)


def audit(manifest_path, corpus, experiment, reference_cache, reference_record, reference_sha):
    check(sha(reference_record) == reference_sha, "reference publication checksum mismatch")
    baseline, _ = read_document(reference_record)
    report, report_sha = read_document(experiment / "report.json")
    manifest, digest = read_document(manifest_path)
    check(report["manifest_sha256"] == digest == baseline["manifest_sha256"], "manifest mismatch")
    check(report["schema_version"] == "jrm-source-transfer-development-v1", "transfer schema")
    check(not report["deployed"] and not report["calibrated"], "unexpected deployment")
    training = selected_samples(manifest, "train")
    rj.load_cache(
        manifest_path,
        reference_cache.parent.parent / "train/cache.json",
        checksum=report["cache_sha256"]["train"],
        split="train",
    )
    features, validation, _ = rj.load_cache(
        manifest_path,
        reference_cache,
        checksum=report["cache_sha256"]["validation"],
        split="validation",
    )
    old, old_sha = read_document(reference_cache.parent.parent / "predictions.json")
    check(
        old_sha == baseline["predictions_sha256"] == report["reference_predictions_sha256"],
        "reference predictions mismatch",
    )
    files = 0
    for sample in manifest["samples"]:
        path = corpus / sample["path"]
        check(
            sha(path) == sample["sha256"] and path.stat().st_size == sample["size"],
            "corpus integrity failure",
        )
        files += 1
    rows_by_source, batches = {}, {}
    artifacts = {"report.json": report_sha}
    sources = rj.training_scope(training)[1]["source_ids"]
    check(len(sources) == 2 and set(report["models"]) == set(sources), "two fits required")
    for source in sources:
        model_dir = experiment / source
        item = report["models"][source]
        card, card_sha = read_document(model_dir / "model-card.json")
        check(card_sha == item["model_card_sha256"] and card == item["training"], "card mismatch")
        check(
            all(
                card[k] == baseline["training"][k]
                for k in (
                    "seed",
                    "learners",
                    "subspace",
                    "sealwatch_version",
                    "feature_version",
                    "manifest_sha256",
                    "calibrated",
                    "deployed",
                )
            ),
            "fixed training parameters changed",
        )
        scope = rj.training_scope(training, source)[1]
        check(scope == card["training_scope"], "fit source scope mismatch")
        check(card["train_cache_sha256"] == report["cache_sha256"]["train"], "train cache mismatch")
        prediction, prediction_sha = read_document(model_dir / "predictions.json")
        check(prediction_sha == item["predictions_sha256"], "predictions mismatch")
        check(
            prediction["model_card_sha256"] == card_sha
            and prediction["manifest_sha256"] == digest
            and prediction["validation_cache_sha256"] == report["cache_sha256"]["validation"],
            "prediction provenance mismatch",
        )
        rows = prediction["predictions"]
        check(len(rows) == len(validation), "row count mismatch")
        for row, sample in zip(rows, validation, strict=True):
            check(
                all(row[k] == sample[k] for k in ("sha256", "lineage", "label", "method")),
                "prediction identity mismatch",
            )
        model = rj.load_model(model_dir / "model.npz", checksum=card["model_sha256"])
        saved = np.array([r["score"] for r in rows])
        np.testing.assert_array_equal(scalar_votes(model, features), saved)
        batches[source] = {}
        for n in (1, 17, len(validation)):
            actual = model.predict(features[:n])
            np.testing.assert_array_equal(actual, saved[:n])
            batches[source][str(n)] = {
                "passed": True,
                "max_score_difference": 0.0,
                "decisions_equal": True,
            }
        rows_by_source[source] = rows
        for name in ("model-card.json", "model.npz", "predictions.json"):
            artifacts[f"{source}/{name}"] = sha(model_dir / name)
    cells = report["comparison_cells"]
    combinations = {(s, t, f) for s in sources for t in sources for f in ("JUNIWARD", "UERD")}
    check(
        len(cells) == 8
        and {
            (c["training_source_id"], c["validation_source_id"], c["method_family"]) for c in cells
        }
        == combinations,
        "incomplete/duplicate cells",
    )
    for cell in cells:
        source, target, family = (
            cell[k] for k in ("training_source_id", "validation_source_id", "method_family")
        )
        indices = HELPERS["context_ids"](validation, "source", target, family)
        before = [old["predictions"][i] for i in indices]
        after = [rows_by_source[source][i] for i in indices]
        previous = next(
            c["new"]
            for c in baseline["comparison_cells"]
            if (
                c["dimension"] == "source"
                and c["context"] == target
                and c["method_family"] == family
            )
        )
        check(previous == cell["reference"], "published reference metrics changed")
        for rows, key in ((before, "reference"), (after, "new")):
            metrics, confusion = HELPERS["independent_metrics"](rows)
            check(confusion == cell[key]["confusion"], "confusion mismatch")
            np.testing.assert_allclose(metrics, [cell[key][k] for k in KEYS], atol=5.01e-7, rtol=0)
        cell["paired_delta_bootstrap_95_percent"] = HELPERS["paired_deltas"](before, after)
    return {
        **report,
        "reference_record_sha256": reference_sha,
        "artifact_sha256": artifacts,
        "independent_audit": {
            "passed": True,
            "files_rehashed": files,
            "metric_cells_checked": 8,
            "scalar_vote_rows_per_model": len(validation),
            "models_checked": 2,
            "numeric_reload_batches": batches,
            "independent_jrm_algorithm_oracle": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("manifest", "corpus", "experiment", "reference-cache", "reference-record", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--reference-record-sha256", required=True)
    args = parser.parse_args()
    report = audit(
        args.manifest,
        args.corpus,
        args.experiment,
        args.reference_cache,
        args.reference_record,
        args.reference_record_sha256,
    )
    write_json(args.out, report)
    print("Complete: source scopes, corpus hashes, scalar votes and all eight metric cells.")


if __name__ == "__main__":
    main()
