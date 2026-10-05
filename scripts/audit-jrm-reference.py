#!/usr/bin/env python3
"""Reproduce local-reference audits and all development comparisons; no training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import math
import random
import sys
from pathlib import Path

import numpy as np

# Use the checked-out implementation, not an older installed wheel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steganography import research_jrm as rj  # noqa: E402
from steganography.research_features import read_document, selected_samples  # noqa: E402
from steganography.research_jpeg import write_json  # noqa: E402
from steganography.research_spatial import cell_metrics, paired_intervals  # noqa: E402

MANIFEST_SHA = "0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14"
REFERENCE_SHA = "839ebbba9f5138f89469585c8422fc9c01e8f9a26e6c05be38a593cef73180c9"
KEYS = (
    "roc_auc",
    "balanced_accuracy",
    "recall",
    "false_positive_rate",
    "expected_calibration_error",
)
LIMITS = (0.90, 0.85, 0.80, 0.03, 0.05)


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def sha(path: Path) -> str:
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("audit input must be a regular non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024**2):
            digest.update(data)
    return digest.hexdigest()


def independent_metrics(rows):
    """Pairwise AUC and direct confusion/bin arithmetic, independent of report helpers."""
    scores = np.array([r["score"] for r in rows])
    labels = np.array([r["label"] == "stego" for r in rows])
    positives, negatives = scores[labels], scores[~labels]
    tp, fp = int((positives >= 0.5).sum()), int((negatives >= 0.5).sum())
    recall, fpr = tp / len(positives), fp / len(negatives)
    auc = float(((positives[:, None] > negatives) + 0.5 * (positives[:, None] == negatives)).mean())
    ece = 0.0
    for b in range(10):
        members = np.minimum((scores * 10).astype(int), 9) == b
        if members.any():
            ece += abs(float((scores[members] - labels[members]).sum())) / len(rows)
    return np.array([auc, 0.5 * (recall + 1 - fpr), recall, fpr, ece]), {
        "tp": tp,
        "tn": len(negatives) - fp,
        "fp": fp,
        "fn": len(positives) - tp,
    }


def context_ids(samples, dimension, context, family):
    result = []
    for i, sample in enumerate(samples):
        if sample["label"] == "stego" and sample["method"] != family:
            continue
        source = "source-" + hashlib.sha256(sample["source_group"].encode()).hexdigest()[:16]
        qf = sample.get("quality_factor")
        quality = f"qf-{qf:g}" if qf is not None else "unknown"
        if dimension == "source" and context != source:
            continue
        if dimension == "quality_factor" and context != quality:
            continue
        result.append(i)
    return result


def paired_deltas(old, new):
    groups = {}
    for i, row in enumerate(new):
        groups.setdefault(row["lineage"], []).append(i)
    lineages = sorted(groups)
    generator = random.Random(20261005)  # noqa: S311 - fixed statistical resampling
    values = []
    for _ in range(200):
        ids = [i for _ in lineages for i in groups[generator.choice(lineages)]]
        values.append(
            independent_metrics([new[i] for i in ids])[0]
            - independent_metrics([old[i] for i in ids])[0]
        )
    intervals = np.quantile(values, [0.025, 0.975], axis=0)
    return {key: intervals[:, j].tolist() for j, key in enumerate(KEYS)}


def audit(
    manifest_path: Path,
    corpus: Path,
    experiment: Path,
    reference_predictions: Path,
    reference_record: Path,
):
    manifest, digest = read_document(manifest_path)
    reference, reference_digest = read_document(reference_record)
    old, old_digest = read_document(reference_predictions)
    check(digest == MANIFEST_SHA and reference_digest == REFERENCE_SHA, "frozen inputs changed")
    check(
        old_digest == reference["artifact_sha256"]["predictions.json"], "reference scores changed"
    )
    training = selected_samples(manifest, "train")
    validation = selected_samples(manifest, "validation")
    new, pred_digest = read_document(experiment / "predictions.json")
    check(new["manifest_sha256"] == digest, "prediction manifest mismatch")
    check(
        len(new["predictions"]) == len(old["predictions"]) == len(validation) == 765, "row mismatch"
    )
    for a, b, s in zip(new["predictions"], old["predictions"], validation, strict=True):
        for key in ("sha256", "lineage", "label", "method"):
            check(a[key] == b[key] == s[key], "prediction identity mismatch")
        check(
            type(a["score"]) in (int, float) and math.isfinite(a["score"]) and 0 <= a["score"] <= 1,
            "nonfinite or invalid score",
        )
    for sample in manifest["samples"]:
        path = corpus / sample["path"]
        check(
            sha(path) == sample["sha256"] and path.stat().st_size == sample["size"],
            "JPEG integrity mismatch",
        )
    card, card_digest = read_document(experiment / "model/model-card.json")
    check(new["model_card_sha256"] == card_digest, "model card mismatch")
    check(
        card["manifest_sha256"] == digest
        and card["train_cache_sha256"] == sha(experiment / "train/cache.json"),
        "training contract mismatch",
    )
    x, _, _ = rj.load_cache(
        manifest_path,
        experiment / "validation/cache.json",
        checksum=new["validation_cache_sha256"],
        split="validation",
    )
    model = rj.load_model(experiment / "model/model.npz", checksum=card["model_sha256"])
    votes = []
    for row in x:
        total = 0
        for space, weights, bias in zip(model.subspaces, model.weights, model.biases, strict=True):
            margin = math.fsum(
                float(row[index]) * float(weight)
                for index, weight in zip(space, weights, strict=True)
            ) - float(bias)
            total += (margin > 0) - (margin < 0)
        votes.append((total / len(model.biases) + 1) / 2)
    saved = np.array([r["score"] for r in new["predictions"]])
    np.testing.assert_array_equal(votes, saved)
    batches = {}
    for n in (1, 17, 765):
        actual = model.predict(x[:n])
        np.testing.assert_array_equal(actual, saved[:n])
        batches[str(n)] = {
            "max_score_difference": float(np.abs(actual - saved[:n]).max()),
            "decisions_equal": bool(np.array_equal(actual >= 0.5, saved[:n] >= 0.5)),
            "passed": True,
        }
    # Oracle shares upstream algorithm, but flattening/native boundary are independent.
    import sealwatch as sw

    from core.jpeg_features import worker

    def oracle(y, qt):
        return (
            np.concatenate(
                [
                    v.ravel(order="C")
                    for v in sw.jrm.extract(y.astype(np.int32), calibrated=False).values()
                ]
            )
            .astype("<f4")
            .tolist()
        )

    example_count = 0
    for split in ("train", "validation"):
        features, samples, _ = rj.load_cache(
            manifest_path,
            experiment / split / "cache.json",
            checksum=sha(experiment / split / "cache.json"),
            split=split,
        )
        seen = set()
        for i, sample in enumerate(samples):
            key = (sample["source_group"], sample.get("quality_factor"), sample["method"])
            if key in seen:
                continue
            seen.add(key)
            expected = worker((corpus / sample["path"]).read_bytes(), oracle)
            np.testing.assert_array_equal(features[i], expected)
            example_count += 1
    cells = []
    for baseline in reference["comparison_cells"]:
        dimension, context, family = (
            baseline[k] for k in ("dimension", "context", "method_family")
        )
        ids = context_ids(validation, dimension, context, family)
        before = [old["predictions"][i] for i in ids]
        after = [new["predictions"][i] for i in ids]
        entry = {
            k: baseline[k]
            for k in (
                "dimension",
                "context",
                "method_family",
                "covers",
                "stego",
                "status",
                "reason",
            )
        }
        if baseline["new"] is None:
            entry.update(reference=None, new=None)
        else:
            metric = cell_metrics(after, threshold=0.5, score_scale=1)
            independent, confusion = independent_metrics(after)
            check(confusion == metric["confusion"], "independent confusion mismatch")
            np.testing.assert_allclose(independent, [metric[k] for k in KEYS], atol=5.01e-7, rtol=0)
            old_metric = cell_metrics(before, threshold=0.5, score_scale=1)
            check(old_metric == baseline["new"], "reference metrics mismatch")
            entry.update(
                reference=old_metric,
                new=metric,
                original_lineages=len({r["lineage"] for r in after}),
                new_bootstrap_95_percent=paired_intervals(after, threshold=0.5, score_scale=1),
                delta_new_minus_reference={k: metric[k] - old_metric[k] for k in KEYS},
                paired_delta_bootstrap_95_percent=paired_deltas(before, after),
                numeric_failures=[
                    k
                    for j, k in enumerate(KEYS)
                    if (metric[k] > LIMITS[j] if j >= 3 else metric[k] < LIMITS[j])
                ],
            )
        cells.append(entry)
    train_sources = {s["source_group"] for s in training}
    val_sources = {s["source_group"] for s in validation}
    return {
        "schema_version": "jrm-reference-development-v1",
        "date": "2026-10-05",
        "manifest_sha256": digest,
        "reference_record_sha256": reference_digest,
        "reference_predictions_sha256": old_digest,
        "predictions_sha256": pred_digest,
        "model_card_sha256": card_digest,
        "training": card,
        "source_records": reference["source_records"],
        "source_id_map": reference["source_id_map"],
        "counts": reference["counts"],
        "source_separation": {
            "training_source_groups": len(train_sources),
            "validation_source_groups": len(val_sources),
            "shared_source_groups": len(train_sources & val_sources),
            "unseen_validation_source_groups": len(val_sources - train_sources),
            "independent_source_evidence": "unavailable",
        },
        "comparison_cells": cells,
        "bootstrap": {
            "resamples": 200,
            "seed": 20261005,
            "unit": "original lineage; correlated qualities stay together",
            "uncorrected_development_intervals": True,
        },
        "independent_audit": {
            "files_rehashed": 3750,
            "metric_cells_checked": 12,
            "upstream_jrm_parity_examples": example_count,
            "jrm_algorithm_independent_oracle": False,
            "scalar_fld_vote_rows": 765,
            "numeric_reload_batches": batches,
            "passed": True,
        },
        "execution": read_document(experiment / "execution-summary.json")[0],
        "artifact_sha256": {
            name: sha(experiment / name)
            for name in (
                "execution-summary.json",
                "model/model-card.json",
                "model/model.npz",
                "predictions.json",
                "train/cache.json",
                "train/features.f32",
                "validation/cache.json",
                "validation/features.f32",
            )
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("sealwatch", "numpy", "scipy", "jpeglib", "pandas", "h5py", "setuptools")
        },
        "support_status": "experimental",
        "deployed": False,
        "calibrated": False,
        "qualification": "unavailable; reused development sources and insufficient scene counts",
        "limitations": [
            "Representation/classifier/objective change together; not an isolated ablation.",
            "Shared training/validation sources; inspected validation is not blind/cross-source.",
            "BOSS: 25 original validation scenes, correlated Q variants; not payload recovery.",
            "ALASKA quality/payload and camera/device/app provenance unavailable. JMiPOD excluded.",
            "FLD scores are vote fractions, not calibrated probabilities; ECE is diagnostic.",
            "JRM: local research-only under upstream terms; no code/weights/data redistribution.",
            "No ONNX export/signed model publication/primary score change.",
            "Upstream parity does not qualify detection or prove MATLAB feature equivalence.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest",
        "corpus",
        "experiment",
        "reference-predictions",
        "reference-record",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = audit(
        args.manifest,
        args.corpus,
        args.experiment,
        args.reference_predictions,
        args.reference_record,
    )
    write_json(args.out, report)
    print("Complete: all files/cells, upstream feature parity and independent scalar FLD votes.")


if __name__ == "__main__":
    main()
