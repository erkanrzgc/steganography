"""Original-pair intervals and the frozen all-source/all-seed research gate."""

from __future__ import annotations

import numpy as np

from core import jpeg_refine_schedule as schedule
from core import jpeg_study as study
from core import jpeg_study_data as data


def measured(pairs):
    negatives, positives = pairs[:, 0], pairs[:, 1]
    tn, tp = float(np.mean(negatives < 0.5)), float(np.mean(positives >= 0.5))
    flat = pairs.ravel()
    labels = np.tile([0, 1], len(pairs))
    bins = np.minimum((flat * 10).astype(int), 9)
    ece = sum(
        float(mask.mean() * abs(flat[mask].mean() - labels[mask].mean()))
        for i in range(10)
        if (mask := bins == i).any()
    )
    return {
        "balanced_accuracy": (tn + tp) / 2,
        "recall": tp,
        "false_positive_rate": 1 - tn,
        "ece": ece,
        "tp": float(np.sum(positives >= 0.5)),
        "tn": float(np.sum(negatives < 0.5)),
        "fp": float(np.sum(negatives >= 0.5)),
        "fn": float(np.sum(positives < 0.5)),
    }


def statistics(samples, scores, *, baseline=None, role="development"):
    if role not in ("fit", "development") or any(r["role"] != role for r in samples):
        raise ValueError("refine metric role mismatch")
    # Historical validator computes paired AUC intervals with identical draws.
    adapted = [{**r, "role": "development"} for r in samples]
    cells = study.statistics(adapted, scores, baseline=baseline)
    for cell in cells:
        indices = [
            i // 2
            for i in range(0, len(samples), 2)
            if samples[i]["source_group"] == cell["source_group"]
        ]
        pairs = np.asarray(scores).reshape(-1, 2)[indices]
        rng = np.random.default_rng(study.SEEDS[0])
        estimates: dict[str, list[float]] = {k: [] for k in measured(pairs)}
        for _ in range(1000):
            draw = rng.integers(len(pairs), size=len(pairs))
            for key, value in measured(pairs[draw]).items():
                estimates[key].append(value)
        cell["metric_ci95"] = {
            k: np.quantile(v, [0.025, 0.975]).tolist() for k, v in estimates.items()
        }
        cell["role"] = role
        if role == "fit":
            cell["learning_gate"] = False
        if baseline is not None:
            prior = np.asarray(baseline).reshape(-1, 2)[indices]
            cell["paired_auc_delta"] = study.auc(pairs) - study.auc(prior)
    return cells


def compare(manifest, evaluations):
    expected = {(a, s) for s in study.SEEDS for a in ("warmup", *schedule.ARMS)}
    if (
        len(evaluations) != 6
        or {(v["metadata"]["arm"], v["metadata"]["seed"]) for v in evaluations} != expected
        or any(
            v.get("status") != "completed"
            or v.get("real_gpu_evaluation") is not True
            or v.get("model_sha256_before") != v.get("model_sha256_after")
            or v["metadata"]["total_updates"]
            != (schedule.WARMUP if v["metadata"]["arm"] == "warmup" else schedule.FINAL)
            for v in evaluations
        )
    ):
        raise ValueError("refine metrics require complete immutable GPU matrix")
    indexed = {(v["metadata"]["arm"], v["metadata"]["seed"]): v for v in evaluations}
    results = []
    for seed in study.SEEDS:
        warmup = indexed[("warmup", seed)]
        control = indexed[("paired", seed)]
        for arm in ("warmup", *schedule.ARMS):
            value = indexed[(arm, seed)]
            scores = value["development"]["scores"]
            cells = statistics(
                manifest["samples"]["development"],
                scores,
                baseline=warmup["development"]["scores"] if arm != "warmup" else None,
            )
            result = {
                "arm": arm,
                "seed": seed,
                "cells": cells,
                "checkpoint_sha256": value["checkpoint_sha256"],
                "model_sha256": value["model_sha256_before"],
            }
            if arm != "warmup":
                result["fit_cells"] = statistics(
                    manifest["samples"]["fit"], value["fit"]["scores"], role="fit"
                )
                result["fit_development_auc_gap"] = [
                    {
                        "source_group": fit["source_group"],
                        "auc_gap": fit["metrics"]["roc_auc"] - dev["metrics"]["roc_auc"],
                    }
                    for fit, dev in zip(result["fit_cells"], cells, strict=True)
                ]
            if arm == "unpaired":
                result["against_paired"] = statistics(
                    manifest["samples"]["development"],
                    scores,
                    baseline=control["development"]["scores"],
                )
            results.append(result)
    passing = [
        arm
        for arm in schedule.ARMS
        if all(
            len(r["cells"]) == len(data.ORIGINS) and all(c["learning_gate"] for c in r["cells"])
            for r in results
            if r["arm"] == arm
        )
    ]
    minima = {
        arm: min(c["metrics"]["roc_auc"] for r in results if r["arm"] == arm for c in r["cells"])
        for arm in passing
    }
    selected = max(passing, key=lambda a: (minima[a], a == "paired")) if passing else None
    superior = all(
        c["paired_auc_delta_ci95"][0] > 0
        for r in results
        if r["arm"] == "unpaired"
        for c in r["against_paired"]
    )
    return {
        "trials": results,
        "passing_arms": passing,
        "selected_arm": selected,
        "learning_gate": bool(passing),
        "unpaired_superiority": superior,
        "accuracy_qualification": "unavailable",
        "deployed": False,
    }
