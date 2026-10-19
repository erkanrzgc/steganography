"""Bounded train-only grouped linear research, independent of production models."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np

EPOCHS = 20
SEEDS = (20261016, 20261017)
LEARNING_RATE = 0.01
L2 = 1e-4
BATCH = 64
SCALE_FLOOR = 1e-6


def row_identity(rows):
    keys = ("sha256", "lineage", "source_group", "split", "label", "format", "method",
            "rate_percent", "channel", "plane")
    return hashlib.sha256(json.dumps(
        [[r.get(k) for k in keys] for r in rows], separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def group_key(row):
    source = row["source_group"]
    key = row.get("speaker") if source == "FSDD-v1.0.10" else (
        row.get("group_key") if source == "ESC-50" else row.get("lineage"))
    if not isinstance(key, str) or not key or len(key) > 256:
        raise ValueError("missing bounded independent group")
    return key


def training_weights(rows):
    if not 2 <= len(rows) <= 4096:
        raise ValueError("linear row limit")
    counts = Counter((r["lineage"], r["label"]) for r in rows)
    lineages = {r["lineage"] for r in rows}
    if any(r["label"] not in {"cover", "stego"} for r in rows) or any(
        not counts[(key, label)] for key in lineages for label in ("cover", "stego")
    ):
        raise ValueError("complete original class families required")
    return np.array([1 / (2 * len(lineages) * counts[(r["lineage"], r["label"])])
                     for r in rows], dtype=np.float64)


def validate(x, y, rows):
    if (
        not isinstance(x, np.ndarray) or x.ndim != 2 or not 1 <= x.shape[1] <= 4096
        or x.shape[0] != len(rows) or not isinstance(y, np.ndarray) or y.shape != (len(rows),)
        or not np.isfinite(x).all() or np.any((x < 0) | (x > 1))
        or not np.array_equal(y, [r["label"] == "stego" for r in rows])
    ):
        raise ValueError("linear feature/label contract mismatch")
    return training_weights(rows)


def group_folds(rows):
    if {r["split"] for r in rows} != {"train"}:
        raise ValueError("cross-validation must be train-only")
    groups = [group_key(r) for r in rows]
    ordered = sorted(set(groups), key=lambda g: hashlib.sha256(
        ("nonjpeg-group-folds-20261018:" + g).encode()).digest())
    if not 2 <= len(ordered) <= 32:
        raise ValueError("insufficient or excessive independent training groups")
    assignment = {key: index % min(4, len(ordered)) for index, key in enumerate(ordered)}
    lineage_groups: dict[str, str] = {}
    for row, group in zip(rows, groups, strict=True):
        if lineage_groups.setdefault(row["lineage"], group) != group:
            raise ValueError("one lineage cannot cross independent groups")
    return np.asarray([assignment[g] for g in groups], dtype=np.int64)


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -40, 40)))


def fit(x, y, rows, seed, *, epochs=EPOCHS, deadline=lambda: None, on_update=lambda: None):
    weights = validate(x, y, rows)
    if seed not in SEEDS or type(epochs) is not int or not 1 <= epochs <= EPOCHS:
        raise ValueError("linear initialization/exposure contract mismatch")
    x = x.astype(np.float64)
    mean = weights @ x
    scale = np.maximum(np.sqrt(weights @ ((x - mean) ** 2)), SCALE_FLOOR)
    z = (x - mean) / scale
    # Separate streams ensure feature dimension cannot change presentation order.
    parameters = np.random.default_rng(seed).normal(0, 0.01 / np.sqrt(x.shape[1]), x.shape[1])
    order = np.random.default_rng(seed ^ 0x4E4A5047)
    bias = 0.0
    exposure = hashlib.sha256(bytes.fromhex(row_identity(rows)))
    losses = []
    updates = 0
    for _ in range(epochs):
        deadline()
        shuffled = order.permutation(len(rows))
        exposure.update(shuffled.astype("<i4").tobytes())
        for start in range(0, len(rows), BATCH):
            deadline()
            indices = shuffled[start : start + BATCH]
            probabilities = sigmoid(z[indices] @ parameters + bias)
            error = weights[indices] * (probabilities - y[indices]) * len(rows) / len(indices)
            gradient = z[indices].T @ error + L2 * parameters
            parameters -= LEARNING_RATE * gradient
            bias -= LEARNING_RATE * float(error.sum())
            updates += 1
            on_update()
            if not np.isfinite(parameters).all() or not np.isfinite(bias):
                raise ValueError("linear nonfinite state")
        logits = z @ parameters + bias
        losses.append(float(weights @ (np.logaddexp(0, logits) - y * logits)))
    return {
        "mean": mean, "scale": scale, "weights": parameters, "bias": bias,
        "epochs": epochs, "optimizer_updates": updates, "losses": losses,
        "exposure_sha256": exposure.hexdigest(), "row_presentations": epochs * len(rows),
        "original_presentations": epochs * len({r["lineage"] for r in rows}),
    }


def predict(model, x):
    d = len(model["weights"])
    if (
        not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != d
        or not 1 <= len(x) <= 4096 or not np.isfinite(x).all() or np.any((x < 0) | (x > 1))
        or any(np.shape(model[k]) != (d,) or not np.isfinite(model[k]).all()
               for k in ("mean", "scale", "weights"))
        or np.any(model["scale"] < SCALE_FLOOR) or not np.isfinite(model["bias"])
    ):
        raise ValueError("linear inference contract mismatch")
    return sigmoid(((x.astype(np.float64) - model["mean"]) / model["scale"])
                   @ model["weights"] + model["bias"])


def select_threshold(scores, y, rows):
    weights = training_weights(rows)
    if {r["split"] for r in rows} != {"train"} or scores.shape != y.shape or (
        not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("threshold selection requires complete train-only OOF scores")
    grid = np.arange(101) / 100
    accuracy = np.array([weights @ ((scores >= t) == y) for t in grid])
    # Fixed higher-threshold tie rule, without inspecting validation values.
    return float(grid[np.flatnonzero(np.round(accuracy, 12) == np.round(accuracy.max(), 12))[-1]])


def grouped_fit(x, y, rows, seed, *, deadline=lambda: None, on_update=lambda: None):
    validate(x, y, rows)
    folds = group_folds(rows)
    oof = np.full(len(rows), np.nan)
    receipts = []
    for fold in sorted(set(folds.tolist())):
        train, held = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        model = fit(x[train], y[train], [rows[i] for i in train], seed,
                    deadline=deadline, on_update=on_update)
        oof[held] = predict(model, x[held])
        receipts.append({"fold": fold, "training_rows": len(train), "held_rows": len(held),
                         "exposure_sha256": model["exposure_sha256"],
                         "optimizer_updates": model["optimizer_updates"],
                         "row_presentations": model["row_presentations"]})
    threshold = select_threshold(oof, y, rows)
    model = fit(x, y, rows, seed, deadline=deadline, on_update=on_update)
    return model, {"threshold": threshold, "folds": receipts,
                   "oof_sha256": hashlib.sha256(oof.astype("<f8").tobytes()).hexdigest(),
                   "oof_scores": oof, "fold_assignments": folds,
                   "all_optimizer_updates": model["optimizer_updates"] + sum(
                       r["optimizer_updates"] for r in receipts)}
