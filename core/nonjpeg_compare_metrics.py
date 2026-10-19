"""Matched cover/stego cells with original/speaker/recording cluster intervals."""

from __future__ import annotations

import numpy as np

from core.nonjpeg_linear import group_key

CELL_KEYS = ("format", "method", "rate_percent", "channel", "plane")
METRICS = ("roc_auc", "balanced_accuracy", "recall", "false_positive_rate", "ece")
DRAWS = 2000


def pair_cells(rows, scores):
    if scores.shape != (len(rows),) or not np.isfinite(scores).all() or np.any(
        (scores < 0) | (scores > 1)
    ):
        raise ValueError("invalid comparison scores")
    covers = {}
    cells: dict[tuple, list[int]] = {}
    for i, row in enumerate(rows):
        if row["label"] == "cover":
            key = (row["lineage"], row["format"])
            if key in covers:
                raise ValueError("duplicate original/format cover")
            covers[key] = i
        else:
            key = tuple(row[k] for k in CELL_KEYS)
            cells.setdefault(key, []).append(i)
    result = []
    for cell, indices in sorted(cells.items()):
        seen = set()
        pairs, groups = [], []
        for index in indices:
            row = rows[index]
            key = (row["lineage"], row["format"])
            if row["lineage"] in seen or key not in covers:
                raise ValueError("incomplete or duplicate matched cell")
            cover = rows[covers[key]]
            if group_key(cover) != group_key(row):
                raise ValueError("cover/stego independent group mismatch")
            seen.add(row["lineage"])
            pairs.append([scores[covers[key]], scores[index]])
            groups.append(group_key(row))
        if len(seen) != len({r["lineage"] for r in rows}):
            raise ValueError("cell lacks original families")
        result.append((dict(zip(CELL_KEYS, cell, strict=True)), np.asarray(pairs), groups))
    if not result:
        raise ValueError("no comparison cells")
    return result


def measured(pairs, weights, threshold):
    """Vectorized cluster draws; ties count as half a win, thresholds use >=."""
    negative, positive = pairs[:, 0], pairs[:, 1]
    total = weights.sum(axis=1)
    wins = (positive[:, None] > negative).astype(float)
    wins += (positive[:, None] == negative) * 0.5
    auc = np.einsum("bi,ij,bj->b", weights, wins, weights) / total**2
    recall = (weights @ (positive >= threshold)) / total
    fpr = (weights @ (negative >= threshold)) / total
    for side, values in enumerate((negative, positive)):
        if side == 0:
            # Bin errors from both labels must cancel within each shared bin.
            errors = np.zeros((len(pairs), 10))
        bins = np.minimum((values * 10).astype(int), 9)
        errors[np.arange(len(pairs)), bins] += values - side
    ece = np.abs(weights @ errors).sum(axis=1) / (2 * total)
    return {"roc_auc": auc, "balanced_accuracy": (recall + 1 - fpr) / 2,
            "recall": recall, "false_positive_rate": fpr, "ece": ece}


def bootstrap_weights(groups):
    unique = sorted(set(groups))
    if not 2 <= len(unique) <= 32:
        raise ValueError("at least two bounded independent validation groups required")
    rng = np.random.default_rng(20261018)
    draw = rng.integers(len(unique), size=(DRAWS, len(unique)))
    counts = np.stack([(draw == i).sum(axis=1) for i in range(len(unique))], axis=1)
    return counts[:, [unique.index(g) for g in groups]].astype(np.float64)


def cell_statistics(pairs, groups, threshold, *, baseline=None, baseline_threshold=0.7):
    if (
        not isinstance(pairs, np.ndarray) or pairs.ndim != 2 or pairs.shape[1] != 2
        or not 2 <= len(pairs) <= 32 or len(groups) != len(pairs)
        or not np.isfinite(pairs).all() or np.any((pairs < 0) | (pairs > 1))
        or not np.isfinite(threshold) or not 0 <= threshold <= 1
    ):
        raise ValueError("invalid bounded matched metrics")
    draws = bootstrap_weights(groups)
    point = measured(pairs, np.ones((1, len(pairs))), threshold)
    intervals = measured(pairs, draws, threshold)
    result = {
        "pairs": len(pairs), "independent_groups": len(set(groups)),
        "uncertainty": "underpowered development clusters", "bootstrap_draws": DRAWS,
        "metrics": {k: float(v[0]) for k, v in point.items()},
        "metric_ci95": {k: np.quantile(v, [0.025, 0.975]).tolist()
                        for k, v in intervals.items()},
        "confusion": {"tp": int(np.sum(pairs[:, 1] >= threshold)),
                      "tn": int(np.sum(pairs[:, 0] < threshold)),
                      "fp": int(np.sum(pairs[:, 0] >= threshold)),
                      "fn": int(np.sum(pairs[:, 1] < threshold))},
        "threshold": threshold,
    }
    if baseline is not None:
        if baseline.shape != pairs.shape or not np.isfinite(baseline).all() or np.any(
            (baseline < 0) | (baseline > 1)
        ) or not np.isfinite(baseline_threshold) or not 0 <= baseline_threshold <= 1:
            raise ValueError("invalid matched reference")
        prior = measured(baseline, np.ones((1, len(pairs))), baseline_threshold)
        sampled = measured(baseline, draws, baseline_threshold)
        result["reference_metrics"] = {k: float(v[0]) for k, v in prior.items()}
        result["paired_delta"] = {k: float(point[k][0] - prior[k][0]) for k in METRICS}
        result["paired_delta_ci95"] = {
            k: np.quantile(intervals[k] - sampled[k], [0.025, 0.975]).tolist() for k in METRICS
        }
        result["reference_threshold"] = baseline_threshold
    return result
