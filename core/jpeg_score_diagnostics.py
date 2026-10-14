"""Bounded descriptive paired-score audit, never standalone accuracy or calibration."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

from core.jpeg_scale import identity

ORIGINS = ("BOSSbase-1.01", "BOWS2")
MAX_ROWS = 1024


def paired_scores(samples, scores):
    """Compare only already-scored original-disjoint development pairs.

    Paired ranking requires a known cover. Midpoint dispersion is descriptive,
    not proof of a causal content/BatchNorm problem. No threshold is selected,
    no score is adjusted, and this evidence cannot qualify a deployed detector.
    """
    if (not isinstance(samples, list) or not isinstance(scores, list)
        or not 8 <= len(samples) <= MAX_ROWS or len(samples) % 4
        or len(scores) != len(samples)):
        raise ValueError("paired score audit row/type bounds")
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
           for v in scores):
        raise ValueError("paired score audit requires finite probabilities")
    seen = set()
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for index in range(0, len(samples), 2):
        cover, stego = samples[index:index + 2]
        if not isinstance(cover, dict) or not isinstance(stego, dict):
            raise ValueError("paired score audit requires object rows")
        lineage = identity(cover.get("lineage"))
        origin = cover.get("source_group")
        if (cover.get("label") != "cover" or stego.get("label") != "stego"
            or cover.get("role") != "development" or stego.get("role") != "development"
            or origin not in ORIGINS or stego.get("source_group") != origin
            or stego.get("lineage") != lineage or lineage in seen):
            raise ValueError("paired score audit ancestry/order/role mismatch")
        seen.add(lineage)
        grouped[origin].append((scores[index], scores[index + 1]))
    if set(grouped) != set(ORIGINS) or len(grouped[ORIGINS[0]]) != len(grouped[ORIGINS[1]]):
        raise ValueError("paired score audit requires both balanced source groups")
    cells = []
    for origin in ORIGINS:
        pairs = grouped[origin]
        deltas = [s - c for c, s in pairs]
        midpoints = [(c + s) / 2 for c, s in pairs]
        mean_absolute = statistics.fmean(abs(d) for d in deltas)
        dispersion = statistics.pstdev(midpoints)
        ratio = dispersion / mean_absolute if mean_absolute else None
        if ratio is not None and not math.isfinite(ratio):
            ratio = None
        cells.append({
            "source_group": origin, "original_pairs": len(pairs),
            "higher_stego_pairs": sum(d > 0 for d in deltas),
            "tied_pairs": sum(d == 0 for d in deltas),
            "lower_stego_pairs": sum(d < 0 for d in deltas),
            "positive_pair_fraction": sum(d > 0 for d in deltas) / len(pairs),
            "mean_cover_probability": statistics.fmean(c for c, _ in pairs),
            "mean_stego_probability": statistics.fmean(s for _, s in pairs),
            "mean_signed_pair_delta": statistics.fmean(deltas),
            "median_signed_pair_delta": statistics.median(deltas),
            "mean_absolute_pair_delta": mean_absolute,
            "pair_midpoint_population_std": dispersion,
            "midpoint_std_over_mean_absolute_delta": ratio,
        })
    return {"schema_version": "jpeg-paired-score-diagnostics-v1", "cells": cells,
            "requires_known_cover": True, "standalone_accuracy_computed": False,
            "threshold_selected": False, "scores_adjusted": False,
            "accuracy_qualification": "unavailable", "deployed": False}
