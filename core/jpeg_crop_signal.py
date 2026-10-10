"""Read-only train-crop signal accounting; no fitting or accuracy assertion."""

from __future__ import annotations

import numpy as np

from core.jpeg_timing_probe import AUDIT_SHA, MANIFEST_SHA, TimingReader
from core.srnet_stream import deadline_after


def summarize(samples, fetch):
    if len(samples) != 480 or any(r["split"] != "train" for r in samples):
        raise ValueError("crop signal requires the complete training kit")
    groups: dict[str, set[str]] = {}
    for row in samples:
        groups.setdefault(row["source_group"], set()).add(row["lineage"])
    if len(groups) != 3 or any(len(v) != 32 for v in groups.values()):
        raise ValueError("crop signal original accounting mismatch")
    fit = {s: set(sorted(v)[:24]) for s, v in groups.items()}
    covers = {
        (r["source_group"], r["lineage"], r["quality_factor"]): i
        for i, r in enumerate(samples)
        if r["label"] == "cover"
    }
    cells: dict[tuple[str, int | None, str], list[tuple[int, float]]] = {}
    for index, row in enumerate(samples):
        if row["label"] != "stego" or row["lineage"] not in fit[row["source_group"]]:
            continue
        key = (row["source_group"], row["quality_factor"], row["method"])
        cover = covers[(row["source_group"], row["lineage"], row["quality_factor"])]
        batch = fetch(np.array([cover, index, cover, index], dtype=np.int64))
        if (
            batch.shape != (4, 1, 256, 256)
            or batch.dtype != np.dtype("<f4")
            or (not np.isfinite(batch).all() or np.any(np.abs(batch) > 2**36))
        ):
            raise ValueError("crop signal invalid decoded batch")
        delta = batch[1].astype(np.float64) - batch[0].astype(np.float64)
        values = (int(np.count_nonzero(delta)), float(np.sqrt(np.mean(delta * delta))))
        cells.setdefault(key, []).append(values)
    if len(cells) != 10 or any(len(v) != 24 for v in cells.values()):
        raise ValueError("crop signal complete fit pair accounting required")
    return [
        {
            "source_group": key[0],
            "quality_factor": key[1],
            "method": key[2],
            "fit_pairs": len(values),
            "identical_crops": sum(count == 0 for count, _ in values),
            "minimum_rms_pixel_difference": min(rms for _, rms in values),
            "maximum_rms_pixel_difference": max(rms for _, rms in values),
        }
        for key, values in sorted(cells.items(), key=lambda item: str(item[0]))
    ]


def inspect_fit_signal(root, audit):
    with TimingReader(root, audit, deadline=deadline_after(120)) as reader:
        cells = summarize(reader.samples, reader.batch)
        reader.verify()
    identical = sum(c["identical_crops"] for c in cells)
    return {
        "schema_version": "jpeg-train-crop-signal-v1",
        "status": "completed",
        "manifest_sha256": MANIFEST_SHA,
        "audit_sha256": AUDIT_SHA,
        "fit_pairs": 240,
        "identical_crops": identical,
        "crop_signal_coverage": "partial" if identical else "complete",
        "cells": cells,
        "probe_pixels_read": 0,
        "validation_test_used": False,
        "optimizer_updates": 0,
        "accuracy_qualification": "unavailable",
        "deployed": False,
    }
