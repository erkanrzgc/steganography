"""Independent scalar metric replay of the fixed inspected-probe learning arms."""

from __future__ import annotations

import hashlib
import json
import math

from core.jpeg_timing_probe import AUDIT_SHA, MANIFEST_SHA
from core.srnet_stream import regular_open


def read(path, checksum):
    with regular_open(path) as stream:
        raw = stream.read(2 * 1024**2 + 1)
    if len(raw) > 2 * 1024**2 or hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("normalization audit input checksum/bound mismatch")
    return json.loads(raw)


def replay(report, samples):
    if (
        report.get("schema_version") != "jpeg-bn-gn-learning-pilot-v1"
        or report.get("status") != "completed"
        or report.get("manifest_sha256") != MANIFEST_SHA
        or report.get("audit_sha256") != AUDIT_SHA
        or report.get("epochs") != 8
        or report.get("optimizer_updates") != 1152
        or report.get("accuracy_qualification") != "unavailable"
        or report.get("deployed") is not False
        or report.get("validation_test_used") is not False
        or report.get("probe_previously_inspected") is not True
    ):
        raise ValueError("normalization audit requires complete fixed nondeployed arm")
    originals: dict[str, set[str]] = {}
    if not isinstance(samples, list) or len(samples) != 480:
        raise ValueError("normalization audit requires complete metadata")
    for sample in samples:
        if sample.get("split") != "train":
            raise ValueError("normalization audit forbids non-training metadata")
        originals.setdefault(sample["source_group"], set()).add(sample["lineage"])
    if len(samples) != 480 or len(originals) != 3 or any(len(v) != 32 for v in originals.values()):
        raise ValueError("normalization audit original accounting mismatch")
    probes = {s: set(sorted(v)[24:]) for s, v in originals.items()}
    expected = {i for i, s in enumerate(samples) if s["lineage"] in probes[s["source_group"]]}
    outputs = report.get("probe_logits")
    if not isinstance(outputs, list) or len(outputs) != len(expected) or len(expected) != 120:
        raise ValueError("normalization audit complete probe required")
    logits = {}
    for output in outputs:
        index, values = output.get("row"), output.get("logits")
        if (
            type(index) is not int
            or index not in expected
            or index in logits
            or not isinstance(values, list)
            or len(values) != 2
            or any(
                type(v) not in (int, float)
                or not math.isfinite(v)
                or abs(v) > 3.4028234663852886e38
                for v in values
            )
        ):
            raise ValueError("normalization audit invalid singleton output")
        logits[index] = values
    cells = report.get("probe_cells")
    if not isinstance(cells, list) or len(cells) != 10:
        raise ValueError("normalization audit complete method cells required")
    seen, comparisons = set(), []
    for cell in cells:
        key = (cell["source_group"], cell["quality_factor"], cell["method"])
        if key in seen or key[2] not in {"JUNIWARD", "UERD"}:
            raise ValueError("normalization audit repeated/unknown cell")
        seen.add(key)
        counts = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
        losses = []
        for index, values in logits.items():
            sample = samples[index]
            if (sample["source_group"], sample["quality_factor"]) != key[:2] or (
                sample["method"] not in (None, key[2])
            ):
                continue
            positive, decision = sample["label"] == "stego", values[1] >= values[0]
            name = ("true_" if decision == positive else "false_") + (
                "positive" if decision else "negative"
            )
            counts[name] += 1
            margin = values[1] - values[0]
            loss = math.log1p(math.exp(-abs(margin))) + max(-margin if positive else margin, 0)
            losses.append(loss)
        if counts["true_positive"] + counts["false_negative"] != 8 or (
            counts["true_negative"] + counts["false_positive"] != 8
        ):
            raise ValueError("normalization audit cell support mismatch")
        expected_metrics = {
            **counts,
            "cover_rows": 8,
            "stego_rows": 8,
            "balanced_accuracy": (counts["true_positive"] + counts["true_negative"]) / 16,
            "recall": counts["true_positive"] / 8,
            "false_positive_rate": counts["false_positive"] / 8,
            "cross_entropy": math.fsum(losses) / 16,
        }
        if any(
            type(cell.get(k)) not in (int, float)
            or not math.isfinite(cell[k])
            or not math.isclose(cell[k], v, abs_tol=1e-12, rel_tol=1e-12)
            for k, v in expected_metrics.items()
        ):
            raise ValueError("normalization scalar metric replay mismatch")
        comparisons.append(
            {
                "source_group": key[0],
                "quality_factor": key[1],
                "method": key[2],
                "counts_exact": True,
                "scalar_metrics_passed": True,
            }
        )
    return comparisons


def audit(bn, bn_sha, gn, gn_sha, manifest):
    samples = read(manifest, MANIFEST_SHA)["samples"]
    baseline, variant = read(bn, bn_sha), read(gn, gn_sha)
    if baseline.get("arm") != "bn" or variant.get("arm") != "gn":
        raise ValueError("normalization audit requires both distinct arms")
    for key in (
        "source_sha256",
        "initial_parameter_sha256",
        "schedule_sha256",
        "schedules",
        "seed",
        "epochs",
        "optimizer_updates",
        "optimizer",
    ):
        if baseline.get(key) is None or baseline[key] != variant.get(key):
            raise ValueError("normalization learning arms not matched")
    return {
        "schema_version": "jpeg-bn-gn-scalar-audit-v1",
        "status": "completed",
        "bn_report_sha256": bn_sha,
        "gn_report_sha256": gn_sha,
        "manifest_sha256": MANIFEST_SHA,
        "audit_sha256": AUDIT_SHA,
        "matched_initialization_schedule_optimizer": True,
        "bn_cells": replay(baseline, samples),
        "gn_cells": replay(variant, samples),
        "arithmetic": "independent Python math stable logistic loss and confusion counts",
        "independent_whole_model_oracle": False,
        "accuracy_qualification": "unavailable",
        "probe_previously_inspected": True,
        "deployed": False,
    }
