#!/usr/bin/env python3
"""Portable aggregation of completed train-only BN probes; never inference."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from steganography.research_features import read_document  # noqa: E402
from steganography.research_jpeg import write_json  # noqa: E402

PROTOCOL_SHA = "ba26860c67e711073c1f7c23ffa453d2ded4f589475000e62059e4653892912f"


def aggregate(path: Path, out: Path, *, checksum: str):
    report, digest = read_document(path)
    if digest != checksum:
        raise ValueError("diagnostic checksum mismatch")
    protocol = ROOT / "docs/SRNET_TRAIN_DIAGNOSTIC_PROTOCOL.md"
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise ValueError("frozen diagnostic protocol changed")
    if (
        report.get("schema_version") != "srnet-train-diagnostic-v1"
        or report.get("status") != "completed"
        or report.get("state_unchanged") is not True
        or report.get("validation_pixels_loaded") is not False
        or report.get("deployed") is not False
        or report.get("model_sha256")
        != "b48faf464c9116fa1778d9cc411e03d154f54f3d31c2fc260cce79424b6acd51"
        or report.get("probe_pairs") != 32
        or not isinstance(report.get("probes"), list)
        or len(report["probes"]) != 32
    ):
        raise ValueError("frozen diagnostic probe contract mismatch")
    probes = report["probes"]
    anchors = {
        "manifest_sha256": "0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14",
        "train_cache_sha256": "828158342b8937844e76f8f879bb41177a32e7ebcb85eef1eef59f2e67692028",
        "training_plan_sha256": "27634b6576a1428669d3d8c7c490544f86a59a3449c5b7e304fbbd2dac38bdda",
        "model_card_sha256": "1e4db84ee3b905c73f04bd7de19691ef34c491af38d5d52929c870fc510dc99f",
    }
    if (
        any(report.get(k) != v for k, v in anchors.items())
        or len({p["stego_sha256"] for p in probes}) != 32
    ):
        raise ValueError("frozen diagnostic provenance/unique selection mismatch")
    result = {
        k: report[k]
        for k in (
            "manifest_sha256",
            "train_cache_sha256",
            "train_data_sha256",
            "training_plan_sha256",
            "model_card_sha256",
            "model_sha256",
            "training_scope",
            "seconds",
        )
    }
    result.update(
        schema_version="srnet-train-diagnostic-evidence-v1",
        protocol_sha256=PROTOCOL_SHA,
        protocol_commit="2cbca3a",
        probes_sha256=digest,
        probe_pairs=32,
        state_unchanged=True,
        validation_pixels_loaded=False,
        calibrated=False,
        deployed=False,
        qualification="unavailable",
        cells=[],
        layers=[],
    )
    result["probe_core_sha256"] = hashlib.sha256(
        (ROOT / "core/srnet_diagnostics.py").read_bytes()
    ).hexdigest()
    result["probe_service_sha256"] = hashlib.sha256(
        (ROOT / "steganography/research_srnet_diagnose.py").read_bytes()
    ).hexdigest()
    result["interpretation"] = (
        "post-hoc in-sample diagnostic; batch-dependent contrast is not primary inference"
    )
    keys = sorted({(p["source_id"], p["quality_factor"], p["method"]) for p in probes}, key=str)
    if len(keys) != 4 or any(p.get("state_unchanged") is not True for p in probes):
        raise ValueError("frozen diagnostic cell/state mismatch")
    for source, quality, method in keys:
        selected = [
            p
            for p in probes
            if (p["source_id"], p["quality_factor"], p["method"]) == (source, quality, method)
        ]
        if (
            source != "source-6a23b26e6f51f8cc"
            or quality not in (75, 95)
            or method not in ("JUNIWARD", "UERD")
            or len(selected) != 8
        ):
            raise ValueError("frozen diagnostic cell coverage mismatch")
        cell = {
            "source_id": source,
            "quality_factor": quality,
            "method": method,
            "pairs": len(selected),
            "mean_input_difference_rms": float(
                np.mean([p["input_difference_rms"] for p in selected])
            ),
        }
        for mode in ("native", "batch_statistics"):
            scores = np.asarray([p[mode]["scores"] for p in selected], dtype=np.float64)
            losses = np.asarray(
                [p[mode]["paired_cross_entropy"] for p in selected], dtype=np.float64
            )
            if (
                scores.shape != (8, 2)
                or losses.shape != (8,)
                or not np.isfinite(scores).all()
                or not np.isfinite(losses).all()
                or np.any(scores < 0)
                or np.any(scores > 1)
                or np.any(losses < 0)
            ):
                raise ValueError("diagnostic nonfinite scores/loss")
            decisions = np.asarray([p[mode]["stego_decisions"] for p in selected])
            if decisions.dtype != np.dtype(bool) or decisions.shape != (8, 2):
                raise ValueError("diagnostic invalid decisions")
            cell[mode] = {
                "mean_paired_cross_entropy": float(losses.mean()),
                "saturated_scores": int(np.sum((scores < 0.001) | (scores > 0.999))),
                "in_sample_label_accuracy": float(np.mean(decisions == [False, True])),
                "pairs_with_different_decisions": int(np.sum(decisions[:, 0] != decisions[:, 1])),
            }
        result["cells"].append(cell)
    names = [layer["layer"] for layer in probes[0]["native"]["layers"]]
    if len(names) != 26 or len(set(names)) != 26:
        raise ValueError("diagnostic incomplete layer coverage")
    for index, name in enumerate(names):
        record = {"layer": name}
        for mode in ("native", "batch_statistics"):
            if any(
                len(p[mode]["layers"]) != 26 or p[mode]["layers"][index]["layer"] != name
                for p in probes
            ):
                raise ValueError("diagnostic layer order mismatch")
            values = np.array(
                [
                    [
                        p[mode]["layers"][index][k]
                        for k in ("median_standardized_mean_shift", "median_variance_ratio")
                    ]
                    for p in probes
                ]
            )
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("diagnostic nonfinite layer summary")
            record[mode] = {
                "median_standardized_mean_shift": float(np.median(values[:, 0])),
                "median_variance_ratio": float(np.median(values[:, 1])),
            }
        result["layers"].append(record)
    write_json(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.probes, args.out, checksum=args.sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
