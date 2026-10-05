"""Opt-in inference-only precision derivation and development export audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np

from core.feature_model import feature_model
from steganography.research import ResearchManifestError, export_onnx
from steganography.research_features import (
    MODEL_DOMAINS,
    feature_inputs,
    read_document,
    read_feature_checkpoint,
)
from steganography.research_jpeg import write_json


def derive_checkpoint(source: Path, out: Path, *, source_sha256: str):
    import torch

    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("derived checkpoint output exists or uses a symlink")
    checkpoint = copy.deepcopy(read_feature_checkpoint(source, expected_sha256=source_sha256))
    feature_model(checkpoint)
    checkpoint["preprocessing"]["inference_arithmetic"] = "float64"
    checkpoint["inference_derivation"] = {
        "source_checkpoint_sha256": source_sha256,
        "method": "float64-normalization-and-accumulation-float32-io-v1",
        "retrained": False,
    }
    feature_model(checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        torch.save(checkpoint, stream)
    return checkpoint["inference_derivation"]


def audit_export(config_path: Path, source: Path, out: Path, *, source_sha256: str):
    import onnxruntime as ort
    import torch

    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ResearchManifestError("symlink audit output forbidden")
    config, _ = read_document(config_path)
    x, _, provenance = feature_inputs(config, split="validation")
    original = read_feature_checkpoint(source, expected_sha256=source_sha256)
    if (
        original["domain"] != MODEL_DOMAINS[provenance["feature_version"]]
        or original["preprocessing"]["feature_version"] != provenance["feature_version"]
        or original["preprocessing"]["feature_names"] != provenance["feature_names"]
        or original["training_provenance"]["manifest_sha256"] != provenance["manifest_sha256"]
    ):
        raise ResearchManifestError("checkpoint/validation contract mismatch")
    out.mkdir(parents=True, exist_ok=False)
    derived = out / "precise.pt"
    derive_checkpoint(source, derived, source_sha256=source_sha256)
    card = export_onnx(derived, out / "precise.onnx")
    precise = read_feature_checkpoint(derived)
    if any(
        not torch.equal(value, precise["state_dict"][key])
        for key, value in original["state_dict"].items()
    ):
        raise ResearchManifestError("inference derivation altered weights")
    reference = feature_model(original).eval()
    model = feature_model(precise).eval()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(out / "precise.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    batches = {}
    for size in sorted({1, min(17, len(x)), len(x)}):
        max_logit = max_probability = 0.0
        decisions_equal = True
        for start in range(0, len(x), size):
            batch = x[start : start + size]
            with torch.no_grad():
                expected = model(torch.from_numpy(batch)).numpy()
            actual = session.run(None, {input_name: batch})[0]
            if not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise ResearchManifestError("nonfinite inference audit")
            max_logit = max(max_logit, float(np.max(np.abs(actual - expected))))
            p = torch.sigmoid(torch.from_numpy(expected)).numpy()
            q = torch.sigmoid(torch.from_numpy(actual)).numpy()
            max_probability = max(max_probability, float(np.max(np.abs(p - q))))
            decisions_equal &= bool(np.array_equal(expected >= 0, actual >= 0))
        batches[str(size)] = {
            "max_logit_difference": max_logit,
            "max_probability_difference": max_probability,
            "threshold_decisions_equal": decisions_equal,
            "passed": max_logit <= 1e-6 and max_probability <= 1e-6 and decisions_equal,
        }
    with torch.no_grad():
        old = torch.sigmoid(reference(torch.from_numpy(x))).numpy()
        new = torch.sigmoid(model(torch.from_numpy(x))).numpy()
    drift = float(np.max(np.abs(new - old)))
    decisions_equal = bool(np.array_equal(new >= 0.5, old >= 0.5))
    report = {
        "schema_version": "inference-precision-audit-v1",
        "split": "validation",
        "samples": len(x),
        "source_checkpoint_sha256": source_sha256,
        "derived_checkpoint_sha256": hashlib.sha256(derived.read_bytes()).hexdigest(),
        "onnx_sha256": card["onnx_sha256"],
        "provenance": provenance,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 0,
        "batches": batches,
        "original_probability_drift": drift,
        "original_threshold_decisions_equal": decisions_equal,
        "passed": all(b["passed"] for b in batches.values()) and drift <= 1e-6 and decisions_equal,
        "deployed": False,
        "retrained": False,
        "cross_source_gate": "unavailable",
    }
    write_json(out / "audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    result = audit_export(args.config, args.checkpoint, args.out, source_sha256=args.source_sha256)
    print(f"precision audit passed={result['passed']}; deployed=False")


if __name__ == "__main__":
    main()
