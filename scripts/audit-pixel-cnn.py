#!/usr/bin/env python3
"""Replay pixel-CNN fits, independent NumPy forward, ONNX and paired metrics."""

from __future__ import annotations

import argparse
import importlib.metadata
import runpy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import jpeg_cnn as cnn  # noqa: E402
from core.jpeg_cnn_model import SHAPES, load_model  # noqa: E402
from steganography.research_features import read_document, selected_samples  # noqa: E402
from steganography.research_jpeg import write_json  # noqa: E402
from steganography.research_jrm import training_scope  # noqa: E402
from steganography.research_pixels import load_pixels  # noqa: E402
from steganography.research_spatial import cell_metrics, paired_intervals  # noqa: E402

H = runpy.run_path(str(ROOT / "scripts/audit-jrm-reference.py"))
sha, check, KEYS = H["sha"], H["check"], H["KEYS"]
EXPECTED = {
    "epochs": 10,
    "batch_size": 32,
    "threads": 2,
    "seed": 20261011,
    "max_seconds": 1800,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
}


def conv(x, w, b=None, padding=0):
    if padding:
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    windows = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(2, 3))
    y = np.einsum("nchwuv,ocuv->nohw", windows, w, optimize=True)
    return y if b is None else y + b[None, :, None, None]


def numpy_logits(raw, weights, architecture=cnn.ARCHITECTURE):
    _, clip = cnn.residual_contract(architecture)
    values = (raw.astype(np.float32) / 255).astype(np.float64)
    values = np.clip(conv(values, weights["filters"]), -clip, clip)
    for index in (0, 3, 6):
        values = np.maximum(
            0, conv(values, weights[f"layers.{index}.weight"], weights[f"layers.{index}.bias"], 1)
        )
        if index != 6:
            n, c, h, w = values.shape
            values = (
                values[:, :, : h // 2 * 2, : w // 2 * 2]
                .reshape(n, c, h // 2, 2, w // 2, 2)
                .mean(axis=(3, 5))
            )
    return values.mean(axis=(2, 3)) @ weights["layers.10.weight"].T + weights["layers.10.bias"]


def probability(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -709, 709)))


def comparison(actual, saved):
    difference = float(np.max(np.abs(actual - saved)))
    decisions = bool(np.array_equal(actual >= 0.5, saved >= 0.5))
    return {
        "passed": difference <= 1e-6 and decisions,
        "maximum_score_difference": difference,
        "decisions_equal": decisions,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 0,
    }


def onnx_audit(model, values, saved, path):
    try:
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError:
        return {
            "status": "unavailable",
            "passed": False,
            "reason": "optional ONNX dependencies absent",
        }
    try:
        check(not path.exists() and not path.is_symlink(), "ONNX output exists")
        torch.onnx.export(
            model,
            torch.zeros((1, 1, 128, 128), dtype=torch.float32),
            str(path),
            input_names=["pixels"],
            output_names=["logit"],
            dynamic_axes={"pixels": {0: "batch"}, "logit": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        graph = onnx.load(str(path))
        graph.doc_string = ""
        for node in graph.graph.node:
            node.doc_string = ""
        onnx.checker.check_model(graph)
        onnx.save(graph, str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        batches = {}
        for batch_size in (1, 17, 64):
            scores = []
            for first in range(0, len(values), batch_size):
                inputs = values[first : first + batch_size].astype(np.float32) / 255
                scores.extend(
                    probability(session.run(None, {"pixels": inputs})[0]).ravel().tolist()
                )
            batches[str(batch_size)] = comparison(np.array(scores), saved)
        return {
            "status": "completed",
            "passed": all(v["passed"] for v in batches.values()),
            "batches": batches,
            "input": "N x 1 x 128 x 128 float32, raw uint8 /255",
            "sha256": sha(path),
        }
    except Exception as exc:
        return {"status": "failed", "passed": False, "reason": type(exc).__name__}


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def audit(
    manifest_path,
    corpus,
    cache_root,
    experiment,
    reference_record,
    exports,
    *,
    architecture=cnn.ARCHITECTURE,
    protocol=None,
    pixel_reference_record=None,
    pixel_reference_experiment=None,
):
    cnn.residual_contract(architecture)
    manifest, digest = read_document(manifest_path)
    check(digest == H["MANIFEST_SHA"], "frozen corpus mismatch")
    baseline, baseline_sha = read_document(reference_record)
    check(
        baseline_sha == "eb7b69e97c8e43d7c46a6deeb29d04b82bfd9972f5cb61bffb1187fa09f8a9cc",
        "reference publication mismatch",
    )
    summary, summary_sha = read_document(experiment / "fit-summary.json")
    pixel_reference = None
    if pixel_reference_record is not None:
        check(pixel_reference_experiment is not None, "missing pixel reference experiment")
        pixel_reference, pixel_reference_hash = read_document(pixel_reference_record)
        check(
            pixel_reference_hash
            == "a10b916f815238e167cc9c166f1d851aea2a98b70a8c14bc6195c46f4e25b5d1",
            "frozen pixel reference publication mismatch",
        )
        check(pixel_reference["manifest_sha256"] == digest, "pixel reference corpus mismatch")
    if protocol is not None:
        check(sha(protocol) == summary["protocol_sha256"], "frozen protocol mismatch")
    check(summary["manifest_sha256"] == digest and not summary["deployed"], "fit summary mismatch")
    training = selected_samples(manifest, "train")
    sources = training_scope(training)[1]["source_ids"]
    check(set(summary["models"]) == {"all", *sources}, "incomplete fits")
    train_values, _, train_descriptor = load_pixels(
        manifest_path,
        cache_root / "train/cache.json",
        checksum="7fa5ab60df47483eedb4e1306dff5a0586b9b8cc2cba431d2d38ec0eb5ad011f",
        split="train",
    )
    del train_values
    values, validation, validation_descriptor = load_pixels(
        manifest_path,
        cache_root / "validation/cache.json",
        checksum="a333104b64139f385f1d1a35a034593917ef5613ba9ed1b29a4f1a21b4b07a21",
        split="validation",
    )
    for sample in manifest["samples"]:
        path = corpus / sample["path"]
        check(
            sha(path) == sample["sha256"] and path.stat().st_size == sample["size"],
            "corpus integrity",
        )
    old, old_hash = read_document(ROOT / ".benchmark/jpeg-jrm-reference-20261005/predictions.json")
    check(old_hash == baseline["predictions_sha256"], "reference predictions mismatch")
    check(
        not exports.exists() and not any(p.is_symlink() for p in (exports, *exports.parents)),
        "export output exists/symlink",
    )
    exports.mkdir(parents=True, exist_ok=False)
    models, artifacts = {}, {"fit-summary.json": summary_sha}
    cells = []
    import torch

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        for name in ("all", *sources):
            item = summary["models"][name]
            directory = experiment / name
            card, card_sha = read_document(directory / "model-card.json")
            pred, pred_sha = read_document(directory / "predictions.json")
            check(
                card_sha == item["model_card_sha256"] and pred_sha == item["predictions_sha256"],
                "fit artifacts mismatch",
            )
            check(
                card["settings"] == EXPECTED
                and card["training_scope"]
                == training_scope(training, None if name == "all" else name)[1],
                "fixed fit settings/scope",
            )
            check(
                card["train_data_sha256"] == train_descriptor["data_sha256"],
                "training pixels changed",
            )
            check(
                card["model_sha256"] == item["model_sha256"]
                and pred["model_card_sha256"] == card_sha,
                "model binding",
            )
            check(
                card["manifest_sha256"] == digest
                and card["architecture"] == architecture
                and card["decoder"] == train_descriptor["decoder"]
                and card["decoder"] == validation_descriptor["decoder"]
                and pred["validation_cache_sha256"]
                == "a333104b64139f385f1d1a35a034593917ef5613ba9ed1b29a4f1a21b4b07a21"
                and card["threshold"] == 0.5
                and not card["deployed"]
                and not card["calibrated"]
                and not pred["deployed"]
                and not pred["calibrated"],
                "experimental model contract",
            )
            check(
                pred["manifest_sha256"] == digest and len(pred["predictions"]) == len(validation),
                "validation binding",
            )
            for row, sample in zip(pred["predictions"], validation, strict=True):
                check(
                    all(row[k] == sample[k] for k in ("sha256", "lineage", "label", "method")),
                    "prediction identity",
                )
            model = load_model(
                directory / "model.npz", checksum=card["model_sha256"], architecture=architecture
            )
            weights = {k: model.state_dict()[k].numpy().astype(np.float64) for k in SHAPES}
            saved = np.array([r["score"] for r in pred["predictions"]])
            previous_pixel = None
            if pixel_reference is not None:
                previous_pixel, previous_hash = read_document(
                    pixel_reference_experiment / name / "predictions.json"
                )
                check(
                    previous_hash == pixel_reference["models"][name]["predictions_sha256"],
                    "pixel reference predictions mismatch",
                )
                check(
                    len(previous_pixel["predictions"]) == len(validation)
                    and all(
                        all(a[k] == b[k] for k in ("sha256", "lineage", "label", "method"))
                        for a, b in zip(
                            previous_pixel["predictions"], pred["predictions"], strict=True
                        )
                    ),
                    "pixel reference identities mismatch",
                )
            native, numpy_scores = [], []
            from core.jpeg_cnn import pixel_logits

            for first in range(0, len(values), 17):
                raw = values[first : first + 17]
                native.extend(
                    torch.sigmoid(torch.from_numpy(pixel_logits(model, raw)))
                    .numpy()
                    .ravel()
                    .tolist()
                )
                numpy_scores.extend(
                    probability(numpy_logits(raw, weights, architecture)).ravel().tolist()
                )
            numeric = {
                "native_reload": comparison(np.array(native), saved),
                "numpy_forward": comparison(np.array(numpy_scores), saved),
            }
            numeric["onnx"] = onnx_audit(model, values, saved, exports / (name + ".onnx"))
            models[name] = {
                "training": card,
                "numeric_audit": numeric,
                "predictions_sha256": pred_sha,
            }
            for filename in ("model.npz", "model-card.json", "predictions.json"):
                artifacts[f"{name}/{filename}"] = sha(directory / filename)
            if (exports / (name + ".onnx")).is_file():
                artifacts[f"onnx/{name}.onnx"] = sha(exports / (name + ".onnx"))
            for target in sources:
                for family in ("JUNIWARD", "UERD"):
                    ids = H["context_ids"](validation, "source", target, family)
                    before = [old["predictions"][i] for i in ids]
                    after = [pred["predictions"][i] for i in ids]
                    metrics = cell_metrics(after, threshold=0.5, score_scale=1)
                    independent, confusion = H["independent_metrics"](after)
                    check(confusion == metrics["confusion"], "independent confusion mismatch")
                    np.testing.assert_allclose(
                        independent, [metrics[k] for k in KEYS], atol=5.01e-7, rtol=0
                    )
                    reference = next(
                        c["new"]
                        for c in baseline["comparison_cells"]
                        if c["dimension"] == "source"
                        and c["context"] == target
                        and c["method_family"] == family
                    )
                    check(
                        cell_metrics(before, threshold=0.5, score_scale=1) == reference,
                        "reference metrics changed",
                    )
                    cells.append(
                        {
                            "fit": name,
                            "validation_source_id": target,
                            "method_family": family,
                            "training_excluded_target": name != "all" and name != target,
                            "reference": reference,
                            "new": metrics,
                            "new_bootstrap_95_percent": paired_intervals(
                                after, threshold=0.5, score_scale=1
                            ),
                            "paired_delta_bootstrap_95_percent": H["paired_deltas"](before, after),
                            "delta_new_minus_reference": {
                                k: metrics[k] - reference[k] for k in KEYS
                            },
                            "numeric_failures": [
                                k
                                for i, k in enumerate(KEYS)
                                if (
                                    metrics[k] > H["LIMITS"][i]
                                    if i >= 3
                                    else metrics[k] < H["LIMITS"][i]
                                )
                            ],
                        }
                    )
                    if previous_pixel is not None:
                        pixel_before = [previous_pixel["predictions"][i] for i in ids]
                        cells[-1]["previous_pixel_reference"] = cell_metrics(
                            pixel_before, threshold=0.5, score_scale=1
                        )
                        cells[-1]["paired_delta_vs_previous_pixel_95_percent"] = H["paired_deltas"](
                            pixel_before, after
                        )
    finally:
        torch.set_num_threads(previous_threads)
    return {
        "schema_version": "pixel-cnn-development-v1",
        "date": "2026-10-06",
        "architecture": architecture,
        "manifest_sha256": digest,
        "protocol_sha256": summary["protocol_sha256"],
        "preregistered_implementation_protocol_commit": summary["implementation_protocol_commit"],
        "reference_record_sha256": baseline_sha,
        "reference_predictions_sha256": old_hash,
        "previous_pixel_reference_sha256": (
            sha(pixel_reference_record) if pixel_reference_record is not None else None
        ),
        "execution": summary,
        "models": models,
        "comparison_cells": cells,
        "artifact_sha256": artifacts,
        "source_id_map": baseline["source_id_map"],
        "counts": baseline["counts"],
        "versions": {k: version(k) for k in ("torch", "numpy", "onnx", "onnxruntime")},
        "independent_audit": {
            "passed": True,
            "files_rehashed": len(manifest["samples"]),
            "metric_cells_checked": 12,
            "numpy_forward_rows_per_model": len(validation),
        },
        "support_status": "experimental",
        "deployed": False,
        "calibrated": False,
        "qualification": "unavailable",
        "primary_detection_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest",
        "corpus",
        "cache-root",
        "experiment",
        "reference-record",
        "exports",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--architecture", choices=cnn.ARCHITECTURES, default=cnn.ARCHITECTURE)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--pixel-reference-record", type=Path)
    parser.add_argument("--pixel-reference-experiment", type=Path)
    args = parser.parse_args()
    write_json(
        args.out,
        audit(
            args.manifest,
            args.corpus,
            args.cache_root,
            args.experiment,
            args.reference_record,
            args.exports,
            architecture=args.architecture,
            protocol=args.protocol,
            pixel_reference_record=args.pixel_reference_record,
            pixel_reference_experiment=args.pixel_reference_experiment,
        ),
    )
    print("Complete: every fit/row/cell audited; numeric and detection gates remain separate.")


if __name__ == "__main__":
    main()
