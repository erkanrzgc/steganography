"""Explicit complete-validation replay, never calibration or deployment."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from core import srnet, srnet_model, srnet_reference, srnet_training
from core.srnet_sampling import source_id
from steganography.research import ResearchManifestError
from steganography.research_features import read_document
from steganography.research_jpeg import write_json
from steganography.research_pixels import load_pixels
from steganography.research_srnet_fit import bound_training, card_contract


def configuration(config):
    required = {
        "manifest",
        "manifest_sha256",
        "cache",
        "cache_sha256",
        "plan",
        "plan_sha256",
        "validation_cache",
        "validation_cache_sha256",
        "model_dir",
        "card_sha256",
    }
    if (
        not isinstance(config, dict)
        or set(config) != required
        or any(not isinstance(config[k], str) or not config[k] for k in required)
        or any(
            len(config[k]) != 64 or any(c not in "0123456789abcdef" for c in config[k])
            for k in required
            if k.endswith("sha256")
        )
    ):
        raise ResearchManifestError("SRNet evaluation requires explicit checksum-bound inputs")
    return {
        k: config[k]
        for k in ("manifest", "manifest_sha256", "cache", "cache_sha256", "plan", "plan_sha256")
    }


def checked_card(card, plan, settings, plan_sha):
    optimizer = card.get("optimizer")
    if not isinstance(optimizer, dict):
        raise ResearchManifestError("SRNet evaluation optimizer contract mismatch")
    params = srnet_training.settings(optimizer)
    expected = {**card_contract(plan, settings, params), "training_plan_sha256": plan_sha}
    dynamic = {"epoch_training", "model_sha256", "seconds", "torch_version"}
    if set(card) != set(expected) | dynamic or any(
        json.dumps(card[k], sort_keys=True) != json.dumps(v, sort_keys=True)
        for k, v in expected.items()
    ):
        raise ResearchManifestError("SRNet evaluation fit-card contract mismatch")
    if (
        type(card["seconds"]) not in (int, float)
        or not math.isfinite(card["seconds"])
        or card["seconds"] < 0
        or not isinstance(card["torch_version"], str)
        or not card["torch_version"]
        or not isinstance(card["model_sha256"], str)
        or len(card["model_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in card["model_sha256"])
        or not isinstance(card["epoch_training"], list)
        or len(card["epoch_training"]) != len(plan["epochs"])
    ):
        raise ResearchManifestError("SRNet evaluation incomplete fit-card")
    for epoch, record in enumerate(card["epoch_training"]):
        if (
            not isinstance(record, dict)
            or set(record) != {"epoch", "updates", "mean_pair_loss"}
            or type(record["epoch"]) is not int
            or record["epoch"] != epoch
            or type(record["updates"]) is not int
            or record["updates"] != plan["epochs"][epoch]["pairs"]
            or type(record["mean_pair_loss"]) not in (int, float)
            or not math.isfinite(record["mean_pair_loss"])
            or record["mean_pair_loss"] < 0
        ):
            raise ResearchManifestError("SRNet evaluation incomplete epoch accounting")
    return params


def oracle_rows(samples):
    """First manifest row per declared source/quality/label/method, before inference."""
    cells: dict[tuple, int] = {}
    for i, sample in enumerate(samples):
        key = (
            sample["source_group"],
            sample.get("quality_factor"),
            sample["label"],
            sample["method"],
        )
        cells.setdefault(key, i)
    if not 1 <= len(cells) <= 24:
        raise ResearchManifestError("SRNet independent replay cell limit exceeded")
    return list(cells.values())


def evaluate(config: dict, out: Path):
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("SRNet evaluation output exists or uses a symlink")
    fit_config = configuration(config)
    started = time.monotonic()
    train, _, _, plan, settings, _ = bound_training(fit_config)
    del train
    card, digest = read_document(Path(config["model_dir"]) / "model-card.json")
    if digest != config["card_sha256"]:
        raise ResearchManifestError("SRNet evaluation card checksum mismatch")
    params = checked_card(card, plan, settings, config["plan_sha256"])
    values, samples, descriptor = load_pixels(
        Path(config["manifest"]),
        Path(config["validation_cache"]),
        checksum=config["validation_cache_sha256"],
        split="validation",
        _float=True,
    )
    if descriptor["decoder"] != plan["decoder"]:
        raise ResearchManifestError("SRNet validation decoder mismatch")
    selected = oracle_rows(samples)
    model = srnet_model.load_model(
        Path(config["model_dir"]) / "model.npz", checksum=card["model_sha256"]
    )
    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    updates = sum(e["pairs"] for e in plan["epochs"])
    if any(int(v) != updates for k, v in arrays.items() if k.endswith("num_batches_tracked")):
        raise ResearchManifestError("SRNet fit BN update accounting mismatch")
    import torch

    previous = torch.get_num_threads()
    logits, audits = [], []

    def deadline():
        if time.monotonic() - started > 1800:
            raise ResearchManifestError("SRNet evaluation deadline exceeded")

    try:
        torch.set_num_threads(params["threads"])
        for first in range(0, len(values), 4):
            deadline()
            logits.append(srnet.float_logits(model, values[first : first + 4]))
            deadline()
        native = np.concatenate(logits)
        for index in selected:
            deadline()
            reference = srnet_reference.reference_logits(arrays, values[index : index + 1])
            deadline()
            audits.append(
                {
                    "sha256": samples[index]["sha256"],
                    **srnet_reference.compare(native[index : index + 1], reference),
                }
            )
    finally:
        torch.set_num_threads(previous)
    passed = all(a["passed"] for a in audits)
    scores = np.exp(-np.logaddexp(0.0, native[:, 0].astype(np.float64) - native[:, 1]))
    report = {
        "schema_version": "srnet-validation-predictions-v1",
        "status": "completed" if passed else "failed_numerical_gate",
        "manifest_sha256": config["manifest_sha256"],
        "train_cache_sha256": config["cache_sha256"],
        "training_plan_sha256": config["plan_sha256"],
        "validation_cache_sha256": config["validation_cache_sha256"],
        "validation_data_sha256": descriptor["data_sha256"],
        "model_card_sha256": digest,
        "model_sha256": card["model_sha256"],
        "training_scope": plan["training_scope"],
        "oracle_selection": "first manifest row per declared source/quality/label/method; max 24",
        "independent_forward_audit": audits,
        "validation_rows": len(samples),
        "native_rows_evaluated": len(native),
        "seconds": time.monotonic() - started,
        "threshold": 0.5,
        "tie_policy": "stego",
        "calibrated": False,
        "deployed": False,
        "support_status": "experimental",
        "qualification": "unavailable",
        "onnx_replay": "unavailable",
        "validation_independence": "reused validation; not blind or untouched",
        "predictions": [
            {
                **{k: sample[k] for k in ("sha256", "lineage", "label", "method")},
                "source_id": source_id(sample["source_group"]),
                "quality_factor": sample.get("quality_factor"),
                "score": float(score),
            }
            for sample, score in zip(samples, scores, strict=True)
        ]
        if passed
        else [],
    }
    write_json(out, report)
    return report


def run_evaluation(config_path: Path, out: Path):
    config, _ = read_document(config_path)
    return evaluate(config, out)
