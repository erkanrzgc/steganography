"""Explicit bounded CPU minibatch pixel learning; never deploys a detector."""

from __future__ import annotations

import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from core import jpeg_cnn as cnn
from core.jpeg_cnn_model import load_model, save_model
from core.jpeg_pixels import FEATURE_VERSION, decoder_contract
from steganography.research import ResearchManifestError
from steganography.research_features import read_document, selected_samples
from steganography.research_jpeg import write_json
from steganography.research_jrm import paired_indices, training_scope
from steganography.research_pixels import load_pixels
from steganography.research_weighting import _jpeg_source_weights


def fresh(path):
    if path.exists() or any(p.is_symlink() for p in (path, *path.parents)):
        raise FileExistsError("pixel CNN output exists or uses a symlink")


def settings(config):
    if not isinstance(config, dict):
        raise ResearchManifestError("pixel CNN settings must be an object")
    values: dict[str, Any] = {
        "epochs": 10,
        "batch_size": 32,
        "threads": 2,
        "seed": 20261011,
        "max_seconds": 1800,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }
    bounds = {
        "epochs": (1, 50),
        "batch_size": (1, 64),
        "threads": (1, 2),
        "seed": (0, 2**32 - 1),
        "max_seconds": (1, 1800),
    }
    for key, (low, high) in bounds.items():
        value = config.get(key, values[key])
        if type(value) is not int or not low <= value <= high:
            raise ResearchManifestError("invalid bounded pixel CNN training setting")
        values[key] = value
    for key, (optimizer_low, optimizer_high) in {
        "learning_rate": (1e-6, 0.1),
        "weight_decay": (0, 1),
    }.items():
        value = config.get(key, values[key])
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not optimizer_low <= value <= optimizer_high
        ):
            raise ResearchManifestError("invalid bounded pixel CNN optimizer setting")
        values[key] = float(value)
    return values


def train_pixels(config: dict, out: Path, *, progress=None):
    fresh(out)
    params = settings(config)
    architecture = config.get("architecture", cnn.ARCHITECTURE)
    try:
        cnn.residual_contract(architecture)
    except ValueError as exc:
        raise ResearchManifestError("pixel CNN architecture contract mismatch") from exc
    manifest_path = Path(config["manifest"])
    manifest, digest = read_document(manifest_path)
    if digest != config["manifest_sha256"]:
        raise ResearchManifestError("pixel CNN manifest checksum mismatch")
    values, samples, descriptor = load_pixels(
        manifest_path, Path(config["cache"]), checksum=config["cache_sha256"], split="train"
    )
    provenance: dict[str, Any] = {}
    _jpeg_source_weights(manifest, samples, provenance)
    paired_indices(samples)
    indices, scope = training_scope(samples, config.get("training_source_id"))
    selected = [samples[i] for i in indices]
    values = values[indices]
    counts = Counter((s["source_group"], s["label"]) for s in selected)
    sources = {s["source_group"] for s in selected}
    weights = np.array(
        [
            len(selected) / (2 * len(sources) * counts[(s["source_group"], s["label"])])
            for s in selected
        ],
        dtype=np.float32,
    )
    labels = np.array([s["label"] == "stego" for s in selected], dtype=np.float32)[:, None]
    import torch

    started = time.monotonic()
    previous_threads = torch.get_num_threads()
    losses = []
    try:
        torch.set_num_threads(params["threads"])
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(params["seed"])
            generator = torch.Generator(device="cpu").manual_seed(params["seed"])
            model = cnn.network() if architecture == cnn.ARCHITECTURE else cnn.network(architecture)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=params["learning_rate"], weight_decay=params["weight_decay"]
            )
            for epoch in range(params["epochs"]):
                order = torch.randperm(len(selected), generator=generator, device="cpu").numpy()
                total = 0.0
                for first in range(0, len(order), params["batch_size"]):
                    if time.monotonic() - started > params["max_seconds"]:
                        raise ResearchManifestError("pixel CNN training deadline exceeded")
                    batch = order[first : first + params["batch_size"]]
                    inputs = torch.from_numpy(values[batch].astype(np.float32) / 255)
                    targets = torch.from_numpy(labels[batch])
                    optimizer.zero_grad()
                    per_row = torch.nn.functional.binary_cross_entropy_with_logits(
                        model(inputs), targets, reduction="none"
                    )
                    loss = (per_row * torch.from_numpy(weights[batch, None])).mean()
                    if not bool(torch.isfinite(loss)):
                        raise ResearchManifestError("pixel CNN training produced nonfinite loss")
                    loss.backward()
                    if not all(
                        p.grad is not None and bool(torch.isfinite(p.grad).all())
                        for p in model.parameters()
                    ):
                        raise ResearchManifestError("pixel CNN training produced invalid gradients")
                    optimizer.step()
                    total += float(loss.detach()) * len(batch)
                losses.append(total / len(selected))
                if progress is not None:
                    progress({"epoch": epoch + 1, "epochs": params["epochs"], "loss": losses[-1]})
            if time.monotonic() - started > params["max_seconds"]:
                raise ResearchManifestError("pixel CNN training deadline exceeded")
    finally:
        torch.set_num_threads(previous_threads)
    out.mkdir(parents=True, exist_ok=False)
    checksum = save_model(model, out / "model.npz")
    card = {
        "schema_version": "jpeg-pixel-cnn-v1",
        "architecture": architecture,
        "feature_version": FEATURE_VERSION,
        "manifest_sha256": digest,
        "train_cache_sha256": config["cache_sha256"],
        "train_data_sha256": descriptor["data_sha256"],
        "model_sha256": checksum,
        "decoder": decoder_contract(),
        "training_scope": scope,
        "source_validation": provenance["source_balance"],
        "settings": params,
        "epoch_losses": losses,
        "seconds": time.monotonic() - started,
        "weighting": {
            "recipe": "source-class-balanced-v1",
            "formula": "n / (2 * sources * source_label_count)",
            "reduction": "mean-over-current-batch",
            "minimum": float(weights.min()),
            "maximum": float(weights.max()),
        },
        "torch_version": str(torch.__version__),
        "score": "sigmoid logit; uncalibrated",
        "threshold": 0.5,
        "support_status": "experimental",
        "calibrated": False,
        "deployed": False,
    }
    write_json(out / "model-card.json", card)
    return card


def predict_pixels(config: dict, out: Path):
    fresh(out)
    manifest_path = Path(config["manifest"])
    manifest, digest = read_document(manifest_path)
    values, samples, _ = load_pixels(
        manifest_path, Path(config["cache"]), checksum=config["cache_sha256"], split="validation"
    )
    model_dir = Path(config["model_dir"])
    card, card_hash = read_document(model_dir / "model-card.json")
    if (
        card_hash != config["card_sha256"]
        or card.get("schema_version") != "jpeg-pixel-cnn-v1"
        or card.get("manifest_sha256") != digest
        or digest != config["manifest_sha256"]
        or card.get("architecture") not in cnn.ARCHITECTURES
        or card.get("feature_version") != FEATURE_VERSION
        or card.get("decoder") != decoder_contract()
        or card.get("calibrated") is not False
        or card.get("deployed") is not False
    ):
        raise ResearchManifestError("pixel CNN model/validation contract mismatch")
    scope = card.get("training_scope")
    if settings(card.get("settings")) != card["settings"]:
        raise ResearchManifestError("pixel CNN settings contract mismatch")
    if not isinstance(scope, dict) or not isinstance(scope.get("source_ids"), list):
        raise ResearchManifestError("pixel CNN training scope mismatch")
    if scope.get("recipe") == "all-declared-sources-v1":
        source_id = None
    elif scope.get("recipe") == "single-declared-source-v1" and len(scope["source_ids"]) == 1:
        source_id = scope["source_ids"][0]
    else:
        raise ResearchManifestError("pixel CNN training scope mismatch")
    if training_scope(selected_samples(manifest, "train"), source_id)[1] != scope:
        raise ResearchManifestError("pixel CNN training scope mismatch")
    model = load_model(
        model_dir / "model.npz", checksum=card["model_sha256"], architecture=card["architecture"]
    )
    import torch

    previous_threads = torch.get_num_threads()
    scores = []
    try:
        torch.set_num_threads(card["settings"]["threads"])
        for first in range(0, len(values), cnn.MAX_INFERENCE_BATCH):
            logits = cnn.pixel_logits(model, values[first : first + cnn.MAX_INFERENCE_BATCH])
            scores.extend(torch.sigmoid(torch.from_numpy(logits)).numpy().ravel().tolist())
    finally:
        torch.set_num_threads(previous_threads)
    report = {
        "schema_version": "jpeg-pixel-cnn-predictions-v1",
        "manifest_sha256": digest,
        "validation_cache_sha256": config["cache_sha256"],
        "model_card_sha256": card_hash,
        "support_status": "experimental",
        "calibrated": False,
        "deployed": False,
        "predictions": [
            {**{k: s[k] for k in ("sha256", "lineage", "label", "method")}, "score": float(score)}
            for s, score in zip(samples, scores, strict=True)
        ],
    }
    write_json(out, report)
    return report


def run_pixel_cnn(config_path: Path, out: Path):
    config, _ = read_document(config_path)
    if config.get("stage") == "train":
        return train_pixels(config, out)
    if config.get("stage") == "predict":
        return predict_pixels(config, out)
    raise ResearchManifestError("pixel CNN stage must be train or predict")
