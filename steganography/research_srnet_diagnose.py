"""Explicit train-only, state-preserving SRNet diagnostic service."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from core import srnet_diagnostics, srnet_model
from core.srnet_sampling import epoch_pairs, source_id
from steganography.research import ResearchManifestError
from steganography.research_features import read_document
from steganography.research_jpeg import write_json
from steganography.research_srnet_evaluate import checked_card
from steganography.research_srnet_fit import bound_training


def configuration(config):
    keys = {
        "manifest",
        "manifest_sha256",
        "cache",
        "cache_sha256",
        "plan",
        "plan_sha256",
        "model_dir",
        "card_sha256",
    }
    if (
        not isinstance(config, dict)
        or set(config) != keys
        or any(not isinstance(config[k], str) or not config[k] for k in keys)
        or any(
            len(config[k]) != 64 or any(c not in "0123456789abcdef" for c in config[k])
            for k in keys
            if k.endswith("sha256")
        )
    ):
        raise ResearchManifestError("SRNet diagnosis requires explicit training-only configuration")
    return {k: config[k] for k in keys - {"model_dir", "card_sha256"}}


def select_pairs(samples, *, seed):
    pairs, _ = epoch_pairs(samples, seed=seed, epoch=0)
    selected, seen = [], set()
    counts: dict[tuple, int] = {}
    for cover, stego in pairs:
        sample = samples[stego]
        key = (sample["source_group"], sample["quality_factor"], sample["method"])
        if sample["sha256"] not in seen and counts.get(key, 0) < 8:
            seen.add(sample["sha256"])
            selected.append((int(cover), int(stego)))
            counts[key] = counts.get(key, 0) + 1
    if not 1 <= len(selected) <= 64:
        raise ResearchManifestError("SRNet diagnostic probe limit exceeded")
    return selected


def diagnose(config, out: Path):
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("SRNet diagnostic output exists or uses a symlink")
    fit_config = configuration(config)
    values, samples, descriptor, plan, settings, indices = bound_training(fit_config)
    card, card_sha = read_document(Path(config["model_dir"]) / "model-card.json")
    if card_sha != config["card_sha256"]:
        raise ResearchManifestError("SRNet diagnostic card checksum mismatch")
    checked_card(card, plan, settings, config["plan_sha256"])
    selected = [samples[i] for i in indices]
    pairs = select_pairs(selected, seed=settings["seed"])
    model = srnet_model.load_model(
        Path(config["model_dir"]) / "model.npz", checksum=card["model_sha256"]
    )
    updates = sum(e["pairs"] for e in plan["epochs"])
    if any(
        int(v) != updates for k, v in model.named_buffers() if k.endswith("num_batches_tracked")
    ):
        raise ResearchManifestError("SRNet diagnostic BN update accounting mismatch")
    import torch

    started, previous_threads = time.monotonic(), torch.get_num_threads()
    records = []
    try:
        torch.set_num_threads(2)
        for cover, stego in pairs:
            if time.monotonic() - started > 180:
                raise ResearchManifestError("SRNet diagnostic deadline exceeded")
            pixel_pair = values[indices[[cover, stego]]]
            result = srnet_diagnostics.paired_probe(model, pixel_pair)
            if time.monotonic() - started > 180:
                raise ResearchManifestError("SRNet diagnostic deadline exceeded")
            sample = selected[stego]
            difference = pixel_pair[1].astype(np.float64) - pixel_pair[0]
            records.append(
                {
                    "cover_sha256": selected[cover]["sha256"],
                    "stego_sha256": sample["sha256"],
                    "source_id": source_id(sample["source_group"]),
                    "quality_factor": sample["quality_factor"],
                    "method": sample["method"],
                    "input_difference_rms": float(np.sqrt(np.mean(difference**2))),
                    **result,
                }
            )
    finally:
        torch.set_num_threads(previous_threads)
    report = {
        "schema_version": "srnet-train-diagnostic-v1",
        "status": "completed",
        "manifest_sha256": config["manifest_sha256"],
        "train_cache_sha256": config["cache_sha256"],
        "train_data_sha256": descriptor["data_sha256"],
        "training_plan_sha256": config["plan_sha256"],
        "model_card_sha256": card_sha,
        "model_sha256": card["model_sha256"],
        "training_scope": plan["training_scope"],
        "selection": "first 8 unique stegos per cell in epoch-zero order",
        "probe_pairs": len(records),
        "probes": records,
        "seconds": time.monotonic() - started,
        "validation_pixels_loaded": False,
        "threshold": 0.5,
        "state_unchanged": True,
        "batch_statistics_inference": "diagnostic-only; batch-dependent; never primary verdict",
        "calibrated": False,
        "deployed": False,
        "qualification": "unavailable",
    }
    write_json(out, report)
    return report


def run_diagnosis(config_path: Path, out: Path):
    config, _ = read_document(config_path)
    return diagnose(config, out)
