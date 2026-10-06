"""Explicit research-only SRNet fitting bound to a verified training plan."""

from __future__ import annotations

import time
from pathlib import Path

from core import srnet_model, srnet_training
from steganography.research import ResearchManifestError
from steganography.research_features import read_document
from steganography.research_jpeg import write_json
from steganography.research_jrm import training_scope
from steganography.research_pixels import load_pixels
from steganography.research_srnet_plan import schedule_record
from steganography.research_srnet_plan import settings as plan_settings


def configuration(config):
    required = {"manifest", "manifest_sha256", "cache", "cache_sha256", "plan", "plan_sha256"}
    if (
        not isinstance(config, dict)
        or not required <= config.keys()
        or config.keys() - required - {"threads", "max_seconds", "learning_rate", "weight_decay"}
        or any(not isinstance(config[k], str) or not config[k] for k in required)
    ):
        raise ResearchManifestError("SRNet fit requires explicit plan-bound configuration")
    if len(config["plan_sha256"]) != 64 or any(
        c not in "0123456789abcdef" for c in config["plan_sha256"]
    ):
        raise ResearchManifestError("SRNet fit requires plan SHA-256")
    plan_settings({k: config[k] for k in ("manifest", "manifest_sha256", "cache", "cache_sha256")})
    return srnet_training.settings(config)


def train_srnet(config: dict, out: Path):
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("SRNet fit output exists or uses a symlink")
    params = configuration(config)
    started = time.monotonic()
    manifest_path = Path(config["manifest"])
    manifest, digest = read_document(manifest_path)
    plan, plan_sha = read_document(Path(config["plan"]))
    if digest != config["manifest_sha256"] or plan_sha != config["plan_sha256"]:
        raise ResearchManifestError("SRNet fit manifest/plan checksum mismatch")
    values, samples, descriptor = load_pixels(
        manifest_path,
        Path(config["cache"]),
        checksum=config["cache_sha256"],
        split="train",
        _float=True,
    )
    scope, schedule_settings = plan.get("training_scope"), plan.get("settings")
    if not isinstance(scope, dict) or not isinstance(schedule_settings, dict):
        raise ResearchManifestError("SRNet fit plan contract mismatch")
    ids = scope.get("source_ids")
    if scope.get("recipe") == "all-declared-sources-v1":
        source_id = None
    elif (
        scope.get("recipe") == "single-declared-source-v1"
        and isinstance(ids, list)
        and len(ids) == 1
    ):
        source_id = ids[0]
    else:
        raise ResearchManifestError("SRNet fit plan scope mismatch")
    checked = plan_settings(
        {
            **{k: config[k] for k in ("manifest", "manifest_sha256", "cache", "cache_sha256")},
            **schedule_settings,
            "training_source_id": source_id,
        }
    )
    expected = schedule_record(
        manifest, digest, samples, descriptor, config["cache_sha256"], checked, source_id
    )
    if plan != expected:
        raise ResearchManifestError("SRNet fit bound schedule/provenance mismatch")
    indices, _ = training_scope(samples, source_id)
    model, records = srnet_training.fit(
        values, samples, indices, seed=checked["seed"], schedule=plan["epochs"], config=params
    )
    out.mkdir(parents=True, exist_ok=False)
    checksum = srnet_model.save_model(model, out / "model.npz")
    card = {
        "schema_version": "srnet-fit-v1",
        "architecture": plan["architecture"],
        "feature_version": plan["feature_version"],
        "decoder": plan["decoder"],
        "manifest_sha256": digest,
        "train_cache_sha256": config["cache_sha256"],
        "train_data_sha256": descriptor["data_sha256"],
        "training_plan_sha256": plan_sha,
        "training_scope": scope,
        "source_validation": plan["source_validation"],
        "sampling_settings": checked,
        "epoch_schedule": plan["epochs"],
        "optimizer": {
            "name": "Adamax",
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "foreach": False,
            "lr_schedule": "constant",
            **params,
        },
        "epoch_training": records,
        "model_sha256": checksum,
        "seconds": time.monotonic() - started,
        "validation_used": False,
        "batch_size": 2,
        "input_units": "unrounded float32 Y pixels; no normalization",
        "batchnorm": "paired training; persisted running statistics for eval",
        "training": "completed",
        "qualification": "unavailable",
        "accuracy_metrics": "unavailable",
        "independent_forward_audit": "unavailable",
        "support_status": "experimental",
        "calibrated": False,
        "deployed": False,
    }
    import torch

    card["torch_version"] = str(torch.__version__)
    write_json(out / "model-card.json", card)
    return card


def run_fit(config_path: Path, out: Path):
    from steganography.research_srnet_job import run_job

    return run_job(config_path, out)
