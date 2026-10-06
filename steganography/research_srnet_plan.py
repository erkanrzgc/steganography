"""Explicit checksum-bound training schedule; does not fit or deploy a model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core import jpeg_float256, srnet
from core.srnet_sampling import epoch_pairs
from steganography.research import ResearchManifestError
from steganography.research_features import read_document
from steganography.research_jpeg import write_json
from steganography.research_jrm import training_scope
from steganography.research_pixels import load_pixels
from steganography.research_weighting import _jpeg_source_weights


def settings(config):
    required = {"manifest", "manifest_sha256", "cache", "cache_sha256"}
    if (
        not isinstance(config, dict)
        or not required <= config.keys()
        or config.keys() - required - {"epochs", "seed", "training_source_id"}
        or any(not isinstance(config[k], str) or not config[k] for k in required)
    ):
        raise ResearchManifestError("SRNet plan requires explicit training-only configuration")
    for key in ("manifest_sha256", "cache_sha256"):
        if len(config[key]) != 64 or any(c not in "0123456789abcdef" for c in config[key]):
            raise ResearchManifestError("SRNet plan requires SHA-256 identities")
    values = {"epochs": config.get("epochs", 10), "seed": config.get("seed", 20261012)}
    for key, high in (("epochs", 50), ("seed", 2**32 - 1)):
        if type(values[key]) is not int or not (1 if key == "epochs" else 0) <= values[key] <= high:
            raise ResearchManifestError("SRNet plan setting outside limits")
    return values


def plan_training(config: dict, out: Path):
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("SRNet plan output exists or uses a symlink")
    params = settings(config)
    manifest_path = Path(config["manifest"])
    manifest, digest = read_document(manifest_path)
    if digest != config["manifest_sha256"]:
        raise ResearchManifestError("SRNet plan manifest checksum mismatch")
    pixels, samples, descriptor = load_pixels(
        manifest_path,
        Path(config["cache"]),
        checksum=config["cache_sha256"],
        split="train",
        _float=True,
    )
    del pixels  # Full cache integrity checked, never retain multiple corpus copies.
    record = schedule_record(
        manifest,
        digest,
        samples,
        descriptor,
        config["cache_sha256"],
        params,
        config.get("training_source_id"),
    )
    write_json(out, record)
    return record


def schedule_record(manifest, digest, samples, descriptor, cache_sha256, params, source_id):
    """One provenance contract for explicit plans and training plan verification."""
    provenance: dict[str, Any] = {}
    _jpeg_source_weights(manifest, samples, provenance)
    epoch_pairs(samples, seed=params["seed"], epoch=0)  # Validate excluded rows too.
    indices, scope = training_scope(samples, source_id)
    selected = [samples[i] for i in indices]
    epochs = [
        epoch_pairs(selected, seed=params["seed"], epoch=e)[1] for e in range(params["epochs"])
    ]
    record = {
        "schema_version": "srnet-training-plan-v1",
        "architecture": srnet.ARCHITECTURE,
        "feature_version": jpeg_float256.FEATURE_VERSION,
        "manifest_sha256": digest,
        "train_cache_sha256": cache_sha256,
        "train_data_sha256": descriptor["data_sha256"],
        "decoder": descriptor["decoder"],
        "training_scope": scope,
        "source_validation": provenance["source_balance"],
        "settings": params,
        "epochs": epochs,
        "batch_contract": "one matched cover/stego pair; labels [0,1]; float32 pixel units",
        "balancing": "equal sources, equal quality/method cells within each source",
        "unknown_quality": "separate declared-unknown cell; never estimated from labels",
        "validation_used": False,
        "real_training": "unavailable",
        "accuracy_metrics": "unavailable",
        "support_status": "experimental",
        "deployed": False,
    }
    return record


def run_plan(config_path: Path, out: Path):
    config, _ = read_document(config_path)
    return plan_training(config, out)
