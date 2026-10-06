"""Explicit bounded train-only learning sanity, never detector accuracy."""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from core import srnet, srnet_model, srnet_multibatch, srnet_sanity, srnet_training
from core.srnet_sampling import epoch_pairs, source_id
from steganography.research_features import read_document
from steganography.research_jpeg import write_json
from steganography.research_pixels import load_pixels

PROTOCOL_SHA = "e76c6af4a3d075ffd2ad0f7647366994943aaee1e858495e63b7a2e3747b9982"
MANIFEST_SHA = "0f45f59007d230c6396a9e995cf7bd465de91795f995ced8f79dcd4e0a58cd14"
CACHE_SHA = "828158342b8937844e76f8f879bb41177a32e7ebcb85eef1eef59f2e67692028"
PARAMS = {"threads": 2, "max_seconds": 1800, "learning_rate": 0.001, "weight_decay": 0.0001}


def configuration(config):
    keys = {"manifest", "manifest_sha256", "cache", "cache_sha256"}
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
        raise ValueError("tiny sanity requires explicit checksum-bound train-only configuration")
    return config


def run(config, out: Path):
    configuration(config)
    if config["manifest_sha256"] != MANIFEST_SHA or config["cache_sha256"] != CACHE_SHA:
        raise ValueError("frozen tiny sanity manifest/cache checksums mismatch")
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("tiny sanity output must be fresh and non-symlink")
    protocol = Path(__file__).resolve().parents[1] / "docs/SRNET_TINY_SANITY_PROTOCOL.md"
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise ValueError("frozen tiny sanity protocol changed")
    _, manifest_sha = read_document(Path(config["manifest"]))
    if manifest_sha != config["manifest_sha256"]:
        raise ValueError("tiny sanity manifest checksum mismatch")
    values, samples, descriptor = load_pixels(
        Path(config["manifest"]),
        Path(config["cache"]),
        checksum=config["cache_sha256"],
        split="train",
        _float=True,
    )
    indices = srnet_sanity.select(samples)
    selected = [samples[i] for i in indices]
    paired = [epoch_pairs(selected, seed=20261012, epoch=e)[1] for e in range(50)]
    batched = [
        srnet_multibatch.epoch_batches(selected, seed=20261012, epoch=e)[1] for e in range(50)
    ]
    started = time.monotonic()
    model, records = srnet_training.fit(
        values,
        samples,
        indices,
        seed=20261012,
        schedule=paired,
        config={**PARAMS, "batch_recipe": srnet_multibatch.RECIPE},
    )
    if any(int(v) != 400 for k, v in model.named_buffers() if k.endswith("num_batches_tracked")):
        raise ValueError("tiny sanity BN update accounting mismatch")
    import torch

    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        logits = []
        for i in indices:
            if time.monotonic() - started > 1860:
                raise ValueError("tiny sanity evaluation deadline exceeded")
            logits.append(srnet.float_logits(model, values[i : i + 1])[0])
        final = srnet_sanity.metrics(np.array(logits), selected)
    finally:
        torch.set_num_threads(previous)
    gates = srnet_sanity.objectives(records, final)
    out.mkdir(parents=True, exist_ok=False)
    model_sha = srnet_model.save_model(model, out / "model.npz")
    report = {
        "schema_version": "srnet-tiny-sanity-v1",
        "status": "completed",
        "protocol_sha256": PROTOCOL_SHA,
        "protocol_commit": "64584b0",
        "manifest_sha256": manifest_sha,
        "train_cache_sha256": config["cache_sha256"],
        "train_data_sha256": descriptor["data_sha256"],
        "decoder": descriptor["decoder"],
        "selection": (
            "metadata-only first 4 ALASKA unknown-Q / first 2 BOSS complete Q75,Q95 lineages"
        ),
        "selected_rows": [
            {
                **{k: s[k] for k in ("sha256", "lineage", "quality_factor", "label", "method")},
                "source_id": source_id(s["source_group"]),
            }
            for s in selected
        ],
        "seed": 20261012,
        "optimizer": PARAMS,
        "epoch_pair_schedule": paired,
        "epoch_batch_schedule": batched,
        "epoch_training": records,
        "optimizer_updates": 400,
        "model_sha256": model_sha,
        "stored_bn_singleton_train_metrics": final,
        "sanity_objectives": gates,
        "sanity_objectives_passed": all(
            v for k, v in gates.items() if k != "relative_loss_reduction"
        ),
        "seconds": time.monotonic() - started,
        "validation_pixels_loaded": False,
        "in_sample_only": True,
        "accuracy_qualification": "unavailable",
        "calibrated": False,
        "deployed": False,
        "primary_detection_changed": False,
    }
    write_json(out / "sanity.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        config, _ = read_document(args.config)
        configuration(config)
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (3660, 3661))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        report = run(config, args.out)
    except Exception:
        print("failed/unavailable: no complete usable train-sanity result")
        return 2
    print("completed: train-only sanity; detector qualification unavailable")
    return 0 if report["sanity_objectives_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
