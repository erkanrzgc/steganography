"""Explicit train-only gradient diagnosis; no fitting or detector changes."""

import argparse
import hashlib
import time
from pathlib import Path

from core import srnet, srnet_gradients, srnet_model, srnet_multibatch, srnet_positive, srnet_sanity
from steganography.research_features import read_document
from steganography.research_jpeg import write_json
from steganography.research_pixels import load_pixels
from steganography.research_srnet_sanity import CACHE_SHA, MANIFEST_SHA, configuration

PROTOCOL_SHA = "3b632bd4181e56a5e1f5fe1a696cf0443222186e419ab8d856e35ce6240f5ad1"
MODEL_SHA = "78dfff3e6da836fcabd3a6e65dc103b0a7508f4fcd9d30ca15ce961984c4b16e"


def run(config, model_path: Path, out: Path):
    configuration(config)
    if config["manifest_sha256"] != MANIFEST_SHA or config["cache_sha256"] != CACHE_SHA:
        raise ValueError("gradient diagnostic frozen input mismatch")
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("gradient output must be fresh and non-symlink")
    root = Path(__file__).resolve().parents[1]
    protocol = root / "docs/SRNET_GRADIENT_PROTOCOL.md"
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise ValueError("gradient diagnostic frozen protocol changed")
    _, checksum = read_document(Path(config["manifest"]))
    if checksum != MANIFEST_SHA:
        raise ValueError("gradient diagnostic manifest checksum mismatch")
    values, samples, descriptor = load_pixels(
        Path(config["manifest"]),
        Path(config["cache"]),
        checksum=CACHE_SHA,
        split="train",
        _float=True,
    )
    indices = srnet_sanity.select(samples)
    selected = [samples[i] for i in indices]
    batches, schedule = srnet_gradients.select_batches(selected)
    trained = srnet_model.load_model(model_path, checksum=MODEL_SHA)
    if any(int(v) != 400 for k, v in trained.named_buffers() if k.endswith("num_batches_tracked")):
        raise ValueError("gradient diagnostic trained BN mismatch")
    import torch

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261012)
        fresh = srnet.network().eval()
    generated, generated_rows = srnet_positive.generate()
    generated_batch = srnet_multibatch.epoch_batches(generated_rows, seed=20261012, epoch=0)[0][0]
    records, started = [], time.monotonic()

    def record(model, pixels, rows, kind):
        if time.monotonic() - started > 180:
            raise ValueError("gradient diagnostic deadline exceeded")
        result = srnet_gradients.probe(model, pixels)
        if time.monotonic() - started > 180:
            raise ValueError("gradient diagnostic deadline exceeded")
        records.append({"kind": kind, "rows": rows, **result})

    for batch in batches:
        metadata = [
            {
                k: selected[i][k]
                for k in ("sha256", "source_group", "lineage", "quality_factor", "label", "method")
            }
            for i in batch
        ]
        for kind, model in (("real_fresh", fresh), ("real_failed_tiny_model", trained)):
            record(model, values[indices[batch]], metadata, kind)
    record(
        fresh,
        generated[generated_batch],
        [generated_rows[i] for i in generated_batch],
        "generated_strong_fresh",
    )
    report = {
        "schema_version": "srnet-train-gradient-diagnostic-v1",
        "status": "completed",
        "protocol_sha256": PROTOCOL_SHA,
        "protocol_commit": "7c365fc",
        "manifest_sha256": MANIFEST_SHA,
        "train_cache_sha256": CACHE_SHA,
        "train_data_sha256": descriptor["data_sha256"],
        "trained_model_sha256": MODEL_SHA,
        "selected_rows": selected,
        "selected_batches": batches,
        "epoch_batch_schedule": schedule,
        "probes": records,
        "seconds": time.monotonic() - started,
        "classifier_directional_checks_passed": all(
            r["classifier_directional_check"]["passed"] for r in records
        ),
        "execution_source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in (
                "core/srnet_gradients.py",
                "core/srnet.py",
                "core/srnet_sampling.py",
                "core/srnet_multibatch.py",
                "core/srnet_positive.py",
                "steganography/research_srnet_gradients.py",
            )
        },
        "validation_pixels_loaded": False,
        "optimizer_steps": 0,
        "model_saved": False,
        "accuracy_qualification": "unavailable",
        "primary_detection_changed": False,
    }
    # Selected training cache metadata can contain local paths; publish identities only.
    report["selected_rows"] = [
        {
            k: s[k]
            for k in ("sha256", "source_group", "lineage", "quality_factor", "label", "method")
        }
        for s in selected
    ]
    write_json(out, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (580, 581))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        config, _ = read_document(args.config)
        report = run(config, args.model, args.out)
    except Exception:
        print("failed/unavailable: no complete usable gradient diagnostic")
        return 2
    print("completed: training-gradient diagnosis; detector qualification unavailable")
    return 0 if report["classifier_directional_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
