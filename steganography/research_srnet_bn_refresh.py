"""Explicit frozen training-only BN clone refresh and batch-context contrast."""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from core import srnet_bn_refresh as bn
from core import srnet_model, srnet_sanity, srnet_signal
from steganography.research_features import read_document
from steganography.research_jpeg import write_json
from steganography.research_pixels import load_pixels
from steganography.research_srnet_sanity import CACHE_SHA, MANIFEST_SHA, configuration

PROTOCOL_SHA = "821ff78fa8ae49ae8d86a76622f49fc394d460b880a65e3c6aa7d5eea67ddb75"
BASELINE_SHA = "a7ce1b3f61a22accb45b73ae8d69ebd3e47e3ee53b428a7e49aabc32d67666cf"


def run(config, model_dir: Path, out: Path):
    configuration(config)
    if config["manifest_sha256"] != MANIFEST_SHA or config["cache_sha256"] != CACHE_SHA:
        raise ValueError("BN refresh frozen input mismatch")
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("BN refresh output must be fresh and non-symlink")
    root = Path(__file__).resolve().parents[1]
    if (
        hashlib.sha256((root / "docs/SRNET_BN_REFRESH_PROTOCOL.md").read_bytes()).hexdigest()
        != PROTOCOL_SHA
    ):
        raise ValueError("BN refresh frozen protocol mismatch")
    baseline, checksum = read_document(root / "benchmarks/srnet-signal-strength-20261007.json")
    if checksum != BASELINE_SHA:
        raise ValueError("BN refresh baseline evidence mismatch")
    _, checksum = read_document(Path(config["manifest"]))
    if checksum != MANIFEST_SHA:
        raise ValueError("BN refresh manifest checksum mismatch")
    values, samples, descriptor = load_pixels(
        Path(config["manifest"]),
        Path(config["cache"]),
        checksum=CACHE_SHA,
        split="train",
        _float=True,
    )
    indices = srnet_sanity.select(samples)
    selected = [samples[i] for i in indices]
    original = values[indices].copy()
    identities = [
        {
            k: s[k]
            for k in ("sha256", "source_group", "lineage", "quality_factor", "label", "method")
        }
        for s in selected
    ]
    if (
        identities != baseline["selected_rows"]
        or descriptor["data_sha256"] != baseline["train_data_sha256"]
    ):
        raise ValueError("BN refresh selected train identity mismatch")
    sources = {
        n: hashlib.sha256((root / n).read_bytes()).hexdigest()
        for n in (
            "core/srnet_bn_refresh.py",
            "core/srnet.py",
            "core/srnet_model.py",
            "core/srnet_signal.py",
            "core/srnet_multibatch.py",
            "core/srnet_sampling.py",
            "core/srnet_reference.py",
            "steganography/research_srnet_bn_refresh.py",
        )
    }
    started, arms = time.monotonic(), []

    def deadline():
        if time.monotonic() - started > 300:
            raise ValueError("BN refresh job deadline exceeded")

    def replay(model, pixels, base, prefix):
        logits, metrics = srnet_signal.evaluate(model, pixels, selected)
        stored = np.asarray(base[prefix + "_singleton_logits"], dtype=np.float64)
        if (
            stored.shape != (24, 2)
            or not np.allclose(logits, stored, atol=1e-6, rtol=0)
            or set(metrics) != set(base[prefix + "_train_metrics"])
            or any(
                not np.allclose(v, base[prefix + "_train_metrics"][k], atol=1e-6, rtol=0)
                for k, v in metrics.items()
            )
        ):
            raise ValueError("BN refresh singleton baseline/reload mismatch")
        deadline()
        return logits, metrics

    out.mkdir(parents=True, exist_ok=False)
    for base in baseline["arms"]:
        deadline()
        factor = base["factor"]
        pixels, hashes = srnet_signal.prepare(original, selected, factor=factor)
        if hashes != base["derived_tensor_sha256"]:
            raise ValueError("BN refresh derived tensor identity mismatch")
        model = srnet_model.load_model(
            model_dir / f"factor-{factor}" / "model.npz", checksum=base["model_sha256"]
        )
        if any(
            int(v) != 160 for k, v in model.named_buffers() if k.endswith("num_batches_tracked")
        ):
            raise ValueError("BN refresh original counter mismatch")
        replay(model, pixels, base, "own_input")
        replay(model, original, base, "original_input")
        contrast = {
            "own_input": bn.context(model, pixels, selected),
            "original_input": bn.context(model, original, selected),
        }
        deadline()
        clone, change = bn.refresh(model, pixels, selected)
        deadline()
        logits, own = srnet_signal.evaluate(clone, pixels, selected)
        original_logits, original_metrics = srnet_signal.evaluate(clone, original, selected)
        arm = {
            "factor": factor,
            "source_model_sha256": base["model_sha256"],
            "derived_tensor_sha256": hashes,
            "baseline_own_metrics": base["own_input_train_metrics"],
            "baseline_original_metrics": base["original_input_train_metrics"],
            "contrast": contrast,
            "refresh": change,
            "own_input_singleton_logits": logits.tolist(),
            "own_input_train_metrics": own,
            "original_input_singleton_logits": original_logits.tolist(),
            "original_input_train_metrics": original_metrics,
            "in_sample_objectives": bn.objectives(own, base["own_input_train_metrics"]),
        }
        arm["in_sample_objectives_passed"] = all(arm["in_sample_objectives"].values())
        arm_dir = out / f"factor-{factor}"
        arm_dir.mkdir(exist_ok=False)
        arm["refreshed_model_sha256"] = srnet_model.save_model(clone, arm_dir / "model.npz")
        reloaded = srnet_model.load_model(
            arm_dir / "model.npz", checksum=arm["refreshed_model_sha256"]
        )
        replay(reloaded, pixels, arm, "own_input")
        replay(reloaded, original, arm, "original_input")
        arm["audit"] = bn.oracles(reloaded, pixels, selected, logits)
        deadline()
        write_json(arm_dir / "arm.json", arm)
        arms.append(arm)
    report = {
        "schema_version": "srnet-bn-refresh-control-v1",
        "status": "completed",
        "protocol_sha256": PROTOCOL_SHA,
        "protocol_commit": "71f3569",
        "baseline_report_sha256": BASELINE_SHA,
        "manifest_sha256": MANIFEST_SHA,
        "train_cache_sha256": CACHE_SHA,
        "train_data_sha256": descriptor["data_sha256"],
        "selected_rows": identities,
        "arms": arms,
        "execution_source_sha256": sources,
        "seconds": time.monotonic() - started,
        "optimizer_steps": 0,
        "validation_pixels_loaded": False,
        "in_sample_only": True,
        "accuracy_qualification": "unavailable",
        "deployed": False,
        "primary_detection_changed": False,
    }
    write_json(out / "control.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (820, 821))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        config, _ = read_document(args.config)
        report = run(config, args.model_dir, args.out)
    except Exception:
        print("failed/unavailable: no complete usable BN refresh control")
        return 2
    print("completed: train-only BN refresh; detector qualification unavailable")
    return (
        0
        if all(
            a["in_sample_objectives_passed"] and a["audit"]["numerical_gates_passed"]
            for a in report["arms"]
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
