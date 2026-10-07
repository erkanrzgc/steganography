"""Explicit frozen two-arm signal-strength learning control, not detection."""

import argparse
import hashlib
import time
from pathlib import Path

from core import srnet_model, srnet_sanity, srnet_signal
from steganography.research_features import read_document
from steganography.research_jpeg import write_json
from steganography.research_pixels import load_pixels
from steganography.research_srnet_sanity import CACHE_SHA, MANIFEST_SHA, configuration

PROTOCOL_SHA = "c1a067ba32ae56d19dffedfec8f20725b5b60e6726e0fb992264ec2b7327e0ca"


def run(config, out: Path):
    configuration(config)
    if config["manifest_sha256"] != MANIFEST_SHA or config["cache_sha256"] != CACHE_SHA:
        raise ValueError("signal control frozen input mismatch")
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("signal output must be fresh and non-symlink")
    root = Path(__file__).resolve().parents[1]
    protocol = root / "docs/SRNET_SIGNAL_STRENGTH_PROTOCOL.md"
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != PROTOCOL_SHA:
        raise ValueError("signal control frozen protocol changed")
    _, checksum = read_document(Path(config["manifest"]))
    if checksum != MANIFEST_SHA:
        raise ValueError("signal control manifest checksum mismatch")
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
    sources = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "core/srnet_signal.py",
            "core/srnet_training.py",
            "core/srnet.py",
            "core/srnet_multibatch.py",
            "core/srnet_sampling.py",
            "core/srnet_positive.py",
            "steganography/research_srnet_signal.py",
        )
    }
    out.mkdir(parents=True, exist_ok=False)
    arms, started = [], time.monotonic()
    for factor in (1, 32):
        if time.monotonic() - started > 3720:
            raise ValueError("signal control total deadline exceeded")
        model, arm = srnet_signal.learn(original, selected, factor=factor)
        arm_dir = out / f"factor-{factor}"
        arm_dir.mkdir(exist_ok=False)
        arm["model_sha256"] = srnet_model.save_model(model, arm_dir / "model.npz")
        reloaded = srnet_model.load_model(arm_dir / "model.npz", checksum=arm["model_sha256"])
        arm["audit"] = srnet_signal.audit(reloaded, original, selected, arm)
        if time.monotonic() - started > 3720:
            raise ValueError("signal control total deadline exceeded")
        write_json(arm_dir / "arm.json", arm)
        arms.append(arm)
    report = {
        "schema_version": "srnet-signal-strength-control-v1",
        "status": "completed",
        "protocol_sha256": PROTOCOL_SHA,
        "protocol_commit": "09e28da",
        "manifest_sha256": MANIFEST_SHA,
        "train_cache_sha256": CACHE_SHA,
        "train_data_sha256": descriptor["data_sha256"],
        "selected_rows": identities,
        "sampler_sha256_is_original_jpeg_identity_not_derived_tensor": True,
        "amplified_inputs_not_encoded_jpeg_or_actual_embedding": True,
        "execution_source_sha256": sources,
        "arms": arms,
        "seconds": time.monotonic() - started,
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (7600, 7601))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        config, _ = read_document(args.config)
        report = run(config, args.out)
    except Exception:
        print("failed/unavailable: no complete usable two-arm signal control")
        return 2
    print("completed: train-only signal control; detector qualification unavailable")
    return (
        0
        if all(
            a["learning_objectives_passed"] and a["audit"]["numerical_gates_passed"]
            for a in report["arms"]
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
