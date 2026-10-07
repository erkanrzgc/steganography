"""Explicit frozen eight-row context learning control, not detector qualification."""

import argparse
import hashlib
import time
from pathlib import Path

from core import srnet_model, srnet_sanity, srnet_widebatch
from steganography.research_features import read_document
from steganography.research_jpeg import write_json
from steganography.research_pixels import load_pixels
from steganography.research_srnet_sanity import CACHE_SHA, MANIFEST_SHA, configuration

PROTOCOL_SHA = "f4aede48cb279a2745c0979b20615f07125bfb8193a0047da00f023d95b7fe9d"
BASELINE_SHA = "a7ce1b3f61a22accb45b73ae8d69ebd3e47e3ee53b428a7e49aabc32d67666cf"


def run(config, out: Path, *, _accumulation=False):
    if type(_accumulation) is not bool:
        raise ValueError("accumulation must be explicit boolean")
    protocol_name, protocol_sha, protocol_commit = (
        "SRNET_WIDE_BATCH_PROTOCOL.md",
        PROTOCOL_SHA,
        "c78a9eb",
    )
    baseline_name, baseline_sha = "srnet-signal-strength-20261007.json", BASELINE_SHA
    if _accumulation:
        from steganography import research_srnet_accumulation as control

        protocol_name, protocol_sha, protocol_commit = (
            "SRNET_ACCUMULATION_PROTOCOL.md",
            control.PROTOCOL_SHA,
            "47217aa",
        )
        baseline_name, baseline_sha = "srnet-wide-batch-20261007.json", control.BASELINE_SHA
    configuration(config)
    if config["manifest_sha256"] != MANIFEST_SHA or config["cache_sha256"] != CACHE_SHA:
        raise ValueError("wide control frozen input mismatch")
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("wide output must be fresh and non-symlink")
    root = Path(__file__).resolve().parents[1]
    if hashlib.sha256((root / "docs" / protocol_name).read_bytes()).hexdigest() != protocol_sha:
        raise ValueError("wide control frozen protocol mismatch")
    baseline, checksum = read_document(root / "benchmarks" / baseline_name)
    if checksum != baseline_sha:
        raise ValueError("wide control baseline evidence mismatch")
    _, checksum = read_document(Path(config["manifest"]))
    if checksum != MANIFEST_SHA:
        raise ValueError("wide control manifest checksum mismatch")
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
        raise ValueError("wide control selected training identity mismatch")
    sources = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            *(
                "core/srnet_widebatch.py",
                "core/srnet_training.py",
                "core/srnet.py",
                "core/srnet_model.py",
                "core/srnet_signal.py",
                "core/srnet_multibatch.py",
                "core/srnet_sampling.py",
                "core/srnet_positive.py",
                "core/srnet_reference.py",
                "steganography/research_srnet_widebatch.py",
            ),
            *(("steganography/research_srnet_accumulation.py",) if _accumulation else ()),
        )
    }
    out.mkdir(parents=True, exist_ok=False)
    arms, started = [], time.monotonic()

    def deadline():
        if time.monotonic() - started > 3720:
            raise ValueError("wide control total deadline exceeded")

    for base in baseline["arms"]:
        deadline()
        if _accumulation:
            from core import srnet_signal

            _, hashes = srnet_signal.prepare(original, selected, factor=base["factor"])
            expected = [
                srnet_widebatch.epoch_batches(selected, seed=20261012, epoch=e)[1]
                for e in range(20)
            ]
            if (
                hashes != base["derived_tensor_sha256"]
                or expected != base["epoch_batch_schedule"]
                or base["optimizer_updates"] != 80
                or base["presented_rows"] != 640
            ):
                raise ValueError("accumulation baseline tensor/order/exposure mismatch")
        model, arm = srnet_widebatch.learn(
            original, selected, factor=base["factor"], accumulation=_accumulation
        )
        if arm["derived_tensor_sha256"] != base["derived_tensor_sha256"]:
            raise ValueError("wide control derived tensor identity mismatch")
        arm["baseline_own_metrics"] = base["own_input_train_metrics"]
        arm["baseline_original_metrics"] = base["original_input_train_metrics"]
        arm_dir = out / f"factor-{arm['factor']}"
        arm_dir.mkdir(exist_ok=False)
        arm["model_sha256"] = srnet_model.save_model(model, arm_dir / "model.npz")
        reloaded = srnet_model.load_model(arm_dir / "model.npz", checksum=arm["model_sha256"])
        arm["audit"] = srnet_widebatch.audit(
            reloaded, original, selected, arm, accumulation=_accumulation
        )
        deadline()
        write_json(arm_dir / "arm.json", arm)
        arms.append(arm)
    report = {
        "schema_version": "srnet-accumulation-control-v1"
        if _accumulation
        else "srnet-wide-batch-control-v1",
        "status": "completed",
        "protocol_sha256": protocol_sha,
        "protocol_commit": protocol_commit,
        "baseline_report_sha256": baseline_sha,
        "manifest_sha256": MANIFEST_SHA,
        "train_cache_sha256": CACHE_SHA,
        "train_data_sha256": descriptor["data_sha256"],
        "selected_rows": identities,
        "execution_source_sha256": sources,
        "arms": arms,
        "seconds": time.monotonic() - started,
        "row_exposure_matched": True,
        "optimizer_steps_matched": _accumulation,
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
        print("failed/unavailable: no complete usable eight-row control")
        return 2
    print("completed: train-only eight-row control; detector qualification unavailable")
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
