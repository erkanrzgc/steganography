"""Read-only generated-control replay; learning and numerical gates stay separate."""

import hashlib
from pathlib import Path

import numpy as np

from core import srnet, srnet_model, srnet_multibatch, srnet_positive, srnet_reference, srnet_sanity
from core.srnet_sampling import epoch_pairs

# Executed source retained in Git; the later list/ndarray variable rename is
# type-checking only. Do not relabel the historical report with today's hash.
HISTORICAL_POSITIVE_SHA = "03b710bd47d5a2761447019b0ca2d3f99175a759ed1d5f721863a46fbb87e430"


def audit(report, result_dir: Path, root: Path):
    if (
        report["schema_version"] != "srnet-generated-positive-control-v1"
        or report["status"] != "completed"
        or report["protocol_sha256"] != srnet_positive.PROTOCOL_SHA
        or report["protocol_commit"] != "ebd8916"
        or report["generated_seed"] != 20261013
        or report["training_seed"] != 20261012
        or report["optimizer"] != srnet_positive.PARAMS
        or report["optimizer_updates"] != 160
        or any(
            report[k] is not v
            for k, v in {
                "generated_tensors_not_jpeg_or_steganography": True,
                "sampler_format_and_method_tags_are_synthetic": True,
                "real_dataset_loaded": False,
                "validation_pixels_loaded": False,
                "in_sample_only": True,
                "deployed": False,
                "primary_detection_changed": False,
            }.items()
        )
        or report["accuracy_qualification"] != "unavailable"
    ):
        raise ValueError("generated control provenance mismatch")
    protocol = root / "docs/SRNET_POSITIVE_CONTROL_PROTOCOL.md"
    if hashlib.sha256(protocol.read_bytes()).hexdigest() != srnet_positive.PROTOCOL_SHA:
        raise ValueError("generated control protocol mismatch")
    expected_sources = {
        "core/srnet_positive.py",
        "core/srnet_training.py",
        "core/srnet.py",
        "core/srnet_multibatch.py",
        "core/srnet_sampling.py",
        "steganography/research_srnet_positive.py",
    }
    if set(report["execution_source_sha256"]) != expected_sources or any(
        hashlib.sha256((root / name).read_bytes()).hexdigest() != checksum
        and not (name == "core/srnet_positive.py" and checksum == HISTORICAL_POSITIVE_SHA)
        for name, checksum in report["execution_source_sha256"].items()
    ):
        raise ValueError("generated control execution source mismatch")
    pixels, samples = srnet_positive.generate()
    if (
        report["selected_rows"] != samples
        or report["epoch_pair_schedule"]
        != [epoch_pairs(samples, seed=20261012, epoch=e)[1] for e in range(20)]
        or report["epoch_batch_schedule"]
        != [srnet_multibatch.epoch_batches(samples, seed=20261012, epoch=e)[1] for e in range(20)]
    ):
        raise ValueError("generated control content/schedule mismatch")
    model = srnet_model.load_model(result_dir / "model.npz", checksum=report["model_sha256"])
    arrays = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    if any(int(v) != 160 for k, v in arrays.items() if k.endswith("num_batches_tracked")):
        raise ValueError("generated control BN accounting mismatch")
    import torch

    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        logits = np.array([srnet.float_logits(model, p[None])[0] for p in pixels])
        stored = np.asarray(report["stored_bn_singleton_train_logits"], dtype=np.float64)
        if stored.shape != (24, 2) or not np.allclose(logits, stored, atol=1e-6, rtol=0):
            raise ValueError("generated control saved model logits mismatch")
        metrics = srnet_sanity.metrics(logits, samples)
        if set(metrics) != set(report["stored_bn_singleton_train_metrics"]) or any(
            not np.allclose(v, report["stored_bn_singleton_train_metrics"][k], atol=1e-6, rtol=0)
            for k, v in metrics.items()
        ):
            raise ValueError("generated control saved model metrics mismatch")
        gates = srnet_positive.objectives(report["epoch_training"], metrics)
        if gates != report["sanity_objectives"] or report["sanity_objectives_passed"] is not all(
            v for k, v in gates.items() if k != "relative_loss_reduction"
        ):
            raise ValueError("generated control objectives mismatch")
        oracles = [
            {
                "row": i,
                "sha256": samples[i]["sha256"],
                **srnet_reference.compare(
                    logits[i : i + 1], srnet_reference.reference_logits(arrays, pixels[i : i + 1])
                ),
            }
            for i in (0, 1, 2, 12, 13, 14)
        ]
    finally:
        torch.set_num_threads(previous)
    return {
        "schema_version": "srnet-generated-positive-audit-v1",
        "model_sha256": report["model_sha256"],
        "singleton_rows_replayed": 24,
        "independent_oracles": oracles,
        "numerical_gates_passed": all(r["passed"] for r in oracles),
        "learning_objectives_passed": report["sanity_objectives_passed"],
        "accuracy_qualification": "unavailable",
        "primary_detection_changed": False,
    }
