"""Artificial real-delta amplification learning controls, never detector accuracy."""

import hashlib
import time

import numpy as np

from core import (
    srnet,
    srnet_multibatch,
    srnet_positive,
    srnet_reference,
    srnet_sanity,
    srnet_training,
)
from core.srnet_sampling import epoch_pairs


def prepare(pixels, samples, *, factor):
    if (
        type(factor) is not int
        or factor not in (1, 32)
        or not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.dtype("<f4")
        or pixels.shape != (24, 1, 256, 256)
        or not np.isfinite(pixels).all()
        or np.any(np.abs(pixels) > 2**36)
    ):
        raise ValueError("signal control requires 24 bounded float256 tensors and frozen factor")
    epoch_pairs(samples, seed=20261012, epoch=0)
    if len(samples) != 24:
        raise ValueError("signal control metadata/tensor cardinality mismatch")
    covers = {
        (s["source_group"], s["lineage"], s["quality_factor"]): i
        for i, s in enumerate(samples)
        if s["label"] == "cover"
    }
    result = pixels.copy()
    for i, s in enumerate(samples):
        if s["label"] == "stego":
            cover = pixels[covers[(s["source_group"], s["lineage"], s["quality_factor"])]].astype(
                np.float64
            )
            result[i] = cover + factor * (pixels[i].astype(np.float64) - cover)
    if not np.isfinite(result).all() or np.any(np.abs(result) > 2**36):
        raise ValueError("amplified signal tensor outside finite bounds")
    result.flags.writeable = False
    return result, [hashlib.sha256(p.tobytes()).hexdigest() for p in result]


def evaluate(model, pixels, samples):
    import torch

    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        logits = np.array([srnet.float_logits(model, p[None])[0] for p in pixels])
    finally:
        torch.set_num_threads(previous)
    return logits, srnet_sanity.metrics(logits, samples)


def learn(original, samples, *, factor):
    pixels, hashes = prepare(original, samples, factor=factor)
    paired = [epoch_pairs(samples, seed=20261012, epoch=e)[1] for e in range(20)]
    batched = [
        srnet_multibatch.epoch_batches(samples, seed=20261012, epoch=e)[1] for e in range(20)
    ]
    started = time.monotonic()
    model, records = srnet_training.fit(
        pixels,
        samples,
        np.arange(24, dtype="<i8"),
        seed=20261012,
        schedule=paired,
        config={**srnet_positive.PARAMS, "batch_recipe": srnet_multibatch.RECIPE},
    )
    if any(int(v) != 160 for k, v in model.named_buffers() if k.endswith("num_batches_tracked")):
        raise ValueError("signal-control BN accounting mismatch")
    logits, metrics = evaluate(model, pixels, samples)
    original_logits, original_metrics = evaluate(model, original, samples)
    gates = srnet_positive.objectives(records, metrics)
    return model, {
        "factor": factor,
        "optimizer": srnet_positive.PARAMS.copy(),
        "seed": 20261012,
        "optimizer_updates": 160,
        "derived_tensor_sha256": hashes,
        "epoch_pair_schedule": paired,
        "epoch_batch_schedule": batched,
        "epoch_training": records,
        "own_input_singleton_logits": logits.tolist(),
        "own_input_train_metrics": metrics,
        "original_input_singleton_logits": original_logits.tolist(),
        "original_input_train_metrics": original_metrics,
        "learning_objectives": gates,
        "learning_objectives_passed": all(
            v for k, v in gates.items() if k != "relative_loss_reduction"
        ),
        "seconds": time.monotonic() - started,
    }


def audit(model, original, samples, report):
    pixels, hashes = prepare(original, samples, factor=report["factor"])
    if (
        hashes != report["derived_tensor_sha256"]
        or report["epoch_pair_schedule"]
        != [epoch_pairs(samples, seed=20261012, epoch=e)[1] for e in range(20)]
        or report["epoch_batch_schedule"]
        != [srnet_multibatch.epoch_batches(samples, seed=20261012, epoch=e)[1] for e in range(20)]
        or report["optimizer"] != srnet_positive.PARAMS
        or report["seed"] != 20261012
        or report["optimizer_updates"] != 160
    ):
        raise ValueError("signal-control content/schedule/optimizer mismatch")
    if any(int(v) != 160 for k, v in model.named_buffers() if k.endswith("num_batches_tracked")):
        raise ValueError("signal-control reloaded BN accounting mismatch")
    own_logits, own = evaluate(model, pixels, samples)
    original_logits, original_metrics = evaluate(model, original, samples)
    for logits, metrics, prefix in (
        (own_logits, own, "own_input"),
        (original_logits, original_metrics, "original_input"),
    ):
        stored = np.asarray(report[prefix + "_singleton_logits"], dtype=np.float64)
        if stored.shape != (24, 2) or not np.allclose(logits, stored, atol=1e-6, rtol=0):
            raise ValueError("signal-control singleton logits replay mismatch")
        if set(metrics) != set(report[prefix + "_train_metrics"]) or any(
            not np.allclose(v, report[prefix + "_train_metrics"][k], atol=1e-6, rtol=0)
            for k, v in metrics.items()
        ):
            raise ValueError("signal-control singleton metric replay mismatch")
    gates = srnet_positive.objectives(report["epoch_training"], own)
    if gates != report["learning_objectives"] or report["learning_objectives_passed"] is not all(
        v for k, v in gates.items() if k != "relative_loss_reduction"
    ):
        raise ValueError("signal-control objective replay mismatch")
    selected, seen = [], set()
    for i, s in enumerate(samples):
        key = (s["source_group"], s["quality_factor"], s["label"], s["method"])
        if key not in seen:
            selected.append(i)
            seen.add(key)
    if len(selected) != 9:
        raise ValueError("signal-control independent oracle coverage incomplete")
    arrays = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    oracles = [
        {
            "row": i,
            "source_sha256": samples[i]["sha256"],
            "derived_tensor_sha256": hashes[i],
            **srnet_reference.compare(
                own_logits[i : i + 1], srnet_reference.reference_logits(arrays, pixels[i : i + 1])
            ),
        }
        for i in selected
    ]
    return {
        "all_own_and_original_singletons_replayed": True,
        "independent_oracles": oracles,
        "numerical_gates_passed": all(r["passed"] for r in oracles),
    }
