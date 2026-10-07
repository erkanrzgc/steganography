"""Generated strong-signal learning control, never steganalysis qualification."""

import hashlib
import math
import time

import numpy as np

from core import srnet, srnet_multibatch, srnet_sanity, srnet_training
from core.srnet_sampling import epoch_pairs

PROTOCOL_SHA = "5f34d707ec8c575359ff0d5c42d38a5791c583dbb02b27185589a4f1a1109ab5"
PARAMS = {"threads": 2, "max_seconds": 1800, "learning_rate": 0.001, "weight_decay": 0.0001}


def generate():
    rng = np.random.Generator(np.random.PCG64(20261013))
    checker = (2 * (np.indices((256, 256)).sum(axis=0) % 2) - 1).astype("<f4")
    pixels, samples = [], []
    for source in range(2):
        for lineage in range(4):
            cover = (96 + 16 * source + 4 * lineage + rng.uniform(-1, 1, (256, 256))).astype("<f4")
            for method, amplitude in ((None, 0), ("JUNIWARD", 48), ("UERD", 56)):
                values = (cover + amplitude * checker).astype("<f4")
                pixels.append(values[None])
                samples.append(
                    {
                        "split": "train",
                        "format": "JPEG",
                        "source_group": f"generated-{source}",
                        "lineage": f"generated-{source}-{lineage}",
                        "quality_factor": None,
                        "method": method,
                        "label": "cover" if method is None else "stego",
                        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                    }
                )
    result = np.array(pixels, dtype="<f4")
    result.flags.writeable = False
    return result, samples


def objectives(records, final):
    if (
        len(records) != 20
        or any(
            type(r.get("epoch")) is not int
            or r["epoch"] != i
            or type(r.get("updates")) is not int
            or r["updates"] != 8
            or type(r.get("mean_pair_loss")) not in (float, int)
            or not math.isfinite(r["mean_pair_loss"])
            or r["mean_pair_loss"] < 0
            for i, r in enumerate(records)
        )
        or type(final.get("balanced_accuracy")) not in (float, int)
        or not 0 <= final["balanced_accuracy"] <= 1
    ):
        raise ValueError("positive control requires complete finite learning records")
    first, last = records[0]["mean_pair_loss"], records[-1]["mean_pair_loss"]
    reduction = (first - last) / first if first else 0.0
    return {
        "final_batch_loss_at_most_0_35": last <= 0.35,
        "relative_loss_reduction_at_least_0_25": reduction >= 0.25,
        "stored_bn_train_balanced_accuracy_at_least_0_90": final["balanced_accuracy"] >= 0.90,
        "relative_loss_reduction": reduction,
    }


def learn():
    pixels, samples = generate()
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
        config={**PARAMS, "batch_recipe": srnet_multibatch.RECIPE},
    )
    if any(int(v) != 160 for k, v in model.named_buffers() if k.endswith("num_batches_tracked")):
        raise ValueError("positive control BN update accounting mismatch")
    import torch

    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        rows_logits = []
        for row in pixels:
            if time.monotonic() - started > 1860:
                raise ValueError("positive control evaluation deadline exceeded")
            rows_logits.append(srnet.float_logits(model, row[None])[0])
    finally:
        torch.set_num_threads(previous)
    logits = np.array(rows_logits)
    final = srnet_sanity.metrics(logits, samples)
    gates = objectives(records, final)
    return model, {
        "schema_version": "srnet-generated-positive-control-v1",
        "status": "completed",
        "protocol_sha256": PROTOCOL_SHA,
        "protocol_commit": "ebd8916",
        "generated_tensors_not_jpeg_or_steganography": True,
        "sampler_format_and_method_tags_are_synthetic": True,
        "generated_seed": 20261013,
        "training_seed": 20261012,
        "selected_rows": samples,
        "optimizer": PARAMS.copy(),
        "optimizer_updates": 160,
        "epoch_pair_schedule": paired,
        "epoch_batch_schedule": batched,
        "epoch_training": records,
        "stored_bn_singleton_train_logits": logits.tolist(),
        "stored_bn_singleton_train_metrics": final,
        "sanity_objectives": gates,
        "sanity_objectives_passed": all(
            v for k, v in gates.items() if k != "relative_loss_reduction"
        ),
        "seconds": time.monotonic() - started,
        "real_dataset_loaded": False,
        "validation_pixels_loaded": False,
        "in_sample_only": True,
        "accuracy_qualification": "unavailable",
        "deployed": False,
        "primary_detection_changed": False,
    }
