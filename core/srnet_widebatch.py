"""Frozen, exposure-matched eight-row learning control; never detector inference."""

import hashlib
import json
import math
import time

import numpy as np

from core import srnet_multibatch, srnet_positive, srnet_reference, srnet_signal
from core.srnet_sampling import epoch_pairs

RECIPE = "two-source-four-distinct-lineages-eight-rows-v1"


def epoch_batches(samples, *, seed, epoch):
    if len(samples) != 24:
        raise ValueError("wide control requires exactly 24 training rows")
    groups, base = srnet_multibatch.epoch_batches(samples, seed=seed, epoch=epoch)
    if groups.shape != (8, 4):
        raise ValueError("wide control requires exactly eight complete base groups")
    keys = [{(samples[i]["source_group"], samples[i]["lineage"]) for i in g[::2]} for g in groups]

    def match(remaining):
        if not remaining:
            return []
        first = remaining[0]
        for partner in remaining[1:]:
            if keys[first].isdisjoint(keys[partner]):
                tail = match([i for i in remaining[1:] if i != partner])
                if tail is not None:
                    return [(first, partner), *tail]
        return None

    pairing = match(list(range(8)))
    if pairing is None:
        raise ValueError("wide control cannot form complete distinct-lineage batches")
    values = np.asarray([np.concatenate((groups[a], groups[b])) for a, b in pairing], dtype="<i8")
    identities = [[samples[i]["sha256"] for i in batch] for batch in values]
    values.flags.writeable = False
    return values, {
        "recipe": RECIPE,
        "epoch": epoch,
        "batch_size": 8,
        "pairs": 16,
        "optimizer_updates": 4,
        "rows": 32,
        "base_group_pairing": [list(pair) for pair in pairing],
        "base_batch_sha256": base["ordered_batch_sha256"],
        "ordered_batch_sha256": hashlib.sha256(
            json.dumps(identities, separators=(",", ":")).encode()
        ).hexdigest(),
        "independence": (
            "distinct declared source/lineage; camera/perceptual independence unverified"
        ),
    }


def objectives(records, final):
    if (
        not isinstance(records, list)
        or len(records) != 20
        or any(
            not isinstance(r, dict)
            or type(r.get("epoch")) is not int
            or r["epoch"] != i
            or type(r.get("updates")) is not int
            or r["updates"] != 4
            or type(r.get("mean_pair_loss")) not in (int, float)
            or not math.isfinite(r["mean_pair_loss"])
            or r["mean_pair_loss"] < 0
            for i, r in enumerate(records)
        )
    ):
        raise ValueError("wide control requires complete finite 80-update records")
    # Reuse the unchanged learning thresholds; only the explicit update count differs.
    return srnet_positive.objectives([{**r, "updates": 8} for r in records], final)


def counters(model):
    found = [int(v) for k, v in model.named_buffers() if k.endswith("num_batches_tracked")]
    if found != [80] * 26:
        raise ValueError("wide control BN accounting mismatch")


def learn(original, samples, *, factor):
    from core import srnet_training

    pixels, hashes = srnet_signal.prepare(original, samples, factor=factor)
    paired = [epoch_pairs(samples, seed=20261012, epoch=e)[1] for e in range(20)]
    batched = [epoch_batches(samples, seed=20261012, epoch=e)[1] for e in range(20)]
    started = time.monotonic()
    model, records = srnet_training.fit(
        pixels,
        samples,
        np.arange(24, dtype="<i8"),
        seed=20261012,
        schedule=paired,
        config={**srnet_positive.PARAMS, "batch_recipe": srnet_multibatch.RECIPE},
        wide_context=True,
    )
    counters(model)
    logits, own = srnet_signal.evaluate(model, pixels, samples)
    original_logits, original_metrics = srnet_signal.evaluate(model, original, samples)
    gates = objectives(records, own)
    return model, {
        "factor": factor,
        "optimizer": srnet_positive.PARAMS.copy(),
        "seed": 20261012,
        "batch_recipe": RECIPE,
        "optimizer_updates": 80,
        "presented_rows": 640,
        "baseline_optimizer_updates": 160,
        "baseline_presented_rows": 640,
        "optimizer_steps_matched": False,
        "derived_tensor_sha256": hashes,
        "epoch_pair_schedule": paired,
        "epoch_batch_schedule": batched,
        "epoch_training": records,
        "own_input_singleton_logits": logits.tolist(),
        "own_input_train_metrics": own,
        "original_input_singleton_logits": original_logits.tolist(),
        "original_input_train_metrics": original_metrics,
        "learning_objectives": gates,
        "learning_objectives_passed": all(
            v for k, v in gates.items() if k != "relative_loss_reduction"
        ),
        "seconds": time.monotonic() - started,
    }


def audit(model, original, samples, report):
    pixels, hashes = srnet_signal.prepare(original, samples, factor=report["factor"])
    if (
        hashes != report["derived_tensor_sha256"]
        or report["epoch_pair_schedule"]
        != [epoch_pairs(samples, seed=20261012, epoch=e)[1] for e in range(20)]
        or report["epoch_batch_schedule"]
        != [epoch_batches(samples, seed=20261012, epoch=e)[1] for e in range(20)]
        or report["optimizer"] != srnet_positive.PARAMS
        or report["seed"] != 20261012
        or report["batch_recipe"] != RECIPE
        or report["optimizer_updates"] != 80
        or report["presented_rows"] != 640
        or report["baseline_optimizer_updates"] != 160
        or report["baseline_presented_rows"] != 640
        or report["optimizer_steps_matched"] is not False
    ):
        raise ValueError("wide control content/schedule/exposure mismatch")
    counters(model)
    own_logits, own = srnet_signal.evaluate(model, pixels, samples)
    original_logits, original_metrics = srnet_signal.evaluate(model, original, samples)
    for logits, metrics, prefix in (
        (own_logits, own, "own_input"),
        (original_logits, original_metrics, "original_input"),
    ):
        stored = np.asarray(report[prefix + "_singleton_logits"], dtype=np.float64)
        if (
            stored.shape != (24, 2)
            or not np.allclose(logits, stored, atol=1e-6, rtol=0)
            or set(metrics) != set(report[prefix + "_train_metrics"])
            or any(
                not np.allclose(v, report[prefix + "_train_metrics"][k], atol=1e-6, rtol=0)
                for k, v in metrics.items()
            )
        ):
            raise ValueError("wide control singleton replay mismatch")
    gates = objectives(report["epoch_training"], own)
    if gates != report["learning_objectives"] or report["learning_objectives_passed"] is not all(
        v for k, v in gates.items() if k != "relative_loss_reduction"
    ):
        raise ValueError("wide control objective replay mismatch")
    selected, seen = [], set()
    for i, s in enumerate(samples):
        key = (s["source_group"], s["quality_factor"], s["label"], s["method"])
        if key not in seen:
            selected.append(i)
            seen.add(key)
    if len(selected) != 9:
        raise ValueError("wide control independent oracle coverage incomplete")
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
