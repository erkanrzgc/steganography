"""Complete four-row batches from two distinct declared source/lineage pairs."""

from __future__ import annotations

import hashlib
import json

from core.srnet_sampling import epoch_pairs

RECIPE = "two-source-two-lineage-pairs-v1"


def recipe(config):
    value = config.get("batch_recipe")
    if "batch_recipe" in config and value != RECIPE:
        raise ValueError("unsupported SRNet batch recipe")
    return value


def epoch_batches(samples, *, seed, epoch):
    pairs, _ = epoch_pairs(samples, seed=seed, epoch=epoch)
    return _group_pairs(samples, pairs, epoch=epoch)


def _group_pairs(samples, pairs, *, epoch):
    if len({s["source_group"] for s in samples}) != 2 or len(pairs) % 2:
        raise ValueError("SRNet multipair batches require exactly two complete sources")
    batches = pairs.reshape(-1, 4).copy()
    for c0, _s0, c1, _s1 in batches:
        first, second = samples[c0], samples[c1]
        if (
            first["source_group"] == second["source_group"]
            or first["sha256"] == second["sha256"]
            or (first["source_group"], first["lineage"])
            == (second["source_group"], second["lineage"])
        ):
            raise ValueError("SRNet multipair batches require distinct declared source lineages")
    identities = [[samples[i]["sha256"] for i in batch] for batch in batches]
    batches.flags.writeable = False
    return batches, {
        "recipe": RECIPE,
        "epoch": epoch,
        "batch_size": 4,
        "pairs": len(pairs),
        "optimizer_updates": len(batches),
        "rows": 4 * len(batches),
        "ordered_batch_sha256": hashlib.sha256(
            json.dumps(identities, separators=(",", ":")).encode()
        ).hexdigest(),
        "independence": (
            "distinct declared source/lineage; camera/perceptual independence unverified"
        ),
    }


def updates(plan):
    if recipe(plan.get("settings", {})) is None:
        return [e["pairs"] for e in plan["epochs"]]
    return [e["optimizer_updates"] for e in plan["epoch_batch_schedule"]]
