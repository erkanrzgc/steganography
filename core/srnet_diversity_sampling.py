"""Opt-in three-origin schedule; no change to historical two-source fitting."""

from __future__ import annotations

import hashlib
import json

from core import srnet_sampling

# 7,380 previous train rows + 807 BOWS originals * six derivatives = 12,222.
# This metadata-only bound does not increase pixel-reader or worker limits.
MAX_ROWS = 15_000
RECIPE = "srnet-three-source-quality-method-four-row-v1"


def epoch_batches(samples, *, seed, epoch):
    """Exhaust all matched pairs, balancing sources then source-local Q/method.

    The shared sampler cycles three distinct origins. Grouping consecutive
    pairs gives AB, CA, BC four-row batches, without dropping the third origin
    or requiring a six-row GPU batch. Each source has an even quota because
    every quality context contains both method families.
    """
    pairs, paired = srnet_sampling._epoch_pairs(samples, seed=seed, epoch=epoch, row_limit=MAX_ROWS)
    if len({s["source_group"] for s in samples}) != 3 or len(pairs) % 6:
        raise ValueError("diversity sampling requires exactly three complete sources")
    owners: dict[str, str] = {}
    for sample in samples:
        source, lineage = sample["source_group"], sample["lineage"]
        if owners.setdefault(lineage, source) != source:
            raise ValueError("diversity sampling rejects cross-source original lineage overlap")
    batches = pairs.reshape(-1, 4).copy()
    for c0, _s0, c1, _s1 in batches:
        first, second = samples[c0], samples[c1]
        if first["source_group"] == second["source_group"]:
            raise ValueError("diversity batches require distinct declared sources")
    identities = [[samples[i]["sha256"] for i in batch] for batch in batches]
    batches.flags.writeable = False
    return batches, {
        "recipe": RECIPE,
        "pair_schedule": paired,
        "batch_schedule": {
            "recipe": RECIPE,
            "epoch": epoch,
            "batch_size": 4,
            "sources": 3,
            "pairs": len(pairs),
            "optimizer_updates": len(batches),
            "rows": 4 * len(batches),
            "ordered_batch_sha256": hashlib.sha256(
                json.dumps(identities, separators=(",", ":")).encode()
            ).hexdigest(),
            "independence": (
                "distinct declared sources and original lineages; "
                "camera/perceptual independence unverified"
            ),
        },
    }
