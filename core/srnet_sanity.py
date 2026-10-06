"""Train-only tiny-subset selection and in-sample learning objectives."""

import math

import numpy as np

from core.srnet_sampling import epoch_pairs


def select(samples):
    epoch_pairs(samples, seed=20261012, epoch=0)  # Validate all rows, including exclusions.
    if {s["source_group"] for s in samples} != {"ALASKA2", "BOSSbase-1.01"}:
        raise ValueError("tiny sanity requires both frozen declared sources")
    groups: dict[tuple[str, str], set[int | None]] = {}
    for row in samples:
        groups.setdefault((row["source_group"], row["lineage"]), set()).add(row["quality_factor"])
    alaska = sorted(
        k for k, qualities in groups.items() if k[0] == "ALASKA2" and qualities == {None}
    )[:4]
    boss = sorted(
        k for k, qualities in groups.items() if k[0] == "BOSSbase-1.01" and qualities == {75, 95}
    )[:2]
    if len(alaska) != 4 or len(boss) != 2:
        raise ValueError("tiny sanity requires four unknown-Q and two complete Q75/Q95 lineages")
    wanted = set(alaska + boss)
    indices = np.array(
        [i for i, s in enumerate(samples) if (s["source_group"], s["lineage"]) in wanted],
        dtype="<i8",
    )
    if len(indices) != 24:
        raise ValueError("tiny sanity requires exactly 24 complete train rows")
    indices.flags.writeable = False
    return indices


def metrics(logits, samples):
    if (
        not isinstance(logits, np.ndarray)
        or logits.dtype.kind != "f"
        or logits.shape != (len(samples), 2)
        or not 1 <= len(samples) <= 24
        or not np.isfinite(logits).all()
        or any(s["split"] != "train" or s["label"] not in {"cover", "stego"} for s in samples)
        or {s["label"] for s in samples} != {"cover", "stego"}
    ):
        raise ValueError("tiny sanity metrics require finite train-only logits")
    margins = logits[:, 1].astype(np.float64) - logits[:, 0]
    if not np.isfinite(margins).all():
        raise ValueError("tiny sanity metrics produced nonfinite margins")
    labels = np.array([s["label"] == "stego" for s in samples])
    losses = np.logaddexp(0, np.where(labels, -margins, margins))
    scores = np.exp(-np.logaddexp(0, -margins))
    decisions = margins >= 0
    return {
        "cross_entropy": float(losses.mean()),
        "balanced_accuracy": float(((decisions[labels]).mean() + (~decisions[~labels]).mean()) / 2),
        "recall": float(decisions[labels].mean()),
        "false_positive_rate": float(decisions[~labels].mean()),
        "scores": scores.tolist(),
    }


def objectives(records, final):
    if (
        type(final.get("balanced_accuracy")) not in (int, float)
        or not 0 <= final["balanced_accuracy"] <= 1
    ):
        raise ValueError("tiny sanity requires finite balanced accuracy")
    if len(records) != 50 or any(
        type(r.get("epoch")) is not int
        or type(r.get("updates")) is not int
        or r.get("epoch") != i
        or r.get("updates") != 8
        or type(r.get("mean_pair_loss")) not in (int, float)
        or not math.isfinite(r["mean_pair_loss"])
        or r["mean_pair_loss"] < 0
        for i, r in enumerate(records)
    ):
        raise ValueError("tiny sanity requires 50 complete finite epochs")
    first, last = records[0]["mean_pair_loss"], records[-1]["mean_pair_loss"]
    reduction = (first - last) / first if first else 0.0
    return {
        "final_batch_loss_at_most_0_35": last <= 0.35,
        "relative_loss_reduction_at_least_0_25": reduction >= 0.25,
        "stored_bn_train_balanced_accuracy_at_least_0_90": final["balanced_accuracy"] >= 0.90,
        "relative_loss_reduction": reduction,
    }
