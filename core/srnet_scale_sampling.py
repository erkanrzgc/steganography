"""Versioned full-block sampling; legacy 4,000-row sampling stays unchanged."""

from core import srnet_multibatch, srnet_sampling

MAX_ROWS = 12_000
RECIPE = "srnet-block-source-quality-method-four-row-v1"


def epoch_batches(samples, *, seed, epoch):
    pairs, paired = srnet_sampling._epoch_pairs(samples, seed=seed, epoch=epoch, row_limit=MAX_ROWS)
    batches, grouped = srnet_multibatch._group_pairs(samples, pairs, epoch=epoch)
    return batches, {"recipe": RECIPE, "pair_schedule": paired, "batch_schedule": grouped}
