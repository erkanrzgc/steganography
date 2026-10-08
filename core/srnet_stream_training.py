"""Train-only full-block fitting through the unchanged numeric SRNet engine."""

from core import srnet_scale_sampling, srnet_training
from core.srnet_stream import TrainBlocks


def fit(reader, *, seed, schedule, config, deadline):
    if (
        not isinstance(reader, TrainBlocks)
        or not isinstance(schedule, list)
        or not 1 <= len(schedule) <= 50
        or not callable(deadline)
        or not isinstance(config, dict)
        or config.keys() - {"threads", "max_seconds", "learning_rate", "weight_decay"}
    ):
        raise ValueError("invalid bounded streaming fit inputs")
    params = srnet_training.settings(config)
    for epoch, record in enumerate(schedule):
        deadline()
        if srnet_scale_sampling.epoch_batches(reader.samples, seed=seed, epoch=epoch)[1] != record:
            raise ValueError("streaming fit schedule mismatch")
    return srnet_training._learn(
        fetch=reader.batch,
        batches=lambda epoch: srnet_scale_sampling.epoch_batches(
            reader.samples, seed=seed, epoch=epoch
        )[0],
        epochs=len(schedule),
        seed=seed,
        params=params,
        target_pairs=2,
        outer_deadline=deadline,
    )
