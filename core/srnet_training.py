"""Optional CPU paired SRNet learning; no validation, calibration or deployment."""

from __future__ import annotations

import math
import time

import numpy as np

from core import srnet, srnet_model, srnet_multibatch
from core.srnet_sampling import epoch_pairs


def settings(config):
    values = {"threads": config.get("threads", 2), "max_seconds": config.get("max_seconds", 1800)}
    for key, high in (("threads", 2), ("max_seconds", 1800)):
        if type(values[key]) is not int or not 1 <= values[key] <= high:
            raise ValueError("SRNet CPU setting outside limits")
    for key, default, optimizer_low, optimizer_high in (
        ("learning_rate", 0.001, 1e-6, 0.01),
        ("weight_decay", 0.0001, 0, 0.1),
    ):
        value = config.get(key, default)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not optimizer_low <= value <= optimizer_high
        ):
            raise ValueError("SRNet optimizer setting outside limits")
        values[key] = float(value)
    return values


def fit(pixels, samples, indices, *, seed, schedule, config, wide_context=False):
    params = settings(config)
    batch_recipe = srnet_multibatch.recipe(config)
    if (
        not isinstance(samples, list)
        or not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.dtype("<f4")
        or pixels.shape != (len(samples), 1, 256, 256)
        or not 1 <= len(samples) <= 4000
        or not isinstance(indices, np.ndarray)
        or indices.dtype.kind not in "iu"
        or indices.ndim != 1
        or not 1 <= len(indices) <= len(samples)
        or len(set(indices.tolist())) != len(indices)
        or np.any(indices >= len(samples))
        or np.any(indices < 0)
        or not isinstance(schedule, list)
        or not 1 <= len(schedule) <= 50
    ):
        raise ValueError("invalid bounded SRNet training inputs")
    for first in range(0, len(pixels), 4):
        batch = pixels[first : first + 4]
        if not np.isfinite(batch).all() or np.any(np.abs(batch) > 2**36):
            raise ValueError("invalid SRNet training pixel values")
    epoch_pairs(samples, seed=seed, epoch=0)  # Excluded rows cannot hide malformed pairs.
    selected = [samples[i] for i in indices]
    if type(wide_context) is not bool or (
        wide_context
        and (batch_recipe is None or len(selected) != 24 or seed != 20261012 or len(schedule) != 20)
    ):
        raise ValueError("wide context is restricted to the frozen research control")
    batch_builder = srnet_multibatch.epoch_batches
    if wide_context:
        from core import srnet_widebatch

        batch_builder = srnet_widebatch.epoch_batches
    for epoch, record in enumerate(schedule):
        if epoch_pairs(selected, seed=seed, epoch=epoch)[1] != record:
            raise ValueError("SRNet training schedule mismatch")
        if batch_recipe is not None:
            batch_builder(selected, seed=seed, epoch=epoch)
    import torch

    started = time.monotonic()
    previous = torch.get_num_threads()
    records = []

    def deadline():
        if time.monotonic() - started > params["max_seconds"]:
            raise ValueError("SRNet training deadline exceeded")

    try:
        torch.set_num_threads(params["threads"])
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = srnet.network().train()
            optimizer = torch.optim.Adamax(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
                betas=(0.9, 0.999),
                eps=1e-8,
                foreach=False,
            )
            targets = torch.tensor(
                [0, 1] * (4 if wide_context else 2 if batch_recipe is not None else 1),
                dtype=torch.int64,
                device="cpu",
            )
            for epoch in range(len(schedule)):
                pairs, _ = epoch_pairs(selected, seed=seed, epoch=epoch)
                if batch_recipe is not None:
                    pairs, _ = batch_builder(selected, seed=seed, epoch=epoch)
                total = 0.0
                for pair in pairs:
                    deadline()
                    inputs = torch.from_numpy(pixels[indices[pair]].copy())
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(inputs)
                    if logits.shape != (len(targets), 2) or not bool(torch.isfinite(logits).all()):
                        raise ValueError("SRNet training produced invalid logits")
                    loss = torch.nn.functional.cross_entropy(logits, targets)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("SRNet training produced nonfinite loss")
                    deadline()
                    loss.backward()
                    if not all(
                        p.grad is not None and bool(torch.isfinite(p.grad).all())
                        for p in model.parameters()
                    ):
                        raise ValueError("SRNet training produced invalid gradients")
                    optimizer.step()
                    srnet_model.validate(
                        {k: v.detach().numpy() for k, v in model.state_dict().items()}
                    )
                    deadline()
                    total += float(loss.detach())
                records.append(
                    {"epoch": epoch, "updates": len(pairs), "mean_pair_loss": total / len(pairs)}
                )
            deadline()
    finally:
        torch.set_num_threads(previous)
    return model.eval(), records
