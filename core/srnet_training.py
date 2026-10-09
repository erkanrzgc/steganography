"""Optional CPU paired SRNet learning; no validation, calibration or deployment."""

from __future__ import annotations

import math
import time
from contextlib import AbstractContextManager, nullcontext

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


def fit(
    pixels,
    samples,
    indices,
    *,
    seed,
    schedule,
    config,
    wide_context=False,
    accumulate_context=False,
):
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
    if type(accumulate_context) is not bool or (accumulate_context and not wide_context):
        raise ValueError("accumulation is restricted to the frozen wide context")
    batch_builder = srnet_multibatch.epoch_batches
    if wide_context:
        from core import srnet_widebatch

        batch_builder = srnet_widebatch.epoch_batches
    for epoch, record in enumerate(schedule):
        if epoch_pairs(selected, seed=seed, epoch=epoch)[1] != record:
            raise ValueError("SRNet training schedule mismatch")
        if batch_recipe is not None:
            batch_builder(selected, seed=seed, epoch=epoch)

    def batches(epoch):
        builder = batch_builder if batch_recipe is not None else epoch_pairs
        return builder(selected, seed=seed, epoch=epoch)[0]

    return _learn(
        fetch=lambda chunk: pixels[indices[chunk]].copy(),
        batches=batches,
        epochs=len(schedule),
        seed=seed,
        params=params,
        target_pairs=4 if wide_context else 2 if batch_recipe is not None else 1,
        accumulate_context=accumulate_context,
    )


def _finite_gradients(model, device):
    """Preserve every gradient check; consolidate CUDA host synchronization."""
    import torch

    parameters = list(model.parameters())
    if not parameters or any(p.grad is None for p in parameters):
        return False
    if device == "cuda:0":
        # Each element is still checked. Only scalar boolean transfer changes,
        # not floating-point gradients, reduction order or optimizer arithmetic.
        return bool(torch.stack([torch.isfinite(p.grad).all() for p in parameters]).all())
    return all(bool(torch.isfinite(p.grad).all()) for p in parameters)


def _learn(
    *,
    fetch,
    batches,
    epochs,
    seed,
    params,
    target_pairs,
    accumulate_context=False,
    outer_deadline=None,
    device="cpu",
    segment=None,
):
    """One numerical optimizer engine shared by legacy and bounded block readers."""
    import torch

    if device not in {"cpu", "cuda:0"}:
        raise ValueError("unsupported explicit training device")
    started = time.monotonic()
    start_epoch, stop_epoch = 0, epochs
    context = None
    if segment is not None:
        from core import srnet_checkpoint
        from core.jpeg_scale import identity

        if (
            not isinstance(segment, dict)
            or set(segment) != {"binding", "stop_epoch", "resume", "sink"}
            or type(epochs) is not int
            or not 1 <= epochs <= 50
            or type(segment["stop_epoch"]) is not int
            or not 1 <= segment["stop_epoch"] <= epochs
            or not callable(segment["sink"])
            or type(target_pairs) is not int
            or target_pairs != 2
            or type(accumulate_context) is not bool
            or accumulate_context
        ):
            raise ValueError("invalid explicit four-row epoch segment")
        identity(segment["binding"])
        counts, digest = srnet_checkpoint.schedule(batches, epochs, deadline=outer_deadline)
        context = {
            "binding": segment["binding"],
            "seed": seed,
            "settings": params,
            "device": device,
            "epochs": epochs,
            "scheduled_updates": counts,
            "schedule_sha256": digest,
        }
        stop_epoch = segment["stop_epoch"]
        if segment["resume"] is not None:
            srnet_checkpoint.validate(segment["resume"])
            start_epoch = segment["resume"]["metadata"]["next_epoch"]
            if start_epoch >= stop_epoch:
                raise ValueError("checkpoint cannot repeat or skip completed segment")
    execution: AbstractContextManager
    if device == "cuda:0":
        from core import srnet_cuda

        execution = srnet_cuda.policy()
    else:
        execution = nullcontext()

    previous = torch.get_num_threads()
    records = []

    def deadline():
        if outer_deadline is not None:
            outer_deadline()
        if time.monotonic() - started > params["max_seconds"]:
            raise ValueError("SRNet training deadline exceeded")

    try:
        torch.set_num_threads(params["threads"])
        with execution, torch.random.fork_rng(devices=[0] if device == "cuda:0" else []):
            torch.random.default_generator.manual_seed(seed)
            if device == "cuda:0":
                torch.cuda.manual_seed(seed)
            model = srnet.network().to(device).train()
            optimizer = torch.optim.Adamax(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
                betas=(0.9, 0.999),
                eps=1e-8,
                foreach=False,
            )
            if segment is not None and segment["resume"] is not None:
                records = srnet_checkpoint.restore(segment["resume"], model, optimizer, context)
            targets = torch.tensor(
                [0, 1] * target_pairs,
                dtype=torch.int64,
                device=device,
            )
            for epoch in range(start_epoch, stop_epoch):
                deadline()
                pairs = batches(epoch)
                total = 0.0
                for pair in pairs:
                    deadline()
                    optimizer.zero_grad(set_to_none=True)
                    chunks = (pair[:4], pair[4:]) if accumulate_context else (pair,)
                    group_loss = 0.0
                    for chunk in chunks:
                        deadline()
                        inputs = torch.from_numpy(fetch(chunk)).to(device)
                        labels = targets[: len(chunk)]
                        logits = model(inputs)
                        if logits.shape != (len(labels), 2) or not bool(
                            torch.isfinite(logits).all()
                        ):
                            raise ValueError("SRNet training produced invalid logits")
                        loss = torch.nn.functional.cross_entropy(logits, labels)
                        if not bool(torch.isfinite(loss)):
                            raise ValueError("SRNet training produced nonfinite loss")
                        deadline()
                        if accumulate_context:
                            (loss / 2).backward()
                        else:
                            loss.backward()
                        if not _finite_gradients(model, device):
                            raise ValueError("SRNet training produced invalid gradients")
                        group_loss += float(loss.detach()) / len(chunks)
                    optimizer.step()
                    if device == "cuda:0":
                        srnet_model.validate_tensors(model.state_dict())
                    else:
                        srnet_model.validate(
                            {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
                        )
                    deadline()
                    total += group_loss
                records.append(
                    {"epoch": epoch, "updates": len(pairs), "mean_pair_loss": total / len(pairs)}
                )
            if segment is not None:
                snapshot = srnet_checkpoint.capture(model, optimizer, records, context)
                deadline()
            if device == "cuda:0":
                torch.cuda.synchronize(0)
                model.cpu()
                srnet_model.validate({k: v.detach().numpy() for k, v in model.state_dict().items()})
            deadline()
            if segment is not None:
                segment["sink"](snapshot)
    finally:
        torch.set_num_threads(previous)
    return model.eval(), records
