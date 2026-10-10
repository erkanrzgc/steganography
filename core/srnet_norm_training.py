"""Versioned normalization learning adapter; legacy BN optimizer is untouched."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np

from core import srnet, srnet_cuda, srnet_groupnorm, srnet_model, srnet_training


def learn(*, mode, fetch, batches, epochs, seed, params, deadline, device="cuda:0"):
    if mode not in {"bn", "gn"} or device not in {"cpu", "cuda:0"}:
        raise ValueError("normalization training requires an explicit supported arm/backend")
    if type(epochs) is not int or not 1 <= epochs <= 8:
        raise ValueError("normalization epoch count outside pilot bounds")
    if mode == "bn":
        return srnet_training._learn(
            fetch=fetch,
            batches=batches,
            epochs=epochs,
            seed=seed,
            params=params,
            target_pairs=2,
            outer_deadline=deadline,
            device=device,
        )
    return kernel(
        fetch=fetch,
        batches=batches,
        epochs=epochs,
        seed=seed,
        params=params,
        deadline=deadline,
        device=device,
        factory=srnet_groupnorm.network,
        validator=srnet_groupnorm.validate,
    )


def kernel(*, fetch, batches, epochs, seed, params, deadline, device, factory, validator):
    """Matched four-row Adamax kernel; factory/validator are trusted core callables."""
    import torch

    previous = torch.get_num_threads()
    records = []
    try:
        torch.set_num_threads(params["threads"])
        policy = srnet_cuda.policy() if device == "cuda:0" else nullcontext()
        with policy, torch.random.fork_rng(devices=[0] if device == "cuda:0" else []):
            torch.random.default_generator.manual_seed(seed)
            if device == "cuda:0":
                torch.cuda.manual_seed(seed)
            model = factory().to(device).train()
            optimizer = torch.optim.Adamax(
                model.parameters(),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
                betas=(0.9, 0.999),
                eps=1e-8,
                foreach=False,
            )
            targets = torch.tensor([0, 1, 0, 1], dtype=torch.int64, device=device)
            for epoch in range(epochs):
                deadline()
                schedule = batches(epoch)
                if (
                    not isinstance(schedule, np.ndarray)
                    or schedule.ndim != 2
                    or (
                        schedule.shape[1] != 4
                        or not 1 <= len(schedule) <= 144
                        or schedule.dtype != np.int64
                    )
                ):
                    raise ValueError("normalization schedule outside four-row pilot bounds")
                total = 0.0
                for batch in schedule:
                    deadline()
                    optimizer.zero_grad(set_to_none=True)
                    pixels = fetch(batch)
                    if (
                        pixels.shape != (4, 1, 256, 256)
                        or pixels.dtype != np.dtype("<f4")
                        or not np.isfinite(pixels).all()
                        or np.any(np.abs(pixels) > 2**36)
                    ):
                        raise ValueError("normalization training pixels outside bounds")
                    logits = model(torch.from_numpy(pixels).to(device))
                    if logits.shape != (4, 2) or not bool(torch.isfinite(logits).all()):
                        raise ValueError("normalization training logits invalid")
                    loss = torch.nn.functional.cross_entropy(logits, targets)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("normalization training loss invalid")
                    deadline()
                    loss.backward()
                    if not srnet_training._finite_gradients(model, device):
                        raise ValueError("normalization training gradients invalid")
                    total += float(loss.detach())
                    optimizer.step()
                    validator(model.state_dict())
                    deadline()
                records.append(
                    {
                        "epoch": epoch,
                        "updates": len(schedule),
                        "mean_pair_loss": total / len(schedule),
                    }
                )
            if device == "cuda:0":
                torch.cuda.synchronize(0)
            model.cpu()
            validator(model.state_dict())
            deadline()
    finally:
        torch.set_num_threads(previous)
    return model.eval(), records


def bn_kernel_oracle(**kwargs):
    """Generated-test adapter only: validate the new arithmetic against legacy BN."""
    return kernel(factory=srnet.network, validator=srnet_model.validate_tensors, **kwargs)
