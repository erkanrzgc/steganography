"""Generated real-batch capacity probe, never a detection accuracy experiment."""

from __future__ import annotations

import time

import numpy as np

from core import srnet_cuda, srnet_model, srnet_samepool
from core.srnet_stream import deadline_after

ROWS = (8, 16, 32)
SEED = 20261012


def generated(rows):
    if type(rows) is not int or rows not in ROWS:
        raise ValueError("batch probe requires 8, 16 or 32 rows")
    yy, xx = np.indices((256, 256))
    base = 128 + 24 * np.sin(xx / 11) + 16 * np.cos(yy / 13)
    return np.stack(
        [base + (i % 2) * 8 * ((xx + yy) % 2) + i // 2 for i in range(rows)]
    ).astype("<f4")[:, None]


def update(model, optimizer, values, device):
    """One genuine batch/BN/optimizer update, not gradient accumulation."""
    import torch

    rows = values.shape[0]
    if (
        type(rows) is not int
        or rows not in ROWS
        or values.shape != (rows, 1, 256, 256)
        or values.dtype != np.dtype("<f4")
        or not np.isfinite(values).all()
        or np.any(np.abs(values) > 2**36)
    ):
        raise ValueError("batch probe pixels invalid")
    optimizer.zero_grad(set_to_none=True)
    targets = torch.tensor([0, 1] * (rows // 2), dtype=torch.int64, device=device)
    logits = model(torch.from_numpy(values).to(device))
    if logits.shape != (rows, 2) or not bool(torch.isfinite(logits).all()):
        raise ValueError("batch probe logits invalid")
    loss = torch.nn.functional.cross_entropy(logits, targets)
    if not bool(torch.isfinite(loss)):
        raise ValueError("batch probe loss invalid")
    loss.backward()
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise ValueError("batch probe gradients invalid")
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("batch probe parameters invalid")
    return float(loss.detach())


def probe(rows):
    values = generated(rows)
    import torch

    started = time.monotonic()
    deadline = deadline_after(120)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with srnet_cuda.policy() as execution, torch.random.fork_rng(devices=[0]):
            torch.random.default_generator.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            torch.cuda.reset_peak_memory_stats(0)
            deadline()
            model = srnet_samepool.network().to("cuda:0").train()
            optimizer = torch.optim.Adamax(
                model.parameters(), lr=0.001, weight_decay=0.0001,
                betas=(0.9, 0.999), eps=1e-8, foreach=False,
            )
            losses = []
            # First update includes allocator/optimizer state initialization.
            # Second checks steady-state execution, not detection learning.
            for _ in range(2):
                deadline()
                losses.append(update(model, optimizer, values, "cuda:0"))
                torch.cuda.synchronize(0)
                srnet_model.validate_tensors(model.state_dict())
                deadline()
            result = {
                "execution": execution,
                "rows": rows,
                "pairs": rows // 2,
                "optimizer_updates": 2,
                "losses": losses,
                "input_tensor_bytes": values.nbytes,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0),
                "seconds": time.monotonic() - started,
                "architecture": srnet_samepool.ARCHITECTURE,
                "gradient_accumulation": False,
            }
    finally:
        torch.set_num_threads(previous)
    return result
