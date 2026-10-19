"""Separate numerical gates for generated derivatives and optimizer arithmetic.

The original, failed preflight remains immutable. This revision never compares
independently rounded CPU/CUDA gradients after a nonlinear optimizer step.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from core import jpeg_refine_state as state
from core import jpeg_refine_training as training
from core import jpeg_study_optimizer as rule
from core import srnet_batch_probe, srnet_samepool
from core.srnet_stream import deadline_after


def optimizer_oracle(device):
    """Check zero/tiny/sign-changing gradients against independent float64 math."""
    import torch

    initial = np.linspace(-0.2, 0.2, 12, dtype=np.float64)
    model = torch.nn.Linear(12, 1, bias=False, dtype=torch.float32, device=device)
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(initial.astype(np.float32)[None]).to(device))
    optimizer = rule.optimizer(model, reference=True, corrected_l2=True)
    expected = initial.astype(np.float32).astype(np.float64)
    average = np.zeros(12, dtype=np.float64)
    infinity = np.zeros(12, dtype=np.float64)
    base = np.array([0, 1e-12, -1e-12, 1e-8, -1e-8, 1e-5, -1e-5, 0.1, -0.1, 1, -1, 0])
    for index in range(4):
        gradient = (np.roll(base, index) * (-1) ** index).astype(np.float32)
        model.weight.grad = torch.from_numpy(gradient[None].copy()).to(device)
        optimizer.step()
        average = 0.9 * average + 0.1 * gradient.astype(np.float64)
        infinity = np.maximum(0.999 * infinity + 1e-8, np.abs(gradient.astype(np.float64)))
        expected -= 0.001 * average / infinity
        slots = optimizer.state[model.weight]
        values = ((model.weight, expected), (slots["exp_avg"], average),
                  (slots["exp_inf"], infinity))
        if float(slots["step"]) != index + 1 or any(
            not np.allclose(value.detach().cpu().numpy().ravel(), reference,
                            rtol=1e-6, atol=1e-8)
            for value, reference in values
        ):
            raise ValueError("generated optimizer scalar oracle failed")
    return True


def backward(model, pixels, indices, device):
    import torch

    model.zero_grad(set_to_none=True)
    labels = torch.tensor(indices % 2, dtype=torch.int64, device=device)
    logits = model(torch.from_numpy(pixels).to(device))
    if logits.shape != (16, 2) or not bool(torch.isfinite(logits).all()):
        raise ValueError("generated logits invalid")
    loss = torch.nn.functional.cross_entropy(logits, labels)
    objective = loss + rule.convolution_penalty(model)
    if not bool(torch.isfinite(objective)):
        raise ValueError("generated objective invalid")
    objective.backward()
    return logits.detach().cpu(), (float(loss.detach()), float(objective.detach()))


def derivative_parity(cpu, model, pixels, indices, device):
    """Same fixed tolerances, applied before Adamax's near-zero sign sensitivity."""
    import torch

    a, losses_a = backward(cpu, pixels, indices, "cpu")
    b, losses_b = backward(model, pixels, indices, device)
    if not torch.allclose(a, b, rtol=1e-3, atol=1e-4) or not np.allclose(
        losses_a, losses_b, rtol=1e-3, atol=1e-4
    ):
        raise ValueError("generated CPU/CUDA forward parity failed")
    for (name, parameter), (other_name, other) in zip(
        cpu.named_parameters(), model.named_parameters(), strict=True
    ):
        if name != other_name or parameter.grad is None or other.grad is None or not torch.allclose(
            parameter.grad, other.grad.cpu(), rtol=1e-3, atol=1e-4
        ):
            raise ValueError("generated CPU/CUDA gradient parity failed")
    for name, value in cpu.state_dict().items():
        if not torch.allclose(value, model.state_dict()[name].cpu(), rtol=1e-3, atol=1e-4):
            raise ValueError("generated CPU/CUDA pre-optimizer state parity failed")
    return float((a - b).abs().max())


def probe(device="cuda:0", *, factory=srnet_samepool.network, clock=time.monotonic):
    import torch

    started = clock()
    deadline = deadline_after(1800)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[0] if device == "cuda:0" else []):
            optimizer_oracle("cpu")
            optimizer_oracle(device)
            torch.manual_seed(20261012)
            cpu = factory().train()
            model = factory().to(device).train()
            model.load_state_dict(cpu.state_dict())
            pixels = srnet_batch_probe.generated(16)
            indices = np.arange(16, dtype=np.int64)
            cpu_optimizer = rule.optimizer(cpu, reference=True, corrected_l2=True)
            optimizer = rule.optimizer(model, reference=True, corrected_l2=True)
            logit_difference = derivative_parity(cpu, model, pixels, indices, device)
            cpu_optimizer.step()
            optimizer.step()
            del cpu, cpu_optimizer
            timings = []
            for _ in range(8):
                deadline()
                tick = clock()
                training.update(model, optimizer, pixels, indices, device)
                if device == "cuda:0":
                    torch.cuda.synchronize(0)
                timings.append(clock() - tick)
            for name, value in model.state_dict().items():
                if name.endswith("num_batches_tracked"):
                    value.fill_(512)
            for slots in optimizer.state.values():
                slots["step"].fill_(512)
            bound = "a" * 64
            meta = state.metadata(bound, "warmup", 20261012, 512, "b" * 64)
            with tempfile.TemporaryDirectory(prefix="jpeg-refine-v2-preflight-") as folder:
                path = Path(folder) / "snapshot.npz"
                tick = clock()
                checksum = state.save(state.capture(model, optimizer, meta), path)
                restored = factory().to(device).train()
                restored_optimizer = rule.optimizer(restored, reference=True, corrected_l2=True)
                snapshot = state.load(path, checksum, bound)
                state.restore(snapshot, restored, restored_optimizer, bound, arm="warmup",
                              seed=20261012, start=512, parent_sha256=checksum,
                              schedule_sha256="b" * 64)
                checkpoint_seconds = clock() - tick
                for _ in range(2):
                    deadline()
                    a = training.update(model, optimizer, pixels, indices, device)
                    b = training.update(restored, restored_optimizer, pixels, indices, device)
                    if a != b or training.model_digest(model) != training.model_digest(restored):
                        raise ValueError("generated save/resume model parity failed")
                    for (parameter, slots), (other, restored_slots) in zip(
                        optimizer.state.items(), restored_optimizer.state.items(), strict=True
                    ):
                        if parameter.shape != other.shape or any(
                            not torch.equal(value, restored_slots[key])
                            for key, value in slots.items()
                        ):
                            raise ValueError("generated save/resume optimizer parity failed")
            return {
                "status": "completed", "preflight_version": "derivatives-and-oracle-v2",
                "generated_parity": True, "derivative_parity": True,
                "optimizer_scalar_oracle": True, "save_resume_parity": True,
                "max_absolute_logit_difference": logit_difference,
                "max_update_seconds": max(timings),
                "startup_checkpoint_seconds": clock() - started + checkpoint_seconds,
                "generated_optimizer_updates": 22, "counter_scaffold": 512,
                "real_data_used": False, "seconds": clock() - started,
            }
    finally:
        torch.set_num_threads(previous)
