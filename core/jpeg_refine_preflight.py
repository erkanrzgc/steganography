"""Generated FP32 parity, numeric save/resume and conservative whole-matrix timing."""

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


def probe(device="cuda:0", *, factory=srnet_samepool.network, clock=time.monotonic):
    import torch

    started = clock()
    deadline = deadline_after(1800)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[0] if device == "cuda:0" else []):
            torch.manual_seed(20261012)
            cpu = factory().train()
            model = factory().to(device).train()
            model.load_state_dict(cpu.state_dict())
            pixels = srnet_batch_probe.generated(16)
            indices = np.arange(16, dtype=np.int64)
            cpu_optimizer = rule.optimizer(cpu, reference=True, corrected_l2=True)
            optimizer = rule.optimizer(model, reference=True, corrected_l2=True)
            cpu_loss = training.update(cpu, cpu_optimizer, pixels, indices, "cpu")
            gpu_loss = training.update(model, optimizer, pixels, indices, device)
            if not np.allclose(cpu_loss, gpu_loss, rtol=1e-3, atol=1e-4):
                raise ValueError("refine generated CPU/CUDA loss parity failed")
            for key, value in cpu.state_dict().items():
                if not torch.allclose(value, model.state_dict()[key].cpu(), rtol=1e-3, atol=1e-4):
                    raise ValueError("refine generated CPU/CUDA state parity failed")
            del cpu, cpu_optimizer
            timings = []
            for _ in range(8):
                deadline()
                tick = clock()
                training.update(model, optimizer, pixels, indices, device)
                if device == "cuda:0":
                    torch.cuda.synchronize(0)
                timings.append(clock() - tick)
            # Counter scaffold permits exercising the production numeric format
            # with generated inputs without spending 512 GPU updates in preflight.
            # Author-rule Adamax has no bias correction; BN uses fixed momentum.
            for name, value in model.state_dict().items():
                if name.endswith("num_batches_tracked"):
                    value.fill_(512)
            for slots in optimizer.state.values():
                slots["step"].fill_(512)
            bound = "a" * 64
            meta = state.metadata(bound, "warmup", 20261012, 512, "b" * 64)
            with tempfile.TemporaryDirectory(prefix="jpeg-refine-preflight-") as folder:
                path = Path(folder) / "snapshot.npz"
                tick = clock()
                checksum = state.save(state.capture(model, optimizer, meta), path)
                restored = factory().to(device).train()
                restored_optimizer = rule.optimizer(restored, reference=True, corrected_l2=True)
                snapshot = state.load(path, checksum, bound)
                state.restore(
                    snapshot,
                    restored,
                    restored_optimizer,
                    bound,
                    arm="warmup",
                    seed=20261012,
                    start=512,
                    parent_sha256=checksum,
                    schedule_sha256="b" * 64,
                )
                checkpoint_seconds = clock() - tick
                for _ in range(2):
                    deadline()
                    a = training.update(model, optimizer, pixels, indices, device)
                    b = training.update(restored, restored_optimizer, pixels, indices, device)
                    if a != b or training.model_digest(model) != training.model_digest(restored):
                        raise ValueError("refine generated save/resume model parity failed")
                    for (parameter, slots), (other, restored_slots) in zip(
                        optimizer.state.items(), restored_optimizer.state.items(), strict=True
                    ):
                        if parameter.shape != other.shape or any(
                            not torch.equal(v, restored_slots[k]) for k, v in slots.items()
                        ):
                            raise ValueError("refine generated save/resume optimizer parity failed")
            return {
                "status": "completed",
                "generated_parity": True,
                "save_resume_parity": True,
                "max_update_seconds": max(timings),
                "startup_checkpoint_seconds": clock() - started + checkpoint_seconds,
                "generated_optimizer_updates": 14,
                "counter_scaffold": 512,
                "real_data_used": False,
                "seconds": clock() - started,
            }
    finally:
        torch.set_num_threads(previous)
