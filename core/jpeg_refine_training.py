"""Bounded genuine-16 refinement kernels and immutable singleton evaluation."""

from __future__ import annotations

import hashlib
import time
from contextlib import nullcontext

import numpy as np

from core import jpeg_refine_schedule as schedule
from core import jpeg_refine_state as state
from core import jpeg_study as study
from core import jpeg_study_data as data
from core import jpeg_study_optimizer as rule
from core import srnet_cuda, srnet_samepool, srnet_training


def update(model, optimizer, pixels, indices, device):
    import torch

    optimizer.zero_grad(set_to_none=True)
    labels = torch.tensor(indices % 2, dtype=torch.int64, device=device)
    logits = model(torch.from_numpy(pixels).to(device))
    if logits.shape != (16, 2) or not bool(torch.isfinite(logits).all()):
        raise ValueError("refine logits invalid")
    loss = torch.nn.functional.cross_entropy(logits, labels)
    objective = loss + rule.convolution_penalty(model)
    if not bool(torch.isfinite(objective)):
        raise ValueError("refine objective nonfinite")
    objective.backward()
    if not srnet_training._finite_gradients(model, device):
        raise ValueError("refine gradients nonfinite")
    optimizer.step()
    if not study._finite_state(model, device):
        raise ValueError("refine state nonfinite")
    return float(loss.detach()), float(objective.detach())


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train(
    config,
    out,
    task,
    binding,
    execution,
    *,
    parent=None,
    device="cuda:0",
    factory=srnet_samepool.network,
    clock=time.monotonic,
):
    """Exactly one checkpoint interval; CPU is an explicit generated-test seam."""
    import torch

    arm, seed, stop = task
    if task not in schedule.segments():
        raise ValueError("refine segment invalid")
    start = stop - schedule.INTERVAL
    if (start == 0) != (parent is None):
        raise ValueError("refine segment parent missing or unexpected")
    data.fresh(out)
    out.mkdir()
    started = clock()

    def deadline():
        if clock() - started >= 1800:
            raise TimeoutError("refine worker wall limit")
        for marker in (out / "CANCEL", out.parent / "CANCEL"):
            if marker.is_symlink():
                raise ValueError("refine cancellation symlink")
            if marker.exists():
                raise InterruptedError

    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        policy = srnet_cuda.policy() if device == "cuda:0" else nullcontext(execution)
        with policy as actual, torch.random.fork_rng(devices=[0] if device == "cuda:0" else []):
            if actual != {
                k: v
                for k, v in execution.items()
                if k not in ("library_versions", "python_version")
            }:
                raise ValueError("refine runtime changed")
            if device == "cuda:0":
                torch.cuda.reset_peak_memory_stats(0)
            torch.manual_seed(seed)
            model = factory().to(device).train()
            initial = model_digest(model)
            optimizer = rule.optimizer(model, reference=True, corrected_l2=True)
            with data.Reader(
                config["root"], config["manifest_sha256"], "fit", deadline=deadline
            ) as reader:
                if reader.view_count != 5:
                    raise ValueError("refine requires five-view inputs")
                values = schedule.schedule(reader.samples, seed, arm)
                schedule_sha = schedule.digest(values)
                exposure_sha = schedule.exposure_digest(values, len(reader.samples))
                warmup = None if arm == "warmup" else parent["warmup_sha256"]
                if parent is not None:
                    inherited = state.load(parent["path"], parent["sha256"], binding)
                    parent_schedule = schedule.digest(
                        schedule.schedule(reader.samples, seed, inherited["metadata"]["arm"])
                    )
                    state.restore(
                        inherited,
                        model,
                        optimizer,
                        binding,
                        arm=arm,
                        seed=seed,
                        start=start,
                        parent_sha256=parent["sha256"],
                        schedule_sha256=parent_schedule,
                        warmup_sha256=warmup,
                    )
                records = []
                for index in range(start, stop):
                    deadline()
                    indices, angles, views = (v[index] for v in values)
                    pixels = study.augment(reader.batch(indices, views), angles)
                    records.append(update(model, optimizer, pixels, indices, device))
                reader.verify()
            deadline()
            meta = state.metadata(
                binding, arm, seed, stop, schedule_sha, parent["sha256"] if parent else None, warmup
            )
            path = out / "snapshot.npz"
            checksum = state.save(state.capture(model, optimizer, meta), path)
            # Receipt only follows a fully validated disk snapshot.
            state.load(path, checksum, binding)
            return {
                "status": "checkpointed",
                "arm": arm,
                "seed": seed,
                "binding": binding,
                "metadata": meta,
                "checkpoint_sha256": checksum,
                "initial_sha256": initial,
                "schedule_sha256": schedule_sha,
                "exposure_sha256": exposure_sha,
                "new_updates": len(records),
                "total_updates": stop,
                "mean_cross_entropy": float(np.mean(records, axis=0)[0]),
                "mean_objective": float(np.mean(records, axis=0)[1]),
                "seconds": clock() - started,
                "execution": execution,
                "real_gpu_segment": device == "cuda:0",
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0)
                if device == "cuda:0"
                else None,
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0)
                if device == "cuda:0"
                else None,
            }
    finally:
        torch.set_num_threads(previous)


def evaluate(
    config, checkpoint, *, device="cuda:0", factory=srnet_samepool.network, deadline=lambda: None
):
    import torch

    snapshot = state.load(checkpoint["path"], checkpoint["sha256"], checkpoint["binding"])
    meta = snapshot["metadata"]
    if meta["total_updates"] not in (schedule.WARMUP, schedule.FINAL):
        raise ValueError("refine evaluation is restricted to frozen endpoints")
    model = factory().to(device)
    model.load_state_dict(
        {
            k.removeprefix("model/"): torch.from_numpy(v.copy())
            for k, v in snapshot["arrays"].items()
            if k.startswith("model/")
        },
        strict=True,
    )
    before = model_digest(model)
    roles = ("development",) if meta["arm"] == "warmup" else ("development", "fit")
    results = {}
    for role in roles:
        deadline()
        with data.Reader(
            config["root"], config["manifest_sha256"], role, deadline=deadline
        ) as reader:
            results[role] = {
                "scores": study.evaluate(model, reader, device),
                "singleton_batch_parity": True,
            }
    after = model_digest(model)
    if before != after:
        raise ValueError("refine evaluation mutated parameters or buffers")
    return {
        "status": "completed",
        "checkpoint_sha256": checkpoint["sha256"],
        "metadata": meta,
        "model_sha256_before": before,
        "model_sha256_after": after,
        "real_gpu_evaluation": device == "cuda:0",
        **results,
    }
