"""Fixed small three-origin BN/GN pilot; inspected probe is not a support gate."""

from __future__ import annotations

import hashlib
import json
import time

from core import (
    jpeg_population_control,
    srnet,
    srnet_cuda,
    srnet_diversity_sampling,
    srnet_groupnorm,
    srnet_norm_training,
    srnet_population_bn,
    srnet_training,
)
from core.jpeg_timing_probe import AUDIT_SHA, MANIFEST_SHA, TimingReader
from core.srnet_stream import deadline_after

SEED = 20261010
EPOCHS = 8


def schedules(samples):
    fit, probe = srnet_population_bn.partition(samples)
    indices = fit.reshape(-1)
    selected = [samples[i] for i in indices]
    batches, records = [], []
    for epoch in range(EPOCHS):
        local, record = srnet_diversity_sampling.epoch_batches(selected, seed=SEED, epoch=epoch)
        if local.shape != (144, 4):
            raise ValueError("normalization pilot requires full fixed 144-update epochs")
        mapped = indices[local].copy()
        if set(mapped.flat) != set(indices.tolist()):
            raise ValueError("normalization fit row exposure incomplete")
        mapped.flags.writeable = False
        batches.append(mapped)
        records.append(record)
    return batches, records, probe


def fit_arm(root, audit, arm):
    import torch

    if arm not in {"bn", "gn"}:
        raise ValueError("normalization pilot arm invalid")
    started = time.monotonic()
    deadline = deadline_after(1800)
    execution = srnet_cuda.inspect()
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with srnet_cuda.policy(), TimingReader(root, audit, deadline=deadline) as reader:
            batches, records, probe = schedules(reader.samples)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(SEED)
                initial = (srnet.network if arm == "bn" else srnet_groupnorm.network)()
                initial_sha = srnet_groupnorm.parameter_sha(initial)
                del initial
            model, losses = srnet_norm_training.learn(
                mode=arm,
                fetch=reader.batch,
                batches=lambda epoch: batches[epoch],
                epochs=EPOCHS,
                seed=SEED,
                params=srnet_training.settings({}),
                deadline=deadline,
                device="cuda:0",
            )
            if len(losses) != EPOCHS or any(r["updates"] != 144 for r in losses):
                raise ValueError("normalization pilot optimizer accounting incomplete")
            learned_sha = srnet_groupnorm.parameter_sha(model)
            if learned_sha == initial_sha:
                raise ValueError("normalization pilot failed to update learned parameters")
            model.to("cuda:0").eval()
            logits, parity = jpeg_population_control.score(model, reader, probe, deadline)
            reader.verify()
            result = {
                "schema_version": "jpeg-bn-gn-learning-pilot-v1",
                "status": "completed",
                "arm": arm,
                "architecture": model.architecture,
                "execution": execution,
                "manifest_sha256": MANIFEST_SHA,
                "audit_sha256": AUDIT_SHA,
                "fit_rows": 360,
                "fit_originals": 72,
                "probe_rows": 120,
                "probe_originals": 24,
                "original_groups_disjoint": True,
                "probe_previously_inspected": True,
                "probe_is_train_role_development": True,
                "validation_test_used": False,
                "initial_parameter_sha256": initial_sha,
                "learned_parameter_sha256": learned_sha,
                "seed": SEED,
                "epochs": EPOCHS,
                "optimizer_updates": EPOCHS * 144,
                "optimizer": {
                    "name": "Adamax",
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "foreach": False,
                },
                "schedules": records,
                "epoch_losses": losses,
                "schedule_sha256": hashlib.sha256(
                    json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "probe_cells": jpeg_population_control.cells(reader.samples, logits),
                "singleton_batch_parity": {"passed": parity, "atol": 1e-4, "rtol": 1e-4},
                "probe_logits": [
                    {"row": int(i), "logits": logits[int(i)].tolist()} for i in probe.reshape(-1)
                ],
                "seconds": time.monotonic() - started,
                "process_peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "process_peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0),
                "real_model_trained": True,
                "accuracy_qualification": "unavailable",
                "deployed": False,
            }
            deadline()
            return model.cpu().eval(), result
    finally:
        torch.set_num_threads(previous)
