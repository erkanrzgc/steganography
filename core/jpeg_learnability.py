"""Fixed real train-pair memorization; never a deployment/accuracy gate."""

from __future__ import annotations

import hashlib
import math
import time

import numpy as np

from core import (
    jpeg_population_control,
    srnet,
    srnet_cuda,
    srnet_groupnorm,
    srnet_model,
    srnet_norm_training,
    srnet_samepool,
    srnet_training,
)
from core.jpeg_timing_probe import AUDIT_SHA, MANIFEST_SHA, TimingReader
from core.srnet_stream import deadline_after

SEED = 20261010
EPOCHS = 8
UPDATES = 128


def select(samples):
    if len(samples) != 480 or any(r["split"] != "train" for r in samples):
        raise ValueError("learnability requires the complete training kit")
    originals = sorted({r["lineage"] for r in samples if r["source_group"] == "BOWS2"})
    if len(originals) != 32:
        raise ValueError("learnability fixed BOWS originals missing")
    indices = []
    for original in originals[:2]:
        for method in (None, "JUNIWARD"):
            matched = [
                i
                for i, r in enumerate(samples)
                if r["source_group"] == "BOWS2"
                and r["lineage"] == original
                and r["quality_factor"] == 75
                and r["method"] == method
                and r["label"] == ("cover" if method is None else "stego")
            ]
            if len(matched) != 1:
                raise ValueError("learnability requires unique matched fit rows")
            indices.append(matched[0])
    return np.array(indices, dtype=np.int64)


def metrics(outputs):
    if set(outputs) != set(range(4)):
        raise ValueError("learnability complete singleton outputs required")
    rows, losses, correct = [], [], 0
    for index in range(4):
        logits = outputs[index]
        if logits.shape != (2,) or not np.isfinite(logits).all() or np.any(np.abs(logits) > 2**36):
            raise ValueError("learnability singleton logits invalid")
        margin = float(logits[1]) - float(logits[0])
        label = index % 2
        prediction = int(margin >= 0)
        correct += prediction == label
        losses.append(math.log1p(math.exp(-abs(margin))) + max(-margin if label else margin, 0))
        rows.append({"position": index, "label": label, "logits": logits.tolist()})
    loss = math.fsum(losses) / 4
    return {
        "rows": rows,
        "correct": correct,
        "cross_entropy": loss,
        "learnability_criterion_passed": correct == 4 and loss < 0.2,
    }


def fit_arm(root, audit, arm):
    import torch

    if arm not in {"legacy", "reference"}:
        raise ValueError("learnability explicit arm required")
    execution = srnet_cuda.inspect()
    started, deadline = time.monotonic(), deadline_after(1800)
    factory = srnet.network if arm == "legacy" else srnet_samepool.network
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with srnet_cuda.policy(), TimingReader(root, audit, deadline=deadline) as reader:
            indices = select(reader.samples)
            pixels = reader.batch(indices)
            if any(np.array_equal(pixels[i], pixels[i + 1]) for i in (0, 2)):
                raise ValueError("learnability selected crop has no input signal")
            schedule = np.tile(indices, (UPDATES, 1))
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(SEED)
                initial = factory().to("cuda:0").eval()
                initial_sha = srnet_groupnorm.parameter_sha(initial)
                outputs, initial_parity = jpeg_population_control.score(
                    initial, reader, indices[None], deadline
                )
                initial_metrics = metrics({i: outputs[int(v)] for i, v in enumerate(indices)})
                del initial
            model, losses = srnet_norm_training.kernel(
                fetch=reader.batch,
                batches=lambda epoch: schedule,
                epochs=EPOCHS,
                seed=SEED,
                params=srnet_training.settings({}),
                deadline=deadline,
                device="cuda:0",
                factory=factory,
                validator=srnet_model.validate_tensors,
            )
            if len(losses) != EPOCHS or any(r["updates"] != UPDATES for r in losses):
                raise ValueError("learnability optimizer accounting incomplete")
            learned_sha = srnet_groupnorm.parameter_sha(model)
            if learned_sha == initial_sha:
                raise ValueError("learnability parameters did not update")
            model.to("cuda:0").eval()
            outputs, parity = jpeg_population_control.score(model, reader, indices[None], deadline)
            final_metrics = metrics({i: outputs[int(v)] for i, v in enumerate(indices)})
            reader.verify()
            deadline()
            return {
                "schema_version": "jpeg-real-pair-learnability-v1",
                "status": "completed",
                "arm": arm,
                "architecture": model.architecture,
                "execution": execution,
                "manifest_sha256": MANIFEST_SHA,
                "audit_sha256": AUDIT_SHA,
                "fit_rows": indices.tolist(),
                "fit_originals": 2,
                "source_group": "BOWS2",
                "quality_factor": 75,
                "method": "JUNIWARD",
                "selected_crops_have_signal": True,
                "initial_parameter_sha256": initial_sha,
                "learned_parameter_sha256": learned_sha,
                "seed": SEED,
                "epochs": EPOCHS,
                "optimizer_updates": EPOCHS * UPDATES,
                "schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
                "epoch_losses": losses,
                "initial": initial_metrics,
                "final": final_metrics,
                "singleton_batch_parity": {
                    "initial": initial_parity,
                    "final": parity,
                    "atol": 1e-4,
                    "rtol": 1e-4,
                },
                "seconds": time.monotonic() - started,
                "process_peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "process_peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0),
                "real_model_trained": True,
                "in_sample_only": True,
                "probe_pixels_read": 0,
                "validation_test_used": False,
                "model_discarded": True,
                "accuracy_qualification": "unavailable",
                "deployed": False,
            }
    finally:
        torch.set_num_threads(previous)
