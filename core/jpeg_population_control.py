"""Fixed train-only layerwise BN intervention on the audited real JPEG kit."""

from __future__ import annotations

import hashlib
import time

import numpy as np

from core import srnet_cuda, srnet_model, srnet_population_bn
from core.jpeg_timing_probe import AUDIT_SHA, MANIFEST_SHA, TimingReader
from core.srnet_stream import deadline_after, regular_open

MODEL_SHA = "71f6f5fe8730944c38d406362bf26e382f348f302b03c7f30809991db73e44b6"


def score(model, reader, batches, deadline):
    import torch

    results, parity = {}, True
    device = next(model.parameters()).device
    if any(m.training for m in model.modules()):
        raise ValueError("population scoring requires singleton eval mode")
    with torch.no_grad():
        for batch in batches:
            deadline()
            pixels = torch.from_numpy(reader.batch(batch)).to(device)
            together = model(pixels).detach().cpu().numpy()
            if together.shape != (4, 2) or not np.isfinite(together).all():
                raise ValueError("population logits invalid")
            for i, index in enumerate(batch):
                deadline()
                logits = model(pixels[i : i + 1]).detach().cpu().numpy()[0]
                if logits.shape != (2,) or not np.isfinite(logits).all():
                    raise ValueError("population singleton logits invalid")
                parity = parity and bool(np.allclose(logits, together[i], atol=1e-4, rtol=1e-4))
                results[int(index)] = logits.astype(np.float64)
    return results, parity


def cells(samples, logits):
    contexts = sorted(
        {(samples[i]["source_group"], samples[i]["quality_factor"]) for i in logits},
        key=lambda v: (v[0], str(v[1])),
    )
    result = []
    for source, quality in contexts:
        for method in ("JUNIWARD", "UERD"):
            selected = [
                i
                for i in logits
                if samples[i]["source_group"] == source
                and samples[i]["quality_factor"] == quality
                and samples[i]["method"] in (None, method)
            ]
            labels = np.array([samples[i]["label"] != "cover" for i in selected])
            margins = np.array([logits[i][1] - logits[i][0] for i in selected])
            if int(labels.sum()) != 8 or int((~labels).sum()) != 8:
                raise ValueError("population probe cell accounting incomplete")
            decisions = margins >= 0
            tp, fp = int((decisions & labels).sum()), int((decisions & ~labels).sum())
            result.append(
                {
                    "source_group": source,
                    "quality_factor": quality,
                    "method": method,
                    "cover_rows": 8,
                    "stego_rows": 8,
                    "true_positive": tp,
                    "false_positive": fp,
                    "true_negative": 8 - fp,
                    "false_negative": 8 - tp,
                    "balanced_accuracy": (tp + 8 - fp) / 16,
                    "recall": tp / 8,
                    "false_positive_rate": fp / 8,
                    "cross_entropy": float(np.mean(np.logaddexp(0, margins) - labels * margins)),
                    "cover_rows_shared_across_method_cells": True,
                }
            )
    if len(result) != 10:
        raise ValueError("population probe context coverage incomplete")
    return result


def control(root, audit, model_path):
    import torch

    started = time.monotonic()
    deadline = deadline_after(1800)
    execution = srnet_cuda.inspect()
    source = srnet_model.load_model(model_path, checksum=MODEL_SHA).to("cuda:0").eval()
    saved = {k: v.detach().clone() for k, v in source.state_dict().items()}
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with srnet_cuda.policy(), TimingReader(root, audit, deadline=deadline) as reader:
            calibration, probe = srnet_population_bn.partition(reader.samples)
            baseline, baseline_parity = score(source, reader, probe, deadline)
            clone, layers = srnet_population_bn.refresh(source, reader.batch, calibration, deadline)
            refreshed, refreshed_parity = score(clone, reader, probe, deadline)
            reader.verify()
            if any(not torch.equal(saved[k], v) for k, v in source.state_dict().items()):
                raise ValueError("population source model mutated")
            with regular_open(model_path) as stream:
                raw = stream.read(srnet_model.MAX_BYTES + 1)
            if hashlib.sha256(raw).hexdigest() != MODEL_SHA:
                raise ValueError("population source file mutated")
            deadline()
            return clone.cpu().eval(), {
                "schema_version": "jpeg-layerwise-population-bn-v1",
                "status": "completed",
                "execution": execution,
                "model_sha256": MODEL_SHA,
                "manifest_sha256": MANIFEST_SHA,
                "audit_sha256": AUDIT_SHA,
                "calibration_rows": 360,
                "probe_rows": 120,
                "calibration_originals": 72,
                "probe_originals": 24,
                "original_groups_disjoint": True,
                "validation_used": False,
                "probe_is_historical_training_data": True,
                "optimizer_updates": 0,
                "learned_parameters_bit_identical": True,
                "source_model_unchanged": True,
                "normalization_refreshed": True,
                "calibration_layers": layers,
                "baseline_cells": cells(reader.samples, baseline),
                "refreshed_cells": cells(reader.samples, refreshed),
                "singleton_batch_parity": {
                    "baseline": baseline_parity,
                    "refreshed": refreshed_parity,
                    "atol": 1e-4,
                    "rtol": 1e-4,
                },
                "numerical_gates_passed": baseline_parity and refreshed_parity,
                "seconds": time.monotonic() - started,
                "process_peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "process_peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(0),
                "accuracy_qualification": "unavailable",
                "deployed": False,
            }
    finally:
        torch.set_num_threads(previous_threads)
