"""Controlled JPEG learning: fixed exposure, singleton evaluation, no deployment."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import nullcontext

import numpy as np

from core import jpeg_study_data as data
from core import (
    jpeg_study_optimizer,
    jpeg_study_state,
    jpeg_study_views,
    srnet_cuda,
    srnet_samepool,
    srnet_training,
)
from core.srnet_stream import document
from steganography.benchmarking.metrics import classification_metrics

ARMS = ("legacy", "reference-adamax", "convolution-l2", "genuine-batch")
SEEDS = (20261012, 20261013)
PRESENTED_ROWS = 32768


def recipe(arm, seed, rows, *, representation=jpeg_study_views.CENTER):
    if (arm not in ARMS or type(seed) is not int or seed not in SEEDS
        or type(rows) is not int or rows not in (4, 8, 16, 32)
        or (arm != "genuine-batch" and rows != 4)
        or (arm == "genuine-batch" and rows == 4)
        or representation not in (jpeg_study_views.CENTER, jpeg_study_views.FIVE)):
        raise ValueError("study arm/seed/genuine-batch settings invalid")
    result = {"arm": arm, "seed": seed, "rows": rows,
            "presented_rows": PRESENTED_ROWS, "learning_rate": 0.001,
            "optimizer_rule": "pytorch-adamax" if arm == "legacy" else jpeg_study_optimizer.RULE,
            "convolution_only_l2": arm in ARMS[2:], "gradient_accumulation": False,
            "architecture": srnet_samepool.ARCHITECTURE,
            "initialization": "unchanged legacy Kaiming; not full upstream parity"}
    if representation == jpeg_study_views.FIVE:
        result["representation"] = jpeg_study_views.contract()
        result["row_unit"] = "one view of one image; two matched rows per original pair"
    return result


def schedule(samples, seed, rows, presented_rows=PRESENTED_ROWS):
    if (type(seed) is not int or seed not in SEEDS or type(rows) is not int
        or rows not in (4, 8, 16, 32) or type(presented_rows) is not int
        or not 1 <= presented_rows <= PRESENTED_ROWS or presented_rows % rows):
        raise ValueError("study schedule settings invalid")
    if len(samples) % 4 or not 8 <= len(samples) <= 1024:
        raise ValueError("study fit sample count invalid")
    groups: dict[str, list[int]] = {origin: [] for origin in data.ORIGINS}
    for i in range(0, len(samples), 2):
        cover, stego = samples[i:i + 2]
        if (cover["label"] != "cover" or stego["label"] != "stego"
            or cover["role"] != "fit" or stego["role"] != "fit"
            or cover["lineage"] != stego["lineage"]
            or cover["source_group"] != stego["source_group"]
            or cover["source_group"] not in groups):
            raise ValueError("study fit pair contract invalid")
        groups[cover["source_group"]].append(i)
    if not groups[data.ORIGINS[0]] or len(groups[data.ORIGINS[0]]) != len(groups[data.ORIGINS[1]]):
        raise ValueError("study schedule requires balanced origins")
    rng = np.random.default_rng(seed)
    pairs: list[int] = []
    orientations: list[int] = []
    while len(pairs) < presented_rows:
        shuffled = [rng.permutation(groups[o]) for o in data.ORIGINS]
        for a, b in zip(*shuffled, strict=True):
            for pair in (int(a), int(b)):
                orientation = int(rng.integers(8))
                pairs.extend((pair, pair + 1))
                orientations.extend((orientation, orientation))
                if len(pairs) >= presented_rows:
                    break
            if len(pairs) >= presented_rows:
                break
    indices = np.asarray(pairs[:presented_rows], dtype=np.int64).reshape(-1, rows)
    angles = np.asarray(orientations[:presented_rows], dtype=np.int64).reshape(-1, rows)
    indices.setflags(write=False)
    angles.setflags(write=False)
    return indices, angles


def view_schedule(samples, seed, indices):
    """Same view for paired rows; cycle all five regardless of content or label."""
    if (type(seed) is not int or seed not in SEEDS or not isinstance(indices, np.ndarray)
        or indices.ndim != 2 or indices.shape[1] not in (4, 8, 16, 32)
        or indices.dtype.kind not in "iu" or not 1 <= indices.size <= PRESENTED_ROWS
        or np.any(indices < 0) or np.any(indices >= len(samples))):
        raise ValueError("study view schedule bounds invalid")
    exposures: dict[str, int] = {}
    result = np.empty(indices.size, dtype=np.int64)
    flat = indices.ravel()
    for position in range(0, len(flat), 2):
        i = int(flat[position])
        if i % 2 or flat[position + 1] != i + 1:
            raise ValueError("study view schedule requires adjacent original pairs")
        lineage = samples[i]["lineage"]
        if samples[i + 1]["lineage"] != lineage:
            raise ValueError("study view schedule pair ancestry mismatch")
        offset = int.from_bytes(hashlib.sha256(
            f"jpeg-five-view:{seed}:{lineage}".encode()).digest()[:8], "little") % 5
        exposure = exposures.get(lineage, 0)
        result[position:position + 2] = (offset + exposure) % 5
        exposures[lineage] = exposure + 1
    result = result.reshape(indices.shape)
    result.setflags(write=False)
    return result


def augment(values, angles):
    if values.shape != (len(angles), 1, 256, 256) or values.dtype != np.dtype("<f4"):
        raise ValueError("study augmentation contract mismatch")
    result = np.empty_like(values)
    for target, value, angle in zip(result, values, angles, strict=True):
        if not 0 <= int(angle) < 8:
            raise ValueError("study D4 orientation invalid")
        oriented = np.rot90(value, int(angle) % 4, axes=(-2, -1))
        target[:] = oriented[..., ::-1] if angle >= 4 else oriented
    return result


def auc(pairs):
    negatives, positives = pairs[:, 0], pairs[:, 1]
    difference = positives[:, None] - negatives[None, :]
    return float(np.mean((difference > 0) + 0.5 * (difference == 0)))


def statistics(samples, scores, *, baseline=None, bootstrap=1000):
    scores = np.asarray(scores, dtype=np.float64)
    if (scores.shape != (len(samples),) or not np.isfinite(scores).all()
        or np.any(scores < 0) or np.any(scores > 1) or len(samples) % 2
        or type(bootstrap) is not int or not 100 <= bootstrap <= 2000):
        raise ValueError("study metric inputs invalid")
    seen = set()
    for i in range(0, len(samples), 2):
        cover, stego = samples[i:i + 2]
        if (cover.get("label") != "cover" or stego.get("label") != "stego"
            or cover.get("role") != "development" or stego.get("role") != "development"
            or cover.get("source_group") not in data.ORIGINS
            or stego.get("source_group") != cover["source_group"]
            or not isinstance(cover.get("lineage"), str)
            or stego.get("lineage") != cover["lineage"] or cover["lineage"] in seen):
            raise ValueError("study metric pair/role/ancestry mismatch")
        seen.add(cover["lineage"])
    pairs = scores.reshape(-1, 2)
    if baseline is not None:
        baseline = np.asarray(baseline, dtype=np.float64)
        if (baseline.shape != scores.shape or not np.isfinite(baseline).all()
            or np.any(baseline < 0) or np.any(baseline > 1)):
            raise ValueError("study paired baseline invalid")
        baseline = baseline.reshape(-1, 2)
    result = []
    for origin in data.ORIGINS:
        indices = [i // 2 for i in range(0, len(samples), 2)
                   if samples[i]["source_group"] == origin]
        if not indices:
            raise ValueError("study metrics require both sources")
        selected = pairs[indices]
        metrics = classification_metrics(
            [(bool(i % 2), float(v) * 100) for i, v in enumerate(selected.ravel())],
            threshold=50, recommend_threshold=False,
        )
        rng = np.random.default_rng(SEEDS[0])
        estimates, deltas = [], []
        for _ in range(bootstrap):
            draw = rng.integers(len(indices), size=len(indices))
            estimate = auc(selected[draw])
            estimates.append(estimate)
            if baseline is not None:
                deltas.append(estimate - auc(baseline[indices][draw]))
        flat = selected.ravel()
        labels = np.tile([0, 1], len(selected))
        ece = 0.0
        bins = np.minimum((flat * 10).astype(int), 9)
        for index in range(10):
            mask = bins == index
            if mask.any():
                ece += float(mask.mean() * abs(flat[mask].mean() - labels[mask].mean()))
        interval = np.quantile(estimates, [0.025, 0.975]).tolist()
        delta_interval = np.quantile(deltas, [0.025, 0.975]).tolist() if deltas else None
        result.append({"source_group": origin, "method": "JUNIWARD", "quality_factor": 75,
                       "payload_rate": 0.4, "original_pairs": len(indices),
                       "metrics": metrics, "ece": ece, "auc_ci95": interval,
                       "paired_auc_delta_ci95": delta_interval,
                       "learning_gate": bool(len(indices) >= 64 and auc(selected) >= 0.7
                                             and interval[0] > 0.5
                                             and delta_interval and delta_interval[0] > 0),
                       "support_status": "experimental"})
    return result


def evaluate(model, reader, device):
    import torch

    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(reader.samples), 4):
            indices = np.arange(start, min(start + 4, len(reader.samples)), dtype=np.int64)
            predictions = []
            for view in range(reader.view_count):
                pixels = (reader.batch(indices, np.full(len(indices), view, dtype=np.int64))
                          if reader.view_count == 5 else reader.batch(indices))
                values = torch.from_numpy(pixels).to(device)
                batch = model(values)
                singleton = torch.cat([model(value[None]) for value in values])
                if (batch.shape != (len(indices), 2) or not bool(torch.isfinite(batch).all())
                    or not torch.allclose(batch, singleton, atol=1e-4, rtol=1e-4)):
                    raise ValueError("study singleton/batch inference parity failed")
                predictions.append(torch.softmax(singleton, dim=1)[:, 1].cpu().numpy())
            scores.extend(np.mean(np.asarray(predictions, dtype=np.float64), axis=0).tolist())
    reader.verify()
    return scores


def _finite_state(model, device):
    """Check every parameter/buffer; CUDA transfers only one scalar boolean."""
    import torch

    values = list(model.state_dict().values())
    if not values:
        return False
    if device == "cuda:0":
        return bool(torch.stack([torch.isfinite(value).all() for value in values]).all())
    return all(bool(torch.isfinite(value).all()) for value in values)


def fit(root, manifest_sha256, out, arm, seed, rows, sources, *, resume=None):
    """CUDA-only production entrypoint; tests exercise kernel explicitly on CPU."""
    srnet_cuda.inspect()  # Fail before opening data or starting any optimizer.
    return kernel(root, manifest_sha256, out, arm, seed, rows, sources, resume=resume)


def kernel(root, manifest_sha256, out, arm, seed, rows, sources, *, resume=None,
           device="cuda:0", factory=srnet_samepool.network, row_budget=PRESENTED_ROWS):
    import torch

    manifest = document(root / "manifest.json", manifest_sha256)
    data.validate_manifest(manifest)
    representation = (jpeg_study_views.FIVE if manifest["schema_version"] == jpeg_study_views.SCHEMA
                      else jpeg_study_views.CENTER)
    selected = recipe(arm, seed, rows, representation=representation)
    if (type(row_budget) is not int or not 1 <= row_budget <= PRESENTED_ROWS
        or row_budget % rows):
        raise ValueError("study row budget invalid")
    # Non-default budgets are test controls, not eligible real-study completion.
    binding = hashlib.sha256(json.dumps(
        {"recipe": selected, "manifest_sha256": manifest_sha256, "sources": sources,
         "torch_version": str(torch.__version__)}, sort_keys=True
    ).encode()).hexdigest()
    started = time.monotonic()

    def deadline():
        if time.monotonic() - started > 1800:
            raise TimeoutError("study trial deadline exceeded")
        for marker in (out / "CANCEL", out.parent / "CANCEL"):
            if marker.is_symlink():
                raise ValueError("study cancellation marker cannot use symlink")
            if marker.exists():
                raise InterruptedError("study trial cancelled")

    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        policy = srnet_cuda.policy() if device == "cuda:0" else nullcontext({"test_only": True})
        with policy as execution, torch.random.fork_rng(devices=[0] if device == "cuda:0" else []):
            if device == "cuda:0":
                torch.cuda.reset_peak_memory_stats(0)
            torch.manual_seed(seed)
            model = factory().to(device).train()
            optimizer = jpeg_study_optimizer.optimizer(
                model, reference=arm != "legacy", corrected_l2=selected["convolution_only_l2"]
            )
            initial_digest = hashlib.sha256()
            for parameter in model.parameters():
                initial_digest.update(parameter.detach().cpu().numpy().tobytes())
            next_row = 0
            if resume is not None:
                snapshot = jpeg_study_state.load(resume["path"], resume["sha256"], binding)
                next_row = jpeg_study_state.restore(snapshot, model, optimizer, binding, rows)
                if next_row >= row_budget:
                    raise ValueError("study resume has no remaining updates")
            records, checkpoints = [], []
            with data.Reader(root, manifest_sha256, "fit", deadline=deadline) as reader:
                batches, angles = schedule(reader.samples, seed, rows, row_budget)
                views = view_schedule(reader.samples, seed, batches) if reader.view_count == 5 \
                    else None
                schedule_hash = hashlib.sha256(
                    batches.tobytes() + angles.tobytes() + (views.tobytes() if views is not None
                                                          else b"")
                ).hexdigest()
                labels = torch.tensor([0, 1] * (rows // 2), device=device)
                for i in range(next_row // rows, len(batches)):
                    deadline()
                    optimizer.zero_grad(set_to_none=True)
                    pixels = augment(reader.batch(batches[i], views[i]) if views is not None
                                     else reader.batch(batches[i]), angles[i])
                    logits = model(torch.from_numpy(pixels).to(device))
                    if logits.shape != (rows, 2) or not bool(torch.isfinite(logits).all()):
                        raise ValueError("study training logits invalid")
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                    objective = loss + (jpeg_study_optimizer.convolution_penalty(model)
                                        if selected["convolution_only_l2"] else 0)
                    if not bool(torch.isfinite(objective)):
                        raise ValueError("study training objective nonfinite")
                    objective.backward()
                    if not srnet_training._finite_gradients(model, device):
                        raise ValueError("study training gradients nonfinite")
                    optimizer.step()
                    if not _finite_state(model, device):
                        raise ValueError("study training state nonfinite")
                    records.append(float(loss.detach()))
                    presented = (i + 1) * rows
                    if presented in {16384, PRESENTED_ROWS}:
                        digest = jpeg_study_state.save(
                            jpeg_study_state.capture(model, optimizer, binding, presented),
                            out / f"rows-{presented}.npz",
                        )
                        checkpoints.append({"path": f"rows-{presented}.npz", "sha256": digest,
                                            "presented_rows": presented})
                reader.verify()
            with data.Reader(root, manifest_sha256, "development", deadline=deadline) as reader:
                scores = evaluate(model, reader, device)
            return {
                "status": "completed", "recipe": selected, "binding": binding,
                "manifest_sha256": manifest_sha256, "execution": execution,
                "initial_learned_sha256": initial_digest.hexdigest(),
                "schedule_sha256": schedule_hash, "presented_rows": row_budget,
                "resumed_from_rows": next_row, "optimizer_updates": len(records),
                "total_optimizer_updates": row_budget // rows,
                "mean_cross_entropy": float(np.mean(records)), "scores": scores,
                "singleton_batch_parity": True, "cells": statistics(reader.samples, scores),
                "checkpoints": checkpoints, "seconds": time.monotonic() - started,
                "real_gpu_fit": device == "cuda:0" and row_budget == PRESENTED_ROWS,
                "process_peak_gpu_allocated_bytes": (
                    torch.cuda.max_memory_allocated(0) if device == "cuda:0" else None),
                "process_peak_gpu_reserved_bytes": (
                    torch.cuda.max_memory_reserved(0) if device == "cuda:0" else None),
                "accuracy_qualification": "unavailable", "deployed": False,
            }
    finally:
        torch.set_num_threads(previous)
