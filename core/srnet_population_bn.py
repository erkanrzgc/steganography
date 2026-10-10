"""Layerwise train-only population statistics, not detector qualification."""

from __future__ import annotations

import copy

import numpy as np

from core import srnet, srnet_model


def partition(samples):
    if not isinstance(samples, list) or len(samples) != 480:
        raise ValueError("population control requires the complete real kit")
    groups: dict[str, set[str]] = {}
    for row in samples:
        if row.get("split") != "train":
            raise ValueError("population control forbids validation/test rows")
        groups.setdefault(row["source_group"], set()).add(row["lineage"])
    if len(groups) != 3 or any(len(v) != 32 for v in groups.values()):
        raise ValueError("population original accounting mismatch")
    if len(set.union(*groups.values())) != 96:
        raise ValueError("population cross-source original leakage")
    selected = {s: set(sorted(v)[:24]) for s, v in groups.items()}
    calibration: list[int] = []
    probe: list[int] = []
    for i, row in enumerate(samples):
        target = calibration if row["lineage"] in selected[row["source_group"]] else probe
        target.append(i)
    if len(calibration) != 360 or len(probe) != 120:
        raise ValueError("population row accounting mismatch")
    return np.array(calibration, dtype=np.int64).reshape(-1, 4), np.array(
        probe, dtype=np.int64
    ).reshape(-1, 4)


def merge(previous, count, mean, m2):
    """Chan's central-moment merge, with count weighting (never mean of variances)."""
    if type(count) is not int or count < 1:
        raise ValueError("population count must be positive")
    if previous is None:
        return count, mean, m2
    old_count, old_mean, old_m2 = previous
    total = old_count + count
    delta = mean - old_mean
    return (
        total,
        old_mean + delta * (count / total),
        old_m2 + m2 + delta * delta * (old_count * count / total),
    )


class _Captured(Exception):
    pass


def refresh(model, fetch, batches, deadline):
    import torch

    if getattr(model, "architecture", None) != srnet.ARCHITECTURE or any(
        m.training for m in model.modules()
    ):
        raise ValueError("population refresh requires the immutable eval architecture")
    srnet_model.validate_tensors(dict(model.state_dict()))
    if (
        not isinstance(batches, np.ndarray)
        or batches.dtype != np.int64
        or batches.shape != (90, 4)
        or len(set(batches.reshape(-1).tolist())) != 360
        or np.any(batches < 0)
        or np.any(batches >= 480)
    ):
        raise ValueError("population calibration must contain 360 distinct bounded rows")
    clone = copy.deepcopy(model).eval()
    layers = [(n, m) for n, m in clone.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
    if len(layers) != 26 or any(
        not m.track_running_stats
        or m.running_mean is None
        or m.running_var is None
        or m.num_batches_tracked is None
        for _, m in layers
    ):
        raise ValueError("population refresh requires exactly 26 tracked layers")
    summaries = []
    device = next(clone.parameters()).device
    with torch.no_grad():
        for name, layer in layers:
            deadline()
            moments = None

            def capture(module, args):
                nonlocal moments
                deadline()
                values = args[0].detach().to(torch.float64)
                if values.ndim != 4 or values.shape[0] != 4:
                    raise ValueError("population activation shape mismatch")
                if not bool(torch.isfinite(values).all()):
                    raise ValueError("population activation nonfinite")
                variance, mean = torch.var_mean(values, dim=(0, 2, 3), correction=0)
                count = values.shape[0] * values.shape[2] * values.shape[3]
                moments = merge(moments, count, mean, variance * count)
                raise _Captured

            hook = layer.register_forward_pre_hook(capture)
            try:
                for batch in batches:
                    deadline()
                    pixels = fetch(batch)
                    if (
                        pixels.shape != (4, 1, 256, 256)
                        or pixels.dtype != np.dtype("<f4")
                        or not np.isfinite(pixels).all()
                        or np.any(np.abs(pixels) > 2**36)
                    ):
                        raise ValueError("population pixels outside bounds")
                    try:
                        clone(torch.from_numpy(pixels).to(device))
                    except _Captured:
                        pass
                    else:
                        raise ValueError("population target layer not reached")
            finally:
                hook.remove()
            if moments is None:
                raise ValueError("population moments absent")
            count, mean, m2 = moments
            if (
                layer.running_mean is None
                or layer.running_var is None
                or layer.num_batches_tracked is None
            ):
                raise ValueError("population running buffers disappeared")
            layer.running_mean.copy_(mean.to(torch.float32))
            layer.running_var.copy_((m2 / count).to(torch.float32))
            layer.num_batches_tracked.fill_(len(batches))
            srnet_model.validate_tensors(dict(clone.state_dict()))
            summaries.append({"layer": name, "activation_count_per_channel": count})
    allowed = ("running_mean", "running_var", "num_batches_tracked")
    if any(
        not torch.equal(value, model.state_dict()[name])
        for name, value in clone.state_dict().items()
        if not name.endswith(allowed)
    ):
        raise ValueError("population refresh changed learned weights")
    deadline()
    return clone, summaries
