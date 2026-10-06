"""Read-only paired BN-mode contrast; never a detector or calibration mode."""

from __future__ import annotations

import numpy as np

from core import srnet, srnet_model


def paired_probe(model, pixels):
    if (
        getattr(model, "architecture", None) != srnet.ARCHITECTURE
        or any(m.training for m in model.modules())
        or not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.dtype("<f4")
        or pixels.shape != (2, 1, 256, 256)
        or not np.isfinite(pixels).all()
        or np.any(np.abs(pixels) > 2**36)
    ):
        raise ValueError("BN diagnosis requires an eval model and one bounded float256 pair")
    import torch

    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    srnet_model.validate(arrays)
    saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
    flags = [(m, m.training) for m in model.modules()]
    layers = [(name, m) for name, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
    if len(layers) != 26:
        raise ValueError("BN diagnosis requires the exact architecture")
    tracking = [(m, m.track_running_stats) for _, m in layers]
    hooks = []
    summaries: dict[str, list[dict]] = {}
    mode = "native"

    def capture(name, module, args):
        values = args[0].detach().to(torch.float64)
        mean = values.mean(dim=(0, 2, 3))
        variance = values.var(dim=(0, 2, 3), unbiased=True)
        shift = torch.abs(mean - saved[name + ".running_mean"]) / torch.sqrt(
            saved[name + ".running_var"] + module.eps
        )
        ratio = (variance + module.eps) / (saved[name + ".running_var"] + module.eps)
        if not bool(torch.isfinite(shift).all() and torch.isfinite(ratio).all()):
            raise ValueError("BN diagnosis produced nonfinite layer statistics")
        summaries[mode].append(
            {
                "layer": name,
                "median_standardized_mean_shift": float(np.median(shift.cpu().numpy())),
                "maximum_standardized_mean_shift": float(shift.max()),
                "median_variance_ratio": float(np.median(ratio.cpu().numpy())),
                "maximum_variance_ratio": float(ratio.max()),
            }
        )

    results = {}
    try:
        for name, module in layers:
            hooks.append(
                module.register_forward_pre_hook(
                    lambda module, args, name=name: capture(name, module, args)
                )
            )
        with torch.no_grad():
            for mode in ("native", "batch_statistics"):
                summaries[mode] = []
                if mode == "batch_statistics":
                    for _, module in layers:
                        module.training, module.track_running_stats = True, False
                logits = model(torch.from_numpy(pixels.copy())).detach().cpu().numpy()
                if logits.shape != (2, 2) or not np.isfinite(logits).all():
                    raise ValueError("BN diagnosis produced invalid logits")
                if len(summaries[mode]) != 26:
                    raise ValueError("BN diagnosis incomplete layer coverage")
                scores = np.exp(-np.logaddexp(0, logits[:, 0].astype(np.float64) - logits[:, 1]))
                margins = logits[:, 1].astype(np.float64) - logits[:, 0]
                loss = (np.logaddexp(0, margins[0]) + np.logaddexp(0, -margins[1])) / 2
                results[mode] = {
                    "logits": logits.tolist(),
                    "scores": scores.tolist(),
                    "stego_decisions": (logits[:, 1] >= logits[:, 0]).tolist(),
                    "paired_cross_entropy": float(loss),
                    "layers": summaries[mode],
                }
    finally:
        for hook in hooks:
            hook.remove()
        for module, training in flags:
            module.training = training
        for module, tracked in tracking:
            module.track_running_stats = tracked
        with torch.no_grad():
            changed = any(
                not torch.equal(value, saved[k]) for k, value in model.state_dict().items()
            )
            model.load_state_dict(saved, strict=True)
    if changed:
        raise ValueError("BN diagnosis attempted to mutate model state")
    return {"state_unchanged": True, **results}
