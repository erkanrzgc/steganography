"""Isolated reference-inspired pooling/BN numerics; not a full SRNet replica."""

from __future__ import annotations

from core import srnet, srnet_model

ARCHITECTURE = "srnet-gray12-samepool-bneps1e3-research-v1"


def same_average(values):
    """TensorFlow SAME 3x3/stride2 geometry, excluding padding from divisors."""
    import torch
    import torch.nn.functional as functional

    if values.ndim != 4 or min(values.shape[-2:]) < 3:
        raise ValueError("reference pooling requires a bounded NCHW feature map")
    height, width = values.shape[-2:]
    ph = max(0, ((height + 1) // 2 - 1) * 2 + 3 - height)
    pw = max(0, ((width + 1) // 2 - 1) * 2 + 3 - width)
    padding = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
    padded = functional.pad(values, padding)
    mask = torch.ones((1, 1, height, width), dtype=values.dtype, device=values.device)
    denominator = functional.avg_pool2d(functional.pad(mask, padding), 3, 2, divisor_override=1)
    return functional.avg_pool2d(padded, 3, 2, divisor_override=1) / denominator


def network():
    import torch

    class SamePool(torch.nn.Module):
        def forward(self, values):
            return same_average(values)

    model = srnet.network()
    norms = 0
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eps = 1e-3
            norms += 1
    if norms != 26 or len(model.middle) != 4:
        raise ValueError("reference numerics require the fixed SRNet-style topology")
    for block in model.middle:
        block.pool = SamePool()
    model.architecture = ARCHITECTURE
    srnet_model.validate_tensors(model.state_dict())
    return model
