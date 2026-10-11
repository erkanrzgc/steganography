"""Isolated author-rule Adamax and convolution-only L2 research controls.

Equations are independently implemented, not copied from research-use SRNet.
This is not a claim of whole-model TensorFlow parity. Legacy trainers stay intact.
"""

from __future__ import annotations

import math

RULE = "srnet-author-adamax-no-bias-correction-v1"
L2_SCALE = 2e-4


def optimizer(model, *, reference=False, corrected_l2=False, learning_rate=0.001):
    import torch

    if (
        type(reference) is not bool or type(corrected_l2) is not bool
        or type(learning_rate) not in (float, int)
        or not math.isfinite(learning_rate) or not 0 < learning_rate <= 0.001
    ):
        raise ValueError("invalid research optimizer settings")
    decay = 0.0 if corrected_l2 else 0.0001
    if not reference:
        return torch.optim.Adamax(
            model.parameters(), lr=learning_rate, weight_decay=decay,
            betas=(0.9, 0.999), eps=1e-8, foreach=False,
        )

    class ReferenceAdamax(torch.optim.Optimizer):
        def __init__(self, parameters):
            super().__init__(parameters, {"lr": learning_rate, "weight_decay": decay})

        @torch.no_grad()
        def step(self, closure=None):
            if closure is not None:
                raise ValueError("research optimizer does not execute closures")
            # Validate every parameter/gradient before changing any parameter.
            for group in self.param_groups:
                for parameter in group["params"]:
                    gradient = parameter.grad
                    if (
                        parameter.dtype != torch.float32 or gradient is None
                        or gradient.is_sparse or gradient.dtype != torch.float32
                        or not bool(torch.isfinite(parameter).all())
                        or not bool(torch.isfinite(gradient).all())
                    ):
                        raise ValueError("reference Adamax requires finite dense FP32 gradients")
                    state = self.state[parameter]
                    if state and (
                        set(state) != {"step", "exp_avg", "exp_inf"}
                        or any(not bool(torch.isfinite(v).all()) for v in state.values())
                        or bool((state["exp_inf"] < 0).any())
                        or state["exp_avg"].shape != parameter.shape
                        or state["exp_inf"].shape != parameter.shape
                        or state["step"].shape != torch.Size([])
                        or any(v.dtype != torch.float32 for v in state.values())
                        or float(state["step"]) < 1
                        or not float(state["step"]).is_integer()
                    ):
                        raise ValueError("reference Adamax state invalid")
            for group in self.param_groups:
                for parameter in group["params"]:
                    gradient = parameter.grad.add(parameter, alpha=group["weight_decay"])
                    state = self.state[parameter]
                    if not state:
                        state["step"] = torch.zeros((), device=parameter.device)
                        state["exp_avg"] = torch.zeros_like(parameter)
                        state["exp_inf"] = torch.zeros_like(parameter)
                    state["step"].add_(1)
                    state["exp_avg"].mul_(0.9).add_(gradient, alpha=0.1)
                    torch.maximum(
                        state["exp_inf"].mul_(0.999).add_(1e-8), gradient.abs(),
                        out=state["exp_inf"],
                    )
                    parameter.addcdiv_(state["exp_avg"], state["exp_inf"], value=-group["lr"])

    return ReferenceAdamax(model.parameters())


def convolution_penalty(model):
    """2e-4 * half squared norm: gradient 2e-4 W, excluding BN/bias/FC."""
    import torch

    weights = [m.weight for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    if not weights:
        raise ValueError("convolution-only regularization requires convolution weights")
    return sum(w.square().sum() for w in weights) * (L2_SCALE / 2)
