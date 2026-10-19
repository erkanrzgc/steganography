"""Generated derivative corruption, scalar arithmetic and exact resume gates."""

import numpy as np
import pytest

from core import jpeg_refine_preflight_v2 as p
from core import jpeg_refine_training as training
from core import srnet_batch_probe
from tests.test_jpeg_refine import tiny as tiny  # noqa: F401


def test_independent_zero_tiny_and_sign_changing_optimizer_equations():
    pytest.importorskip("torch")
    assert p.optimizer_oracle("cpu")


@pytest.mark.parametrize("fault", ["parameter", "average", "infinity", "step"])
def test_scalar_oracle_rejects_wrong_arithmetic(monkeypatch, fault):
    torch = pytest.importorskip("torch")
    original = p.rule.optimizer

    def optimizer(model, **kwargs):
        result = original(model, **kwargs)
        step = result.step

        def broken():
            step()
            parameter = next(model.parameters())
            with torch.no_grad():
                if fault == "parameter":
                    parameter.add_(0.01)
                else:
                    key = {"average": "exp_avg", "infinity": "exp_inf", "step": "step"}[fault]
                    result.state[parameter][key].add_(0.01)

        result.step = broken
        return result

    monkeypatch.setattr(p.rule, "optimizer", optimizer)
    with pytest.raises(ValueError, match="scalar oracle"):
        p.optimizer_oracle("cpu")


def test_generated_full_gate_restores_threads_and_preserves_exact_resume(tiny):
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    result = p.probe(device="cpu", factory=tiny)
    assert result["generated_optimizer_updates"] == 22
    assert result["derivative_parity"] and result["optimizer_scalar_oracle"]
    assert result["save_resume_parity"] and not result["real_data_used"]
    assert result["max_update_seconds"] > 0
    assert result["startup_checkpoint_seconds"] > 0
    assert torch.get_num_threads() == previous


@pytest.mark.parametrize("fault", ["logits", "loss", "gradient", "buffer", "missing_gradient"])
def test_corrupt_derivatives_or_state_fail_before_optimizer(tiny, monkeypatch, fault):
    torch = pytest.importorskip("torch")
    cpu, other = tiny().train(), tiny().train()
    other.load_state_dict(cpu.state_dict())
    original = p.backward

    def backward(model, *args):
        logits, losses = original(model, *args)
        if model is other:
            if fault == "logits":
                logits = logits + 1
            elif fault == "loss":
                losses = (100.0, 100.0)
            elif fault == "gradient":
                next(model.parameters()).grad.add_(1)
            elif fault == "missing_gradient":
                next(model.parameters()).grad = None
            else:
                with torch.no_grad():
                    next(b for n, b in model.named_buffers() if n.endswith("running_mean")).add_(1)
        return logits, losses

    monkeypatch.setattr(p, "backward", backward)
    with pytest.raises(ValueError, match="parity"):
        p.derivative_parity(cpu, other, srnet_batch_probe.generated(16), np.arange(16), "cpu")


@pytest.mark.parametrize("fault", ["logits", "objective"])
def test_nonfinite_or_wrong_output_is_rejected(tiny, fault):
    torch = pytest.importorskip("torch")
    model = tiny().train()
    if fault == "logits":
        model.register_forward_hook(lambda *args: torch.ones(16, 3))
    else:
        with torch.no_grad():
            next(model.parameters()).fill_(float("inf"))
        # Keep logits finite while independently making regularization invalid.
        model.register_forward_hook(lambda *args: torch.ones(16, 2, requires_grad=True))
    with pytest.raises(ValueError, match="invalid"):
        p.backward(model, srnet_batch_probe.generated(16), np.arange(16), "cpu")


@pytest.mark.parametrize("fault", ["model", "optimizer", "shape"])
def test_resume_rejects_state_corruption(tiny, monkeypatch, fault):
    torch = pytest.importorskip("torch")
    original = training.update
    calls = 0

    def update(model, optimizer, *args):
        nonlocal calls
        calls += 1
        result = original(model, optimizer, *args)
        # Eight timing updates, then original and restored models.
        if calls == 10:
            with torch.no_grad():
                if fault == "model":
                    next(model.parameters()).add_(1)
                elif fault == "optimizer":
                    next(iter(optimizer.state.values()))["step"].add_(1)
                else:
                    parameter = next(model.parameters())
                    slots = optimizer.state[next(iter(optimizer.state))]
                    slots["step"] = torch.ones_like(parameter)
        return result

    monkeypatch.setattr(training, "update", update)
    with pytest.raises(ValueError, match="parity"):
        p.probe(device="cpu", factory=tiny)
