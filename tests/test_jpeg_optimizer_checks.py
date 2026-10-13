"""Optimizer safety equivalence and atomic rejection; not GPU throughput."""

from collections import defaultdict

import numpy as np
import pytest

from core import jpeg_study_optimizer as optimizer

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("consolidate", [False, True])
@pytest.mark.parametrize("fault", [None, "missing", "sparse", "parameter_nan", "gradient_inf",
    "extra_key", "state_nan", "negative_norm", "shape", "dtype", "step_shape", "step_zero",
    "step_fraction", "step_inf"])
def test_every_parameter_gradient_and_slot_checked_before_update(consolidate, fault):
    parameters = [torch.nn.Parameter(torch.ones(2)) for _ in range(3)]
    states = defaultdict(dict)
    for parameter in parameters:
        parameter.grad = torch.tensor([-.1, .2])
        states[parameter] = {"step": torch.tensor(2.), "exp_avg": torch.ones(2),
                             "exp_inf": torch.ones(2)}
    last, state = parameters[-1], states[parameters[-1]]
    if fault == "missing":
        last.grad = None
    elif fault == "sparse":
        last.grad = last.grad.to_sparse()
    elif fault == "parameter_nan":
        with torch.no_grad():
            last[-1] = float("nan")
    elif fault == "gradient_inf":
        last.grad[-1] = float("inf")
    elif fault == "extra_key":
        state["unexpected"] = torch.tensor(1.)
    elif fault == "state_nan":
        state["exp_avg"][-1] = float("nan")
    elif fault == "negative_norm":
        state["exp_inf"][-1] = -1
    elif fault == "shape":
        state["exp_avg"] = torch.ones(3)
    elif fault == "dtype":
        state["exp_inf"] = state["exp_inf"].double()
    elif fault == "step_shape":
        state["step"] = torch.ones(2)
    elif fault is not None:
        state["step"] = torch.tensor({"step_zero": 0., "step_fraction": 1.5,
                                      "step_inf": float("inf")}[fault])
    before = [parameter.detach().clone() for parameter in parameters]
    if fault is None:
        optimizer.validate_reference(parameters, states, consolidate=consolidate)
    else:
        with pytest.raises(ValueError):
            optimizer.validate_reference(parameters, states, consolidate=consolidate)
    for parameter, original in zip(parameters, before, strict=True):
        torch.testing.assert_close(parameter, original, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("consolidate", [False, True])
def test_empty_slots_allowed_and_reference_equations_unchanged(consolidate, monkeypatch):
    original = optimizer.validate_reference
    monkeypatch.setattr(optimizer, "validate_reference", lambda params, states:
                        original(params, states, consolidate=consolidate))
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    instance = optimizer.optimizer(model, reference=True, corrected_l2=True)
    weights, mean, norm = np.zeros((1, 2)), np.zeros((1, 2)), np.zeros((1, 2))
    for values in ([.01, -.02], [.005, -.00001], [-.04, .03], [0., 0.]):
        gradient = np.asarray([values], dtype=np.float64)
        model.weight.grad = torch.tensor(gradient, dtype=torch.float32)
        mean = .9*mean + .1*gradient
        norm = np.maximum(.999*norm + 1e-8, np.abs(gradient))
        weights -= .001*mean/norm
        instance.step()
        np.testing.assert_allclose(model.weight.detach().numpy(), weights, rtol=1e-6, atol=1e-9)


def test_all_parameters_remain_unchanged_if_late_slot_invalid():
    model = torch.nn.Linear(2, 2)
    instance = optimizer.optimizer(model, reference=True)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    instance.step()
    parameters = list(model.parameters())
    before = [parameter.detach().clone() for parameter in parameters]
    instance.state[parameters[-1]]["exp_inf"].fill_(-1)
    with pytest.raises(ValueError):
        instance.step()
    assert all(torch.equal(parameter, original)
               for parameter, original in zip(parameters, before, strict=True))
