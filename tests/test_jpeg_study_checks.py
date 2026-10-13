"""Complete safety-check semantics; CPU tensors do not prove CUDA speedup."""

from types import SimpleNamespace

import numpy as np
import pytest

from core import jpeg_study as study

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("fault", [None, "nan", "inf", "negative_inf", "empty"])
def test_all_state_tensors_and_integer_buffers_are_checked_without_mutation(device, fault):
    values = {"weight": torch.tensor([0., torch.finfo(torch.float32).max]),
              "counter": torch.tensor(8, dtype=torch.int64),
              "running_mean": torch.tensor([-1., 2.])}
    if fault == "empty":
        values = {}
    elif fault is not None:
        values["running_mean"][-1] = {"nan": float("nan"), "inf": float("inf"),
                                      "negative_inf": -float("inf")}[fault]
    model = SimpleNamespace(state_dict=lambda: values)
    before = {name: value.clone() for name, value in values.items()}
    assert study._finite_state(model, device) is (fault is None)
    for name, value in values.items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0, equal_nan=True)


def test_state_consolidation_reduces_scalar_reads_not_elements_checked(monkeypatch):
    model = SimpleNamespace(state_dict=lambda: {str(i): torch.ones(8) for i in range(13)})
    original = torch.Tensor.__bool__
    reads = []

    def counted(value):
        reads.append(value.shape)
        return original(value)

    monkeypatch.setattr(torch.Tensor, "__bool__", counted)
    assert study._finite_state(model, "cpu") and len(reads) == 13
    reads.clear()
    assert study._finite_state(model, "cuda:0") and len(reads) == 1


@pytest.mark.parametrize("arm,rows", [("legacy", 4), ("reference-adamax", 4),
                                    ("convolution-l2", 4), ("genuine-batch", 8)])
def test_actual_cpu_updates_losses_and_state_match_serial_checks(arm, rows, monkeypatch):
    from core import jpeg_study_optimizer

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        def learn(consolidated):
            torch.manual_seed(20261012)
            model = torch.nn.Sequential(torch.nn.Conv2d(1, 3, 3), torch.nn.BatchNorm2d(3),
                                        torch.nn.ReLU(), torch.nn.AdaptiveAvgPool2d(1),
                                        torch.nn.Flatten(), torch.nn.Linear(3, 2))
            optimizer = jpeg_study_optimizer.optimizer(
                model, reference=arm != "legacy", corrected_l2=arm in study.ARMS[2:])
            values = torch.from_numpy(np.random.default_rng(42).normal(
                size=(rows, 1, 8, 8)).astype("f4"))
            labels = torch.tensor([0, 1] * (rows // 2))
            records = []
            for _ in range(4):
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(values), labels)
                objective = loss + (jpeg_study_optimizer.convolution_penalty(model)
                                    if arm in study.ARMS[2:] else 0)
                objective.backward()
                if consolidated:
                    assert study.srnet_training._finite_gradients(model, "cuda:0")
                else:
                    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                               for p in model.parameters())
                optimizer.step()
                if consolidated:
                    assert study._finite_state(model, "cuda:0")
                else:
                    assert all(bool(torch.isfinite(value).all())
                               for value in model.state_dict().values())
                records.append(float(loss.detach()))
            return model.state_dict(), optimizer.state_dict(), records

        baseline, old_optimizer, old_losses = learn(False)
        candidate, new_optimizer, new_losses = learn(True)
        assert old_losses == new_losses
        for name, value in baseline.items():
            assert torch.equal(value, candidate[name])
        assert old_optimizer["param_groups"] == new_optimizer["param_groups"]
        for parameter, slots in old_optimizer["state"].items():
            for slot, value in slots.items():
                assert torch.equal(value, new_optimizer["state"][parameter][slot])
    finally:
        torch.set_num_threads(previous)
