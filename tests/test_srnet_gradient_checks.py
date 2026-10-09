"""Consolidated-check semantics on CPU tensors, not physical CUDA timing."""

from types import SimpleNamespace

import numpy as np
import pytest

from core import srnet_training as training

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("fault", [None, "missing", "nan", "inf", "negative_inf", "empty"])
def test_all_gradients_checked_including_last_parameter(device, fault):
    tensors = [torch.nn.Parameter(torch.zeros(3)) for _ in range(3)]
    for tensor in tensors:
        tensor.grad = torch.tensor([0.0, -1.0, torch.finfo(torch.float32).max])
    if fault == "missing":
        tensors[-1].grad = None
    elif fault in {"nan", "inf", "negative_inf"}:
        tensors[-1].grad[-1] = {
            "nan": float("nan"),
            "inf": float("inf"),
            "negative_inf": -float("inf"),
        }[fault]
    elif fault == "empty":
        tensors = []
    model = SimpleNamespace(parameters=lambda: iter(tensors))
    before = [None if p.grad is None else p.grad.clone() for p in tensors]
    assert training._finite_gradients(model, device) is (fault is None)
    for parameter, original in zip(tensors, before, strict=True):
        if original is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, original, rtol=0, atol=0, equal_nan=True)


def test_consolidated_boolean_reduces_host_scalar_reads(monkeypatch):
    tensors = [torch.nn.Parameter(torch.zeros(2)) for _ in range(9)]
    for tensor in tensors:
        tensor.grad = torch.ones_like(tensor)
    model = SimpleNamespace(parameters=lambda: iter(tensors))
    original = torch.Tensor.__bool__
    reads = []

    def counted(value):
        reads.append(value.shape)
        return original(value)

    monkeypatch.setattr(torch.Tensor, "__bool__", counted)
    assert training._finite_gradients(model, "cpu")
    assert len(reads) == 9
    reads.clear()
    assert training._finite_gradients(model, "cuda:0")
    assert len(reads) == 1


def test_actual_generated_updates_identical_with_consolidated_check(monkeypatch):
    """Actual CPU arithmetic, injected check only; no emulated GPU speed claim."""
    previous = torch.get_num_threads()
    inputs = np.arange(4 * 256 * 256, dtype=np.float32).reshape(4, 1, 256, 256) % 255
    original = training._finite_gradients

    def learn():
        return training._learn(
            fetch=lambda _: inputs.copy(),
            batches=lambda _: np.array([[0, 1, 2, 3], [2, 3, 0, 1]]),
            epochs=1,
            seed=91,
            params=training.settings({"threads": 1, "max_seconds": 180}),
            target_pairs=2,
        )

    baseline, records = learn()
    monkeypatch.setattr(training, "_finite_gradients", lambda model, _: original(model, "cuda:0"))
    candidate, candidate_records = learn()
    assert records == candidate_records and records[0]["updates"] == 2
    assert torch.get_num_threads() == previous
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name
