"""CPU tensor contract parity; not CUDA hardware correctness or speed evidence."""

import numpy as np
import pytest

from core import srnet_model as state
from core import srnet_training as training

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def valid():
    return {
        name: torch.zeros(
            shape, dtype=torch.int64 if name.endswith("num_batches_tracked") else torch.float32
        )
        for name, shape in state.SHAPES.items()
    }


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "nan",
        "inf",
        "negative_inf",
        "variance",
        "counter_low",
        "counter_high",
        "shape",
        "dtype",
        "missing",
        "extra",
    ],
)
def test_matches_numpy_state_contract(fault):
    tensors = valid()
    name = "classifier.weight"
    if fault in {"nan", "inf", "negative_inf"}:
        tensors[name][0, 0] = -float("inf") if fault == "negative_inf" else float(fault)
    elif fault == "variance":
        tensors["tail.1.1.running_var"][-1] = -1
    elif fault in {"counter_low", "counter_high"}:
        tensors["tail.1.1.num_batches_tracked"].fill_(-1 if fault == "counter_low" else 10_000_001)
    elif fault == "shape":
        tensors[name] = tensors[name][:1]
    elif fault == "dtype":
        tensors[name] = tensors[name].double()
    elif fault == "missing":
        del tensors[name]
    elif fault == "extra":
        tensors["unexpected"] = torch.zeros(1)
    arrays = {k: v.numpy() for k, v in tensors.items()}
    if fault is None:
        state.validate(arrays)
        state.validate_tensors(tensors)
    else:
        with pytest.raises(ValueError):
            state.validate(arrays)
        with pytest.raises(ValueError):
            state.validate_tensors(tensors)


@pytest.mark.parametrize("fault", ["not_tensor", "meta", "sparse", "not_dict"])
def test_rejects_unsupported_tensor_containers(fault):
    tensors = valid()
    if fault == "not_dict":
        tensors = list(tensors.values())
    else:
        tensors["classifier.weight"] = {
            "not_tensor": np.zeros((2, 512)),
            "meta": torch.empty((2, 512), device="meta"),
            "sparse": torch.zeros((2, 512)).to_sparse(),
        }[fault]
    with pytest.raises(ValueError):
        state.validate_tensors(tensors)


def test_boundary_values_and_no_state_mutation_or_cpu_transfer(monkeypatch):
    tensors = valid()
    tensors["classifier.weight"].fill_(torch.finfo(torch.float32).max)
    tensors["tail.1.1.num_batches_tracked"].fill_(10_000_000)
    arrays = {k: v.numpy().copy() for k, v in tensors.items()}
    state.validate(arrays)
    reads = []
    original = torch.Tensor.__bool__

    def scalar(value):
        reads.append(value.numel())
        return original(value)

    monkeypatch.setattr(torch.Tensor, "__bool__", scalar)
    monkeypatch.setattr(torch.Tensor, "cpu", lambda _: pytest.fail("unexpected CPU transfer"))
    state.validate_tensors(tensors)
    assert reads == [1]
    for name, value in tensors.items():
        np.testing.assert_array_equal(value.numpy(), arrays[name])


@pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
def test_rejects_mixed_and_nonprimary_devices_before_numeric_work(monkeypatch, device):
    """CPU metadata stand-in only; no CUDA tensors or hardware claim."""
    tensors = valid()
    rogue = tensors["classifier.weight"]
    monkeypatch.setattr(
        torch.Tensor,
        "device",
        property(lambda value: torch.device(device if value is rogue else "cpu")),
    )
    with pytest.raises(ValueError, match="device"):
        state.validate_tensors(tensors)


def test_accepts_valid_noncontiguous_values():
    tensors = valid()
    tensors["classifier.weight"] = torch.arange(1024, dtype=torch.float32).reshape(512, 2).T
    assert not tensors["classifier.weight"].is_contiguous()
    state.validate({k: v.numpy() for k, v in tensors.items()})
    state.validate_tensors(tensors)


def test_generated_updates_preserve_exact_state_and_losses(monkeypatch):
    inputs = np.arange(4 * 256 * 256, dtype=np.float32).reshape(4, 1, 256, 256) % 255

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
    original = state.validate_tensors
    monkeypatch.setattr(
        state,
        "validate",
        lambda arrays: original({k: torch.from_numpy(v) for k, v in arrays.items()}),
    )
    candidate, other = learn()
    assert records == other
    for name, value in baseline.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[name]), name
