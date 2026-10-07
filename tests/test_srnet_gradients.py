"""State restoration, bounded training-only probes and finite-difference checks."""

import copy
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from core import srnet, srnet_positive
from core import srnet_gradients as gradients
from steganography import research_srnet_gradients as service


@pytest.fixture
def model():
    import torch

    previous = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261012)
        net = srnet.network().eval()
    torch.set_num_threads(2)
    yield net
    torch.set_num_threads(previous)


def pixels():
    values, samples = srnet_positive.generate()
    batch = gradients.srnet_multibatch.epoch_batches(samples, seed=20261012, epoch=0)[0][0]
    return values[batch].copy()


def samples():
    rows = []
    for source, count, qualities in (("ALASKA2", 4, (None,)), ("BOSSbase-1.01", 2, (75, 95))):
        for lineage in range(count):
            for quality in qualities:
                for method in (None, "JUNIWARD", "UERD"):
                    rows.append(
                        {
                            "split": "train",
                            "format": "JPEG",
                            "source_group": source,
                            "lineage": str(lineage),
                            "quality_factor": quality,
                            "method": method,
                            "label": "cover" if method is None else "stego",
                            "sha256": hashlib.sha256(
                                f"{source}/{lineage}/{quality}/{method}".encode()
                            ).hexdigest(),
                        }
                    )
    return rows


def test_selection_metadata_only_complete_cells_and_no_mutation():
    rows = samples()
    before = copy.deepcopy(rows)
    chosen, record = gradients.select_batches(rows)
    assert rows == before and 1 <= len(chosen) <= 6 and record["optimizer_updates"] == 8
    cells = {
        (rows[i]["source_group"], rows[i]["quality_factor"], rows[i]["method"])
        for batch in chosen
        for i in (batch[1], batch[3])
    }
    assert len(cells) == 6
    assert len({tuple(b) for b in chosen}) == len(chosen)
    for row in rows:
        row["score"] = -1000
    assert gradients.select_batches(rows)[0] == chosen
    with pytest.raises(ValueError):
        gradients.select_batches(srnet_positive.generate()[1])


def snapshot(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def assert_restored(model, before):
    import torch

    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert all(not m.training for m in model.modules())
    assert all(
        m.track_running_stats for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)
    )


def test_actual_gradient_and_classifier_central_difference_preserve_everything(model):
    import torch

    original = snapshot(model)
    data = pixels()
    unchanged = data.copy()
    model.classifier.weight.grad = torch.ones_like(model.classifier.weight)
    existing = model.classifier.weight.grad
    rng = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    result = gradients.probe(model, data)
    assert_restored(model, original)
    assert model.classifier.weight.grad is existing and bool((existing == 1).all())
    assert torch.equal(rng, torch.random.get_rng_state()) and torch.get_num_threads() == threads
    np.testing.assert_array_equal(data, unchanged)
    assert result["classifier_directional_check"]["passed"]
    assert result["true_gradient_l2"] > 0 and result["gradient_difference_l2"] > 0
    assert result["input_gradient_l2"] > 0 and result["input_difference_rms"] > 0
    assert len(result["parameters"]) == len(list(model.parameters()))
    assert result["state_unchanged"] and result["batch_statistics_training_diagnostic_only"]


def test_invalid_inputs_architecture_cpu_and_disabled_parameters(model, monkeypatch):
    import torch

    for data in (
        np.zeros((2, 1, 256, 256), dtype="<f4"),
        np.zeros((4, 1, 256, 256), dtype="u1"),
        np.full((4, 1, 256, 256), np.nan, dtype="<f4"),
        np.full((4, 1, 256, 256), 2**37, dtype="<f4"),
    ):
        with pytest.raises(ValueError):
            gradients.probe(model, data)
    model.train()
    with pytest.raises(ValueError):
        gradients.probe(model, pixels())
    model.eval()
    model.classifier.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="enabled"):
        gradients.probe(model, pixels())
    model.classifier.weight.requires_grad_(True)
    original_modules = model.modules
    monkeypatch.setattr(model, "modules", lambda: iter([model]))
    with pytest.raises(ValueError, match="architecture"):
        gradients.probe(model, pixels())
    monkeypatch.setattr(model, "modules", original_modules)
    original_state = model.state_dict
    monkeypatch.setattr(model, "state_dict", lambda: {"x": torch.empty(1, device="meta")})
    with pytest.raises(ValueError, match="CPU"):
        gradients.probe(model, pixels())
    monkeypatch.setattr(model, "state_dict", original_state)


@pytest.mark.parametrize("failure", ["logits", "loss", "gradients", "zero_classifier", "exception"])
def test_failure_restores_state_flags_threads_rng(model, monkeypatch, failure):
    import torch

    saved, rng = snapshot(model), torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    if failure == "logits":
        monkeypatch.setattr(model, "forward", lambda data: torch.zeros((4, 3)))
    elif failure == "loss":
        monkeypatch.setattr(
            torch.nn.functional, "cross_entropy", lambda *a: torch.tensor(float("nan"))
        )
    elif failure in {"gradients", "zero_classifier"}:
        original = torch.autograd.grad

        def replaced(*args, **kwargs):
            result = original(*args, **kwargs)
            return tuple(
                torch.full_like(g, float("nan")) if failure == "gradients" else torch.zeros_like(g)
                for g in result
            )

        monkeypatch.setattr(torch.autograd, "grad", replaced)
    else:

        def fail(data):
            torch.rand(1)
            raise RuntimeError("unavailable")

        monkeypatch.setattr(model, "forward", fail)
    with pytest.raises((ValueError, RuntimeError)):
        gradients.probe(model, pixels())
    assert_restored(model, saved)
    assert torch.equal(rng, torch.random.get_rng_state()) and torch.get_num_threads() == threads


def test_mutation_rejected_restored_and_bad_finite_difference_retained(model, monkeypatch):
    import torch

    saved = snapshot(model)

    def mutate(module, args, output):
        with torch.no_grad():
            model.front[0][1].running_mean.add_(1)

    hook = model.register_forward_hook(mutate)
    with pytest.raises(ValueError, match="mutation"):
        gradients.probe(model, pixels())
    assert_restored(model, saved)
    hook.remove()
    ce, count = torch.nn.functional.cross_entropy, 0

    def biased(*args, **kwargs):
        nonlocal count
        count += 1
        return ce(*args, **kwargs) + (1 if count == 3 else 0)

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", biased)
    assert not gradients.probe(model, pixels())["classifier_directional_check"]["passed"]
    assert_restored(model, saved)


@pytest.fixture
def fake_service(monkeypatch):
    model = SimpleNamespace(
        named_buffers=lambda: [("bn.num_batches_tracked", 400)], eval=lambda: model
    )
    rows = samples()
    config = {
        "manifest": "manifest",
        "manifest_sha256": service.MANIFEST_SHA,
        "cache": "cache",
        "cache_sha256": service.CACHE_SHA,
    }
    monkeypatch.setattr(service, "read_document", lambda path: ({}, service.MANIFEST_SHA))
    monkeypatch.setattr(
        service,
        "load_pixels",
        lambda *a, **kw: (
            np.zeros((24, 1, 256, 256), dtype="<f4"),
            rows,
            {"data_sha256": "a" * 64},
        ),
    )
    monkeypatch.setattr(service.srnet_model, "load_model", lambda *a, **kw: model)
    monkeypatch.setattr(service.srnet, "network", lambda: model)
    monkeypatch.setattr(
        service.srnet_gradients,
        "probe",
        lambda *a: {"classifier_directional_check": {"passed": True}, "state_unchanged": True},
    )
    return config, model


def test_service_all_cells_both_models_generated_and_no_paths(fake_service, tmp_path):
    config, _ = fake_service
    result = service.run(config, tmp_path / "model.npz", tmp_path / "result.json")
    assert result["status"] == "completed" and result["classifier_directional_checks_passed"]
    assert len(result["probes"]) == 2 * len(result["selected_batches"]) + 1
    assert {p["kind"] for p in result["probes"]} == {
        "real_fresh",
        "real_failed_tiny_model",
        "generated_strong_fresh",
    }
    assert result["optimizer_steps"] == 0 and not result["model_saved"]
    assert not result["validation_pixels_loaded"] and not result["primary_detection_changed"]
    assert result["accuracy_qualification"] == "unavailable"
    assert str(tmp_path) not in (tmp_path / "result.json").read_text()
    with pytest.raises(FileExistsError):
        service.run(config, tmp_path / "model.npz", tmp_path / "result.json")
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(FileExistsError):
        service.run(config, tmp_path / "model.npz", tmp_path / "link/new.json")


def test_service_identity_protocol_bn_and_deadline_rejections(fake_service, tmp_path, monkeypatch):
    config, model = fake_service
    with pytest.raises(ValueError):
        service.run({**config, "validation": "forbidden"}, tmp_path / "model", tmp_path / "out")
    with pytest.raises(ValueError):
        service.run({**config, "cache_sha256": "b" * 64}, tmp_path / "model", tmp_path / "out")
    monkeypatch.setattr(service, "PROTOCOL_SHA", "b" * 64)
    with pytest.raises(ValueError, match="protocol"):
        service.run(config, tmp_path / "model", tmp_path / "out")
    monkeypatch.setattr(
        service, "PROTOCOL_SHA", "3b632bd4181e56a5e1f5fe1a696cf0443222186e419ab8d856e35ce6240f5ad1"
    )
    monkeypatch.setattr(service, "read_document", lambda path: ({}, "b" * 64))
    with pytest.raises(ValueError, match="manifest"):
        service.run(config, tmp_path / "model", tmp_path / "out")
    monkeypatch.setattr(service, "read_document", lambda path: ({}, service.MANIFEST_SHA))
    model.named_buffers = lambda: [("bn.num_batches_tracked", 399)]
    with pytest.raises(ValueError, match="BN"):
        service.run(config, tmp_path / "model", tmp_path / "out")
    model.named_buffers = lambda: [("bn.num_batches_tracked", 400)]
    times = iter((0, 181))
    monkeypatch.setattr(service.time, "monotonic", lambda: next(times))
    with pytest.raises(ValueError, match="deadline"):
        service.run(config, tmp_path / "model", tmp_path / "out")
    times = iter((0, 0, 181))
    with pytest.raises(ValueError, match="deadline"):
        service.run(config, tmp_path / "model", tmp_path / "out")


def test_cli_hard_limits_success_failed_gate_and_secret_redaction(
    fake_service, tmp_path, monkeypatch, capsys
):
    import resource

    config, _ = fake_service
    limits = []
    monkeypatch.setattr(resource, "setrlimit", lambda k, v: limits.append((k, v)))
    monkeypatch.setattr(
        sys,
        "argv",
        ["gradient", "--config", "config", "--model", "model", "--out", str(tmp_path / "out")],
    )
    monkeypatch.setattr(service, "read_document", lambda path: (config, "a" * 64))
    monkeypatch.setattr(service, "run", lambda *a: {"classifier_directional_checks_passed": True})
    assert service.main() == 0 and len(limits) == 4
    assert (resource.RLIMIT_CPU, (580, 581)) in limits
    monkeypatch.setattr(service, "run", lambda *a: {"classifier_directional_checks_passed": False})
    assert service.main() == 2

    def fail(*args):
        raise ValueError("secret /host/path")

    monkeypatch.setattr(service, "run", fail)
    assert service.main() == 2 and "secret" not in capsys.readouterr().out
