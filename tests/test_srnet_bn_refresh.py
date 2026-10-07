"""Frozen BN controls preserve source state and never qualify detection."""

import copy
import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from core import srnet, srnet_positive, srnet_sanity
from core import srnet_bn_refresh as bn
from steganography import research_srnet_bn_refresh as service


def data():
    pixels, _ = srnet_positive.generate()
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
    return pixels, rows


@pytest.fixture
def model():
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261012)
        net = srnet.network().eval()
    for k, v in net.named_buffers():
        if k.endswith("num_batches_tracked"):
            v.fill_(160)
    yield net
    torch.set_num_threads(previous)


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def restored(model, before):
    import torch

    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    assert all(not m.training for m in model.modules())
    assert all(
        m.track_running_stats and m.momentum == 0.1
        for m in model.modules()
        if isinstance(m, torch.nn.BatchNorm2d)
    )


def test_actual_context_refresh_parameter_identity_rng_grad_threads_and_inputs(model):
    import torch

    pixels, rows = data()
    before, rng = snapshot(model), torch.random.get_rng_state().clone()
    original, metadata = pixels.copy(), copy.deepcopy(rows)
    model.classifier.weight.grad = torch.ones_like(model.classifier.weight)
    grad = model.classifier.weight.grad
    threads = torch.get_num_threads()
    contrast = bn.context(model, pixels, rows)
    assert contrast["state_unchanged"]
    for mode in ("stored_bn", "batch_statistics_diagnostic_only"):
        assert len(contrast[mode]["batches"]) == 8 and contrast[mode]["presented_rows"] == 32
    clone, report = bn.refresh(model, pixels, rows)
    assert clone is not model and report["learned_parameters_bit_identical"]
    assert report["source_state_unchanged"] and report["optimizer_steps"] == 0
    assert report["bn_refresh_batches"] == 8 and not report["exact_population_variance"]
    assert all(
        k.endswith(("running_mean", "running_var", "num_batches_tracked"))
        for k in report["changed_buffers"]
    )
    for name, value in clone.named_parameters():
        assert torch.equal(value, before[name]) and value.grad is None
    assert all(int(v) == 8 for k, v in clone.named_buffers() if k.endswith("num_batches_tracked"))
    assert all(not m.training for m in clone.modules())
    assert all(m.momentum == 0.1 for m in clone.modules() if isinstance(m, torch.nn.BatchNorm2d))
    restored(model, before)
    assert torch.equal(rng, torch.random.get_rng_state()) and torch.get_num_threads() == threads
    assert model.classifier.weight.grad is grad and bool((grad == 1).all())
    np.testing.assert_array_equal(original, pixels)
    assert metadata == rows


def test_invalid_inputs_tracking_cpu_layers_and_complete_schedule(model, monkeypatch):
    import torch

    pixels, rows = data()
    for bad in (
        pixels[:23],
        pixels.astype("f8"),
        np.full_like(pixels, np.nan),
        np.full_like(pixels, 2**37),
    ):
        with pytest.raises(ValueError):
            bn.checked(model, bad, rows)
    with pytest.raises(ValueError):
        bn.checked(model, pixels, rows[:-1])
    model.train()
    with pytest.raises(ValueError):
        bn.checked(model, pixels, rows)
    model.eval()
    layer = model.front[0][1]
    layer.track_running_stats = False
    with pytest.raises(ValueError, match="tracked"):
        bn.checked(model, pixels, rows)
    layer.track_running_stats = True
    modules = model.modules
    monkeypatch.setattr(model, "modules", lambda: iter([model]))
    with pytest.raises(ValueError, match="tracked"):
        bn.checked(model, pixels, rows)
    monkeypatch.setattr(model, "modules", modules)
    state = model.state_dict
    monkeypatch.setattr(model, "state_dict", lambda: {"bad": torch.empty(1, device="meta")})
    with pytest.raises(ValueError, match="CPU"):
        bn.checked(model, pixels, rows)
    monkeypatch.setattr(model, "state_dict", state)
    batches, schedule = bn.srnet_multibatch.epoch_batches(rows, seed=20261012, epoch=19)
    monkeypatch.setattr(
        bn.srnet_multibatch, "epoch_batches", lambda *a, **k: (batches[:7], schedule)
    )
    with pytest.raises(ValueError, match="eight"):
        bn.checked(model, pixels, rows)


def test_context_mutation_failure_hook_and_rng_restoration(model, monkeypatch):
    import torch

    pixels, rows = data()
    before, rng = snapshot(model), torch.random.get_rng_state().clone()

    def mutate(module, args, output):
        with torch.no_grad():
            model.front[0][1].running_mean.add_(1)

    hook = model.register_forward_hook(mutate)
    with pytest.raises(ValueError, match="mutation"):
        bn.context(model, pixels, rows)
    assert hook.id in model._forward_hooks
    restored(model, before)
    hook.remove()

    def fail(values):
        torch.rand(1)
        raise RuntimeError("unavailable")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError):
        bn.context(model, pixels, rows)
    restored(model, before)
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("mutation", ["source", "parameters", "counter", "exception"])
def test_refresh_rejects_mutation_and_restores_source_even_on_exception(
    model, monkeypatch, mutation
):
    import torch

    pixels, rows = data()
    before, rng = snapshot(model), torch.random.get_rng_state().clone()
    network = srnet.network
    clones = []

    def hooked():
        clone = network()
        clones.append(clone)

        def mutate(module, args, output):
            with torch.no_grad():
                if mutation == "source":
                    model.front[0][1].running_mean.add_(1)
                elif mutation == "parameters":
                    module.classifier.weight.add_(1)
                elif mutation == "counter":
                    module.front[0][1].num_batches_tracked.zero_()
                else:
                    torch.rand(1)
                    model.front[0][1].running_mean.add_(1)
                    raise RuntimeError("unavailable")

        clone.register_forward_hook(mutate)
        return clone

    monkeypatch.setattr(srnet, "network", hooked)
    with pytest.raises((ValueError, RuntimeError)):
        bn.refresh(model, pixels, rows)
    restored(model, before)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(not m.training for m in clones[0].modules())
    assert all(
        m.momentum == 0.1 for m in clones[0].modules() if isinstance(m, torch.nn.BatchNorm2d)
    )


def test_objectives_and_numeric_gates_are_separate(model, monkeypatch):
    own = {"balanced_accuracy": 0.95, "cross_entropy": 0.1}
    baseline = {"balanced_accuracy": 0.5, "cross_entropy": 0.7}
    assert all(bn.objectives(own, baseline).values())
    assert not any(bn.objectives(baseline, own).values())
    for bad in (
        {**own, "balanced_accuracy": True},
        {**own, "balanced_accuracy": 2},
        {**own, "cross_entropy": -1},
        {**own, "cross_entropy": float("nan")},
    ):
        with pytest.raises(ValueError):
            bn.objectives(bad, baseline)
    pixels, rows = data()
    monkeypatch.setattr(bn.srnet_reference, "reference_logits", lambda *a: np.zeros((1, 2)))
    assert bn.oracles(model, pixels, rows, np.zeros((24, 2)))["numerical_gates_passed"]
    monkeypatch.setattr(bn.srnet_reference, "reference_logits", lambda *a: np.ones((1, 2)))
    assert not bn.oracles(model, pixels, rows, np.zeros((24, 2)))["numerical_gates_passed"]
    generated, metadata = srnet_positive.generate()
    with pytest.raises(ValueError, match="coverage"):
        bn.oracles(model, generated, metadata, np.zeros((24, 2)))


@pytest.fixture
def fake_service(monkeypatch):
    pixels, rows = data()
    config = {
        "manifest": "manifest.json",
        "manifest_sha256": service.MANIFEST_SHA,
        "cache": "cache.json",
        "cache_sha256": service.CACHE_SHA,
    }
    metadata = [
        {
            k: s[k]
            for k in ("sha256", "source_group", "lineage", "quality_factor", "label", "method")
        }
        for s in rows
    ]
    metrics = srnet_sanity.metrics(np.zeros((24, 2)), rows)
    arms = []
    for factor in (1, 32):
        _, hashes = service.srnet_signal.prepare(pixels, rows, factor=factor)
        arms.append(
            {
                "factor": factor,
                "model_sha256": "a" * 64,
                "derived_tensor_sha256": hashes,
                "own_input_singleton_logits": np.zeros((24, 2)).tolist(),
                "original_input_singleton_logits": np.zeros((24, 2)).tolist(),
                "own_input_train_metrics": metrics,
                "original_input_train_metrics": metrics,
            }
        )
    baseline = {"selected_rows": metadata, "train_data_sha256": "a" * 64, "arms": arms}

    def read(path):
        return (
            (baseline, service.BASELINE_SHA)
            if path.name.startswith("srnet-signal")
            else ({}, service.MANIFEST_SHA)
        )

    monkeypatch.setattr(service, "read_document", read)
    monkeypatch.setattr(
        service, "load_pixels", lambda *a, **k: (pixels, rows, {"data_sha256": "a" * 64})
    )
    model = SimpleNamespace(named_buffers=lambda: [("bn.num_batches_tracked", 160)])
    monkeypatch.setattr(service.srnet_model, "load_model", lambda *a, **k: model)
    monkeypatch.setattr(service.srnet_model, "save_model", lambda *a: "b" * 64)
    monkeypatch.setattr(service.srnet_signal, "evaluate", lambda *a: (np.zeros((24, 2)), metrics))
    monkeypatch.setattr(service.bn, "context", lambda *a: {"state_unchanged": True})
    monkeypatch.setattr(service.bn, "refresh", lambda *a: (model, {"optimizer_steps": 0}))
    monkeypatch.setattr(service.bn, "oracles", lambda *a: {"numerical_gates_passed": True})
    return config, baseline, model


def test_service_complete_two_arms_and_fresh_paths_no_secrets(fake_service, tmp_path):
    config, _, _ = fake_service
    report = service.run(config, tmp_path / "models", tmp_path / "result")
    assert [a["factor"] for a in report["arms"]] == [1, 32]
    assert not any(a["in_sample_objectives_passed"] for a in report["arms"])
    assert report["status"] == "completed" and report["optimizer_steps"] == 0
    assert report["accuracy_qualification"] == "unavailable" and not report["deployed"]
    assert not report["validation_pixels_loaded"] and not report["primary_detection_changed"]
    assert str(tmp_path) not in (tmp_path / "result/control.json").read_text()
    with pytest.raises(FileExistsError):
        service.run(config, tmp_path / "models", tmp_path / "result")
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(FileExistsError):
        service.run(config, tmp_path / "models", tmp_path / "link/new")


@pytest.mark.parametrize(
    "failure",
    [
        "configuration",
        "cache",
        "protocol",
        "baseline",
        "manifest",
        "selection",
        "data",
        "tensor",
        "counter",
        "logits",
        "metrics",
        "deadline",
    ],
)
def test_service_rejects_unbound_incomplete_or_changed_inputs(
    fake_service, tmp_path, monkeypatch, failure
):
    config, baseline, model = fake_service
    if failure == "configuration":
        config = {**config, "validation": "forbidden"}
    elif failure == "cache":
        config = {**config, "cache_sha256": "b" * 64}
    elif failure == "protocol":
        monkeypatch.setattr(service, "PROTOCOL_SHA", "b" * 64)
    elif failure in {"baseline", "manifest"}:
        reader = service.read_document

        def bad_read(path):
            value, sha = reader(path)
            if path.name.startswith("srnet-signal") == (failure == "baseline"):
                sha = "b" * 64
            return value, sha

        monkeypatch.setattr(service, "read_document", bad_read)
    elif failure == "selection":
        baseline["selected_rows"] = []
    elif failure == "data":
        baseline["train_data_sha256"] = "b" * 64
    elif failure == "tensor":
        baseline["arms"][0]["derived_tensor_sha256"] = []
    elif failure == "counter":
        model.named_buffers = lambda: [("bn.num_batches_tracked", 159)]
    elif failure == "logits":
        baseline["arms"][0]["own_input_singleton_logits"] = [[0.0, 0.0]]
    elif failure == "metrics":
        baseline["arms"][0]["own_input_train_metrics"] = {}
    else:
        clock = iter((0, 301))
        monkeypatch.setattr(service.time, "monotonic", lambda: next(clock))
    with pytest.raises(ValueError):
        service.run(config, tmp_path / "models", tmp_path / "result")
    assert not (tmp_path / "result/control.json").exists()


def test_cli_limit_guards_failed_goals_and_secret_redaction(
    fake_service, tmp_path, monkeypatch, capsys
):
    import resource

    config, _, _ = fake_service
    limits = []
    monkeypatch.setattr(resource, "setrlimit", lambda k, v: limits.append((k, v)))
    monkeypatch.setattr(
        sys, "argv", ["bn", "--config", "config", "--model-dir", "models", "--out", str(tmp_path)]
    )
    monkeypatch.setattr(service, "read_document", lambda path: (config, "a" * 64))
    monkeypatch.setattr(
        service,
        "run",
        lambda *a: {
            "arms": [
                {"in_sample_objectives_passed": True, "audit": {"numerical_gates_passed": True}}
            ]
            * 2
        },
    )
    assert service.main() == 0 and len(limits) == 4
    assert (resource.RLIMIT_CPU, (820, 821)) in limits
    monkeypatch.setattr(
        service,
        "run",
        lambda *a: {
            "arms": [
                {"in_sample_objectives_passed": False, "audit": {"numerical_gates_passed": True}}
            ]
            * 2
        },
    )
    assert service.main() == 2

    def fail(*args):
        raise ValueError("secret /host/path")

    monkeypatch.setattr(service, "run", fail)
    assert service.main() == 2 and "secret" not in capsys.readouterr().out
