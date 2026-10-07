"""Independent full-network accumulation math, atomic updates and provenance."""

import copy
import json
import sys

import numpy as np
import pytest

from core import srnet, srnet_positive, srnet_training
from core import srnet_widebatch as wide
from steganography import research_srnet_accumulation as service
from steganography import research_srnet_widebatch as context
from tests.test_srnet_signal import data
from tests.test_srnet_widebatch import fake, records, setup_service  # noqa: F401


def plan(rows):
    return [wide.epoch_pairs(rows, seed=20261012, epoch=e)[1] for e in range(20)]


def fit(pixels, rows, **kwargs):
    return srnet_training.fit(
        pixels,
        rows,
        np.arange(24),
        seed=20261012,
        schedule=plan(rows),
        config={**srnet_positive.PARAMS, "batch_recipe": wide.srnet_multibatch.RECIPE},
        wide_context=True,
        accumulate_context=True,
        **kwargs,
    )


def test_actual_two_microbatch_gradients_match_independent_manual_full_network_step(monkeypatch):
    torch = pytest.importorskip("torch")
    pixels, rows = data()
    original, metadata = pixels.copy(), copy.deepcopy(rows)
    rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    factory, opt_factory = srnet.network, torch.optim.Adamax
    batches, _ = wide.epoch_batches(rows, seed=20261012, epoch=0)
    observed, nets, steps = [], [], []

    def network():
        model = factory()
        nets.append(model)

        def hook(_, args):
            observed.append(args[0].detach().numpy().copy())
            if len(observed) == 3:
                raise RuntimeError("stop after one atomic optimizer group")

        model.register_forward_pre_hook(hook)
        return model

    def optimizer(*args, **kwargs):
        opt = opt_factory(*args, **kwargs)
        old = opt.step

        def step(*a, **kw):
            steps.append(1)
            return old(*a, **kw)

        opt.step = step
        return opt

    monkeypatch.setattr(srnet, "network", network)
    monkeypatch.setattr(torch.optim, "Adamax", optimizer)
    with pytest.raises(RuntimeError, match="atomic"):
        fit(pixels, rows)
    assert len(steps) == 1 and len(observed) == 3
    assert torch.equal(rng, torch.get_rng_state()) and torch.get_num_threads() == threads
    for actual, indices in zip(
        observed, (batches[0, :4], batches[0, 4:], batches[1, :4]), strict=True
    ):
        np.testing.assert_array_equal(actual, pixels[indices])
    try:
        torch.set_num_threads(2)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261012)
            manual = factory().train()
            opt = opt_factory(
                manual.parameters(),
                lr=0.001,
                weight_decay=0.0001,
                betas=(0.9, 0.999),
                eps=1e-8,
                foreach=False,
            )
            opt.zero_grad(set_to_none=True)
            labels = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
            for indices in (batches[0, :4], batches[0, 4:]):
                inputs = torch.from_numpy(pixels[indices].copy())
                (torch.nn.functional.cross_entropy(manual(inputs), labels) * 0.5).backward()
            opt.step()
            for name, value in manual.state_dict().items():
                np.testing.assert_array_equal(
                    value.detach().numpy(), nets[0].state_dict()[name].detach().numpy()
                )
    finally:
        torch.set_num_threads(threads)
    assert [int(v) for k, v in nets[0].named_buffers() if k.endswith("num_batches_tracked")] == [
        2
    ] * 26
    assert torch.equal(rng, torch.get_rng_state())
    np.testing.assert_array_equal(pixels, original)
    assert rows == metadata


@pytest.mark.parametrize("fault", ["logits", "gradient", "exception", "deadline"])
def test_invalid_second_microbatch_never_updates_optimizer_and_restores_rng(monkeypatch, fault):
    torch = pytest.importorskip("torch")
    pixels, rows = data()
    factory, opt_factory = srnet.network, torch.optim.Adamax
    forwards, steps = [], []
    rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    expired = [False]

    def network():
        model = factory()

        def hook(_, args, result):
            forwards.append(1)
            if len(forwards) == 2:
                if fault == "logits":
                    return result * float("nan")
                if fault == "exception":
                    raise RuntimeError("private secret")
                if fault == "deadline":
                    expired[0] = True
            return result

        model.register_forward_hook(hook)
        if fault == "gradient":
            next(model.parameters()).register_hook(
                lambda grad: grad * float("nan") if len(forwards) == 2 else grad
            )
        return model

    def optimizer(*args, **kwargs):
        opt = opt_factory(*args, **kwargs)
        opt.step = lambda *a, **kw: steps.append(1)
        return opt

    monkeypatch.setattr(srnet, "network", network)
    monkeypatch.setattr(torch.optim, "Adamax", optimizer)
    if fault == "deadline":
        monkeypatch.setattr(srnet_training.time, "monotonic", lambda: 2000 if expired[0] else 0)
    with pytest.raises((RuntimeError, ValueError)):
        fit(pixels, rows)
    assert len(forwards) == 2 and not steps
    assert torch.equal(rng, torch.get_rng_state()) and torch.get_num_threads() == threads


@pytest.mark.parametrize("flag", [None, 1, "true"])
def test_accumulation_explicit_boolean_and_wide_only(flag):
    pixels, rows = data()
    with pytest.raises(ValueError, match="accumulation"):
        srnet_training.fit(
            pixels,
            rows,
            np.arange(24),
            seed=20261012,
            schedule=plan(rows),
            config={"batch_recipe": wide.srnet_multibatch.RECIPE},
            wide_context=True,
            accumulate_context=flag,
        )
    with pytest.raises(ValueError, match="accumulation"):
        srnet_training.fit(
            pixels,
            rows,
            np.arange(24),
            seed=20261012,
            schedule=plan(rows),
            config={"batch_recipe": wide.srnet_multibatch.RECIPE},
            accumulate_context=True,
        )
    with pytest.raises(ValueError, match="boolean"):
        wide.learn(pixels, rows, factor=1, accumulation=flag)
    with pytest.raises(ValueError, match="boolean"):
        wide.audit(None, pixels, rows, {}, accumulation=flag)


@pytest.fixture
def accumulated(fake, monkeypatch):  # noqa: F811
    fake.named_buffers = lambda: [(f"bn{i}.num_batches_tracked", 160) for i in range(26)]

    def training(pixels, rows, indices, **kwargs):
        assert kwargs["wide_context"] is True and kwargs["accumulate_context"] is True
        assert len(kwargs["schedule"]) == 20 and indices.tolist() == list(range(24))
        return fake, records()

    monkeypatch.setattr(srnet_training, "fit", training)
    return fake


def test_accumulated_learning_replay_counter_and_matched_exposure(accumulated):
    pixels, rows = data()
    model, arm = wide.learn(pixels, rows, factor=32, accumulation=True)
    assert (
        model is accumulated and arm["optimizer_updates"] == arm["baseline_optimizer_updates"] == 80
    )
    assert arm["presented_rows"] == arm["baseline_presented_rows"] == 640
    assert (
        arm["optimizer_steps_matched"]
        and arm["gradient_accumulation_steps"] == 2
        and arm["microbatch_size"] == 4
        and arm["bn_forward_batches"] == 160
    )
    audit = wide.audit(model, pixels, rows, arm, accumulation=True)
    assert audit["numerical_gates_passed"] and len(audit["independent_oracles"]) == 9
    assert not arm["learning_objectives_passed"]
    with pytest.raises(ValueError):
        wide.audit(model, pixels, rows, arm)
    model.named_buffers = lambda: [(f"bn{i}.num_batches_tracked", 80) for i in range(26)]
    with pytest.raises(ValueError, match="BN"):
        wide.audit(model, pixels, rows, arm, accumulation=True)


@pytest.mark.parametrize(
    "field",
    [
        "microbatch_size",
        "gradient_accumulation_steps",
        "bn_forward_batches",
        "baseline_optimizer_updates",
        "optimizer_steps_matched",
    ],
)
def test_forged_microbatch_accounting_fails(accumulated, field):
    pixels, rows = data()
    _, arm = wide.learn(pixels, rows, factor=1, accumulation=True)
    arm[field] = None
    with pytest.raises(ValueError):
        wide.audit(accumulated, pixels, rows, arm, accumulation=True)


@pytest.fixture
def accumulation_service(setup_service, accumulated, monkeypatch):  # noqa: F811
    config, baseline = setup_service
    _, rows = data()
    for arm in baseline["arms"]:
        arm["optimizer_updates"] = 80
        arm["presented_rows"] = 640
        arm["epoch_batch_schedule"] = [
            wide.epoch_batches(rows, seed=20261012, epoch=e)[1] for e in range(20)
        ]
    monkeypatch.setattr(
        context,
        "read_document",
        lambda p: (
            (baseline, service.BASELINE_SHA)
            if p.name.startswith("srnet-wide-batch")
            else ({}, context.MANIFEST_SHA)
        ),
    )
    return config, baseline


def test_service_wrapper_complete_two_arms_no_validation_or_secrets(accumulation_service, tmp_path):
    config, _ = accumulation_service
    report = service.run(config, tmp_path / "out")
    assert (
        report["schema_version"] == "srnet-accumulation-control-v1"
        and report["status"] == "completed"
    )
    assert report["optimizer_steps_matched"] and report["row_exposure_matched"]
    assert len(report["arms"]) == 2 and all(
        not a["learning_objectives_passed"] for a in report["arms"]
    )
    assert not report["validation_pixels_loaded"] and not report["deployed"]
    assert "secret/" not in json.dumps(report)


@pytest.mark.parametrize(
    "bad", ["protocol", "baseline", "order", "tensor", "updates", "exposure", "mode"]
)
def test_accumulation_service_fail_closed_on_historical_mismatch(
    accumulation_service, monkeypatch, tmp_path, bad
):
    config, baseline = accumulation_service
    if bad == "protocol":
        monkeypatch.setattr(service, "PROTOCOL_SHA", "0" * 64)
    if bad == "baseline":
        monkeypatch.setattr(context, "read_document", lambda p: (baseline, "0" * 64))
    if bad == "order":
        baseline["arms"][0]["epoch_batch_schedule"][0]["optimizer_updates"] = 3
    if bad == "tensor":
        baseline["arms"][0]["derived_tensor_sha256"] = ["0" * 64] * 24
    if bad == "updates":
        baseline["arms"][0]["optimizer_updates"] = 160
    if bad == "exposure":
        baseline["arms"][0]["presented_rows"] = 1280
    with pytest.raises(ValueError):
        if bad == "mode":
            context.run(config, tmp_path / "out", _accumulation=1)
        else:
            service.run(config, tmp_path / "out")
    assert not (tmp_path / "out/control.json").exists()


def test_cli_limits_success_failed_goals_and_redaction(monkeypatch, capsys):
    import resource

    calls = []
    monkeypatch.setattr(resource, "setrlimit", lambda *a: calls.append(a))
    monkeypatch.setattr(sys, "argv", ["accumulate", "--config", "config.json", "--out", "out"])
    monkeypatch.setattr(service, "read_document", lambda *a: ({}, ""))
    monkeypatch.setattr(
        service,
        "run",
        lambda *a: {
            "arms": [
                {"learning_objectives_passed": True, "audit": {"numerical_gates_passed": True}}
            ]
        },
    )
    assert service.main() == 0 and len(calls) == 4
    monkeypatch.setattr(
        service,
        "run",
        lambda *a: {
            "arms": [
                {"learning_objectives_passed": False, "audit": {"numerical_gates_passed": True}}
            ]
        },
    )
    assert service.main() == 2

    def fail(*a):
        raise ValueError("private/password=secret")

    monkeypatch.setattr(service, "run", fail)
    assert service.main() == 2 and "secret" not in capsys.readouterr().out
