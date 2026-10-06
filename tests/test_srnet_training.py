"""Generated paired learning/provenance tests; never real accuracy evidence."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import cli
from core import srnet, srnet_model
from core import srnet_training as training
from core.srnet_sampling import epoch_pairs
from steganography import research_pixels as rp
from steganography import research_srnet_fit as service
from steganography import research_srnet_plan as planner
from tests.test_jrm_reference import corpus as corpus
from tests.test_jrm_reference import sha
from tests.test_srnet_sampling import config as plan_config  # noqa: F401 — shared pytest fixture

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config(plan_config, tmp_path):  # noqa: F811 — injected pytest fixture
    plan_path = tmp_path / "plan.json"
    planner.plan_training({**plan_config, "epochs": 1}, plan_path)
    return {
        **{k: plan_config[k] for k in ("manifest", "manifest_sha256", "cache", "cache_sha256")},
        "plan": str(plan_path),
        "plan_sha256": sha(plan_path),
        "threads": 1,
    }


def inputs(config):
    values, samples, _ = rp.load_pixels(
        Path(config["manifest"]),
        Path(config["cache"]),
        checksum=config["cache_sha256"],
        split="train",
        _float=True,
    )
    with open(config["plan"]) as f:
        plan = json.load(f)
    return values, samples, np.arange(len(samples)), plan["epochs"]


def test_actual_paired_fit_cli_card_bn_and_no_rng_or_thread_leak(config, tmp_path, capsys):
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(config))
    out = tmp_path / "fit"
    rng = torch.get_rng_state().clone()
    assert cli.main(["research", "srnet-fit", "--config", str(path), "--out", str(out)]) == 0
    assert torch.equal(rng, torch.get_rng_state()) and torch.get_num_threads() == 2
    card = json.loads(capsys.readouterr().out)
    assert card == json.loads((out / "model-card.json").read_bytes())
    assert card["training_plan_sha256"] == config["plan_sha256"]
    assert card["training"] == "completed" and not card["validation_used"]
    assert card["accuracy_metrics"] == card["qualification"] == "unavailable"
    assert not card["deployed"] and not card["calibrated"] and str(tmp_path) not in json.dumps(card)
    assert card["epoch_training"][0]["updates"] == 4
    assert np.isfinite(card["epoch_training"][0]["mean_pair_loss"])
    model = srnet_model.load_model(out / "model.npz", checksum=card["model_sha256"])
    assert all(not m.training for m in model.modules())
    assert all(
        int(v) == 4 for name, v in model.named_buffers() if name.endswith("num_batches_tracked")
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(91)
        initial = srnet.network()
    assert not torch.equal(initial.front[0][0].weight, model.front[0][0].weight)
    assert np.isfinite(
        srnet.float_logits(model, np.full((1, 1, 256, 256), 128.125, dtype="<f4"))
    ).all()
    with pytest.raises(FileExistsError):
        service.train_srnet(config, out)
    link = tmp_path / "link"
    link.symlink_to(out)
    with pytest.raises(FileExistsError):
        service.train_srnet(config, link)


def test_source_exclusion_only_updates_selected_pairs(config, tmp_path, monkeypatch):
    cache = Path(config["cache"])
    raw = cache.parent / "pixels.f32"
    pixels = np.frombuffer(raw.read_bytes(), dtype="<f4").copy().reshape(6, 1, 256, 256)
    for i, row in enumerate(pixels):
        row.fill(128.125 + i)
    raw.write_bytes(pixels.tobytes())
    descriptor = json.loads(cache.read_bytes())
    descriptor["data_sha256"] = sha(raw)
    cache.write_text(json.dumps(descriptor))
    config = {**config, "cache_sha256": sha(cache)}
    pconfig = {k: config[k] for k in ("manifest", "manifest_sha256", "cache", "cache_sha256")}
    pconfig.update(
        epochs=1, seed=91, training_source_id="source-" + hashlib.sha256(b"one").hexdigest()[:16]
    )
    scoped = tmp_path / "scoped-plan.json"
    planner.plan_training(pconfig, scoped)
    observed = []
    factory = srnet.network

    def network():
        model = factory()
        model.register_forward_pre_hook(
            lambda _, args: observed.append(args[0].detach().numpy().copy())
        )
        return model

    monkeypatch.setattr(srnet, "network", network)
    card = service.train_srnet(
        {**config, "plan": str(scoped), "plan_sha256": sha(scoped)}, tmp_path / "scope"
    )
    assert card["training_scope"]["rows"] == 3 and card["epoch_training"][0]["updates"] == 2
    assert len(observed) == 2
    assert {(float(batch[0, 0, 0, 0]), float(batch[1, 0, 0, 0])) for batch in observed} == {
        (128.125, 129.125),
        (128.125, 130.125),
    }


@pytest.mark.parametrize(
    "edit",
    [
        {"validation_cache": "forbidden"},
        {"plan_sha256": "bad"},
        {"manifest_sha256": "0" * 64},
        {"plan_sha256": "0" * 64},
        {"threads": True},
        {"threads": 3},
        {"max_seconds": 0},
        {"max_seconds": 1801},
        {"learning_rate": float("nan")},
        {"learning_rate": 0.1},
        {"weight_decay": -1},
        {"weight_decay": float("inf")},
        {"learning_rate": True},
    ],
)
def test_invalid_configs_fail_without_model(config, edit, tmp_path):
    out = tmp_path / "bad"
    with pytest.raises(ValueError):
        service.train_srnet({**config, **edit}, out)
    assert not out.exists()


@pytest.mark.parametrize(
    "fault", ["settings", "scope", "ids", "recipe", "order", "provenance", "schema"]
)
def test_rehashed_forged_plan_rejected_before_learning(config, fault, tmp_path, monkeypatch):
    path = Path(config["plan"])
    plan = json.loads(path.read_bytes())
    if fault == "settings":
        plan["settings"] = []
    if fault == "scope":
        plan["training_scope"] = []
    if fault == "ids":
        plan["training_scope"]["recipe"] = "single-declared-source-v1"
        plan["training_scope"]["source_ids"] = []
    if fault == "recipe":
        plan["training_scope"]["recipe"] = "unknown"
    if fault == "order":
        plan["epochs"][0]["ordered_pair_sha256"] = "0" * 64
    if fault == "provenance":
        plan["decoder"] = {"forged": True}
    if fault == "schema":
        plan["schema_version"] = "wrong"
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(training, "fit", lambda *a, **k: pytest.fail("forged plan reached trainer"))
    with pytest.raises(ValueError):
        service.train_srnet({**config, "plan_sha256": sha(path)}, tmp_path / "bad")


@pytest.mark.parametrize(
    "fault",
    [
        "dtype",
        "shape",
        "indices",
        "duplicate",
        "negative",
        "range",
        "float",
        "nan",
        "magnitude",
        "schedule",
    ],
)
def test_core_input_guards(config, fault):
    pixels, samples, indices, schedule = inputs(config)
    if fault == "dtype":
        pixels = pixels.astype("f8")
    if fault == "shape":
        pixels = pixels[:, :, :128, :128]
    if fault == "indices":
        indices = []
    if fault == "duplicate":
        indices[1] = indices[0]
    if fault == "negative":
        indices[0] = -1
    if fault == "range":
        indices[0] = len(samples)
    if fault == "float":
        indices = indices.astype(float)
    if fault in {"nan", "magnitude"}:
        pixels = pixels.copy()
        pixels[0, 0, 0, 0] = float("nan") if fault == "nan" else 2**37
    if fault == "schedule":
        schedule = copy.deepcopy(schedule)
        schedule[0]["ordered_pair_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        training.fit(pixels, samples, indices, seed=91, schedule=schedule, config={})


@pytest.mark.parametrize("fault", ["logits", "loss", "gradient", "state", "deadline"])
def test_learning_failures_restore_state_and_publish_nothing(config, fault, tmp_path, monkeypatch):
    previous = torch.get_rng_state().clone()
    factory = srnet.network
    clock = {"expired": False}

    def network():
        model = factory()
        if fault == "logits":
            model.register_forward_hook(lambda _, args, output: output * float("nan"))
        if fault == "gradient":
            model.front[0][0].weight.register_hook(lambda grad: grad * float("nan"))
        if fault == "deadline":
            clock["expired"] = True
        return model

    monkeypatch.setattr(srnet, "network", network)
    if fault == "deadline":
        monkeypatch.setattr(training.time, "monotonic", lambda: 2 if clock["expired"] else 0)
    if fault == "loss":
        original = torch.nn.functional.cross_entropy
        monkeypatch.setattr(
            torch.nn.functional, "cross_entropy", lambda *a, **k: original(*a, **k) * float("nan")
        )
    if fault == "state":
        original_step = torch.optim.Adamax.step

        def corrupt(self, *args, **kwargs):
            original_step(self, *args, **kwargs)
            self.param_groups[0]["params"][0].data.fill_(float("nan"))

        monkeypatch.setattr(torch.optim.Adamax, "step", corrupt)
    out = tmp_path / "failed"
    reason = {
        "logits": "invalid logits",
        "loss": "nonfinite loss",
        "gradient": "invalid gradients",
        "state": "finite mismatch",
        "deadline": "deadline exceeded",
    }[fault]
    with pytest.raises(ValueError, match=reason):
        service.train_srnet(
            {**config, "max_seconds": 1 if fault == "deadline" else 60, "threads": 1}, out
        )
    assert (
        not out.exists()
        and torch.get_num_threads() == 2
        and torch.equal(previous, torch.get_rng_state())
    )


def test_configuration_defaults_and_missing_fields():
    with pytest.raises(ValueError):
        service.configuration([])
    with pytest.raises(ValueError):
        service.configuration({})
    assert training.settings({}) == {
        "threads": 2,
        "max_seconds": 1800,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
    }


def test_core_excluded_rows_cannot_hide_incomplete_families(config):
    pixels, samples, _, _ = inputs(config)
    indices = np.arange(3)
    schedule = [epoch_pairs(samples[:3], seed=91, epoch=0)[1]]
    samples[4]["method"] = "unmatched"
    with pytest.raises(ValueError, match="complete matched-family pairs"):
        training.fit(pixels, samples, indices, seed=91, schedule=schedule, config={})
