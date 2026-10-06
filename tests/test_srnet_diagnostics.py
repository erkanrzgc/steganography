"""BN contrast mutation/failure guards; generated probes are not accuracy."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import cli
from core import srnet
from core import srnet_diagnostics as core
from steganography import research_srnet_diagnose as service
from tests.test_jrm_reference import corpus as corpus
from tests.test_jrm_reference import sha
from tests.test_srnet_evaluate import config as evaluation_config  # noqa: F401
from tests.test_srnet_sampling import config as plan_config  # noqa: F401
from tests.test_srnet_sampling import rows
from tests.test_srnet_training import config as fit_config  # noqa: F401

torch = pytest.importorskip("torch")


@pytest.fixture
def model():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        network = srnet.network().eval()
    yield network
    torch.set_num_threads(previous)


def pair():
    return np.random.default_rng(41).normal(0.125, 0.25, (2, 1, 256, 256)).astype("<f4")


def test_actual_two_modes_statistics_loss_and_exact_restoration(model):
    pixels = pair()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = core.paired_probe(model, pixels)
    assert result["state_unchanged"]
    assert all(torch.equal(v, initial[k]) for k, v in model.state_dict().items())
    assert all(not m.training for m in model.modules())
    assert all(
        m.track_running_stats for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)
    )
    assert torch.equal(rng, torch.get_rng_state())
    for mode in ("native", "batch_statistics"):
        record = result[mode]
        assert len(record["layers"]) == 26
        expected = torch.nn.functional.cross_entropy(
            torch.tensor(record["logits"]), torch.tensor([0, 1])
        )
        assert record["paired_cross_entropy"] == pytest.approx(float(expected), rel=1e-5)
    with torch.no_grad():
        values = model.front[0][0](torch.from_numpy(pixels)).double()
    ratio = (values.var(dim=(0, 2, 3), unbiased=True) + 1e-5) / (
        initial["front.0.1.running_var"] + 1e-5
    )
    assert result["native"]["layers"][0]["median_variance_ratio"] == pytest.approx(
        float(ratio.median()), rel=0.1
    )
    np.testing.assert_array_equal(srnet.float_logits(model, pixels), result["native"]["logits"])


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((1, 1, 256, 256), dtype="<f4"),
        np.zeros((2, 1, 256, 256), dtype="f8"),
        np.full((2, 1, 256, 256), np.nan, dtype="<f4"),
        np.full((2, 1, 256, 256), 2**37, dtype="<f4"),
        None,
    ],
)
def test_invalid_pair_rejected(model, invalid):
    with pytest.raises(ValueError):
        core.paired_probe(model, invalid)


def test_eval_architecture_and_layer_coverage_guards(model, monkeypatch):
    model.train()
    with pytest.raises(ValueError):
        core.paired_probe(model, pair())
    model.eval()
    monkeypatch.setattr(model, "architecture", "wrong")
    with pytest.raises(ValueError):
        core.paired_probe(model, pair())
    monkeypatch.setattr(model, "architecture", srnet.ARCHITECTURE)
    monkeypatch.setattr(model, "named_modules", lambda *args, **kwargs: iter([]))
    with pytest.raises(ValueError, match="exact architecture"):
        core.paired_probe(model, pair())


def test_exception_nan_and_mutation_restore_everything(model, monkeypatch):
    original = model.forward
    state = {k: v.clone() for k, v in model.state_dict().items()}

    def failure(values):
        if model.front[0][1].training:
            raise ValueError("injected")
        return original(values)

    monkeypatch.setattr(model, "forward", failure)
    with pytest.raises(ValueError, match="injected"):
        core.paired_probe(model, pair())
    assert all(not m.training for m in model.modules())
    assert all(not m._forward_pre_hooks for m in model.modules())
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())

    def mutation(values):
        result = original(values)
        model.front[0][1].num_batches_tracked.add_(1)
        return result

    monkeypatch.setattr(model, "forward", mutation)
    with pytest.raises(ValueError, match="mutate"):
        core.paired_probe(model, pair())
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    monkeypatch.setattr(model, "forward", lambda values: torch.full((2, 2), float("nan")))
    with pytest.raises(ValueError, match="invalid logits"):
        core.paired_probe(model, pair())
    monkeypatch.setattr(model, "forward", lambda values: torch.zeros((2, 2)))
    with pytest.raises(ValueError, match="incomplete"):
        core.paired_probe(model, pair())


@pytest.fixture
def config(evaluation_config):  # noqa: F811
    return {k: v for k, v in evaluation_config.items() if not k.startswith("validation_")}


def test_train_only_cli_complete_and_no_model_change(config, tmp_path, monkeypatch, capsys):
    observed = []
    from steganography import research_srnet_fit as fitting

    load = fitting.load_pixels

    def tracked(*args, **kwargs):
        observed.append(kwargs["split"])
        return load(*args, **kwargs)

    monkeypatch.setattr(fitting, "load_pixels", tracked)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    state_sha = sha(Path(config["model_dir"]) / "model.npz")
    assert (
        cli.main(
            [
                "research",
                "srnet-diagnose",
                "--config",
                str(path),
                "--out",
                str(tmp_path / "probes.json"),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["probe_pairs"] == 4 and not report["validation_pixels_loaded"]
    assert report["state_unchanged"] and not report["deployed"] and not report["calibrated"]
    assert str(tmp_path) not in json.dumps(report) and observed == ["train"]
    assert state_sha == sha(Path(config["model_dir"]) / "model.npz")
    with pytest.raises(FileExistsError):
        service.diagnose(config, tmp_path / "probes.json")


@pytest.mark.parametrize(
    "edit",
    [
        {"validation_cache": "forbidden"},
        {"card_sha256": "0" * 64},
        {"plan_sha256": "0" * 64},
        {"card_sha256": "bad"},
        {"model_dir": ""},
    ],
)
def test_reject_wrong_bindings_and_validation(config, tmp_path, edit):
    with pytest.raises(ValueError):
        service.diagnose({**config, **edit}, tmp_path / "out.json")


def test_scope_sampling_counts_metadata_only_and_cap(monkeypatch):
    samples = rows()
    saved = copy.deepcopy(samples)
    pairs = service.select_pairs(samples, seed=91)
    assert samples == saved and pairs == service.select_pairs(samples, seed=91)
    assert len({samples[s]["sha256"] for _, s in pairs}) == len(pairs)
    expanded = [
        {
            **samples[0],
            "source_group": str(i),
            "quality_factor": 75,
            "method": "UERD",
            "sha256": str(i),
        }
        for i in range(65)
    ]
    monkeypatch.setattr(service, "epoch_pairs", lambda *a, **kw: ([(0, i) for i in range(65)], {}))
    with pytest.raises(ValueError, match="limit"):
        service.select_pairs(expanded, seed=91)


def test_deadline_restores_threads_and_no_partial_output(config, tmp_path, monkeypatch):
    previous = torch.get_num_threads()
    moments = iter([0, 181])
    monkeypatch.setattr(service.time, "monotonic", lambda: next(moments))
    with pytest.raises(ValueError, match="deadline"):
        service.diagnose(config, tmp_path / "out.json")
    assert torch.get_num_threads() == previous and not (tmp_path / "out.json").exists()
    moments = iter([0, 0, 181])
    monkeypatch.setattr(service.time, "monotonic", lambda: next(moments))
    monkeypatch.setattr(core, "paired_probe", lambda *args: {})
    with pytest.raises(ValueError, match="deadline"):
        service.diagnose(config, tmp_path / "out.json")
    assert torch.get_num_threads() == previous


def test_nonfinite_layer_stats_restore_state(model):
    before = {k: v.clone() for k, v in model.state_dict().items()}
    hook = model.front[0][0].register_forward_hook(
        lambda module, args, output: torch.full_like(output, float("inf"))
    )
    try:
        with pytest.raises(ValueError, match="nonfinite layer"):
            core.paired_probe(model, pair())
        assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
        assert all(not m.training for m in model.modules())
    finally:
        hook.remove()


def test_wrong_bn_counts_fail_before_probes(config, tmp_path, monkeypatch):
    from core import srnet_model

    load = srnet_model.load_model

    def wrong(*args, **kwargs):
        model = load(*args, **kwargs)
        model.front[0][1].num_batches_tracked.zero_()
        return model

    monkeypatch.setattr(srnet_model, "load_model", wrong)
    with pytest.raises(ValueError, match="BN update"):
        service.diagnose(config, tmp_path / "out.json")
