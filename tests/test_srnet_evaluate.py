"""Generated complete-split/provenance regressions, not detector accuracy."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import cli
from core import srnet, srnet_reference, srnet_training
from core.srnet_reference import reference_logits as numpy_logits
from core.srnet_training import fit as paired_fit
from steganography import research_pixels as pixels
from steganography import research_srnet_evaluate as service
from steganography import research_srnet_fit as fitting
from tests.test_jrm_reference import corpus as corpus
from tests.test_jrm_reference import sha
from tests.test_srnet_sampling import config as plan_config  # noqa: F401
from tests.test_srnet_training import config as fit_config  # noqa: F401

torch = pytest.importorskip("torch")


@pytest.fixture
def config(fit_config, corpus, tmp_path, monkeypatch):  # noqa: F811 — pytest fixture
    # Card/parser adversarial cases do not each repeat gradient fitting. The
    # actual trained-state/native/NumPy integration below restores real fit.
    def snapshot(values, samples, indices, *, seed, schedule, config):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = srnet.network().eval()
        for name, value in model.named_buffers():
            if name.endswith("num_batches_tracked"):
                value.fill_(sum(e["pairs"] for e in schedule))
        return model, [
            {"epoch": i, "updates": e["pairs"], "mean_pair_loss": 0.1}
            for i, e in enumerate(schedule)
        ]

    monkeypatch.setattr(srnet_training, "fit", snapshot)
    model_dir = tmp_path / "model"
    fitting.train_srnet(fit_config, model_dir)
    manifest, source = corpus
    validation = tmp_path / "validation"
    pixels.extract_pixels(manifest, validation, source=source, split="validation", _float=True)
    monkeypatch.setattr(
        srnet_reference, "reference_logits", lambda arrays, values: np.zeros((1, 2))
    )
    return {
        **{
            k: fit_config[k]
            for k in ("manifest", "manifest_sha256", "cache", "cache_sha256", "plan", "plan_sha256")
        },
        "validation_cache": str(validation / "cache.json"),
        "validation_cache_sha256": sha(validation / "cache.json"),
        "model_dir": str(model_dir),
        "card_sha256": sha(model_dir / "model-card.json"),
    }


def test_complete_cli_predictions_redaction_and_failure_gate(config, tmp_path, monkeypatch, capsys):
    from core import srnet

    monkeypatch.setattr(srnet, "float_logits", lambda model, values: np.zeros((len(values), 2)))
    path = tmp_path / "evaluate.json"
    path.write_text(json.dumps(config))
    previous = torch.get_num_threads()
    rng = torch.get_rng_state().clone()
    out = tmp_path / "predictions.json"
    assert cli.main(["research", "srnet-evaluate", "--config", str(path), "--out", str(out)]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record == json.loads(out.read_bytes())
    assert (
        record["validation_rows"]
        == record["native_rows_evaluated"]
        == len(record["predictions"])
        == 6
    )
    assert len(record["independent_forward_audit"]) == 6
    assert all(r["score"] == 0.5 for r in record["predictions"])
    assert not record["calibrated"] and not record["deployed"]
    assert str(tmp_path) not in json.dumps(record)
    assert torch.get_num_threads() == previous and torch.equal(rng, torch.get_rng_state())
    monkeypatch.setattr(srnet_reference, "reference_logits", lambda arrays, values: np.ones((1, 2)))
    assert (
        cli.main(
            [
                "research",
                "srnet-evaluate",
                "--config",
                str(path),
                "--out",
                str(tmp_path / "failed.json"),
            ]
        )
        == 2
    )
    failed = json.loads(capsys.readouterr().out)
    assert failed["status"] == "failed_numerical_gate" and failed["predictions"] == []
    with pytest.raises(FileExistsError):
        service.evaluate(config, out)
    link = tmp_path / "link"
    link.symlink_to(tmp_path)
    with pytest.raises(FileExistsError):
        service.evaluate(config, link / "new.json")


@pytest.mark.parametrize(
    "edit",
    [
        {"card_sha256": "0" * 64},
        {"manifest_sha256": "0" * 64},
        {"validation_cache_sha256": "0" * 64},
        {"plan_sha256": "0" * 64},
        {"validation_cache_sha256": "wrong"},
        {"threads": 8},
        {"model_dir": ""},
    ],
)
def test_invalid_bound_configuration(config, tmp_path, edit):
    with pytest.raises(ValueError):
        service.evaluate({**config, **edit}, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize(
    "edit",
    [
        {"deployed": True},
        {"validation_used": 0},
        {"input_units": "uint8"},
        {"epoch_training": []},
        {"epoch_training": [{"epoch": 0, "updates": 3, "mean_pair_loss": 0.1}]},
        {"epoch_training": [{"epoch": 0, "updates": 4, "mean_pair_loss": float("nan")}]},
        {"seconds": -1},
        {"model_sha256": "bad"},
        {"torch_version": ""},
        {"optimizer": None},
        {"training_scope": {}},
    ],
)
def test_rehashed_forged_card_rejected(config, tmp_path, edit):
    path = Path(config["model_dir"]) / "model-card.json"
    card = json.loads(path.read_bytes())
    path.write_text(json.dumps({**card, **edit}))
    with pytest.raises(ValueError):
        service.evaluate({**config, "card_sha256": sha(path)}, tmp_path / "out.json")


def test_validation_cannot_be_training_cache(config, tmp_path):
    with pytest.raises(ValueError):
        service.evaluate(
            {
                **config,
                "validation_cache": config["cache"],
                "validation_cache_sha256": config["cache_sha256"],
            },
            tmp_path / "out.json",
        )


def test_bn_counter_integrity_and_decoder_guard(config, tmp_path, monkeypatch):
    from core import srnet_model

    load = srnet_model.load_model

    def changed(*args, **kwargs):
        model = load(*args, **kwargs)
        model.front[0][1].num_batches_tracked.zero_()
        return model

    monkeypatch.setattr(srnet_model, "load_model", changed)
    with pytest.raises(ValueError, match="BN update"):
        service.evaluate(config, tmp_path / "out.json")
    original = service.load_pixels

    def bad_decoder(*args, **kwargs):
        values, rows, descriptor = original(*args, **kwargs)
        return values, rows, {**descriptor, "decoder": {}}

    monkeypatch.setattr(service, "load_pixels", bad_decoder)
    with pytest.raises(ValueError, match="decoder"):
        service.evaluate(config, tmp_path / "out.json")


def test_deadline_and_inference_failure_restore_threads(config, tmp_path, monkeypatch):
    from core import srnet

    previous = torch.get_num_threads()
    times = iter([0, 1801])
    monkeypatch.setattr(service.time, "monotonic", lambda: next(times))
    with pytest.raises(ValueError, match="deadline"):
        service.evaluate(config, tmp_path / "out.json")
    assert torch.get_num_threads() == previous
    monkeypatch.setattr(service.time, "monotonic", lambda: 0)
    monkeypatch.setattr(
        srnet, "float_logits", lambda *args: (_ for _ in ()).throw(ValueError("nonfinite"))
    )
    with pytest.raises(ValueError, match="nonfinite"):
        service.evaluate(config, tmp_path / "out.json")
    assert torch.get_num_threads() == previous and not (tmp_path / "out.json").exists()


def test_oracle_selection_metadata_only_and_bounded():
    rows = [{"source_group": "a", "quality_factor": None, "label": "cover", "method": None}] * 4
    assert service.oracle_rows(rows) == [0]
    original = copy.deepcopy(rows)
    service.oracle_rows(rows)
    assert rows == original
    for invalid in ([], [{**rows[0], "source_group": str(i)} for i in range(25)]):
        with pytest.raises(ValueError, match="cell limit"):
            service.oracle_rows(invalid)


def test_actual_native_split_with_independent_numpy(config, tmp_path, monkeypatch):
    from core import srnet_model

    monkeypatch.setattr(srnet_training, "fit", paired_fit)
    bound_config = service.configuration(config)
    card = fitting.train_srnet(bound_config, tmp_path / "actual-fit")
    config = {
        **config,
        "model_dir": str(tmp_path / "actual-fit"),
        "card_sha256": sha(tmp_path / "actual-fit/model-card.json"),
    }

    model = srnet_model.load_model(
        Path(config["model_dir"]) / "model.npz",
        checksum=card["model_sha256"],
    )
    arrays = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    # Every generated row has exactly the same fractional input; calculate the
    # independent full forward once, without calling native model.forward.
    independent = numpy_logits(arrays, np.full((1, 1, 256, 256), 128.125, dtype="<f4"))
    monkeypatch.setattr(srnet_reference, "reference_logits", lambda arrays, values: independent)
    report = service.evaluate(config, tmp_path / "actual.json")
    assert report["native_rows_evaluated"] == 6
    assert len(report["independent_forward_audit"]) == 6
    assert bool(report["predictions"]) == all(
        a["passed"] for a in report["independent_forward_audit"]
    )


def test_distinct_rows_preserve_manifest_order_and_scores(config, tmp_path, monkeypatch):
    cache = Path(config["validation_cache"])
    data = cache.parent / "pixels.f32"
    values = np.empty((6, 1, 256, 256), dtype="<f4")
    for i, row in enumerate(values):
        row.fill(i + 0.125)
    data.write_bytes(values.tobytes())
    descriptor = json.loads(cache.read_bytes())
    descriptor["data_sha256"] = sha(data)
    cache.write_text(json.dumps(descriptor))

    def forward(model, batch):
        return np.column_stack((np.zeros(len(batch)), batch[:, 0, 0, 0]))

    monkeypatch.setattr(srnet, "float_logits", forward)
    monkeypatch.setattr(srnet_reference, "reference_logits", forward)
    report = service.evaluate(
        {**config, "validation_cache_sha256": sha(cache)}, tmp_path / "ordered.json"
    )
    predictions = report["predictions"]
    assert [p["sha256"] for p in predictions] == [r["sha256"] for r in descriptor["rows"]]
    np.testing.assert_allclose(
        [p["score"] for p in predictions],
        1 / (1 + np.exp(-np.arange(6) - 0.125)),
        rtol=0,
        atol=1e-15,
    )
