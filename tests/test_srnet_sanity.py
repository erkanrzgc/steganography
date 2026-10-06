"""Metadata-only tiny selection and in-sample-only learning checks."""

import copy
import hashlib

import numpy as np
import pytest

from core import srnet_sanity as sanity
from steganography import research_srnet_sanity as service


def rows():
    result = []
    for source, count, qualities in (("ALASKA2", 5, (None,)), ("BOSSbase-1.01", 3, (75, 95))):
        for lineage in range(count):
            for q in qualities:
                for method in (None, "JUNIWARD", "UERD"):
                    result.append(
                        {
                            "sha256": hashlib.sha256(
                                f"{source}/{lineage}/{q}/{method}".encode()
                            ).hexdigest(),
                            "lineage": str(lineage),
                            "source_group": source,
                            "quality_factor": q,
                            "method": method,
                            "label": "cover" if method is None else "stego",
                            "split": "train",
                            "format": "JPEG",
                        }
                    )
    return result


def records():
    return [{"epoch": e, "updates": 8, "mean_pair_loss": 0.8 - 0.6 * e / 49} for e in range(50)]


def test_metadata_selection_full_lineages_original_order_and_no_mutation():
    samples = rows()
    saved = copy.deepcopy(samples)
    indices = sanity.select(samples)
    assert len(indices) == 24 and not indices.flags.writeable
    assert samples == saved and sorted(indices.tolist()) == indices.tolist()
    assert all(samples[i]["lineage"] in ("0", "1", "2", "3") for i in indices)
    assert len({(samples[i]["source_group"], samples[i]["lineage"]) for i in indices}) == 6
    assert (
        len(
            {
                (samples[i]["source_group"], samples[i]["lineage"], samples[i]["quality_factor"])
                for i in indices
            }
        )
        == 8
    )
    altered = copy.deepcopy(samples)
    for s in altered:
        s["score"] = 999  # Selection cannot depend on prior model predictions.
    np.testing.assert_array_equal(sanity.select(altered), indices)


def test_incomplete_wrong_sources_and_validation_fail():
    samples = rows()
    for bad in (
        [s for s in samples if s["source_group"] == "ALASKA2"],
        [s for s in samples if s["lineage"] == "0"],
        samples[:-1],
    ):
        with pytest.raises(ValueError):
            sanity.select(bad)
    samples[0]["split"] = "validation"
    with pytest.raises(ValueError):
        sanity.select(samples)


def test_scalar_loss_imbalance_ties_and_objective_failures():
    samples = [rows()[i] for i in sanity.select(rows())]
    result = sanity.metrics(np.zeros((24, 2)), samples)
    assert result["balanced_accuracy"] == 0.5
    assert result["recall"] == result["false_positive_rate"] == 1
    assert result["cross_entropy"] == pytest.approx(np.log(2))
    assert all(result["scores"][i] == 0.5 for i in range(24))
    objectives = sanity.objectives(records(), {"balanced_accuracy": 0.95})
    assert all(v for k, v in objectives.items() if k != "relative_loss_reduction")
    assert objectives["relative_loss_reduction"] == pytest.approx(0.75)
    flat = [{**r, "mean_pair_loss": 0.69} for r in records()]
    assert not sanity.objectives(flat, result)["final_batch_loss_at_most_0_35"]
    assert not sanity.objectives(flat, result)["relative_loss_reduction_at_least_0_25"]
    assert not sanity.objectives(flat, result)["stored_bn_train_balanced_accuracy_at_least_0_90"]
    zeros = [{**r, "mean_pair_loss": 0.0} for r in records()]
    assert sanity.objectives(zeros, result)["relative_loss_reduction"] == 0
    for bad in (
        records()[:-1],
        [{**r, "updates": 16} for r in records()],
        [{**r, "mean_pair_loss": float("nan")} for r in records()],
    ):
        with pytest.raises(ValueError):
            sanity.objectives(bad, result)
    with pytest.raises(ValueError):
        sanity.objectives(records(), {"balanced_accuracy": float("nan")})


@pytest.mark.parametrize(
    "bad", [np.zeros((24, 1)), np.full((24, 2), np.nan), np.zeros((24, 2), dtype=complex)]
)
def test_bad_logits_rejected(bad):
    with pytest.raises(ValueError):
        sanity.metrics(bad, [rows()[i] for i in sanity.select(rows())])


def test_metadata_cardinality_defense_and_metrics_roles(monkeypatch):
    samples = rows()
    monkeypatch.setattr(sanity, "epoch_pairs", lambda *a, **kw: None)
    with pytest.raises(ValueError, match="24"):
        sanity.select(samples[1:])
    samples = rows()[:3]
    samples[0]["split"] = "validation"
    with pytest.raises(ValueError):
        sanity.metrics(np.zeros((3, 2)), samples)
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="margins"):
        sanity.metrics(np.array([[-1e308, 1e308]] * 3), rows()[:3])


@pytest.fixture
def inputs(monkeypatch, tmp_path):
    config = {
        "manifest": str(tmp_path / "manifest.json"),
        "manifest_sha256": "0" * 64,
        "cache": str(tmp_path / "cache.json"),
        "cache_sha256": "1" * 64,
    }
    monkeypatch.setattr(service, "MANIFEST_SHA", config["manifest_sha256"])
    monkeypatch.setattr(service, "CACHE_SHA", config["cache_sha256"])
    monkeypatch.setattr(service, "read_document", lambda p: ({}, config["manifest_sha256"]))
    samples = rows()
    pixels = np.zeros((len(samples), 1, 256, 256), dtype="<f4")
    for i, s in enumerate(samples):
        pixels[i].fill(s["label"] == "stego")

    def load(*a, **kw):
        assert kw["split"] == "train" and kw["_float"] is True
        assert kw["checksum"] == config["cache_sha256"]
        return pixels, samples, {"data_sha256": "2" * 64, "decoder": {}}

    monkeypatch.setattr(service, "load_pixels", load)

    class Model:
        def named_buffers(self):
            return [("bn.num_batches_tracked", np.array(400))]

    def fit(values, all_samples, indices, *, seed, schedule, config):
        assert values is pixels and all_samples is samples and len(indices) == 24
        assert seed == 20261012 and len(schedule) == 50
        assert all(r["pairs"] == 16 for r in schedule)
        assert config["batch_recipe"] == service.srnet_multibatch.RECIPE
        return Model(), records()

    monkeypatch.setattr(service.srnet_training, "fit", fit)
    monkeypatch.setattr(
        service.srnet,
        "float_logits",
        lambda m, v: np.array([[-5.0, 5.0]]) if v[0, 0, 0, 0] else np.array([[5.0, -5.0]]),
    )

    def save(model, path):
        path.write_bytes(b"generated-numeric-fixture")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(service.srnet_model, "save_model", save)
    return config, tmp_path / "out"


def test_service_complete_train_only_report_no_paths_and_thread_restore(inputs):
    torch = pytest.importorskip("torch")
    previous, rng = torch.get_num_threads(), torch.get_rng_state().clone()
    config, out = inputs
    report = service.run(config, out)
    assert torch.get_num_threads() == previous and torch.equal(rng, torch.get_rng_state())
    assert report["status"] == "completed" and report["sanity_objectives_passed"]
    assert report["in_sample_only"] and not report["validation_pixels_loaded"]
    assert report["accuracy_qualification"] == "unavailable" and not report["deployed"]
    assert len(report["selected_rows"]) == 24 and report["optimizer_updates"] == 400
    assert config["manifest"] not in (out / "sanity.json").read_text()
    with pytest.raises(FileExistsError):
        service.run(config, out)


def test_checksum_protocol_deadline_and_secret_config_rejected(inputs, monkeypatch):
    pytest.importorskip("torch")
    config, out = inputs
    with pytest.raises(ValueError, match="train-only"):
        service.run({**config, "validation_cache": "do-not-read"}, out)
    with pytest.raises(ValueError, match="manifest"):
        service.run({**config, "manifest_sha256": "f" * 64}, out)
    monkeypatch.setattr(service.time, "monotonic", iter([0, 1861]).__next__)
    with pytest.raises(ValueError, match="deadline"):
        service.run(config, out)
    assert not out.exists()
    monkeypatch.setattr(service, "PROTOCOL_SHA", "f" * 64)
    with pytest.raises(ValueError, match="protocol"):
        service.run(config, out)


def test_input_contract_actual_manifest_bn_and_symlink_failures(inputs, monkeypatch):
    config, out = inputs
    for bad in (None, {}, {**config, "cache_sha256": True}, {**config, "cache": ""}):
        with pytest.raises(ValueError):
            service.configuration(bad)
    monkeypatch.setattr(service, "read_document", lambda p: ({}, "f" * 64))
    with pytest.raises(ValueError, match="manifest"):
        service.run(config, out)
    monkeypatch.setattr(service, "read_document", lambda p: ({}, config["manifest_sha256"]))

    class Model:
        def named_buffers(self):
            return [("bn.num_batches_tracked", np.array(401))]

    monkeypatch.setattr(service.srnet_training, "fit", lambda *a, **kw: (Model(), records()))
    with pytest.raises(ValueError, match="BN"):
        service.run(config, out)
    assert not out.exists()
    link = out.parent / "link"
    link.symlink_to(out.parent, target_is_directory=True)
    with pytest.raises(FileExistsError):
        service.run(config, link / "new")


def test_worker_limits_cli_status_and_secret_redaction(inputs, monkeypatch, capsys):
    import resource
    import sys

    config, out = inputs
    monkeypatch.setattr(sys, "argv", ["sanity", "--config", config["manifest"], "--out", str(out)])
    monkeypatch.setattr(service, "read_document", lambda p: (config, "0" * 64))
    observed = []
    monkeypatch.setattr(resource, "setrlimit", lambda which, value: observed.append((which, value)))
    monkeypatch.setattr(service, "run", lambda *a: {"sanity_objectives_passed": True})
    assert service.main() == 0
    assert observed == [
        (resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3)),
        (resource.RLIMIT_CPU, (3660, 3661)),
        (resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2)),
        (resource.RLIMIT_CORE, (0, 0)),
    ]
    monkeypatch.setattr(service, "run", lambda *a: {"sanity_objectives_passed": False})
    assert service.main() == 2

    def broken(*a):
        raise RuntimeError("password-SECRET /private/host/path")

    monkeypatch.setattr(service, "run", broken)
    assert service.main() == 2
    text = capsys.readouterr().out
    assert "password-SECRET" not in text and "/private/" not in text
