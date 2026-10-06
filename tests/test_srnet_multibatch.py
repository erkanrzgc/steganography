"""Independent full-batch accounting and generated actual four-row learning."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from core import srnet, srnet_model
from core import srnet_multibatch as batches
from core.srnet_sampling import epoch_pairs
from steganography import research_srnet_evaluate as evaluator
from steganography import research_srnet_fit as fitting
from steganography import research_srnet_plan as planner
from tests.test_jrm_reference import corpus as corpus
from tests.test_jrm_reference import sha
from tests.test_srnet_sampling import config as plan_config  # noqa: F401
from tests.test_srnet_sampling import rows

torch = pytest.importorskip("torch")


def test_full_order_labels_balance_no_truncation_or_rng_change():
    samples = rows()
    original = copy.deepcopy(samples)
    rng = np.random.get_state()
    pairs, paired = epoch_pairs(samples, seed=91, epoch=0)
    values, record = batches.epoch_batches(samples, seed=91, epoch=0)
    assert not values.flags.writeable and values.shape == (12, 4)
    np.testing.assert_array_equal(values.reshape(-1, 2), pairs)
    assert samples == original and np.array_equal(np.random.get_state()[1], rng[1])
    assert record["optimizer_updates"] == 12 and record["pairs"] == paired["pairs"] == 24
    identities = []
    for c0, s0, c1, s1 in values:
        assert [samples[i]["label"] for i in (c0, s0, c1, s1)] == [
            "cover",
            "stego",
            "cover",
            "stego",
        ]
        assert samples[c0]["source_group"] != samples[c1]["source_group"]
        assert all(samples[c]["lineage"] == samples[s]["lineage"] for c, s in ((c0, s0), (c1, s1)))
        identities.append([samples[i]["sha256"] for i in (c0, s0, c1, s1)])
    assert (
        record["ordered_batch_sha256"]
        == hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()
    )
    assert batches.epoch_batches(samples, seed=91, epoch=0)[1] == record


@pytest.mark.parametrize("bad", [None, "wrong", 4, True])
def test_recipe_is_explicit_and_strict(bad):
    with pytest.raises(ValueError):
        batches.recipe({"batch_recipe": bad})
    assert batches.recipe({}) is None


def test_reject_incomplete_source_or_fake_adjacency(monkeypatch):
    with pytest.raises(ValueError, match="exactly two"):
        batches.epoch_batches([s for s in rows() if s["source_group"] == "A"], seed=91, epoch=0)
    samples = rows()
    monkeypatch.setattr(batches, "epoch_pairs", lambda *a, **kw: (np.array([[0, 1], [0, 2]]), {}))
    with pytest.raises(ValueError, match="distinct"):
        batches.epoch_batches(samples, seed=91, epoch=0)
    monkeypatch.setattr(batches, "epoch_pairs", lambda *a, **kw: (np.array([[0, 1]]), {}))
    with pytest.raises(ValueError, match="exactly two"):
        batches.epoch_batches(samples, seed=91, epoch=0)


@pytest.fixture
def config(plan_config, tmp_path):  # noqa: F811
    plan_path = tmp_path / "multi-plan.json"
    planner.plan_training({**plan_config, "epochs": 1, "batch_recipe": batches.RECIPE}, plan_path)
    return {
        **{k: plan_config[k] for k in ("manifest", "manifest_sha256", "cache", "cache_sha256")},
        "plan": str(plan_path),
        "plan_sha256": sha(plan_path),
        "batch_recipe": batches.RECIPE,
        "threads": 2,
    }


def test_actual_four_row_fit_card_updates_input_mapping_and_legacy_survives(
    config, tmp_path, monkeypatch
):
    cache = Path(config["cache"])
    data = cache.parent / "pixels.f32"
    values = np.frombuffer(data.read_bytes(), dtype="<f4").copy().reshape(6, 1, 256, 256)
    for i, row in enumerate(values):
        row.fill(128.125 + i)
    data.write_bytes(values.tobytes())
    descriptor = json.loads(cache.read_bytes())
    descriptor["data_sha256"] = sha(data)
    cache.write_text(json.dumps(descriptor))
    config = {**config, "cache_sha256": sha(cache)}
    plan_path = tmp_path / "mapped-plan.json"
    planner.plan_training(
        {
            **{k: config[k] for k in ("manifest", "manifest_sha256", "cache", "cache_sha256")},
            "epochs": 1,
            "seed": 91,
            "batch_recipe": batches.RECIPE,
        },
        plan_path,
    )
    config.update(plan=str(plan_path), plan_sha256=sha(plan_path))
    observed = []
    factory = srnet.network

    def network():
        model = factory()
        model.register_forward_pre_hook(
            lambda _, args: observed.append(args[0].detach().numpy().copy())
        )
        return model

    monkeypatch.setattr(srnet, "network", network)
    rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    card = fitting.train_srnet(config, tmp_path / "model")
    assert torch.equal(rng, torch.get_rng_state()) and torch.get_num_threads() == threads
    assert card["schema_version"] == "srnet-fit-v2" and card["batch_size"] == 4
    assert card["epoch_training"][0]["updates"] == 2 and len(observed) == 2
    assert all(x.shape == (4, 1, 256, 256) for x in observed)
    samples = json.loads(Path(config["manifest"]).read_bytes())["samples"][:6]
    expected, record = batches.epoch_batches(samples, seed=91, epoch=0)
    for actual, indices in zip(observed, expected, strict=True):
        np.testing.assert_array_equal(actual, values[indices])
    assert card["epoch_batch_schedule"] == [record]
    plan = json.loads(plan_path.read_bytes())
    evaluator.checked_card(card, plan, plan["settings"], config["plan_sha256"])
    forged = copy.deepcopy(card)
    forged["epoch_training"][0]["updates"] = 4  # Pair count is not the BN update count.
    with pytest.raises(ValueError):
        evaluator.checked_card(forged, plan, plan["settings"], config["plan_sha256"])
    forged = copy.deepcopy(card)
    forged["epoch_batch_schedule"][0]["ordered_batch_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        evaluator.checked_card(forged, plan, plan["settings"], config["plan_sha256"])
    loaded = srnet_model.load_model(tmp_path / "model/model.npz", checksum=card["model_sha256"])
    assert all(int(v) == 2 for k, v in loaded.named_buffers() if k.endswith("num_batches_tracked"))
    assert not card["deployed"] and not card["validation_used"]
    legacy_path = tmp_path / "legacy-plan.json"
    legacy = planner.plan_training(
        {
            **{k: config[k] for k in ("manifest", "manifest_sha256", "cache", "cache_sha256")},
            "epochs": 1,
            "seed": 91,
        },
        legacy_path,
    )
    legacy_card = fitting.card_contract(legacy, legacy["settings"], fitting.configuration(config))
    assert legacy["schema_version"] == "srnet-training-plan-v1"
    assert legacy_card["schema_version"] == "srnet-fit-v1" and legacy_card["batch_size"] == 2
    assert "epoch_batch_schedule" not in legacy and "batch_recipe" not in legacy["settings"]
    from core import srnet_reference
    from steganography import research_pixels

    manifest = Path(config["manifest"])
    validation_dir = tmp_path / "validation"
    research_pixels.extract_pixels(
        manifest, validation_dir, source=manifest.parent, split="validation", _float=True
    )
    monkeypatch.setattr(srnet, "float_logits", lambda model, values: np.zeros((len(values), 2)))
    monkeypatch.setattr(
        srnet_reference, "reference_logits", lambda arrays, values: np.zeros((1, 2))
    )
    report = evaluator.evaluate(
        {
            **{
                k: config[k]
                for k in (
                    "manifest",
                    "manifest_sha256",
                    "cache",
                    "cache_sha256",
                    "plan",
                    "plan_sha256",
                )
            },
            "validation_cache": str(validation_dir / "cache.json"),
            "validation_cache_sha256": sha(validation_dir / "cache.json"),
            "model_dir": str(tmp_path / "model"),
            "card_sha256": sha(tmp_path / "model/model-card.json"),
        },
        tmp_path / "predictions.json",
    )
    assert report["status"] == "completed" and len(report["predictions"]) == 6
    assert not report["deployed"] and not report["calibrated"]


def test_wrong_or_omitted_recipe_and_forged_grouping_fail(config, tmp_path):
    missing = {k: v for k, v in config.items() if k != "batch_recipe"}
    with pytest.raises(ValueError, match="recipe/plan"):
        fitting.train_srnet(missing, tmp_path / "out")
    plan_path = Path(config["plan"])
    plan = json.loads(plan_path.read_bytes())
    plan["epoch_batch_schedule"][0]["optimizer_updates"] = 1
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="provenance"):
        fitting.train_srnet({**config, "plan_sha256": sha(plan_path)}, tmp_path / "out")
