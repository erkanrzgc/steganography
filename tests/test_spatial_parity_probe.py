"""One-variable RGB parity-input probe: generated CPU QA only."""

import dataclasses

import numpy as np
import pytest
from PIL import Image

from core import colab_artifacts as a
from core import nonjpeg_models as models
from core import spatial_native_data as d
from core import spatial_native_training as t
from research import spatial_parity_probe as probe

torch = pytest.importorskip("torch")


def prepare(root, train=4):
    refs = []
    for i in range(train + 2):
        pixels = np.random.default_rng(300 + i).integers(0, 256, (256, 256, 3), dtype="u1")
        path = root / f"fixture-{i}-cover.png"
        Image.fromarray(pixels).save(path)
        source = d.Source(
            f"fixture-{i}",
            "RGB",
            f"group-{i}",
            f"Author {i}",
            f"https://commons.wikimedia.org/wiki/File:Fixture_{i}.png",
            a.digest(f"original-{i}".encode()),
            a.digest(pixels.tobytes()),
            "CC0-1.0",
            d.native.LICENSES["CC0-1.0"],
            "0" * 64,
        )
        refs.append(d.Input(source, path, a.digest(path.read_bytes())))
    out = root / "cache"
    d.prepare(tuple(refs), out)
    cache = d.load(out, a.digest((out / "cache.json").read_bytes()))
    split = t.Split(
        cache.manifest_sha256,
        "RGB",
        13,
        tuple(r.source.id for r in refs[:train]),
        (refs[train].source.id,),
        (refs[train + 1].source.id,),
        (),
    )
    return cache, split


def toy():
    return torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(6, 2)
    )


def test_parity_channels_append_scaled_lsb_sign_and_keep_rgb():
    x = np.arange(2 * 3 * 4 * 4, dtype="<f4").reshape(2, 3, 4, 4) % 256
    y = probe.with_parity(x)
    assert y.shape == (2, 6, 4, 4) and y.dtype == np.float32
    assert np.array_equal(y[:, :3], x)
    expected = np.where(x.astype(np.int64) % 2 == 1, probe.PARITY_SCALE, -probe.PARITY_SCALE)
    assert np.array_equal(y[:, 3:], expected)


@pytest.mark.parametrize(
    "value",
    [
        np.full((1, 3, 4, 4), 0.5, dtype="<f4"),
        np.full((1, 3, 4, 4), 256, dtype="<f4"),
        np.full((1, 3, 4, 4), -1, dtype="<f4"),
        np.zeros((1, 1, 4, 4), dtype="<f4"),
        np.zeros((3, 4, 4), dtype="<f4"),
        np.zeros((1, 3, 4, 4), dtype="<f8"),
        np.full((1, 3, 4, 4), np.nan, dtype="<f4"),
    ],
)
def test_parity_rejects_non_sample_inputs(value):
    with pytest.raises(ValueError):
        probe.with_parity(value)


def test_network_keeps_original_initialisation_and_widens_only_first_conv():
    torch.manual_seed(2026102301)
    base = models.network(models.RGB_DOMAIN)
    torch.manual_seed(2026102301)
    widened = probe.network()
    first = widened.front[0][0]
    assert first.in_channels == 6 and first.out_channels == 64
    assert torch.equal(first.weight[:, :3], base.front[0][0].weight)
    assert torch.equal(first.bias, base.front[0][0].bias)
    base_rest = {k: v for k, v in base.state_dict().items() if not k.startswith("front.0.0.")}
    probe_rest = {k: v for k, v in widened.state_dict().items() if not k.startswith("front.0.0.")}
    assert base_rest.keys() == probe_rest.keys()
    assert all(torch.equal(base_rest[k], probe_rest[k]) for k in base_rest)
    torch.manual_seed(1)
    again = probe.network()
    assert torch.equal(again.front[0][0].weight[:, 3:], first.weight[:, 3:])
    assert 0.15 < float(first.weight[:, 3:].detach().std()) < 0.4
    assert widened.architecture == probe.ARCHITECTURE


def test_generated_cpu_run_follows_original_schedule_and_is_deterministic(tmp_path):
    cache, split = prepare(tmp_path)
    first = probe.run(cache, split, device="cpu", epochs=2, max_seconds=120, build=toy)
    second = probe.run(cache, split, device="cpu", epochs=2, max_seconds=120, build=toy)
    assert first["losses"] == second["losses"]
    assert first["schema"] == "rgb-parity-probe-result-v1"
    assert first["optimizer_updates"] == 2 and first["presentations"] == 16
    plan = t.Plan(split, probe.MODEL_SEED, probe.PAYLOAD_SEED, 2, 8, probe.binding())
    assert first["schedule"] == [int(v % 4) for v in t.order(plan)]
    assert len(first["epochs"]) == 2
    for role, count in (("train", 4), ("development", 1)):
        block = first["final"][role]
        assert len(block["scores"]) == len(block["margins"]) == count
        assert 0 <= block["auc"] <= 1 and 0 <= block["margin_auc"] <= 1
        assert 0 <= first["ws_baseline"][role]["auc"] <= 1
    assert set(first["gate"]) == {"train_auc", "development_auc", "loss_ratio", "passed"}
    assert first["future_reserved_opened"] is False
    assert first["checkpoints"] == 0


def test_deadline_and_device_contract_fail_closed(tmp_path):
    cache, split = prepare(tmp_path)
    with pytest.raises(ValueError, match="deadline"):
        probe.run(cache, split, device="cpu", epochs=1, max_seconds=1e-9, build=toy)
    with pytest.raises(ValueError, match="device"):
        probe.run(cache, split, device="cuda:1", epochs=1, max_seconds=60, build=toy)
    with pytest.raises(ValueError, match="twenty"):
        probe.run(cache, split, device="cuda:0", epochs=2, max_seconds=60, build=toy)
    l_split = dataclasses.replace(split, mode="L")
    with pytest.raises(ValueError):
        probe.run(cache, l_split, device="cpu", epochs=1, max_seconds=60, build=toy)


def test_pairwise_auc_counts_ties_as_half():
    assert probe.pairwise_auc([[0.0, 1.0], [0.0, 1.0]]) == 1.0
    assert probe.pairwise_auc([[1.0, 0.0], [1.0, 0.0]]) == 0.0
    assert probe.pairwise_auc([[0.5, 0.5]]) == 0.5


def test_real_widened_srnet_takes_a_generated_cpu_step(tmp_path):
    cache, split = prepare(tmp_path)
    result = probe.run(cache, split, device="cpu", epochs=1, max_seconds=600)
    assert result["optimizer_updates"] == 1
    assert all(np.isfinite(result["losses"]))
    assert result["environment"]["torch"] == torch.__version__


def test_fresh_payload_changes_only_training_stego_and_keeps_fixed_scoring(tmp_path):
    cache, split = prepare(tmp_path)
    assert probe.training_seed(0, False) == probe.training_seed(5, False) == probe.PAYLOAD_SEED
    assert probe.training_seed(0, True) != probe.training_seed(1, True)
    assert probe.PAYLOAD_SEED not in {probe.training_seed(i, True) for i in range(probe.EPOCHS)}
    assert probe.binding(True) != probe.binding() == probe.binding(False)
    fixed = probe.run(cache, split, device="cpu", epochs=2, max_seconds=120, build=toy)
    fresh = probe.run(
        cache, split, device="cpu", epochs=2, max_seconds=120, build=toy, fresh_payload=True
    )
    again = probe.run(
        cache, split, device="cpu", epochs=2, max_seconds=120, build=toy, fresh_payload=True
    )
    assert fresh["losses"] == again["losses"]
    first = (0,)
    fixed_x, _ = d.batch(cache, "RGB", first, probe.training_seed(0, False))
    fresh_x, _ = d.batch(cache, "RGB", first, probe.training_seed(0, True))
    assert np.array_equal(fixed_x[0], fresh_x[0])  # identical cover
    assert not np.array_equal(fixed_x[1], fresh_x[1])  # fresh stego
    assert fresh["fresh_payload"] is True and fixed["fresh_payload"] is False
    assert fresh["training_payload_seeds"] == [probe.training_seed(i, True) for i in range(2)]
    assert fixed["training_payload_seeds"] == [probe.PAYLOAD_SEED] * 2
    assert fresh["initial"] == fixed["initial"]
    assert fresh["schedule"] == fixed["schedule"]


def test_learning_rate_is_bounded_bound_and_recorded(tmp_path):
    cache, split = prepare(tmp_path)
    assert probe.binding(False, probe.LEARNING_RATE) == probe.binding()
    assert probe.binding(True, 1e-3) not in {probe.binding(), probe.binding(True)}
    for bad in (0.0, -1e-3, 0.5, 1):
        with pytest.raises(ValueError, match="learning rate"):
            probe.run(cache, split, device="cpu", epochs=1, max_seconds=60, build=toy,
                      learning_rate=bad)
    result = probe.run(cache, split, device="cpu", epochs=1, max_seconds=60, build=toy,
                       fresh_payload=True, learning_rate=1e-3)
    assert result["learning_rate"] == 1e-3
    assert result["binding_sha256"] == probe.binding(True, 1e-3)


def test_second_original_seed_changes_order_and_only_original_seeds_allowed(tmp_path):
    cache, split = prepare(tmp_path)
    with pytest.raises(ValueError, match="model seed"):
        probe.run(cache, split, device="cpu", epochs=1, max_seconds=60, build=toy, model_seed=7)
    second = probe.run(cache, split, device="cpu", epochs=2, max_seconds=60, build=toy,
                       model_seed=probe.MODEL_SEEDS[1])
    plan = t.Plan(split, probe.MODEL_SEEDS[1], probe.PAYLOAD_SEED, 2, 8, probe.binding())
    assert second["schedule"] == [int(v % 4) for v in t.order(plan)]
    assert second["model_seed"] == probe.MODEL_SEEDS[1]
    assert probe.binding(model_seed=probe.MODEL_SEEDS[1]) != probe.binding()


@pytest.mark.parametrize(("epochs", "seconds"), [(0, 60), (21, 60), ("2", 60), (1, 0), (1, 1801)])
def test_epoch_and_deadline_bounds_fail_closed(tmp_path, epochs, seconds):
    cache, split = prepare(tmp_path)
    with pytest.raises(ValueError, match="bounded"):
        probe.run(cache, split, device="cpu", epochs=epochs, max_seconds=seconds, build=toy)


def test_nonfinite_training_loss_is_rejected(tmp_path):
    cache, split = prepare(tmp_path)

    def poisoned():
        model = toy()
        with torch.no_grad():
            model[2].weight.fill_(float("nan"))
        return model

    with pytest.raises(ValueError, match="nonfinite"):
        probe.run(cache, split, device="cpu", epochs=1, max_seconds=60, build=poisoned)


def test_unbiased_first_convolution_is_rejected(monkeypatch):
    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            conv = torch.nn.Conv2d(3, 64, 3, padding=1, bias=False)
            self.front = torch.nn.Sequential(torch.nn.Sequential(conv))

    monkeypatch.setattr(probe.models, "network", lambda _domain: Stub())
    with pytest.raises(ValueError, match="biased"):
        probe.network()
