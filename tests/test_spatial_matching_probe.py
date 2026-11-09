"""LSB-matching fine-tune probe: generated CPU QA only."""

import numpy as np
import pytest

from core import colab_artifacts as a
from research import spatial_matching_probe as m
from research import spatial_parity_probe as probe
from tests.test_spatial_parity_probe import prepare, toy

torch = pytest.importorskip("torch")


def cover(seed=1, shape=(16, 16, 3)):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


@pytest.mark.parametrize("kind", m.KINDS)
def test_embedding_writes_message_lsbs_only_at_chosen_positions(kind):
    image = cover()
    out = m.embed(image, kind, 0.4, np.random.default_rng(5))
    replay = np.random.default_rng(5)
    chosen = replay.permutation(image.size)[: int(round(image.size * 0.4))]
    bits = replay.integers(0, 2, chosen.size)
    flat, before = out.reshape(-1).astype(int), image.reshape(-1).astype(int)
    assert np.array_equal(flat[chosen] & 1, bits)
    untouched = np.setdiff1d(np.arange(image.size), chosen)
    assert np.array_equal(flat[untouched], before[untouched])
    assert np.abs(flat - before).max() <= 1
    if kind == "replacement":
        assert np.array_equal(flat >> 1, before >> 1)


def test_matching_moves_both_ways_and_stays_in_range():
    image = cover(2, (64, 64, 3))
    diff = m.embed(image, "matching", 1.0, np.random.default_rng(3)).astype(int) - image
    assert set(np.unique(diff)) == {-1, 0, 1}
    low = m.embed(np.zeros((8, 8, 3), np.uint8), "matching", 1.0, np.random.default_rng(4))
    high = m.embed(np.full((8, 8, 3), 255, np.uint8), "matching", 1.0, np.random.default_rng(4))
    assert set(np.unique(low)) <= {0, 1} and set(np.unique(high)) <= {254, 255}
    assert low.max() == 1 and high.min() == 254


@pytest.mark.parametrize(
    "image,kind,rate",
    [
        (np.zeros((4, 4, 3), np.int16), "matching", 0.4),
        (np.zeros((4, 4), np.uint8), "matching", 0.4),
        (np.zeros((4, 4, 3), np.uint8), "plusminus", 0.4),
        (np.zeros((4, 4, 3), np.uint8), "matching", 0.0),
        (np.zeros((4, 4, 3), np.uint8), "matching", 1.5),
        (np.zeros((4, 4, 3), np.uint8), "matching", 1),
    ],
)
def test_embedding_rejects_invalid_inputs(image, kind, rate):
    with pytest.raises(ValueError):
        m.embed(image, kind, rate, np.random.default_rng(0))


def test_d4_gives_eight_distinct_pixel_exact_variants():
    image = cover(3, (5, 5, 3))
    variants = [m.d4(image, k) for k in range(8)]
    assert len({v.tobytes() for v in variants}) == 8
    assert all(np.array_equal(np.sort(v, axis=None), np.sort(image, axis=None)) for v in variants)
    assert np.array_equal(variants[0], image) and variants[1].flags["C_CONTIGUOUS"]
    for bad in (-1, 8, 1.0):
        with pytest.raises(ValueError):
            m.d4(image, bad)


def test_curriculum_lowers_rates_by_epoch():
    assert m.rates(0) == m.rates(9) == (0.4,)
    assert m.rates(10) == (0.2, 0.4)
    assert m.rates(20) == m.rates(39) == (0.1, 0.2, 0.4)
    for bad in (-1, 1.0):
        with pytest.raises(ValueError):
            m.rates(bad)


def test_eval_stegos_reproduce_the_hard_condition_seeds():
    covers = [cover(10), cover(11)]
    stegos = m.eval_stegos(covers)
    assert list(stegos) == [f"{k}-{r}" for k, r in m.CONDITIONS]
    k = m.CONDITIONS.index(("matching", 0.2))
    rng = np.random.default_rng(m.EVAL_SEED + 1000 * k + 1)
    expected = m.embed(covers[1], "matching", 0.2, rng)
    assert np.array_equal(stegos["matching-0.2"][1], expected)
    assert all(np.array_equal(x, y) for x, y in zip(stegos["matching-0.1"],
                                                     m.eval_stegos(covers)["matching-0.1"],
                                                     strict=True))


def test_training_pairs_are_deterministic_d4_covers_with_matching_stegos():
    image = cover(4, (8, 8, 3))
    first = m.training_pair(image, 25, 3, probe.MODEL_SEEDS[0])
    again = m.training_pair(image, 25, 3, probe.MODEL_SEEDS[0])
    assert all(np.array_equal(x, y) for x, y in zip(first, again, strict=True))
    assert any(np.array_equal(first[0], m.d4(image, k)) for k in range(8))
    assert np.abs(first[1].astype(int) - first[0]).max() <= 1
    other = m.training_pair(image, 25, 3, probe.MODEL_SEEDS[1])
    assert not all(np.array_equal(x, y) for x, y in zip(first, other, strict=True))


def kwargs(**extra):
    base = {"model_seed": probe.MODEL_SEEDS[0], "init_state": None, "init_sha256": None,
            "epochs": 3, "decay_epoch": 2, "eval_every": 2, "build": toy}
    base.update(extra)
    return base


def test_toy_run_is_deterministic_decays_scores_and_saves_once(tmp_path):
    cache, split = prepare(tmp_path, train=5)
    seen = []
    first = m.run(cache, split, **kwargs(state_path=tmp_path / "state.pt", progress=seen.append))
    assert seen == first["history"] and seen[0]["loss"] is None
    assert seen[-1]["loss"] == first["losses"][-1]
    second = m.run(cache, split, **kwargs())
    assert first["losses"] == second["losses"] and first["history"] == second["history"]
    assert [h["epoch"] for h in first["history"]] == [0, 2, 3]
    assert set(first["history"][0]["auc"]) == {f"{k}-{r}" for k, r in m.CONDITIONS}
    assert first["learning_rates"] == [1e-3, 1e-3, pytest.approx(1e-4)]
    assert first["updates"] == 3 * 2 and set(first["gate"]) == set(m.GATE)
    assert first["passed"] == all(g["passed"] for g in first["gate"].values())
    assert first["state_sha256"] == a.digest((tmp_path / "state.pt").read_bytes())
    assert second["state_sha256"] is None and not first["future_reserved_opened"]
    with pytest.raises(ValueError):
        m.run(cache, split, **kwargs(state_path=tmp_path / "state.pt"))


def test_warm_start_requires_matching_hash(tmp_path):
    cache, split = prepare(tmp_path)
    path = tmp_path / "init.pt"
    digest = probe._save_state(toy(), path)
    result = m.run(cache, split, **kwargs(init_state=path, init_sha256=digest, epochs=1,
                                          decay_epoch=None))
    assert result["init_sha256"] == digest and result["learning_rates"] == [1e-3]
    with pytest.raises(ValueError, match="hash"):
        m.run(cache, split, **kwargs(init_state=path, init_sha256="0" * 64))


@pytest.mark.parametrize(
    "extra",
    [
        {"epochs": 0}, {"epochs": 61}, {"device": "cuda:1"}, {"max_seconds": 0},
        {"max_seconds": m.MAX_SECONDS + 1}, {"decay_epoch": 3}, {"decay_epoch": 0},
        {"learning_rate": 0.1}, {"learning_rate": 1}, {"model_seed": 7}, {"eval_every": 0},
        {"init_sha256": "0" * 64},
    ],
)
def test_invalid_contracts_fail_closed(tmp_path, extra):
    cache, split = prepare(tmp_path)
    with pytest.raises(ValueError):
        m.run(cache, split, **kwargs(**extra))


def test_deadline_and_nonfinite_loss_fail_closed(tmp_path):
    cache, split = prepare(tmp_path)
    with pytest.raises(ValueError, match="deadline"):
        m.run(cache, split, **kwargs(max_seconds=1e-9))

    def broken():
        model = toy()
        with torch.no_grad():
            model[2].weight.fill_(float("nan"))
        return model

    with pytest.raises(ValueError, match="nonfinite"):
        m.run(cache, split, **kwargs(build=broken))


def test_real_widened_srnet_takes_a_matching_step(tmp_path):
    cache, split = prepare(tmp_path, train=1)
    result = m.run(cache, split, **kwargs(build=probe.network, epochs=1, decay_epoch=None))
    assert result["updates"] == 1 and np.isfinite(result["losses"][0])
