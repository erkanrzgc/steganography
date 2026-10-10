"""Independent metadata accounting, not evidence of detector learning."""

import copy
import hashlib
import json
from collections import Counter

import numpy as np
import pytest

from core import srnet_diversity_sampling as sampling
from core import srnet_scale_sampling as legacy


def rows(contexts=(("A", None, 5), ("B", 75, 1), ("B", 95, 2), ("C", 75, 3))):
    result = []
    for source, quality, count in contexts:
        for original in range(count):
            for method in (None, "JUNIWARD", "UERD"):
                result.append(
                    {
                        "sha256": hashlib.sha256(str(len(result)).encode()).hexdigest(),
                        "lineage": f"{source}-{quality}-{original}",
                        "source_group": source,
                        "quality_factor": quality,
                        "label": "cover" if method is None else "stego",
                        "method": method,
                        "split": "train",
                        "format": "JPEG",
                    }
                )
    return result


def test_independent_counts_hashes_coverage_immutability_and_rng():
    samples = rows()
    original = copy.deepcopy(samples)
    state = np.random.get_state()
    batches, record = sampling.epoch_batches(samples, seed=91, epoch=0)
    assert samples == original and np.array_equal(np.random.get_state()[1], state[1])
    assert not batches.flags.writeable and batches.shape == (18, 4)
    counts, combinations, seen, identities = Counter(), Counter(), set(), []
    for c0, s0, c1, s1 in batches:
        assert samples[c0]["source_group"] != samples[c1]["source_group"]
        assert samples[c0]["lineage"] != samples[c1]["lineage"]
        combinations[tuple(sorted((samples[c0]["source_group"], samples[c1]["source_group"])))] += 1
        for c, s in ((c0, s0), (c1, s1)):
            cover, stego = samples[c], samples[s]
            assert cover["label"] == "cover" and stego["label"] == "stego"
            assert all(cover[k] == stego[k] for k in ("source_group", "lineage", "quality_factor"))
            counts[(stego["source_group"], stego["quality_factor"], stego["method"])] += 1
            seen.add(int(s))
        identities.append([samples[i]["sha256"] for i in (c0, s0, c1, s1)])
    assert seen == {i for i, row in enumerate(samples) if row["label"] == "stego"}
    assert combinations == {("A", "B"): 6, ("A", "C"): 6, ("B", "C"): 6}
    assert counts == {
        ("A", None, "JUNIWARD"): 6,
        ("A", None, "UERD"): 6,
        ("B", 75, "JUNIWARD"): 3,
        ("B", 75, "UERD"): 3,
        ("B", 95, "JUNIWARD"): 3,
        ("B", 95, "UERD"): 3,
        ("C", 75, "JUNIWARD"): 6,
        ("C", 75, "UERD"): 6,
    }
    assert record["pair_schedule"]["pairs_per_source"] == 12
    assert record["batch_schedule"]["optimizer_updates"] == 18
    assert (
        record["batch_schedule"]["ordered_batch_sha256"]
        == hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()
    )
    repeated, again = sampling.epoch_batches(samples, seed=91, epoch=0)
    np.testing.assert_array_equal(batches, repeated)
    assert record == again
    assert not np.array_equal(batches, sampling.epoch_batches(samples, seed=91, epoch=1)[0])


def test_expanded_train_population_and_old_limits_are_separate():
    samples = rows(
        (
            ("ALASKA2", None, 816),
            ("BOSS", 75, 822),
            ("BOSS", 95, 822),
            ("BOWS2", 75, 807),
            ("BOWS2", 95, 807),
        )
    )
    assert len(samples) == 12_222
    batches, record = sampling.epoch_batches(samples, seed=20261010, epoch=0)
    assert batches.shape == (4932, 4)
    assert record["pair_schedule"]["pairs_per_source"] == 3288
    assert {int(s) for _, s in batches.reshape(-1, 2)} == {
        i for i, row in enumerate(samples) if row["label"] == "stego"
    }
    assert legacy.MAX_ROWS == 12_000
    with pytest.raises(ValueError, match="row limit"):
        legacy.epoch_batches(samples, seed=20261010, epoch=0)
    with pytest.raises(ValueError, match="exactly two"):
        legacy.epoch_batches(rows(), seed=91, epoch=0)


@pytest.mark.parametrize(
    "fault",
    ["two", "four", "validation", "duplicate", "overlap", "incomplete", "limit", "seed", "epoch"],
)
def test_reject_bad_roles_origins_lineage_families_and_bounds(fault):
    samples, seed, epoch = rows(), 91, 0
    if fault == "two":
        samples = [s for s in samples if s["source_group"] != "C"]
    elif fault == "four":
        samples = rows((("A", None, 1), ("B", 75, 1), ("C", 75, 1), ("D", 75, 1)))
    elif fault == "validation":
        samples[0]["split"] = "validation"
    elif fault == "duplicate":
        samples[1]["sha256"] = samples[0]["sha256"]
    elif fault == "overlap":
        for row in samples:
            if row["lineage"] == "C-75-0":
                row["lineage"] = "A-None-0"
    elif fault == "incomplete":
        samples.pop()
    elif fault == "limit":
        samples = [samples[0]] * (sampling.MAX_ROWS + 1)
    elif fault == "seed":
        seed = True
    elif fault == "epoch":
        epoch = 50
    with pytest.raises(ValueError):
        sampling.epoch_batches(samples, seed=seed, epoch=epoch)


def test_corrupt_shared_schedule_cannot_make_same_source_batch(monkeypatch):
    monkeypatch.setattr(
        sampling.srnet_sampling, "_epoch_pairs", lambda *a, **k: (np.array([[0, 1]] * 6), {})
    )
    with pytest.raises(ValueError, match="distinct declared sources"):
        sampling.epoch_batches(rows(), seed=91, epoch=0)


def test_same_original_can_keep_multiple_quality_contexts():
    samples = rows()
    for row in samples:
        if row["lineage"] == "B-95-0":
            row["lineage"] = "B-75-0"
    batches, _ = sampling.epoch_batches(samples, seed=91, epoch=0)
    assert {int(s) for _, s in batches.reshape(-1, 2)} == {
        i for i, row in enumerate(samples) if row["label"] == "stego"
    }


def test_shared_pair_budget_still_applies(monkeypatch):
    monkeypatch.setattr(sampling.srnet_sampling, "MAX_PAIRS", 35)
    with pytest.raises(ValueError, match="pair limit"):
        sampling.epoch_batches(rows(), seed=91, epoch=0)


def test_incomplete_shared_cycle_is_rejected(monkeypatch):
    monkeypatch.setattr(
        sampling.srnet_sampling,
        "_epoch_pairs",
        lambda *a, **k: (np.array([[0, 1]] * 3), {}),
    )
    with pytest.raises(ValueError, match="three complete sources"):
        sampling.epoch_batches(rows(), seed=91, epoch=0)
