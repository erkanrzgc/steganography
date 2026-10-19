"""Scalar weighted-SGD oracle, group isolation and finite/exposure contracts."""

import copy
import hashlib
import math

import numpy as np
import pytest

from core import nonjpeg_linear as linear


def rows(count=4, source="BOWS2", split="train"):
    return [{"sha256": hashlib.sha256(f"{i}:{label}:{split}".encode()).hexdigest(),
             "lineage": hashlib.sha256(f"original:{i}:{split}".encode()).hexdigest(),
             "source_group": source, "split": split, "label": label, "format": "PNG",
             "method": None if label == "cover" else "sequential", "rate_percent": 0 if
             label == "cover" else 5, "channel": None if label == "cover" else "gray",
             "plane": None if label == "cover" else 0,
             "speaker": f"speaker-{i % 2}", "group_key": f"recording-{i % 3}"}
            for i in range(count) for label in ("cover", "stego")]


def inputs(count=4):
    records = rows(count)
    x = np.array([[i / (2 * count), label] for i in range(count) for label in (0, 1)])
    y = np.array([r["label"] == "stego" for r in records], dtype=np.float64)
    return x, y, records


def test_weighting_does_not_multiply_cover_or_original_denominators():
    records = rows(2)
    records.extend([copy.deepcopy(records[1]) for _ in range(6)])
    weights = linear.training_weights(records)
    for original in {r["lineage"] for r in records}:
        for label in ("cover", "stego"):
            assert sum(w for w, r in zip(weights, records, strict=True)
                       if r["lineage"] == original and r["label"] == label) == pytest.approx(0.25)


def test_one_step_matches_independent_scalar_weighted_equations():
    x, y, records = inputs()
    n, d = x.shape
    mean = [sum(float(row[j]) for row in x) / n for j in range(d)]
    scale = [math.sqrt(sum((float(row[j]) - mean[j]) ** 2 for row in x) / n)
             for j in range(d)]
    initial = np.random.default_rng(linear.SEEDS[0]).normal(0, 0.01 / math.sqrt(d), d)
    gradients = [linear.L2 * float(w) for w in initial]
    bias_gradient = 0.0
    for row, target in zip(x, y, strict=True):
        z = [(float(row[j]) - mean[j]) / scale[j] for j in range(d)]
        margin = sum(a * float(b) for a, b in zip(z, initial, strict=True))
        error = (1 / (1 + math.exp(-margin)) - float(target)) / n
        for j in range(d):
            gradients[j] += z[j] * error
        bias_gradient += error
    model = linear.fit(x, y, records, linear.SEEDS[0], epochs=1)
    np.testing.assert_allclose(
        model["weights"], initial - linear.LEARNING_RATE * np.array(gradients))
    assert model["bias"] == pytest.approx(-linear.LEARNING_RATE * bias_gradient)
    np.testing.assert_allclose(model["mean"], mean)
    np.testing.assert_allclose(model["scale"], scale)


def test_dimension_and_rng_cannot_change_matched_presentations():
    x, y, records = inputs(8)
    a = linear.fit(x, y, records, linear.SEEDS[0])
    b = linear.fit(np.column_stack([x, x]), y, records, linear.SEEDS[0])
    c = linear.fit(x, y, records, linear.SEEDS[1])
    assert a["exposure_sha256"] == b["exposure_sha256"] != c["exposure_sha256"]
    assert a["row_presentations"] == len(records) * 20
    assert a["original_presentations"] == 8 * 20
    assert a["losses"][-1] < a["losses"][0]


@pytest.mark.parametrize("source,expected", [("BOWS2", 4), ("FSDD-v1.0.10", 2), ("ESC-50", 3)])
def test_every_speaker_recording_and_derivative_stays_in_one_fold(source, expected):
    records = rows(8, source)
    folds = linear.group_folds(records)
    assert len(set(folds)) == expected
    for group in {linear.group_key(r) for r in records}:
        assert len({fold for fold, row in zip(folds, records, strict=True)
                    if linear.group_key(row) == group}) == 1


def test_scaling_sees_only_fitting_fold_and_oof_is_complete(monkeypatch):
    x, y, records = inputs(4)
    original = linear.fit
    calls = []

    def fit(values, labels, rows, *args, **kwargs):
        model = original(values, labels, rows, *args, **kwargs)
        np.testing.assert_allclose(model["mean"], linear.training_weights(rows) @ values)
        calls.append({r["lineage"] for r in rows})
        return model

    monkeypatch.setattr(linear, "fit", fit)
    model, cv = linear.grouped_fit(x, y, records, linear.SEEDS[0])
    assert len(calls) == 5 and len(calls[-1]) == 4
    assert all(len(keys) == 3 for keys in calls[:-1])
    assert np.isfinite(cv["oof_scores"]).all()
    assert cv["all_optimizer_updates"] == 5 * 20
    assert model["epochs"] == 20


@pytest.mark.parametrize("fault", ["rows", "dimension", "shape", "y", "nan", "range", "label",
                                  "incomplete", "seed", "epochs", "boolean_epochs"])
def test_bad_training_contracts_are_rejected(fault):
    x, y, records = inputs()
    kwargs = {"epochs": 20}
    seed = linear.SEEDS[0]
    if fault == "rows":
        records = records[:1]
    elif fault == "dimension":
        x = x[:, :0]
    elif fault == "shape":
        x = x.ravel()
    elif fault == "y":
        y = list(y)
    elif fault == "nan":
        x[0, 0] = np.nan
    elif fault == "range":
        x[0, 0] = 2
    elif fault == "label":
        records[0]["label"] = "unknown"
    elif fault == "incomplete":
        records[0]["lineage"] = "orphan"
    elif fault == "seed":
        seed = 5
    elif fault == "epochs":
        kwargs["epochs"] = 21
    else:
        kwargs["epochs"] = True
    with pytest.raises(ValueError):
        linear.fit(x, y, records, seed, **kwargs)


@pytest.mark.parametrize("fault", ["split", "group", "cross_group", "one_group", "too_many"])
def test_cv_rejects_missing_cross_role_or_ambiguous_groups(fault):
    records = rows(33 if fault == "too_many" else 4, "FSDD-v1.0.10")
    if fault == "split":
        records[0]["split"] = "validation"
    elif fault == "group":
        records[0]["speaker"] = None
    elif fault == "cross_group":
        records[0]["speaker"] = "other"
    elif fault == "one_group":
        for row in records:
            row["speaker"] = "one"
    else:
        for i, row in enumerate(records):
            row["speaker"] = str(i)
    with pytest.raises(ValueError):
        linear.group_folds(records)


@pytest.mark.parametrize("fault", ["dimension", "range", "nan", "scale", "weights", "bias"])
def test_inference_contract_failures(fault):
    x, y, records = inputs()
    model = linear.fit(x, y, records, linear.SEEDS[0])
    if fault == "dimension":
        x = x[:, :1]
    elif fault == "range":
        x[0, 0] = -1
    elif fault == "nan":
        x[0, 0] = np.nan
    elif fault == "scale":
        model["scale"][0] = 0
    elif fault == "weights":
        model["weights"][0] = np.nan
    else:
        model["bias"] = np.nan
    with pytest.raises(ValueError):
        linear.predict(model, x)


def test_threshold_tie_rule_and_closed_validation():
    _, y, records = inputs()
    scores = np.full(len(y), 0.5)
    assert linear.select_threshold(scores, y, records) == 1
    records[0]["split"] = "validation"
    with pytest.raises(ValueError):
        linear.select_threshold(scores, y, records)


def test_deadline_and_partial_update_accounting():
    x, y, records = inputs()
    count = 0

    def update():
        nonlocal count
        count += 1

    def deadline():
        if count == 2:
            raise TimeoutError

    with pytest.raises(TimeoutError):
        linear.fit(x, y, records, linear.SEEDS[0], deadline=deadline, on_update=update)
    assert count == 2


def test_nonfinite_update_is_not_silently_accepted(monkeypatch):
    x, y, records = inputs()
    monkeypatch.setattr(linear, "LEARNING_RATE", float("inf"))
    with np.errstate(invalid="ignore"), pytest.raises(ValueError, match="nonfinite"):
        linear.fit(x, y, records, linear.SEEDS[0])


def test_weights_reject_a_single_unpaired_row():
    with pytest.raises(ValueError, match="row limit"):
        linear.training_weights(rows()[:1])
