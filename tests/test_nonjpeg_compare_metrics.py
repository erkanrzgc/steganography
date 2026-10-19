"""Independent scalar AUC/ECE and matched cluster-bootstrap checks."""

import copy

import numpy as np
import pytest

from core import nonjpeg_compare_metrics as m
from tests.test_nonjpeg_linear import rows


def scalar(pairs, weights, threshold):
    n = sum(weights)
    auc = sum(w * v * (float(b > a) + 0.5 * float(b == a))
              for (_, b), w in zip(pairs, weights, strict=True)
              for (a, _), v in zip(pairs, weights, strict=True)) / n**2
    recall = sum(w for (_, b), w in zip(pairs, weights, strict=True) if b >= threshold) / n
    fpr = sum(w for (a, _), w in zip(pairs, weights, strict=True) if a >= threshold) / n
    errors = [0.0] * 10
    for (a, b), weight in zip(pairs, weights, strict=True):
        for label, probability in ((0, a), (1, b)):
            errors[min(int(probability * 10), 9)] += weight * (probability - label)
    return {"roc_auc": auc, "balanced_accuracy": (recall + 1 - fpr) / 2,
            "recall": recall, "false_positive_rate": fpr,
            "ece": sum(abs(error) for error in errors) / (2 * n)}


def test_vectorized_statistics_match_scalar_ties_thresholds_and_ece():
    pairs = np.array([[0.2, 0.2], [0.1, 0.9], [0.5, 0.5], [1.0, 0.4]])
    weights = np.array([[1, 1, 1, 1], [0, 2, 1, 3], [2, 2, 0, 0]], dtype=float)
    actual = m.measured(pairs, weights, 0.5)
    for i, draw in enumerate(weights):
        for key, value in scalar(pairs, draw, 0.5).items():
            assert actual[key][i] == pytest.approx(value)


def test_speaker_clusters_are_drawn_together_with_matched_reference():
    groups = ["speaker-a", "speaker-a", "speaker-b", "speaker-b"]
    pairs = np.array([[0.1, 0.9], [0.8, 0.8], [0.3, 0.4], [0.5, 0.7]])
    draws = m.bootstrap_weights(groups)
    np.testing.assert_array_equal(draws[:, 0], draws[:, 1])
    np.testing.assert_array_equal(draws[:, 2], draws[:, 3])
    result = m.cell_statistics(pairs, groups, 0.5, baseline=pairs, baseline_threshold=0.5)
    assert result["independent_groups"] == 2
    assert result["bootstrap_draws"] == 2000
    assert result["uncertainty"].startswith("underpowered")
    assert all(value == 0 for value in result["paired_delta"].values())
    assert all(value == [0, 0] for value in result["paired_delta_ci95"].values())
    assert result["confusion"] == {"tp": 3, "tn": 2, "fp": 2, "fn": 1}
    for key, value in scalar(pairs, [1, 1, 1, 1], 0.5).items():
        assert result["metrics"][key] == pytest.approx(value)


def test_interval_draws_match_independent_cluster_replication():
    pairs = np.array([[0.1, 0.7], [0.2, 0.3], [0.8, 0.9]])
    groups = ["recording-a", "recording-a", "recording-b"]
    result = m.cell_statistics(pairs, groups, 0.5)
    rng = np.random.default_rng(20261018)
    values = {key: [] for key in m.METRICS}
    for draw in rng.integers(2, size=(2000, 2)):
        weights = [sum(draw == 0), sum(draw == 0), sum(draw == 1)]
        for key, value in scalar(pairs, weights, 0.5).items():
            values[key].append(value)
    for key in m.METRICS:
        np.testing.assert_allclose(
            result["metric_ci95"][key], np.quantile(values[key], [0.025, 0.975]))


def test_complete_same_format_pairing():
    records = rows(4, split="validation")
    scores = np.array([0.1, 0.9] * 4)
    cells = m.pair_cells(records, scores)
    assert len(cells) == 1 and cells[0][0]["format"] == "PNG"
    np.testing.assert_array_equal(cells[0][1], [[0.1, 0.9]] * 4)


@pytest.mark.parametrize("fault", ["shape", "nan", "range", "cover", "duplicate", "missing",
                                  "group", "no_cells"])
def test_pairing_rejects_ambiguous_or_incomplete_cells(fault):
    records = rows(4, source="FSDD-v1.0.10", split="validation")
    scores = np.array([0.1, 0.9] * 4)
    if fault == "shape":
        scores = scores[:-1]
    elif fault == "nan":
        scores[0] = np.nan
    elif fault == "range":
        scores[0] = 2
    elif fault == "cover":
        records[-2] = copy.deepcopy(records[0])
    elif fault == "duplicate":
        records[-1] = copy.deepcopy(records[1])
    elif fault == "missing":
        records[-1]["method"] = "scattered"
    elif fault == "group":
        records[1]["speaker"] = "other"
    else:
        records = [r for r in records if r["label"] == "cover"]
        scores = scores[::2]
    with pytest.raises(ValueError):
        m.pair_cells(records, scores)


@pytest.mark.parametrize("fault", ["shape", "small", "groups", "range", "nan", "threshold",
                                  "baseline_shape", "baseline_range", "baseline_threshold"])
def test_metric_limits_and_reference_contract(fault):
    pairs = np.array([[0.1, 0.9], [0.3, 0.6]])
    groups = ["a", "b"]
    threshold, prior_threshold = 0.5, 0.5
    baseline = pairs.copy()
    if fault == "shape":
        pairs = pairs.ravel()
    elif fault == "small":
        pairs = pairs[:1]
    elif fault == "groups":
        groups = ["a", "a"]
    elif fault == "range":
        pairs[0, 0] = -1
    elif fault == "nan":
        pairs[0, 0] = np.nan
    elif fault == "threshold":
        threshold = 2
    elif fault == "baseline_shape":
        baseline = baseline[:1]
    elif fault == "baseline_range":
        baseline[0, 0] = 2
    else:
        prior_threshold = float("nan")
    with pytest.raises(ValueError):
        m.cell_statistics(pairs, groups, threshold, baseline=baseline,
                          baseline_threshold=prior_threshold)
