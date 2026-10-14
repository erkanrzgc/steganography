"""Independent paired arithmetic and adversarial contracts, not learning claims."""

import copy
import hashlib
import math

import pytest

from core.jpeg_score_diagnostics import ORIGINS, paired_scores


def fixture():
    samples = []
    for origin in ORIGINS:
        for index in range(2):
            lineage = hashlib.sha256(f"{origin}:{index}".encode()).hexdigest()
            samples.extend({"source_group": origin, "lineage": lineage,
                            "role": "development", "label": label,
                            "path": "/private/never-report.jpg"} for label in ("cover", "stego"))
    return samples, [.1, .2, .7, .8, .2, .2, .8, .6]


def test_independent_arithmetic_and_no_standalone_or_calibration_claims():
    samples, scores = fixture()
    original = copy.deepcopy((samples, scores))
    result = paired_scores(samples, scores)
    boss, bows = result["cells"]
    assert boss["higher_stego_pairs"] == 2 and boss["positive_pair_fraction"] == 1
    assert boss["mean_signed_pair_delta"] == pytest.approx(.1)
    assert boss["median_signed_pair_delta"] == pytest.approx(.1)
    assert boss["mean_cover_probability"] == pytest.approx(.4)
    assert boss["mean_stego_probability"] == pytest.approx(.5)
    assert boss["pair_midpoint_population_std"] == pytest.approx(.3)
    assert boss["midpoint_std_over_mean_absolute_delta"] == pytest.approx(3)
    assert bows["tied_pairs"] == bows["lower_stego_pairs"] == 1
    assert bows["higher_stego_pairs"] == 0
    assert bows["mean_signed_pair_delta"] == pytest.approx(-.1)
    assert bows["pair_midpoint_population_std"] == pytest.approx(.25)
    assert result["requires_known_cover"] and not result["standalone_accuracy_computed"]
    assert not result["threshold_selected"] and not result["scores_adjusted"]
    assert result["accuracy_qualification"] == "unavailable" and not result["deployed"]
    assert "/private/" not in str(result)
    assert (samples, scores) == original


def test_zero_deltas_have_null_ratio_not_infinite_evidence():
    samples, _ = fixture()
    value = paired_scores(samples, [.5] * 8)
    assert all(c["midpoint_std_over_mean_absolute_delta"] is None for c in value["cells"])
    assert all(c["tied_pairs"] == 2 for c in value["cells"])


@pytest.mark.parametrize("score", [True, None, "0.5", -1, 2, float("nan"), float("inf"), []])
def test_probabilities_are_bounded_numeric_not_labels_or_percentages(score):
    samples, scores = fixture()
    scores[0] = score
    with pytest.raises(ValueError, match="probabilities"):
        paired_scores(samples, scores)


@pytest.mark.parametrize("mutation", [
    lambda s: s.__setitem__(0, None),
    lambda s: s[0].update(lineage="bad"),
    lambda s: s[0].update(label="stego"),
    lambda s: s[1].update(label="cover"),
    lambda s: s[0].update(role="fit"),
    lambda s: s[1].update(role="fit"),
    lambda s: s[0].update(source_group="/private/source"),
    lambda s: s[1].update(source_group="BOWS2"),
    lambda s: s[1].update(lineage="e" * 64),
    lambda s: [r.update(lineage=s[0]["lineage"]) for r in s[2:4]],
    lambda s: [r.update(source_group=ORIGINS[0]) for r in s],
])
def test_invalid_ancestry_roles_order_and_sources_rejected(mutation):
    samples, scores = fixture()
    mutation(samples)
    with pytest.raises(ValueError):
        paired_scores(samples, scores)


@pytest.mark.parametrize("samples,scores", [(None, []), ([], None), ([], []),
                                           ([{}] * 8, [.5] * 7), ([{}] * 9, [.5] * 9),
                                           ([{}] * 1028, [.5] * 1028)])
def test_row_type_count_and_expansion_bounds(samples, scores):
    with pytest.raises(ValueError, match="bounds"):
        paired_scores(samples, scores)


def test_unbalanced_source_groups_rejected():
    samples, scores = fixture()
    for index in (4, 5):
        samples[index]["source_group"] = ORIGINS[0]
    with pytest.raises(ValueError, match="balanced"):
        paired_scores(samples, scores)


def test_standalone_random_order_can_be_poor_despite_all_positive_pairs():
    samples, _ = fixture()
    value = paired_scores(samples, [.01, .02, .98, .99] * 2)
    assert all(c["positive_pair_fraction"] == 1 for c in value["cells"])
    assert all(c["midpoint_std_over_mean_absolute_delta"] > 40 for c in value["cells"])
    # Independent four-comparison AUC: three successes /four cross-image pairs.
    covers, stegos = [.01, .98], [.02, .99]
    assert sum(s > c for s in stegos for c in covers) / 4 == .75
    assert not value["standalone_accuracy_computed"] and not value["deployed"]


def test_numerical_metrics_are_finite_even_at_probability_edges():
    samples, _ = fixture()
    result = paired_scores(samples, [0, 1, 1, 0] * 2)
    assert all(math.isfinite(v) for cell in result["cells"] for v in cell.values()
               if isinstance(v, (int, float)))


def test_subnormal_positive_delta_cannot_overflow_ratio():
    samples, _ = fixture()
    result = paired_scores(samples, [0, 1e-320, .99, .99] * 2)
    assert all(c["higher_stego_pairs"] == 1 for c in result["cells"])
    assert all(c["midpoint_std_over_mean_absolute_delta"] is None for c in result["cells"])
