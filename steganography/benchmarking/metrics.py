"""Dependency-free binary classification metrics for steganalysis scores."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def classification_metrics(
    observations: Iterable[tuple[bool, float]], *, threshold: int, recommend_threshold: bool = True
) -> dict[str, Any]:
    values = list(observations)
    if not values:
        raise ValueError("cannot calculate metrics for an empty observation set")
    if not 0 <= threshold <= 100:
        raise ValueError("threshold must be between 0 and 100")
    tp = sum(label and score >= threshold for label, score in values)
    tn = sum(not label and score < threshold for label, score in values)
    fp = sum(not label and score >= threshold for label, score in values)
    fn = sum(label and score < threshold for label, score in values)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    fpr = _ratio(fp, fp + tn)
    accuracy = _ratio(tp + tn, len(values))
    balanced_accuracy = (recall + specificity) / 2
    f1 = _ratio(2 * precision * recall, precision + recall)
    return {
        "threshold": threshold,
        "samples": len(values),
        "positives": tp + fn,
        "negatives": tn + fp,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "precision": _rounded(precision),
        "recall": _rounded(recall),
        "specificity": _rounded(specificity),
        "false_positive_rate": _rounded(fpr),
        "accuracy": _rounded(accuracy),
        "balanced_accuracy": _rounded(balanced_accuracy),
        "f1": _rounded(f1),
        "roc_auc": _optional_rounded(_roc_auc(values)),
        "average_precision": _optional_rounded(_average_precision(values)),
        "recommended_threshold": _recommended_threshold(values) if recommend_threshold else None,
    }


def _roc_auc(values: list[tuple[bool, float]]) -> float | None:
    positive_count = sum(label for label, _score in values)
    negative_count = len(values) - positive_count
    if not positive_count or not negative_count:
        return None
    ranked = sorted(values, key=lambda item: item[1])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][1] == ranked[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        positive_rank_sum += average_rank * sum(label for label, _score in ranked[index:end])
        index = end
    wins = positive_rank_sum - positive_count * (positive_count + 1) / 2
    return wins / (positive_count * negative_count)


def _average_precision(values: list[tuple[bool, float]]) -> float | None:
    positive_count = sum(label for label, _ in values)
    if positive_count == 0:
        return None
    ranked = sorted(values, key=lambda item: item[1], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, (label, _) in enumerate(ranked, start=1):
        if label:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positive_count


def _recommended_threshold(values: list[tuple[bool, float]]) -> int:
    best: tuple[float, float, int] | None = None
    best_threshold = 70
    for threshold in range(101):
        tp = sum(label and score >= threshold for label, score in values)
        tn = sum(not label and score < threshold for label, score in values)
        fp = sum(not label and score >= threshold for label, score in values)
        fn = sum(label and score < threshold for label, score in values)
        recall = _ratio(tp, tp + fn)
        specificity = _ratio(tn, tn + fp)
        precision = _ratio(tp, tp + fp)
        f1 = _ratio(2 * precision * recall, precision + recall)
        # Prefer Youden's J, then F1, then the stricter threshold.
        candidate = (recall + specificity - 1.0, f1, threshold)
        if best is None or candidate > best:
            best = candidate
            best_threshold = threshold
    return best_threshold


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _rounded(value: float) -> float:
    return round(value, 6)


def _optional_rounded(value: float | None) -> float | None:
    return None if value is None else _rounded(value)
