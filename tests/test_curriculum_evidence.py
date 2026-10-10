"""Portable numerical evidence replay; no score/support gate is inferred."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pytest

from steganography.research_jpeg_curriculum import sources

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    raw = (ROOT / "benchmarks" / f"jpeg-q75-curriculum-{name}-20261010.json").read_bytes()
    assert b"/home/" not in raw and b"/tmp/" not in raw
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def test_physical_cache_has_all_fit_originals_and_independent_idct():
    high, checksum = read("high-audit")
    assert checksum == "8e551b93a69eab1e4499a6863a8dac8bf35f5e55f4a4f5cc167b6fc42e9d3c16"
    assert high["source_sha256"] == sources()
    assert high["status"] == "completed" and high["independent_idct_rows"] == 96
    assert high["payload_rate"] == 0.4 and high["payload_unit"] == "bpnzAC"
    assert high["probe_pixels_read"] == 0 and high["deployed"] is False
    assert high["independent_native_parser"] is False
    assert Counter(r["source_group"] for r in high["rows"]) == {"BOSSbase-1.01": 24, "BOWS2": 24}
    assert len({r["lineage"] for r in high["rows"]}) == 48
    assert all(
        r["coefficient_changes"] > 0 and r["idct_max_difference"] <= 1e-4 for r in high["rows"]
    )


def test_physical_training_accounting_and_retained_contexts():
    report, checksum = read("fit")
    audit, _ = read("independent-metrics")
    high, high_sha = read("high-audit")
    assert checksum == audit["report_sha256"]
    assert report["high_manifest_sha256"] == high_sha
    assert report["source_sha256"] == sources() == high["source_sha256"]
    assert report["status"] == "completed" and report["optimizer_updates"] == 7680
    assert report["fit_originals"] == 48 and report["fit_rows"] == [
        i for r in high["rows"] for i in r["indices"]
    ]
    assert report["probe_originals"] == 24 and report["probe_rows"] == 120
    assert report["original_groups_disjoint"] is True
    assert report["probe_previously_inspected"] is True and report["validation_test_used"] is False
    assert report["model_discarded"] is True and report["deployed"] is False
    assert report["accuracy_qualification"] == "unavailable"
    assert (
        report["paired_d4_augmentation"] is True
        and report["singleton_batch_parity"]["passed"] is True
    )
    assert (
        report["execution"]["device"] == "cuda:0" and report["execution"]["cpu_fallback"] is False
    )
    assert report["execution"]["tf32"] is False and report["execution"]["mixed_precision"] is False
    assert 0 < report["seconds"] < 2400
    assert report["process_peak_gpu_reserved_bytes"] <= 4 * 1024**3
    first, second = report["stages"]
    assert first["name"] == "warmup-0.4" and second["name"] == "adapt-0.2"
    assert first["learning_rate"] == 0.001 and second["learning_rate"] == 0.0001
    assert first["starting_parameter_sha256"] == report["initial_parameter_sha256"]
    assert second["starting_parameter_sha256"] == first["learned_parameter_sha256"]
    for stage in (first, second):
        assert stage["optimizer_reset"] is True and len(stage["epoch_losses"]) == 40
        assert [r["epoch"] for r in stage["epoch_losses"]] == list(range(40))
        assert all(
            r["updates"] == 96 and math.isfinite(r["mean_pair_loss"]) for r in stage["epoch_losses"]
        )
        assert stage["starting_parameter_sha256"] != stage["learned_parameter_sha256"]
    assert len(report["probe_cells"]) == 10
    assert audit["independent_whole_model_oracle"] is False and audit["deployed"] is False


def test_independent_scalar_and_rank_auc_replay_every_cell():
    report, _ = read("fit")
    audit, _ = read("independent-metrics")
    metadata = {r["row"]: r for r in audit["probe_metadata"]}
    logits = {r["row"]: r["logits"] for r in report["probe_logits"]}
    assert len(metadata) == len(logits) == 120 and set(metadata) == set(logits)
    assert set(logits).isdisjoint(report["fit_rows"])
    assert len({r["lineage"] for r in metadata.values()}) == 24
    high, _ = read("high-audit")
    assert {r["lineage"] for r in metadata.values()}.isdisjoint(
        {r["lineage"] for r in high["rows"]}
    )
    seen = set()
    for cell in report["probe_cells"]:
        key = (cell["source_group"], cell["quality_factor"], cell["method"])
        assert key not in seen
        seen.add(key)
        counts = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0}
        losses = []
        scores = []
        for index, row in metadata.items():
            if (row["source_group"], row["quality_factor"]) != key[:2] or row["method"] not in (
                None,
                key[2],
            ):
                continue
            values = logits[index]
            assert len(values) == 2 and all(math.isfinite(v) for v in values)
            positive = row["label"] == "stego"
            decision = values[1] >= values[0]
            name = ("true_" if decision == positive else "false_") + (
                "positive" if decision else "negative"
            )
            counts[name] += 1
            top = max(values)
            losses.append(
                top + math.log(sum(math.exp(v - top) for v in values)) - values[int(positive)]
            )
            scores.append((values[1] - values[0], positive))
        assert len(scores) == 16 and sum(label for _, label in scores) == 8
        ordered = sorted(scores)
        rank_sum = 0.0
        i = 0
        while i < len(ordered):
            j = i + 1
            while j < len(ordered) and ordered[j][0] == ordered[i][0]:
                j += 1
            rank_sum += (i + 1 + j) / 2 * sum(label for _, label in ordered[i:j])
            i = j
        expected = {
            **counts,
            "balanced_accuracy": (counts["true_positive"] + counts["true_negative"]) / 16,
            "recall": counts["true_positive"] / 8,
            "false_positive_rate": counts["false_positive"] / 8,
            "cross_entropy": sum(losses) / 16,
            "roc_auc": (rank_sum - 36) / 64,
        }
        for name, value in expected.items():
            assert cell[name] == pytest.approx(value, abs=1e-10)
        assert audit["all_cells_scalar_and_rank_metrics_passed"] is True
    assert (
        audit["auditor_source_sha256"]["tests/test_curriculum_evidence.py"]
        == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )
