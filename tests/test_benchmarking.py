import json
from pathlib import Path

import pytest

from steganography.benchmarking.corpus import generate_corpus, load_manifest
from steganography.benchmarking.metrics import classification_metrics
from steganography.benchmarking.runner import (
    run_benchmark,
    write_benchmark_html,
    write_benchmark_json,
)


def test_classification_metrics_and_validation():
    metrics = classification_metrics(
        [(True, 90), (True, 60), (False, 80), (False, 10)], threshold=70
    )
    assert metrics["confusion"] == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    assert metrics["precision"] == metrics["recall"] == 0.5
    assert metrics["specificity"] == metrics["accuracy"] == metrics["f1"] == 0.5
    assert metrics["roc_auc"] == 0.75
    assert metrics["average_precision"] == pytest.approx(0.833333, abs=1e-6)
    assert 0 <= metrics["recommended_threshold"] <= 100

    positive_only = classification_metrics([(True, 50)], threshold=50)
    assert positive_only["roc_auc"] is None
    assert positive_only["average_precision"] == 1.0
    with pytest.raises(ValueError, match="empty"):
        classification_metrics([], threshold=70)
    with pytest.raises(ValueError, match="between"):
        classification_metrics([(True, 1)], threshold=101)


def test_corpus_is_deterministic_and_tamper_evident(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    options = {
        "seed": 1234,
        "include_dct": False,
        "methods": {"image_lsb_png"},
        "densities": (("test", 0.10),),
    }
    first_manifest = generate_corpus(first, **options)
    second_manifest = generate_corpus(second, **options)
    assert first_manifest["sample_count"] == 2
    assert first_manifest["corpus_digest"] == second_manifest["corpus_digest"]
    for sample in first_manifest["samples"]:
        counterpart = second / sample["path"]
        assert (first / sample["path"]).read_bytes() == counterpart.read_bytes()

    with pytest.raises(FileExistsError):
        generate_corpus(first, **options)
    generate_corpus(first, force=True, **options)
    target = first / first_manifest["samples"][0]["path"]
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        load_manifest(first)


def test_corpus_refuses_unsafe_replacement_and_manifest_paths(tmp_path: Path):
    arbitrary = tmp_path / "arbitrary"
    arbitrary.mkdir()
    (arbitrary / "important.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing"):
        generate_corpus(arbitrary, force=True, methods={"image_lsb_png"})
    assert (arbitrary / "important.txt").read_text(encoding="utf-8") == "keep"

    corpus = tmp_path / "unsafe"
    manifest = generate_corpus(
        corpus,
        include_dct=False,
        methods={"image_lsb_png"},
        densities=(("test", 0.1),),
    )
    manifest["samples"][0]["path"] = "../escape.png"
    (corpus / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe sample path"):
        load_manifest(corpus, verify_files=False)


@pytest.fixture
def evaluation_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "evaluation"
    manifest = generate_corpus(
        corpus,
        seed=20260813,
        include_dct=False,
        densities=(("medium", 0.25),),
    )
    assert manifest["sample_count"] == 20
    return corpus


def test_benchmark_reports_groups_gates_and_baseline(
    evaluation_corpus: Path, tmp_path: Path
):
    report = run_benchmark(
        evaluation_corpus,
        jobs=2,
        min_recall=0.95,
        max_false_positive_rate=0.05,
    )
    assert report["gates"] == {"passed": True, "failures": []}
    assert report["corpus"]["sample_count"] == 20
    assert set(report["profiles"]) == {"sensitive", "balanced", "strict"}
    for profile in report["profiles"].values():
        assert profile["overall"]["recall"] == 1.0
        assert profile["overall"]["false_positive_rate"] == 0.0
        assert set(profile["by_density"]) == {"medium"}

    json_path = tmp_path / "nested" / "report.json"
    html_path = tmp_path / "nested" / "report.html"
    write_benchmark_json(report, json_path)
    write_benchmark_html(report, html_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["gates"]["passed"]
    assert "Gates: PASS" in html_path.read_text(encoding="utf-8")

    compared = run_benchmark(evaluation_corpus, baseline=json_path)
    assert compared["baseline_comparison"]["profiles"]["balanced"] == {
        "recall_delta": 0.0,
        "false_positive_rate_delta": 0.0,
        "roc_auc_delta": 0.0,
    }


def test_benchmark_failure_and_configuration_validation(
    evaluation_corpus: Path, tmp_path: Path
):
    failed = run_benchmark(
        evaluation_corpus,
        profiles=("balanced",),
        threshold=99,
        min_recall=1.0,
    )
    assert not failed["gates"]["passed"]
    assert "recall" in failed["gates"]["failures"][0]

    baseline = run_benchmark(evaluation_corpus)
    baseline["corpus"]["digest"] = "different"
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    with pytest.raises(ValueError, match="different corpus"):
        run_benchmark(evaluation_corpus, baseline=baseline_path)

    for arguments in (
        {"profiles": ()},
        {"profiles": ("unknown",)},
        {"profiles": ("strict", "strict")},
        {"threshold": -1},
        {"jobs": 0},
        {"min_recall": 1.1},
        {"max_false_positive_rate": -0.1},
    ):
        with pytest.raises(ValueError):
            run_benchmark(evaluation_corpus, **arguments)
