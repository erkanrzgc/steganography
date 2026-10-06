"""Generated context diagnostics; perfect tiny fixtures cannot qualify support."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from core import srnet
from steganography import research_srnet_evaluate as evaluator
from steganography import research_srnet_results as service
from tests.test_jrm_reference import corpus as corpus
from tests.test_jrm_reference import sha
from tests.test_srnet_evaluate import config as evaluation_config  # noqa: F401
from tests.test_srnet_sampling import config as plan_config  # noqa: F401
from tests.test_srnet_training import config as fit_config  # noqa: F401

pytest.importorskip("torch")


@pytest.fixture
def inputs(evaluation_config, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(srnet, "float_logits", lambda model, values: np.zeros((len(values), 2)))
    path = tmp_path / "predictions.json"
    record = evaluator.evaluate(evaluation_config, path)
    return evaluation_config, path, record


def run(inputs, tmp_path):
    config, path, _ = inputs
    return service.summarize(config, path, tmp_path / "results.json", predictions_sha256=sha(path))


def test_complete_cells_bootstrap_no_tuning_and_no_false_qualification(inputs, tmp_path):
    result = run(inputs, tmp_path)
    assert len(result["cells"]) == 4 and result["validation_rows"] == 6
    assert result["status"] == "completed"
    for cell in result["cells"]:
        assert cell["metrics"]["roc_auc"] == 0.5
        assert cell["metrics"]["balanced_accuracy"] == 0.5
        assert cell["metrics"]["recall"] == cell["metrics"]["false_positive_rate"] == 1
        assert cell["independent_lineages"] == 1
        assert cell["bootstrap_95_percent"]["roc_auc"] == [0.5, 0.5]
        assert (
            not cell["metric_gates_passed"]
            and not cell["sample_size_gate"]
            and not cell["qualified"]
        )
    assert not result["qualified"] and not result["deployed"] and not result["calibrated"]
    assert str(tmp_path) not in json.dumps(result)
    config, path, _ = inputs
    with pytest.raises(FileExistsError):
        service.summarize(config, path, tmp_path / "results.json", predictions_sha256=sha(path))
    link = tmp_path / "link"
    link.symlink_to(tmp_path)
    with pytest.raises(FileExistsError):
        service.summarize(config, path, link / "new.json", predictions_sha256=sha(path))


def test_perfect_tiny_fixture_is_not_supported(inputs, tmp_path):
    _, path, report = inputs
    for row in report["predictions"]:
        row["score"] = float(row["label"] == "stego")
    path.write_text(json.dumps(report))
    result = run(inputs, tmp_path)
    assert all(c["metric_gates_passed"] for c in result["cells"])
    assert all(not c["qualified"] and not c["sample_size_gate"] for c in result["cells"])


@pytest.mark.parametrize(
    "edit",
    [
        {"manifest_sha256": "0" * 64},
        {"deployed": 0},
        {"threshold": 0.6},
        {"validation_data_sha256": "0" * 64},
        {"status": "failed_numerical_gate"},
        {"predictions": []},
        {"independent_forward_audit": []},
        {"independent_forward_audit": [None] * 6},
    ],
)
def test_rehashed_false_provenance_rejected(inputs, tmp_path, edit):
    _, path, report = inputs
    path.write_text(json.dumps({**report, **edit}))
    with pytest.raises(ValueError):
        run(inputs, tmp_path)
    assert not (tmp_path / "results.json").exists()


@pytest.mark.parametrize(
    "edit",
    [
        {"sha256": "0" * 64},
        {"score": float("nan")},
        {"score": True},
        {"score": 1.1},
        {"lineage": "wrong"},
        {"source_id": "wrong"},
        {"quality_factor": 95},
    ],
)
def test_row_identity_finite_probability_and_context(inputs, tmp_path, edit):
    _, path, report = inputs
    report["predictions"][0].update(edit)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        run(inputs, tmp_path)


@pytest.mark.parametrize(
    "edit",
    [
        {"passed": 1},
        {"decisions_equal": False},
        {"absolute_tolerance": 1},
        {"maximum_score_difference": 1},
        {"maximum_logit_difference": float("nan")},
    ],
)
def test_audit_contract_is_not_relaxed(inputs, tmp_path, edit):
    _, path, report = inputs
    report["independent_forward_audit"][0].update(edit)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        run(inputs, tmp_path)


def test_failed_numerical_gate_is_retained_without_metrics(inputs, tmp_path):
    _, path, report = inputs
    report["status"] = "failed_numerical_gate"
    report["independent_forward_audit"][0].update(passed=False, decisions_equal=False)
    original = copy.deepcopy(report)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="cannot publish"):
        run(inputs, tmp_path)
    report["predictions"] = []
    path.write_text(json.dumps(report))
    result = run(inputs, tmp_path)
    assert result["cells"] == [] and result["status"] == "failed_numerical_gate"
    assert result["independent_forward_audit"] == original["independent_forward_audit"]


def test_checksum_and_cache_card_failure(inputs, tmp_path):
    config, path, _ = inputs
    with pytest.raises(ValueError, match="provenance"):
        service.summarize(config, path, tmp_path / "results.json", predictions_sha256="0" * 64)
    with pytest.raises(ValueError, match="card checksum"):
        service.summarize(
            {**config, "card_sha256": "0" * 64},
            path,
            tmp_path / "results.json",
            predictions_sha256=sha(path),
        )
    cache = Path(config["validation_cache"])
    descriptor = json.loads(cache.read_bytes())
    descriptor["decoder"] = {}
    cache.write_text(json.dumps(descriptor))
    with pytest.raises(ValueError, match="cache mismatch"):
        run(inputs, tmp_path)


def test_unmatched_cover_lineage_cannot_generate_a_cell(inputs, tmp_path, monkeypatch):
    _, path, report = inputs
    original = service.selected_samples

    def unmatched(manifest, split):
        rows = copy.deepcopy(original(manifest, split))
        rows[0]["lineage"] = "unmatched"
        return rows

    monkeypatch.setattr(service, "selected_samples", unmatched)
    report["predictions"][0]["lineage"] = "unmatched"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="unmatched validation cell"):
        run(inputs, tmp_path)


def test_pilot_script_help_runs_from_checkout():
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts/audit-srnet-real-pilot.py"
    result = subprocess.run(  # noqa: S603 — fixed trusted checkout script
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0 and "--job" in result.stdout and "--out" in result.stdout
