"""Frozen real-run audit rejects altered recipes and incomplete comparisons."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def audit_script():
    path = Path(__file__).resolve().parents[1] / "scripts/audit-srnet-multipair-real.py"
    spec = importlib.util.spec_from_file_location("srnet_multipair_real_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_contract_and_all_altered_fields(audit_script):
    plan = {"settings": {"epochs": 1, "seed": 20261012, "batch_recipe": audit_script.RECIPE}}
    card = {
        "schema_version": "srnet-fit-v2",
        "optimizer": copy.deepcopy(audit_script.FIXED_OPTIMIZER),
        "batch_size": 4,
        "training_scope": {"rows": 2985},
        "epoch_training": [{"updates": 1580}],
    }
    audit_script.check_frozen(plan, audit_script.PLAN_SHA, card)
    with pytest.raises(ValueError, match="contract"):
        audit_script.check_frozen(plan, "0" * 64, card)
    with pytest.raises(ValueError, match="contract"):
        audit_script.check_frozen({"settings": {"epochs": 2}}, audit_script.PLAN_SHA, card)
    for field, value in (
        ("schema_version", "srnet-fit-v1"),
        ("batch_size", 2),
        ("training_scope", {"rows": 618}),
        ("optimizer", {}),
        ("epoch_training", [{"updates": 3160}]),
    ):
        altered = {**card, field: value}
        with pytest.raises(ValueError, match="contract"):
            audit_script.check_frozen(plan, audit_script.PLAN_SHA, altered)


def test_comparison_has_complete_cells_and_no_qualification(audit_script):
    metrics = {
        "roc_auc": 0.5,
        "balanced_accuracy": 0.5,
        "recall": 0.24,
        "false_positive_rate": 0.24,
        "expected_calibration_error": 0.49,
    }
    cells = [
        {"source_id": "A", "quality_factor": q, "method": m, "metrics": dict(metrics)}
        for q in (None, 75, 95)
        for m in ("JUNIWARD", "UERD")
    ]
    changed = copy.deepcopy(cells)
    changed[0]["metrics"]["roc_auc"] += 0.1
    result = audit_script.comparison({"cells": changed}, {"cells": cells})
    assert len(result) == 6 and sum(c["point_change"]["roc_auc"] for c in result) == pytest.approx(
        0.1
    )
    assert not any("qualified" in c for c in result)
    with pytest.raises(ValueError, match="complete"):
        audit_script.comparison({"cells": cells[:-1]}, {"cells": cells})
    with pytest.raises(ValueError, match="complete"):
        audit_script.comparison({"cells": [*cells[:-1], cells[0]]}, {"cells": cells})
    changed[0]["source_id"] = "other"
    with pytest.raises(ValueError, match="complete"):
        audit_script.comparison({"cells": changed}, {"cells": cells})


def test_output_overwrite_and_symlink_rejected_before_evaluation(audit_script, tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        audit_script.audit(tmp_path, existing)
    link = tmp_path / "link"
    link.symlink_to(existing, target_is_directory=True)
    with pytest.raises(FileExistsError):
        audit_script.audit(tmp_path, link / "new")


@pytest.mark.parametrize("status", ["completed", "failed_numerical_gate"])
def test_audit_retains_failed_numerical_gates_and_local_paths_only(
    audit_script, tmp_path, monkeypatch, status
):
    root = Path(__file__).resolve().parents[1]
    job = tmp_path / "job"
    (job / "model").mkdir(parents=True)
    card = {
        "schema_version": "srnet-fit-v2",
        "optimizer": audit_script.FIXED_OPTIMIZER,
        "batch_size": 4,
        "training_scope": {"rows": 2985},
        "epoch_training": [{"updates": 1580}],
        "model_sha256": "a" * 64,
    }
    (job / "model/model-card.json").write_text(json.dumps(card))
    original_read = audit_script.read_document

    def read(path):
        if path.name == "plan.json":
            return {
                "settings": {"epochs": 1, "seed": 20261012, "batch_recipe": audit_script.RECIPE}
            }, audit_script.PLAN_SHA
        return original_read(path)

    monkeypatch.setattr(audit_script, "read_document", read)
    prior = json.loads((root / "benchmarks/srnet-real-pilot-20261006.json").read_bytes())
    diagnostics = {**prior["diagnostics"], "status": status}
    if status != "completed":
        diagnostics["cells"] = []
    captured = []

    def evaluate(config, out):
        captured.append(config)
        out.write_text(json.dumps({"status": status, "predictions": []}))
        return {"seconds": 1.0}

    def summarize(config, path, out, *, predictions_sha256):
        import hashlib

        assert predictions_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert config == captured[0]
        out.write_text(json.dumps(diagnostics))
        return diagnostics

    monkeypatch.setattr(audit_script, "evaluate", evaluate)
    monkeypatch.setattr(audit_script, "summarize", summarize)
    out = tmp_path / "evaluation"
    record = audit_script.audit(job, out)
    assert record["diagnostics"]["status"] == status
    assert len(record["comparison"]) == (6 if status == "completed" else 0)
    assert not record["deployed"] and record["qualification"] == "unavailable"
    assert len(record["execution_source_sha256"]) == 7
    assert captured[0]["validation_cache_sha256"] == audit_script.VALIDATION_CACHE_SHA
    assert captured[0]["plan_sha256"] == audit_script.PLAN_SHA
    assert "batch_recipe" not in captured[0]  # Recovered from the bound plan.
    assert str(tmp_path) not in json.dumps(record)
    assert str(tmp_path) in (out / "evaluation-config.json").read_text()


def test_entrypoint_failed_status_is_nonzero(audit_script, tmp_path, monkeypatch):
    monkeypatch.setattr(
        audit_script.sys, "argv", ["audit", "--job", str(tmp_path), "--out", str(tmp_path)]
    )
    monkeypatch.setattr(
        audit_script, "audit", lambda *a: {"diagnostics": {"status": "failed_numerical_gate"}}
    )
    assert audit_script.main() == 2
    monkeypatch.setattr(audit_script, "audit", lambda *a: {"diagnostics": {"status": "completed"}})
    assert audit_script.main() == 0
