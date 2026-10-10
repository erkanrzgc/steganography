"""CI must test optional research without weakening the complete coverage gate."""

import ast
from pathlib import Path

import pytest

from tests.evidence_assertions import assert_metrics_replay

ROOT = Path(__file__).resolve().parents[1]


def test_torch_checks_skip_explicitly_in_lightweight_environment():
    # Raw imports make the lightweight matrix fail before tests can run.
    for path in (ROOT / "tests").glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name != "torch" for alias in node.names), path.name


def test_complete_research_job_retains_coverage_and_dependency_preflight():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    light = workflow.split("  test:\n", 1)[1].split("  research:\n", 1)[0]
    complete = workflow.split("  research:\n", 1)[1].split("  package:\n", 1)[0]
    for version in ("3.11", "3.12", "3.13", "3.14"):
        assert f'"{version}"' in light
    assert "pytest --no-cov" in light
    assert "requirements-research-ci.txt" in complete
    assert "import torch, onnx, onnxruntime, conseal, sealwatch" in complete
    assert "run: pytest\n" in complete
    assert "--no-cov" not in complete
    assert "continue-on-error" not in complete
    assert "--cov-fail-under" not in complete


def test_release_provisions_same_optional_stack_before_verification():
    workflow = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert workflow.index("requirements-research-ci.txt") < workflow.index("- run: mypy")
    assert "- run: pytest\n" in workflow


def test_evidence_replay_accepts_roundoff_not_metric_or_decision_changes():
    expected = {"scores": [0.4, 0.6], "cross_entropy": 0.7, "recall": 1.0}
    assert_metrics_replay({**expected, "scores": [0.4 + 1e-16, 0.6]}, expected)
    for changes in (
        {"scores": [0.4001, 0.6]},
        {"cross_entropy": 0.7001},
        {"recall": 0.0},
        {"scores": [float("nan"), 0.6]},
        {"scores": [0.4]},
    ):
        with pytest.raises(AssertionError):
            assert_metrics_replay({**expected, **changes}, expected)
