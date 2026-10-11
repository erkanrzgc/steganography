"""Independent fixture recipes, bounded failures and measured recovery regressions."""

import hashlib
import json
import runpy
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.ctf_evaluation import evaluate_suite, load_suite
from steganography.benchmarking import ctf_matrix


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_independent_native_matrix_is_reproducible_and_recovers_exact_bytes(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    document = ctf_matrix.generate_matrix(first)
    assert ctf_matrix.generate_matrix(second) == document
    with zipfile.ZipFile(first / "case-05.zip") as archive:
        assert archive.infolist()[0].date_time == (1980, 1, 1, 0, 0, 0)
    path = first / "manifest.json"
    assert load_suite(path, digest(path)) == document
    result = evaluate_suite(path, digest(path), tmp_path / "jobs", hardware="unit-cpu")
    assert result["case_count"] == 17
    assert result["completed_exact_rate"] == 1
    assert result["source_groups"] == 1 and result["blind_gate"] == "unavailable"
    assert result["by_method"]["wav-lsb-channel"]["completed_exact_count"] == 2
    assert result["by_method"]["bmp-appended"]["completed_exact_count"] == 1
    assert all(row["status"] == "completed" for row in result["cases"])
    with pytest.raises(FileExistsError):
        ctf_matrix.generate_matrix(first)
    link = tmp_path / "link"
    link.symlink_to(first, target_is_directory=True)
    with pytest.raises(ValueError):
        ctf_matrix.generate_matrix(link / "child")


@pytest.mark.parametrize("fault", [None, "failed", "missing_output", "symlink"])
def test_external_recipe_orchestration_does_not_fake_missing_tools_or_outputs(
    tmp_path, monkeypatch, fault
):
    root = tmp_path / "external"
    calls = []

    class Runner:
        def __init__(self, **kwargs):
            assert kwargs == {"timeout": 30}

        def run(self, tool, args, **kwargs):
            assert kwargs["cwd"] == root
            calls.append((tool, args))
            target = root / (args[-1] if tool == "outguess" else args[args.index("-sf") + 1])
            if fault == "symlink":
                target.symlink_to(root / "message.bin")
            elif fault != "missing_output":
                target.write_bytes(b"fixture orchestration only:"
                                   + (root / "message.bin").read_bytes())
            return SimpleNamespace(status="failed" if fault == "failed" else "completed")

    monkeypatch.setattr(ctf_matrix, "ToolRunner", Runner)
    if fault:
        with pytest.raises(RuntimeError):
            ctf_matrix.generate_matrix(root, external=True)
        assert not (root / "manifest.json").exists()
    else:
        value = ctf_matrix.generate_matrix(root, external=True)
        assert len(value["cases"]) == 4 and value["mode"] == "balanced"
        assert [tool for tool, _ in calls] == ["steghide", "steghide", "outguess", "openstego"]
        assert all(case["required_tools"] for case in value["cases"])
        path = root / "manifest.json"
        assert load_suite(path, digest(path)) == value


def test_generator_cli_only_wires_explicit_inputs(tmp_path, monkeypatch, capsys):
    root = Path(__file__).resolve().parents[1]
    main = runpy.run_path(str(root / "scripts/generate-ctf-matrix.py"))["main"]
    calls = []

    def generate(path, **kwargs):
        calls.append((path, kwargs))
        return {"cases": [1, 2]}

    monkeypatch.setitem(main.__globals__, "generate_matrix", generate)
    monkeypatch.setattr(sys, "argv", ["generate", "--out", str(tmp_path), "--external"])
    main()
    assert calls == [(tmp_path, {"external": True})]
    assert "development" in capsys.readouterr().out


def test_published_before_after_evidence_is_paired_and_keeps_blind_unavailable():
    root = Path(__file__).resolve().parents[1]
    path = root / "benchmarks/ctf-matrix-20261011.json"
    evidence = json.loads(path.read_text())
    before, after = evidence["before"], evidence["after"]
    assert before["manifest_sha256"] == after["manifest_sha256"]
    assert before["limits"] == after["limits"] and before["mode"] == after["mode"]
    assert before["case_count"] == after["case_count"] == 17
    assert before["recovered_including_incomplete"] == 14
    assert after["recovered_including_incomplete"] == 17
    for old, new in zip(before["cases"], after["cases"], strict=True):
        for key in ("id", "input_sha256", "payload_sha256", "method_family"):
            assert old[key] == new[key]
        if old["completed_exact"]:
            assert new["completed_exact"]
    assert before["blind_gate"] == after["blind_gate"] == "unavailable"
