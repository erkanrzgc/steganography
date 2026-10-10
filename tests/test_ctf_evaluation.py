"""Frozen recovery evidence: independent decoder fixtures and hostile manifests/artifacts."""

import base64
import copy
import hashlib
import io
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import ctf_evaluation as evaluation
from core.ctf_types import Artifact, Recommendation, ToolExecution


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def corpus(tmp_path):
    payload = b"flag{frozen_independent_base64_case}"
    source = tmp_path / "challenge.txt"
    source.write_bytes(base64.b64encode(payload))
    value = {
        "schema_version": "ctf-evaluation-manifest-v1", "suite_id": "unit-development",
        "mode": "quick", "limits": dict(max_depth=3, max_artifacts=256, max_bytes=1024**3,
                                        tool_timeout=30, job_timeout=180),
        "cases": [{"id": "case-01", "source_group": "independent-encoding",
                   "method_family": "base64", "input": source.name,
                   "input_sha256": sha(source.read_bytes()), "payload_sha256": sha(payload),
                   "required_tools": []}],
    }
    return tmp_path, value, payload


def manifest(corpus, value=None):
    root, original, _ = corpus
    path = root / "manifest.json"
    path.write_text(json.dumps(original if value is None else value))
    return path, sha(path.read_bytes())


def test_actual_independent_encoding_exact_recovery_is_not_blind(corpus):
    path, digest = manifest(corpus)
    root, value, _ = corpus
    output = root / "run"
    result = evaluation.evaluate_suite(path, digest, output, hardware="cpu-unit")
    assert result["completed_exact_rate"] == 1
    assert result["recovered_including_incomplete"] == 1
    assert result["blind_gate"] == "unavailable" and result["qualification"] == "development_only"
    assert result["timed_case_count"] == 1 and result["median_seconds"] >= 0
    assert result["limits"] == value["limits"]
    assert result["by_method"]["base64"]["completed_exact_rate"] == 1
    assert json.loads((output / "evaluation.json").read_text()) == result
    assert str(root) not in json.dumps(result)
    with pytest.raises(FileExistsError):
        evaluation.evaluate_suite(path, digest, output, hardware="cpu-unit")


@pytest.mark.parametrize("fault", ["schema", "id", "hash", "mode", "limits", "nan", "bool",
                                   "fraction", "budget", "empty", "many", "case", "duplicate",
                                   "self", "tools", "tool", "absolute", "parent", "backslash",
                                   "empty_path", "normalize", "source_hash", "wordlist_hash"])
def test_invalid_manifests_fail_before_output_or_solver(corpus, fault):
    root, original, _ = corpus
    value = copy.deepcopy(original)
    case = value["cases"][0]
    if fault == "schema":
        value = []
    elif fault == "id":
        case["id"] = "evaluation.json"
    elif fault == "hash":
        case["payload_sha256"] = "not-sha256"
    elif fault == "mode":
        value["mode"] = "fast"
    elif fault == "limits":
        value["limits"] = {}
    elif fault in {"nan", "bool", "fraction", "budget"}:
        key = "max_depth" if fault == "fraction" else "job_timeout"
        value["limits"][key] = {"nan": float("nan"), "bool": True, "fraction": 1.5,
                                "budget": 181}[fault]
    elif fault == "empty":
        value["cases"] = []
    elif fault == "many":
        value["cases"] = [case] * 121
    elif fault == "case":
        case["password"] = "not-allowed"  # noqa: S105 - rejected manifest field
    elif fault == "duplicate":
        value["cases"].append(case.copy())
    elif fault == "self":
        case["payload_sha256"] = case["input_sha256"]
    elif fault == "tools":
        case["required_tools"] = ["steghide", "steghide"]
    elif fault == "tool":
        case["required_tools"] = ["../unsafe"]
    elif fault == "source_hash":
        case["input_sha256"] = "0" * 64
    elif fault == "wordlist_hash":
        case.update(wordlist="challenge.txt", wordlist_sha256="0" * 64)
    else:
        case["input"] = {"absolute": "/etc/passwd", "parent": "../outside",
                         "backslash": "x\\y", "empty_path": "", "normalize": "./x"}[fault]
    path, digest = manifest(corpus, value)
    with pytest.raises(ValueError):
        evaluation.evaluate_suite(path, digest, root / "run", hardware="cpu-unit")
    assert not (root / "run").exists()


def test_file_bounds_digest_identity_and_symlink_inputs(corpus):
    root, _, _ = corpus
    path, digest = manifest(corpus)
    with pytest.raises(ValueError, match="checksum"):
        evaluation.load_suite(path, "0" * 64)
    with pytest.raises(ValueError):
        evaluation.load_suite(path, "invalid")
    with pytest.raises(ValueError):
        evaluation._digest(path, 1)
    with pytest.raises(ValueError):
        evaluation._digest(root, 65536)
    link = root / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        evaluation.load_suite(link, digest)
    with pytest.raises(ValueError):
        evaluation.evaluate_suite(path, digest, link / "run", hardware="cpu-unit")
    with pytest.raises(ValueError):
        evaluation.evaluate_suite(path, digest, root / "run", hardware="/private/host")


@pytest.mark.parametrize("fault", [None, "missing", "cancelled", "no_payload", "escape",
                                   "symlink", "identity", "exception", "duration", "mutate",
                                   "wordlist_mutate"])
def test_oracle_never_reaches_solver_and_all_cases_stay_in_denominator(corpus, monkeypatch, fault):
    root, value, payload = corpus
    wordlist = root / "wordlist.txt"
    wordlist.write_text("user-supplied-example\n")
    value["cases"][0].update(wordlist=wordlist.name, wordlist_sha256=sha(wordlist.read_bytes()),
                             required_tools=["steghide"])
    path, digest = manifest(corpus)
    calls = []

    def which(tool):
        if fault == "mutate":
            (root / "challenge.txt").write_bytes(b"mutated")
        if fault == "wordlist_mutate":
            wordlist.write_text("mutated")
        return None if fault == "missing" else "/unused"

    monkeypatch.setattr(evaluation.shutil, "which", which)

    class Solver:
        def solve(self, source, output, **kwargs):
            calls.append(kwargs)
            assert set(kwargs) == {"mode", "limits", "wordlist"}
            assert kwargs["wordlist"] == wordlist
            if fault == "exception":
                raise RuntimeError(str(root) + " secret")
            output.mkdir()
            recovered = output / "payload.bin"
            recovered.write_bytes(payload)
            if fault == "escape":
                recovered = root / "challenge.txt"
            if fault == "symlink":
                recovered.unlink()
                recovered.symlink_to(source)
            item = Artifact("a", "payload.bin", "application/octet-stream", len(payload),
                            "0" * 64 if fault == "identity" else sha(payload), 1,
                            "independent-extraction", parent_id="root", path=recovered)
            # A matching root must never count as extraction.
            root_artifact = Artifact("root", "input", "text/plain", len(payload), sha(payload),
                                     0, "input", path=source)
            return SimpleNamespace(
                status="cancelled" if fault == "cancelled" else "completed",
                duration_ms=float("nan") if fault == "duration" else 1000,
                artifacts=[root_artifact] if fault == "no_payload" else [root_artifact, item],
                recommendations=[Recommendation(1, "decode", "fixture", "base64")],
                tools=[ToolExecution("steghide", "test", "completed", 1, 0, (), "secret")],
            )

    result = evaluation.evaluate_suite(path, digest, root / "run", hardware="cpu-unit",
                                       service=Solver())
    row = result["cases"][0]
    success = fault is None
    assert result["case_count"] == 1
    assert result["completed_exact_rate"] == int(success)
    assert result["recovered_including_incomplete"] == int(fault in {None, "cancelled"})
    assert "secret" not in json.dumps(result) and str(root) not in json.dumps(result)
    if fault == "missing":
        assert not calls and row["status"] == "unavailable"
        assert result["median_seconds"] is None and result["p95_seconds"] is None
        assert result["by_method"]["base64"]["unavailable_count"] == 1
    elif success:
        assert row["method_top3"]


def test_manifest_and_file_growth_are_detected(corpus, monkeypatch):
    path, digest = manifest(corpus)
    original_open = Path.open

    def changed(self, *args, **kwargs):
        if self == path:
            return io.BytesIO(b"x" * 65537)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", changed)
    with pytest.raises(ValueError, match="grew"):
        evaluation._digest(path, 65536)
    monkeypatch.setattr(evaluation, "_digest", lambda *a: digest)
    with pytest.raises(ValueError, match="changed"):
        evaluation.load_suite(path, digest)


def test_identity_and_tool_types_rejected(corpus):
    _, value, _ = corpus
    for invalid in (123, [], "--unsafe"):
        value["cases"][0]["source_group"] = invalid
        path, digest = manifest(corpus)
        with pytest.raises(ValueError):
            evaluation.load_suite(path, digest)


@pytest.mark.parametrize("status", ["completed", "unavailable", "failed", "cancelled"])
def test_cli_exit_is_completion_not_recovery_or_blind_gate(tmp_path, monkeypatch, capsys, status):
    code = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/evaluate-ctf.py"))
    main = code["main"]
    result = dict(case_count=1, completed_exact_rate=0, blind_gate="unavailable",
                  qualification="development_only", cases=[dict(status=status)])
    monkeypatch.setitem(main.__globals__, "evaluate_suite", lambda *a, **k: result)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--manifest", "manifest.json",
                                    "--manifest-sha256", "0" * 64, "--out", str(tmp_path),
                                    "--hardware", "unit-cpu"])
    assert main() == (0 if status == "completed" else 1)
    assert json.loads(capsys.readouterr().out)["completed_exact_rate"] == 0


def test_checked_in_manifest_and_two_independent_decoder_examples(tmp_path):
    source = Path(__file__).resolve().parents[1] / "examples/ctf-evaluation/manifest.json"
    result = evaluation.evaluate_suite(source, sha(source.read_bytes()), tmp_path / "run",
                                       hardware="unit-cpu")
    assert result["case_count"] == 2 and result["source_groups"] == 1
    assert result["completed_exact_rate"] == 1 and result["blind_gate"] == "unavailable"
    assert set(result["by_method"]) == {"base64", "base16"}
