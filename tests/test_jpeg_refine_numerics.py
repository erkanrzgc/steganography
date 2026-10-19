"""Bounded revision dispatch, portable failure codes and cumulative accounting."""

import copy
import io
import json
from contextlib import nullcontext

import pytest

from steganography import research_jpeg_refine_numerics as f


def test_explicit_prior_debit_changes_start_only():
    values = iter([1000.0, 1001.0, 1002.0])
    clock = f.ChargedClock(604.309597279, lambda: next(values))
    start = clock()
    assert clock() - start == pytest.approx(605.309597279)
    assert clock() - start == pytest.approx(606.309597279)


@pytest.fixture
def worker(monkeypatch, tmp_path):
    monkeypatch.setattr(f.resource, "setrlimit", lambda *args: None)
    monkeypatch.setattr(f, "sources", lambda: {"fixture": "a" * 64})
    monkeypatch.setattr(f.old.service, "configuration", lambda *args: args[0])
    monkeypatch.setattr(f.old, "inspect", lambda *args: {"device": "cuda:0"})
    monkeypatch.setattr(f.old.srnet_cuda, "policy", lambda: nullcontext())
    monkeypatch.setattr(f.preflight, "probe", lambda: {"status": "completed"})
    monkeypatch.setattr(f.old.jpeg_study_audit, "audit", lambda *a, **k: {"status": "completed"})
    monkeypatch.setattr(f.old.training, "train", lambda *a, **k: {"status": "checkpointed"})
    monkeypatch.setattr(f.old.training, "evaluate", lambda *a, **k: {"status": "completed"})
    return {"config": {"expected_gpu": {}, "root": tmp_path, "manifest_sha256": "a" * 64},
            "parent": {"path": "snapshot.npz"}, "task": ["warmup", 20261012, 512],
            "binding": "a" * 64, "execution": {},
            "checkpoint": {"path": "snapshot.npz"}}


@pytest.mark.parametrize("mode", ["audit", "preflight", "train", "evaluate"])
def test_worker_modes_bind_unchanged_sources(worker, tmp_path, mode):
    result = f.worker(mode, worker, tmp_path)
    assert result["status"] in {"completed", "checkpointed"}
    assert result["source_sha256"] == {"fixture": "a" * 64}


@pytest.mark.parametrize("error,reason", [
    (ValueError("generated CPU/CUDA gradient parity failed"), "cpu_cuda_gradient_parity_failed"),
    (ValueError("/private/password=secret"), "worker_contract_failed"),
    (RuntimeError("/private/password=secret"), "worker_runtime_failed"),
    (OSError("/private/password=secret"), "worker_os_failed"),
    (ImportError(), "optional_dependency_unavailable"),
    (TimeoutError(), "worker_interrupted_or_timeout"),
    (f.old.srnet_cuda.CUDAUnavailable("bounded_cgroup_v2_required"), "bounded_cgroup_v2_required"),
])
def test_failure_codes_never_echo_exception_text(worker, monkeypatch, tmp_path, error, reason):
    def fail():
        raise error

    monkeypatch.setattr(f.preflight, "probe", fail)
    result = f.worker("preflight", worker, tmp_path)
    assert result == {"status": "unavailable", "reason": reason}
    assert "private" not in json.dumps(result) and "secret" not in json.dumps(result)


def test_real_oom_code(worker, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")

    def fail():
        raise torch.cuda.OutOfMemoryError()

    monkeypatch.setattr(f.preflight, "probe", fail)
    assert f.worker("preflight", worker, tmp_path)["reason"] == "gpu_allocator_out_of_memory"


def test_mutation_and_unknown_mode_fail_closed(worker, monkeypatch, tmp_path):
    assert f.worker("invalid", worker, tmp_path)["reason"] == "worker_contract_failed"
    sequence = iter([{}, {"changed": "b" * 64}])
    monkeypatch.setattr(f, "sources", lambda: next(sequence))
    assert f.worker("preflight", worker, tmp_path)["reason"] == "worker_contract_failed"


def test_source_bounds_and_manifest_binding(tmp_path, monkeypatch):
    assert len(f.sources()) == 35
    monkeypatch.setattr(f, "ROOT", tmp_path)
    monkeypatch.setattr(f, "NAMES", ("large.py",))
    (tmp_path / "large.py").write_bytes(b"x" * (1024**2 + 1))
    with pytest.raises(ValueError, match="bound"):
        f.sources()


def test_launch_uses_protocol_json_even_for_unavailable(monkeypatch, tmp_path):
    def child(command, **kwargs):
        assert "steganography.research_jpeg_refine_numerics" in command
        assert json.loads(kwargs["payload"]) == {"path": str(tmp_path)}
        return b'{"status":"unavailable","reason":"cpu_cuda_gradient_parity_failed"}'

    monkeypatch.setattr(f.old, "child", child)
    assert f.launch("preflight", {"path": tmp_path}, tmp_path, 10, lambda: False) == {
        "status": "unavailable", "reason": "cpu_cuda_gradient_parity_failed", "exit_status": 0}


@pytest.mark.parametrize("fault", [None, "resume", "original", "revision", "seconds", "storage"])
def test_run_preserves_ancestry_accounting_and_storage_guard(tmp_path, monkeypatch, fault):
    amendment = json.loads((f.ROOT / f.AMENDMENT).read_bytes())
    before = f.sources()
    settings = {}
    if fault == "resume":
        settings["resume"] = {}
    if fault == "original":
        amendment["original_source_sha256"]["core/jpeg_refine.py"] = "0" * 64
    if fault == "revision":
        amendment["revision_source_sha256"]["core/jpeg_refine_preflight_v2.py"] = "0" * 64
    if fault == "seconds":
        amendment["accounting"]["prior_seconds_debit"] = 0
    def read(path):
        return (copy.deepcopy(settings), "a" * 64) if path == tmp_path / "config" else (
            amendment, "b" * 64)

    monkeypatch.setattr(f.old.service, "read", read)
    monkeypatch.setattr(f.old, "inspect", lambda expected: {})
    monkeypatch.setattr(f, "launch", lambda *a, **k: {"status": "completed"})

    def execute(config, out, **kwargs):
        assert kwargs["sources"] == before
        assert kwargs["current_sources"]() == before
        assert kwargs["clock"].debit == 604.309597279
        out.mkdir()
        kwargs["inspect"]({})
        if fault == "storage":
            monkeypatch.setattr(f.old.service, "storage", lambda path: 10 * 1024**3)
        kwargs["launch"]("preflight", {}, out, 10, lambda: False)
        return {"status": "completed"}

    monkeypatch.setattr(f.old.service, "execute", execute)
    if fault:
        with pytest.raises(ValueError):
            f.run(tmp_path / "config", tmp_path / "out")
    else:
        assert f.run(tmp_path / "config", tmp_path / "out")["status"] == "completed"
        assert json.loads((tmp_path / "out/external-accounting.json").read_bytes()) == (
            amendment["accounting"])


def test_cli_missing_config_and_worker_payload_bounds(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        f.main(["--out", str(tmp_path)])
    stream = type("Input", (), {"buffer": io.BytesIO(b"x" * (1024**2 + 1))})
    monkeypatch.setattr(f.sys, "stdin", stream)
    with pytest.raises(SystemExit):
        f.main(["--worker", "preflight", "--out", str(tmp_path)])


def test_cli_handled_failure_is_valid_worker_response(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(f.sys, "stdin", type("Input", (), {"buffer": io.BytesIO(b"{}")}))
    monkeypatch.setattr(f, "worker", lambda *args: {"status": "unavailable", "reason": "fixture"})
    assert f.main(["--worker", "preflight", "--out", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["reason"] == "fixture"
    monkeypatch.setattr(f, "run", lambda *args: {"status": "unavailable", "reason": "fixture"})
    assert f.main(["--config", "private", "--out", str(tmp_path)]) == 2
