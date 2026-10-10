import hashlib
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import srnet_cuda
from steganography import research_jpeg_timing_probe as cli


def complete():
    return {
        "manifest_sha256": cli.probe.MANIFEST_SHA,
        "audit_sha256": cli.probe.AUDIT_SHA,
        "real_optimizer_updates": 66,
        "steady_intervals_seconds": [0.1] * 47,
        "projection": cli.probe.projection([0.1] * 47),
        "execution": {"device": "cuda:0"},
        "real_data_used": True,
        "real_model_trained": True,
        "production_model_trained": False,
        "deployed": False,
        "accuracy_qualification": "unavailable",
    }


def test_sources_size_and_content(monkeypatch):
    monkeypatch.setattr(cli, "existing_sources", lambda: {"old": "a" * 64})
    result = cli.sources()
    assert len(result) == 5
    assert (
        result["core/jpeg_timing_probe.py"]
        == hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "core/jpeg_timing_probe.py").read_bytes()
        ).hexdigest()
    )
    monkeypatch.setattr(cli, "regular_open", lambda path: io.BytesIO(bytes(1024**2 + 1)))
    with pytest.raises(ValueError, match="bound"):
        cli.sources()


def test_execute_binds_sources_and_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "sources", lambda: {"old": "a" * 64})
    monkeypatch.setattr(cli.probe, "profile", lambda *args: complete())
    out = tmp_path / "out.json"
    report = cli.execute(tmp_path, tmp_path / "audit", out)
    assert report["status"] == "completed" and report["source_sha256"] == {"old": "a" * 64}
    with pytest.raises(FileExistsError):
        cli.execute(tmp_path, tmp_path / "audit", out)
    calls = iter([{}, {"changed": "b" * 64}])
    monkeypatch.setattr(cli, "sources", lambda: next(calls))
    with pytest.raises(ValueError, match="changed"):
        cli.execute(tmp_path, tmp_path / "audit", tmp_path / "changed.json")


@pytest.mark.parametrize(
    "fault",
    [
        "success",
        "unavailable",
        "timeout",
        "bytes",
        "object",
        "exit",
        "status",
        "source",
        "binding",
        "projection",
        "cpu",
        "after-source",
    ],
)
def test_parent_protocol_boundaries(tmp_path, monkeypatch, fault):
    source = {"source": "a" * 64}
    calls = 0

    def sources():
        nonlocal calls
        calls += 1
        return {"changed": "b" * 64} if fault == "after-source" and calls == 2 else source

    monkeypatch.setattr(cli, "sources", sources)
    out = tmp_path / "out.json"

    def run(command, **kw):
        assert command[1:4] == ["-m", "steganography.research_jpeg_timing_probe", "--worker"]
        assert kw["timeout"] == 210 and kw["stderr"] == subprocess.DEVNULL
        assert kw["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert "shell" not in kw
        if fault == "timeout":
            raise subprocess.TimeoutExpired(command, 210)
        if fault == "unavailable":
            kw["stdout"].write(b'{"status":"unavailable"}')
            return SimpleNamespace(returncode=2)
        if fault == "bytes":
            kw["stdout"].write(bytes(65537))
            return SimpleNamespace(returncode=0)
        if fault == "object":
            kw["stdout"].write(b"[]")
            return SimpleNamespace(returncode=0)
        report = complete()
        report.update(
            schema_version="srnet-cuda-real-timing-v1", status="completed", source_sha256=source
        )
        if fault == "source":
            report["source_sha256"] = {}
        elif fault == "binding":
            report["manifest_sha256"] = "b" * 64
        elif fault == "projection":
            report["projection"] = {}
        elif fault == "cpu":
            report["execution"]["device"] = "cpu"
        cli.write_json(out, report)
        response = {
            "status": "failed" if fault == "status" else "completed",
            "report_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        }
        kw["stdout"].write(json.dumps(response).encode())
        return SimpleNamespace(returncode=1 if fault == "exit" else 0)

    monkeypatch.setattr(cli.subprocess, "run", run)
    if fault in {"success", "unavailable"}:
        report = cli.run_job(tmp_path, tmp_path / "audit.json", out)
        assert report["status"] == ("completed" if fault == "success" else "unavailable")
        assert str(tmp_path) not in out.read_text()
        if fault == "unavailable":
            assert not report["cpu_fallback"] and not report["real_data_used"]
            assert "projection" not in report
    else:
        with pytest.raises(RuntimeError, match="partial evidence"):
            cli.run_job(tmp_path, tmp_path / "audit.json", out)


def test_main_worker_and_unavailable_status(tmp_path, monkeypatch, capsys):
    import resource

    options = [
        "--root",
        str(tmp_path),
        "--audit",
        str(tmp_path / "audit"),
        "--out",
        str(tmp_path / "out"),
    ]
    monkeypatch.setattr(cli, "run_job", lambda *a: {"status": "completed"})
    assert cli.main(options) == 0
    monkeypatch.setattr(cli, "run_job", lambda *a: {"status": "unavailable"})
    assert cli.main(options) == 2
    limits = []
    monkeypatch.setattr(resource, "setrlimit", lambda *a: limits.append(a))
    monkeypatch.setattr(srnet_cuda, "host_bound", lambda: 8 * 1024**3)
    monkeypatch.setattr(cli, "execute", lambda root, audit, out: out.write_text("{}"))
    assert cli.main(options + ["--worker"]) == 0
    assert dict(limits)[resource.RLIMIT_FSIZE] == (1024**2,) * 2

    def unavailable(*a):
        raise srnet_cuda.CUDAUnavailable("gpu_absent")

    monkeypatch.setattr(srnet_cuda, "host_bound", unavailable)
    assert cli.main(options + ["--worker"]) == 2
    monkeypatch.setattr(cli, "run_job", lambda *a: (_ for _ in ()).throw(ValueError("secret")))
    assert cli.main(options) == 2
    assert "secret" not in capsys.readouterr().out
