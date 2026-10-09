"""Generated/emulated controls; no physical CUDA success inferred here."""

import hashlib
import io
import itertools
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from core import srnet_cuda
from core import srnet_profile as core
from steganography import research_cuda_profile as cli


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        [],
        [0.1] * 46,
        [0] * 47,
        [True] * 47,
        [float("nan")] * 47,
        [float("inf")] * 47,
        [-1] * 47,
    ],
)
def test_invalid_intervals(value):
    with pytest.raises(ValueError):
        core.budget(value)


def test_budget_is_fixed_and_estimate_only():
    result = core.budget([0.01] * 47)
    assert result["planned_updates"] == 16440
    assert result["planned_epochs"] == 5
    assert result["estimated_fit_seconds"] == pytest.approx(448.8)
    assert result["eligible_to_attempt_fixed_fit"] and result["estimate_only"]
    assert not core.budget([0.1] * 47)["eligible_to_attempt_fixed_fit"]
    values = list(range(1, 48))
    assert core.budget(values)["steady_interval_p95_seconds"] == pytest.approx(44.7)
    with pytest.raises(ValueError):
        core.budget([1e308] * 47)


@pytest.mark.parametrize(
    "count,updates,records", [(64, 64, 1), (63, 64, 1), (64, 63, 1), (64, 64, 2)]
)
def test_generated_profile_control_flow(monkeypatch, count, updates, records):
    torch = pytest.importorskip("torch")
    ticks = itertools.count(1, 0.01)
    monkeypatch.setattr(core.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(srnet_cuda, "inspect", lambda: {"device": "cuda:0"})
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 128)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 256)

    def learn(**kwargs):
        assert kwargs["seed"] == 20261008 and kwargs["device"] == "cuda:0"
        assert kwargs["params"]["max_seconds"] == 180
        assert kwargs["epochs"] == 1 and kwargs["target_pairs"] == 2
        assert kwargs["batches"](0).shape == (64, 4)
        for i in range(count):
            pixels = kwargs["fetch"](i)
            assert pixels.shape == (4, 1, 256, 256) and pixels.dtype == np.dtype("<f4")
            kwargs["outer_deadline"]()
        return None, [{"updates": updates}] * records

    monkeypatch.setattr(core.srnet_training, "_learn", learn)
    if (count, updates, records) != (64, 64, 1):
        with pytest.raises(ValueError, match="accounting"):
            core.profile()
    else:
        result = core.profile()
        assert len(result["steady_intervals_seconds"]) == 47
        assert result["generated_optimizer_updates"] == 64
        assert result["process_peak_gpu_reserved_bytes"] == 256
        assert not result["real_data_used"] and not result["real_model_trained"]


def test_source_snapshot_and_oversize(monkeypatch):
    assert len(cli.sources()) == 16
    monkeypatch.setattr(cli, "probe_sources", lambda: {})
    monkeypatch.setattr(cli, "regular_open", lambda _: io.BytesIO(b"x" * (1024**2 + 1)))
    with pytest.raises(ValueError, match="size"):
        cli.sources()


def test_execute_fresh_and_immutable_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.srnet_profile, "profile", lambda: {"real_data_used": False})
    out = tmp_path / "result.json"
    report = cli.execute(out)
    assert report["status"] == "completed" and len(report["source_sha256"]) == 16
    with pytest.raises(FileExistsError):
        cli.execute(out)
    target = tmp_path / "target"
    target.write_text("untouched")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(FileExistsError):
        cli.execute(link)
    assert target.read_text() == "untouched"
    snapshots = iter([{"a": "old"}, {"a": "new"}])
    monkeypatch.setattr(cli, "sources", lambda: next(snapshots))
    with pytest.raises(ValueError, match="changed"):
        cli.execute(tmp_path / "changed")
    assert not (tmp_path / "changed").exists()


def fake_worker(monkeypatch, *, raw=None, status="completed", code=0, tamper=False):
    def run(args, **kwargs):
        assert "research_cuda_profile" in args[2]
        assert kwargs["timeout"] == 210 and kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert kwargs["stderr"] == subprocess.DEVNULL
        report = {"source_sha256": cli.sources(), "status": "completed"}
        if tamper:
            report["source_sha256"] = {}
        path = cli.Path(args[-1])
        if status == "completed" and raw is None:
            cli.write_json(path, report)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            data = {"status": status, "report_sha256": digest}
        else:
            data = {"status": status}
        kwargs["stdout"].write(raw if raw is not None else json.dumps(data).encode())
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(cli.subprocess, "run", run)


def test_parent_completed_unavailable_and_stale(tmp_path, monkeypatch):
    fake_worker(monkeypatch)
    assert cli.run_job(tmp_path / "pass.json")["status"] == "completed"
    fake_worker(monkeypatch, status="unavailable", code=2)
    report = cli.run_job(tmp_path / "absent.json")
    assert report["status"] == "unavailable" and not report["cpu_fallback"]
    assert not report["profile_completed"]
    fake_worker(monkeypatch, tamper=True)
    with pytest.raises(RuntimeError, match="partial"):
        cli.run_job(tmp_path / "stale.json")


@pytest.mark.parametrize(
    "raw,code",
    [
        (b"x" * 65537, 0),
        (b"[]", 0),
        (b"oops", 0),
        (b'{"status":"failed"}', 2),
        (b"{}", 0),
        (b'{"status":"completed"}', 1),
        (b'{"status":"completed"}', 0),
    ],
)
def test_worker_response_failures(tmp_path, monkeypatch, raw, code):
    fake_worker(monkeypatch, raw=raw, code=code)
    with pytest.raises(RuntimeError, match="partial"):
        cli.run_job(tmp_path / "bad.json")


def test_timeout_and_cli_redaction(tmp_path, monkeypatch, capsys):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 210)

    monkeypatch.setattr(cli.subprocess, "run", timeout)
    assert cli.main(["--out", str(tmp_path / "timeout")]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "failed"

    def unavailable():
        raise srnet_cuda.CUDAUnavailable("bounded_cgroup_v2_required")

    monkeypatch.setattr(srnet_cuda, "host_bound", unavailable)
    assert cli.main(["--worker", "--out", str(tmp_path / "absent")]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "unavailable"


def test_main_success_unavailable_and_bounded_worker(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_job", lambda _: {"status": "completed"})
    assert cli.main(["--out", str(tmp_path / "one")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    monkeypatch.setattr(cli, "run_job", lambda _: {"status": "unavailable"})
    assert cli.main(["--out", str(tmp_path / "two")]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "unavailable"
    import resource

    limits = []
    monkeypatch.setattr(srnet_cuda, "host_bound", lambda: 8 * 1024**3)
    monkeypatch.setattr(resource, "setrlimit", lambda key, value: limits.append((key, value)))
    monkeypatch.setattr(cli.srnet_profile, "profile", lambda: {"generated_optimizer_updates": 64})
    assert cli.main(["--worker", "--out", str(tmp_path / "worker")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert (resource.RLIMIT_CPU, (360, 361)) in limits
    assert (resource.RLIMIT_FSIZE, (1024**2, 1024**2)) in limits
    assert (resource.RLIMIT_CORE, (0, 0)) in limits


def test_missing_optional_torch_is_unavailable_and_redacted(tmp_path, monkeypatch, capsys):
    import resource

    monkeypatch.setattr(srnet_cuda, "host_bound", lambda: 8 * 1024**3)
    monkeypatch.setattr(resource, "setrlimit", lambda *_: None)

    def missing(_):
        raise ImportError("PRIVATE HOST PATH MUST NOT LEAK")

    monkeypatch.setattr(cli, "execute", missing)
    assert cli.main(["--worker", "--out", str(tmp_path / "missing")]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "unavailable",
        "reason": "torch_not_installed",
    }
