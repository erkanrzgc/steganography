"""Generated complete epoch jobs and fail-closed provenance; not accuracy."""

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core import srnet_checkpoint, srnet_resume_probe
from steganography import research_srnet_epochs as service
from tests.test_srnet_stream import corpus as corpus


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def config(corpus, tmp_path):
    config, _, _ = corpus
    protocol = tmp_path / "protocol.md"
    protocol.write_text("generated two-epoch test only")
    report = {
        "schema_version": "srnet-epoch-resume-probe-v1",
        "status": "completed",
        "source_sha256": service.sources(),
        "exact_model_bn_optimizer_rng_loss": True,
        "execution": {"device": "cpu"},
        "real_data_used": False,
        "real_model_trained": False,
        "generated_optimizer_updates": 128,
        "uninterrupted_updates": 64,
        "steady_intervals_seconds": [0.001] * 47,
        "steady_interval_p95_seconds": 0.001,
    }
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps(report))
    return {
        **config,
        "epochs": 2,
        "protocol": str(protocol),
        "protocol_sha256": sha(protocol),
        "resume_probe": str(probe),
        "resume_probe_sha256": sha(probe),
    }


def plan(config, folder):
    target = folder / "plan.json"
    value = service.execute(config, target, operation="plan")
    return target, sha(target), value


def test_two_real_engine_jobs_complete_exact_chain_and_no_validation(config, tmp_path):
    pytest.importorskip("torch")
    target, digest, record = plan(config, tmp_path)
    assert record["train_rows"] == 9 and record["validation_used"] is False
    assert len(record["epoch_estimated_seconds"]) == 2
    first = tmp_path / "first"
    card = service.execute(
        config, first, operation="fit", epoch=0, plan_path=target, plan_sha=digest
    )
    second = tmp_path / "second"
    resumed = service.execute(
        config,
        second,
        operation="fit",
        epoch=1,
        plan_path=target,
        plan_sha=digest,
        parent=first,
        parent_sha=sha(first / "epoch.json"),
    )
    assert resumed["epoch_training"][:1] == card["epoch_training"]
    assert resumed["next_epoch"] == 2
    assert resumed["parent_report_sha256"] == sha(first / "epoch.json")
    snapshot = srnet_checkpoint.load(
        second / "checkpoint.npz", checksum=resumed["checkpoint_sha256"]
    )
    assert snapshot["metadata"]["binding"] == digest
    assert snapshot["metadata"]["epoch_training"] == resumed["epoch_training"]
    assert resumed["accuracy_qualification"] == "unavailable"


@pytest.mark.parametrize("fault", ["missing", "path", "sha", "protocol", "unknown", "epochs"])
def test_config_rejects(config, fault):
    changed = copy.deepcopy(config)
    if fault == "missing":
        del changed["resume_probe"]
    elif fault == "path":
        changed["protocol"] = ""
    elif fault == "sha":
        changed["protocol_sha256"] = "bad"
    elif fault == "protocol":
        changed["protocol_sha256"] = "0" * 64
    elif fault == "unknown":
        changed["unknown"] = "bad"
    else:
        changed["epochs"] = 0
    with pytest.raises((ValueError, KeyError)):
        service.configuration(changed)


@pytest.mark.parametrize(
    "fault",
    ["schema", "sources", "parity", "device", "updates", "intervals", "nan", "p95", "budget"],
)
def test_probe_gate_rejects_without_outputs(config, tmp_path, fault):
    path = Path(config["resume_probe"])
    record = json.loads(path.read_text())
    changes = {
        "schema": ("schema_version", "bad"),
        "sources": ("source_sha256", {}),
        "parity": ("exact_model_bn_optimizer_rng_loss", False),
        "device": ("execution", {"device": "cuda:0"}),
        "updates": ("generated_optimizer_updates", 1),
        "intervals": ("steady_intervals_seconds", [1.0]),
        "nan": ("steady_intervals_seconds", [float("nan")] * 47),
        "p95": ("steady_interval_p95_seconds", 1.0),
    }
    if fault == "budget":
        record.update(steady_interval_p95_seconds=100, steady_intervals_seconds=[100] * 47)
    else:
        key, value = changes[fault]
        record[key] = value
    path.write_text(json.dumps(record))
    config["resume_probe_sha256"] = sha(path)
    out = tmp_path / "out.json"
    with pytest.raises(ValueError):
        service.execute(config, out, operation="plan")
    assert not out.exists()


@pytest.mark.parametrize(
    "fault",
    [
        "operation",
        "plan_epoch",
        "missing_plan",
        "wrong_plan",
        "epoch_bool",
        "later_parent",
        "first_parent",
    ],
)
def test_job_gate_rejects(config, tmp_path, fault):
    target, digest, _ = plan(config, tmp_path)
    args = {"operation": "fit", "epoch": 0, "plan_path": target, "plan_sha": digest}
    if fault == "operation":
        args["operation"] = "bad"
    elif fault == "plan_epoch":
        args.update(operation="plan", epoch=0)
    elif fault == "missing_plan":
        args["plan_path"] = None
    elif fault == "wrong_plan":
        args["plan_sha"] = "0" * 64
    elif fault == "epoch_bool":
        args["epoch"] = True
    elif fault == "later_parent":
        args["epoch"] = 1
    else:
        args["parent"] = tmp_path / "bad"
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        service.execute(config, out, **args)
    assert not out.exists()


def test_parent_card_and_numeric_state_binding_rejected(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    card = {
        "schema_version": "srnet-epoch-job-v1",
        "status": "completed",
        "plan_sha256": "1" * 64,
        "source_sha256": {},
        "next_epoch": 1,
        "validation_used": False,
        "checkpoint_sha256": "2" * 64,
        "epoch_training": [],
    }
    path = parent / "epoch.json"
    path.write_text(json.dumps(card))
    with pytest.raises(ValueError, match="chain"):
        service.parent_state(
            parent, sha(path), {"source_sha256": {"different": "3" * 64}}, "1" * 64, 1
        )
    monkeypatch.setattr(
        srnet_checkpoint, "load", lambda *a, **k: {"metadata": {"binding": "0" * 64}}
    )
    with pytest.raises(ValueError, match="numeric"):
        service.parent_state(parent, sha(path), {"source_sha256": {}}, "1" * 64, 1)


def test_input_sources_change_prevents_publication(config, tmp_path, monkeypatch):
    before = service.sources()
    calls = iter([before, {}])
    monkeypatch.setattr(service, "sources", lambda: next(calls))
    with pytest.raises(ValueError, match="sources"):
        service.execute(config, tmp_path / "plan.json", operation="plan")
    assert not (tmp_path / "plan.json").exists()


def test_source_and_file_bounds(tmp_path, monkeypatch):
    target = tmp_path / "bytes"
    target.write_bytes(b"0123")
    with pytest.raises(ValueError, match="size"):
        service.file_sha(target, maximum=3)
    result = service.sources()
    assert "core/srnet_checkpoint.py" in result and len(result) == 16
    monkeypatch.setattr(
        service, "regular_open", lambda _: __import__("io").BytesIO(b"x" * (1024**2 + 1))
    )
    with pytest.raises(ValueError, match="size"):
        service.sources()


def test_probe_publication_protocol_guard(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol"
    protocol.write_text("frozen")
    out = tmp_path / "probe.json"
    with pytest.raises(ValueError, match="protocol"):
        service.execute_probe(out, device="cpu", protocol=protocol, protocol_sha="0" * 64)
    monkeypatch.setattr(
        srnet_resume_probe, "probe", lambda *a, **k: {"accuracy_qualification": "unavailable"}
    )
    record = service.execute_probe(out, device="cpu", protocol=protocol, protocol_sha=sha(protocol))
    assert record["status"] == "completed" and record["source_sha256"] == service.sources()


def arguments(out, config=None, operation="probe"):
    return argparse.Namespace(
        out=out,
        config=config,
        operation=operation,
        epoch=None,
        plan=None,
        plan_sha=None,
        parent=None,
        parent_sha=None,
        protocol=None,
        protocol_sha=None,
        device="cpu",
        worker=False,
        config_sha256=None,
    )


@pytest.mark.parametrize("fault", ["returncode", "oversized", "json", "timeout", "sources"])
def test_isolated_worker_failure(tmp_path, monkeypatch, fault):
    def run(command, **kwargs):
        if fault == "timeout":
            raise subprocess.TimeoutExpired(command, 1)
        raw = b"x" * 65537 if fault == "oversized" else b"{}"
        if fault == "json":
            raw = b"bad-json"
        if fault == "sources":
            out = tmp_path / "out.json"
            out.write_text(json.dumps({"source_sha256": {}}))
            raw = json.dumps({"status": "completed", "report_sha256": sha(out)}).encode()
        kwargs["stdout"].write(raw)
        return SimpleNamespace(returncode=1 if fault == "returncode" else 0)

    monkeypatch.setattr(service.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="incomplete"):
        service.run_job(arguments(tmp_path / "out.json"))


def test_generated_probe_real_cpu_exact_disk_resume(tmp_path):
    pytest.importorskip("torch")
    report = srnet_resume_probe.probe(tmp_path, binding="1" * 64, device="cpu")
    assert report["exact_model_bn_optimizer_rng_loss"] is True
    assert report["generated_optimizer_updates"] == 128
    assert len(report["steady_intervals_seconds"]) == 47
    assert report["real_model_trained"] is False


def test_main_fail_closed_without_config(tmp_path, capsys):
    assert service.main(["--operation", "plan", "--out", str(tmp_path / "out")]) == 2
    assert '"status":"failed"' in capsys.readouterr().out


@pytest.mark.parametrize("operation,device", [("probe", "cuda:0"), ("plan", "cpu"), ("fit", "cpu")])
def test_worker_limits_and_completion_response(
    config, tmp_path, monkeypatch, capsys, operation, device
):
    import resource

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    limits = []
    monkeypatch.setattr(resource, "setrlimit", lambda kind, value: limits.append((kind, value)))
    monkeypatch.setattr(service.srnet_cuda, "host_bound", lambda: 8 * 1024**3)
    out = tmp_path / "result"

    def emit(*args, **kwargs):
        if operation == "fit":
            out.mkdir()
            (out / "epoch.json").write_text("{}")
        else:
            out.write_text("{}")

    monkeypatch.setattr(service, "execute_probe", emit)
    monkeypatch.setattr(service, "execute", emit)
    assert (
        service.main(
            [
                "--worker",
                "--operation",
                operation,
                "--out",
                str(out),
                "--config",
                str(config_path),
                "--config-sha256",
                sha(config_path),
                "--device",
                device,
            ]
        )
        == 0
    )
    assert any(
        kind == resource.RLIMIT_FSIZE and value == (96 * 1024**2,) * 2 for kind, value in limits
    )
    assert '"completed"' in capsys.readouterr().out


def test_parent_cli_completed_and_backend_unavailable(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(service, "run_job", lambda _: {})
    argv = ["--operation", "probe", "--out", str(tmp_path / "out")]
    assert service.main(argv) == 0
    assert '"completed"' in capsys.readouterr().out

    def unavailable(_):
        raise service.srnet_cuda.CUDAUnavailable("test_only")

    monkeypatch.setattr(service, "run_job", unavailable)
    assert service.main(argv) == 2
    assert '"unavailable"' in capsys.readouterr().out


@pytest.mark.parametrize("fault", [None, "config", "cursor"])
def test_parent_verifies_worker_fit_result(config, tmp_path, monkeypatch, fault):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    out = tmp_path / "result"
    args = arguments(out, config_path, "fit")
    args.epoch = 0

    def run(command, **kwargs):
        assert "--config-sha256" in command and kwargs["timeout"] == 210
        out.mkdir()
        report = {
            "source_sha256": service.sources(),
            "checkpoint_sha256": "1" * 64,
            "model_sha256": "2" * 64,
        }
        target = out / "epoch.json"
        target.write_text(json.dumps(report))
        kwargs["stdout"].write(
            json.dumps({"status": "completed", "report_sha256": sha(target)}).encode()
        )
        if fault == "config":
            config_path.write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(service.subprocess, "run", run)
    monkeypatch.setattr(
        srnet_checkpoint,
        "load",
        lambda *a, **k: {"metadata": {"next_epoch": 2 if fault == "cursor" else 1}},
    )
    monkeypatch.setattr(service.srnet_model, "load_model", lambda *a, **k: None)
    if fault:
        with pytest.raises(RuntimeError):
            service.run_job(args)
    else:
        assert service.run_job(args)["source_sha256"] == service.sources()


def test_probe_source_mutation_rejected(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol"
    protocol.write_text("frozen")
    calls = iter([{}, {"changed": "0" * 64}])
    monkeypatch.setattr(service, "sources", lambda: next(calls))
    monkeypatch.setattr(srnet_resume_probe, "probe", lambda *a, **k: {})
    with pytest.raises(ValueError, match="dependencies"):
        service.execute_probe(
            tmp_path / "out", device="cpu", protocol=protocol, protocol_sha=sha(protocol)
        )


def test_real_fit_mutation_and_completion_checks(config, tmp_path, monkeypatch):
    target, digest, record = plan(config, tmp_path)
    monkeypatch.setattr(service.srnet_stream_training, "fit", lambda *a, **k: (None, []))
    with pytest.raises(ValueError, match="accounting"):
        service.execute(
            config,
            tmp_path / "incomplete",
            operation="fit",
            epoch=0,
            plan_path=target,
            plan_sha=digest,
        )

    def fake_fit(*a, **k):
        k["segment"]["sink"]({})
        return None, [{}]

    monkeypatch.setattr(service.srnet_stream_training, "fit", fake_fit)
    calls = iter([record["source_sha256"], {}])
    monkeypatch.setattr(service, "sources", lambda: next(calls))
    with pytest.raises(ValueError, match="dependencies"):
        service.execute(
            config,
            tmp_path / "changed",
            operation="fit",
            epoch=0,
            plan_path=target,
            plan_sha=digest,
        )


@pytest.mark.parametrize("fault", [None, "parity", "timing"])
def test_generated_cuda_probe_control_flow_only(tmp_path, monkeypatch, fault):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(service.srnet_cuda, "inspect", lambda: {"device": "cuda:0"})
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 456)
    if fault == "timing":
        monkeypatch.setattr(srnet_resume_probe.time, "monotonic", lambda: 0.0)

    def fake_learn(**kwargs):
        stop = kwargs["segment"]["stop_epoch"]
        for _ in range(64 if stop == 2 else 32):
            kwargs["fetch"](None)
        arrays = {"test": np.array([1], dtype="<f4")}
        if fault == "parity" and kwargs["segment"]["resume"] is not None:
            arrays["test"][0] = 2
        records = list(range(stop))
        kwargs["segment"]["sink"]({"metadata": {"next_epoch": stop}, "arrays": arrays})
        return None, records

    monkeypatch.setattr(srnet_resume_probe.srnet_training, "_learn", fake_learn)
    monkeypatch.setattr(srnet_checkpoint, "save", lambda *a, **k: "1" * 64)
    monkeypatch.setattr(srnet_checkpoint, "load", lambda *a, **k: {"metadata": {"next_epoch": 1}})
    if fault:
        with pytest.raises(ValueError):
            srnet_resume_probe.probe(tmp_path, binding="1" * 64, device="cuda:0")
    else:
        report = srnet_resume_probe.probe(tmp_path, binding="1" * 64, device="cuda:0")
        assert report["process_peak_gpu_allocated_bytes"] == 123
