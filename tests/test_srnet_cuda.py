"""CPU-generated/emulated controls are not an actual CUDA hardware pass."""

import copy
import json
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from core import srnet, srnet_training
from core import srnet_cuda as backend
from steganography import research_cuda_probe as probe
from steganography import research_srnet_stream as service
from tests.test_srnet_stream import corpus as corpus  # noqa: F401
from tests.test_srnet_stream import digest, save


@pytest.fixture
def fake_torch(monkeypatch):
    """Policy API emulation only: contains no GPU tensor implementation."""
    state = {"fraction": 0.9, "deterministic": False, "warn": True, "current": 1}
    cuda = NS(
        is_available=lambda: True,
        get_device_capability=lambda _: (12, 0),
        get_arch_list=lambda: ["sm_120"],
        get_device_properties=lambda _: NS(total_memory=8 * 1024**3, name="generated-gpu"),
        memory_reserved=lambda _: 0,
        get_per_process_memory_fraction=lambda _: state["fraction"],
    )

    def fraction(value, _):
        state["fraction"] = value

    @contextmanager
    def select(value):
        previous = state["current"]
        state["current"] = value
        try:
            yield
        finally:
            state["current"] = previous

    def deterministic(value, *, warn_only):
        state.update(deterministic=value, warn=warn_only)

    cuda.set_per_process_memory_fraction, cuda.device = fraction, select
    module = NS(
        __version__="generated",
        version=NS(cuda="13.0"),
        cuda=cuda,
        backends=NS(
            cuda=NS(matmul=NS(fp32_precision="none")),
            cudnn=NS(conv=NS(fp32_precision="tf32"), benchmark=True, deterministic=False),
        ),
        are_deterministic_algorithms_enabled=lambda: state["deterministic"],
        is_deterministic_algorithms_warn_only_enabled=lambda: state["warn"],
        use_deterministic_algorithms=deterministic,
        autocast=lambda **kw: nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "torch", module)
    monkeypatch.setattr(backend, "host_bound", lambda: backend.HOST_BYTES)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return module, state


def test_policy_precision_budget_current_device_and_flags_restored(fake_torch):
    module, state = fake_torch
    before = copy.deepcopy(state)
    info = backend.inspect()
    assert info["torch_allocator_limit_bytes"] == 4 * 1024**3 and not info["cpu_fallback"]
    for error in (False, True):
        try:
            with backend.policy() as execution:
                assert execution == info and state["current"] == 0
                assert state["fraction"] == 0.5
                assert module.backends.cuda.matmul.fp32_precision == "ieee"
                assert module.backends.cudnn.conv.fp32_precision == "ieee"
                assert not module.backends.cudnn.benchmark and module.backends.cudnn.deterministic
                assert state["deterministic"] and not state["warn"]
                if error:
                    raise ValueError("injected failure")
        except ValueError:
            assert error
        assert state == before
        assert module.backends.cuda.matmul.fp32_precision == "none"
        assert module.backends.cudnn.conv.fp32_precision == "tf32"
        assert module.backends.cudnn.benchmark and not module.backends.cudnn.deterministic


@pytest.mark.parametrize(
    "fault",
    [
        "not-visible",
        "cpu-torch",
        "environment",
        "precision",
        "architecture",
        "small-vram",
        "reserved",
    ],
)
def test_missing_prerequisites_fail_closed_without_policy_changes(fake_torch, fault, monkeypatch):
    module, state = fake_torch
    before = copy.deepcopy(state)
    if fault == "not-visible":
        module.cuda.is_available = lambda: False
    if fault == "cpu-torch":
        module.version.cuda = None
    if fault == "environment":
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    if fault == "precision":
        del module.backends.cudnn.conv.fp32_precision
    if fault == "architecture":
        module.cuda.get_arch_list = lambda: ["sm_90"]
    if fault == "small-vram":
        module.cuda.get_device_properties = lambda _: NS(total_memory=1024**3, name="small")
    if fault == "reserved":
        module.cuda.memory_reserved = lambda _: 5 * 1024**3
    with pytest.raises(backend.CUDAUnavailable), backend.policy():
        pass
    assert state == before


@pytest.fixture
def cgroup(tmp_path):
    root = tmp_path / "cgroups"
    folder = root / "user/job"
    folder.mkdir(parents=True)
    proc = tmp_path / "membership"
    proc.write_text("0::/user/job\n")
    (folder / "memory.max").write_text("max\n")
    (root / "user/memory.max").write_text(str(8 * 1024**3))
    return proc, root, folder


def test_kernel_ancestor_ram_limit_without_root_file(cgroup):
    proc, root, folder = cgroup
    assert backend.host_bound(proc=proc, root=root) == 8 * 1024**3
    (folder / "memory.max").write_text(str(4 * 1024**3))
    assert backend.host_bound(proc=proc, root=root) == 4 * 1024**3


@pytest.mark.parametrize(
    "fault",
    [
        "v1",
        "escape",
        "relative",
        "membership-large",
        "depth",
        "unlimited",
        "too-high",
        "zero",
        "bad-number",
        "large-limit",
        "symlink",
        "missing",
    ],
)
def test_invalid_or_missing_kernel_memory_bound(cgroup, fault, tmp_path):
    proc, root, folder = cgroup
    if fault == "v1":
        proc.write_text("3:memory:/user/job\n")
    if fault == "escape":
        proc.write_text("0::/../escape\n")
    if fault == "relative":
        proc.write_text("0::user/job\n")
    if fault == "membership-large":
        proc.write_text("x" * 4097)
    if fault == "depth":
        proc.write_text("0::/" + "/".join("x" for _ in range(33)))
    if fault == "unlimited":
        (root / "user/memory.max").write_text("max")
    if fault == "too-high":
        (root / "user/memory.max").write_text(str(9 * 1024**3))
    if fault == "zero":
        (folder / "memory.max").write_text("0")
    if fault == "bad-number":
        (folder / "memory.max").write_text("NaN")
    if fault == "large-limit":
        (folder / "memory.max").write_text("9" * 33)
    if fault == "missing":
        proc.write_text("0::/missing\n")
    if fault == "symlink":
        path = folder / "memory.max"
        target = tmp_path / "target"
        target.write_text("1")
        path.unlink()
        path.symlink_to(target)
    with pytest.raises(backend.CUDAUnavailable, match="bounded_cgroup"):
        backend.host_bound(proc=proc, root=root)


@pytest.mark.parametrize("value", [None, True, "auto", "cuda", "cuda:1", [], 0])
def test_device_choice_is_explicit(value):
    with pytest.raises(ValueError):
        backend.device(value)
    assert backend.device("cpu") == "cpu" and backend.device("cuda:0") == "cuda:0"


def test_cpu_only_cuda_request_never_opens_corpus_or_falls_back(corpus, monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbidden(*a, **kw):
        raise AssertionError("must not read training data")

    monkeypatch.setattr(service, "TrainBlocks", forbidden)
    with pytest.raises(backend.CUDAUnavailable, match="cuda_not_visible"):
        service.execute({**corpus[0], "device": "cuda:0"}, tmp_path / "out", operation="plan")
    assert not (tmp_path / "out").exists()


def test_versioned_cuda_plan_no_cpu_schema_change(corpus, monkeypatch, tmp_path):
    info = {"device": "cuda:0", "precision": "float32-ieee"}
    monkeypatch.setattr(backend, "inspect", lambda: info)
    plan = service.execute(
        {**corpus[0], "device": "cuda:0"}, tmp_path / "cuda-plan", operation="plan"
    )
    assert plan["schema_version"] == "srnet-stream-plan-v2" and plan["execution"] == info
    cpu = service.execute(corpus[0], tmp_path / "cpu-plan", operation="plan")
    explicit_cpu = service.execute(
        {**corpus[0], "device": "cpu"}, tmp_path / "explicit-cpu", operation="plan"
    )
    assert cpu == explicit_cpu and cpu["schema_version"] == "srnet-stream-plan-v1"
    assert "device" not in cpu["settings"] and "execution" not in cpu


def test_cuda_job_uses_kernel_ram_bound_not_cpu_virtual_cap(corpus, tmp_path, monkeypatch, capsys):
    import resource

    config = {**corpus[0], "device": "cuda:0"}
    path = tmp_path / "config.json"
    save(path, config)
    calls, limits = [], []
    monkeypatch.setattr(backend, "host_bound", lambda: calls.append("kernel") or backend.HOST_BYTES)
    monkeypatch.setattr(resource, "setrlimit", lambda key, value: limits.append((key, value)))
    monkeypatch.setattr(
        service,
        "execute",
        lambda *a, **kw: save(tmp_path / "report.json", {"status": "fake-control"}),
    )
    args = [
        "--worker",
        "--operation",
        "plan",
        "--config",
        str(path),
        "--config-sha256",
        digest(path),
        "--out",
        str(tmp_path / "report.json"),
    ]
    assert service.main(args) == 0
    assert calls == ["kernel"] and all(key != resource.RLIMIT_AS for key, _ in limits)
    capsys.readouterr()

    def unavailable(*a, **kw):
        raise backend.CUDAUnavailable("bounded_cgroup_v2_required")

    monkeypatch.setattr(backend, "host_bound", unavailable)
    assert service.main(args) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "unavailable"


def test_worker_unavailability_stays_unavailable(corpus, tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.json"
    save(path, {**corpus[0], "device": "cuda:0"})

    def unavailable(*a, **kw):
        kw["stdout"].write(b'{"status":"unavailable","reason":"not disclosed"}')
        return subprocess.CompletedProcess(a[0], 2)

    monkeypatch.setattr(service.subprocess, "run", unavailable)
    assert (
        service.main(["--operation", "plan", "--config", str(path), "--out", str(tmp_path / "out")])
        == 2
    )
    assert "CUDA backend unavailable" in capsys.readouterr().out


def test_emulated_cuda_engine_control_flow_only(corpus, monkeypatch):
    """Redirect device transfers to CPU. NOT a CUDA numerical/hardware oracle."""
    torch = pytest.importorskip("torch")
    events = []
    original_tensor_to, original_module_to, original_tensor = (
        torch.Tensor.to,
        torch.nn.Module.to,
        torch.tensor,
    )

    def tensor_to(self, *a, **kw):
        return (
            original_tensor_to(self, "cpu")
            if a == ("cuda:0",)
            else original_tensor_to(self, *a, **kw)
        )

    def module_to(self, *a, **kw):
        return (
            original_module_to(self, "cpu")
            if a == ("cuda:0",)
            else original_module_to(self, *a, **kw)
        )

    def tensor(*a, **kw):
        if kw.get("device") == "cuda:0":
            kw["device"] = "cpu"
        return original_tensor(*a, **kw)

    monkeypatch.setattr(torch.Tensor, "to", tensor_to)
    monkeypatch.setattr(torch.nn.Module, "to", module_to)
    monkeypatch.setattr(torch, "tensor", tensor)
    original_fork = torch.random.fork_rng
    monkeypatch.setattr(torch.random, "fork_rng", lambda **kw: original_fork(devices=[]))
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda seed: events.append(("seed", seed)))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda i: events.append(("sync", i)))
    monkeypatch.setattr(backend, "policy", lambda: nullcontext())
    model, records = srnet_training._learn(
        fetch=lambda _: corpus[1][:4].copy(),
        batches=lambda _: np.array([[0, 1, 2, 3]]),
        epochs=1,
        seed=91,
        params=srnet_training.settings({"threads": 1, "max_seconds": 180}),
        target_pairs=2,
        device="cuda:0",
    )
    assert records[0]["updates"] == 1 and events == [("seed", 91), ("sync", 0)]
    assert all(p.device.type == "cpu" for p in model.parameters())
    with pytest.raises(ValueError, match="device"):
        srnet_training._learn(
            fetch=None, batches=None, epochs=1, seed=91, params={}, target_pairs=2, device="auto"
        )


def test_generated_probe_cpu_emulation_is_not_published_as_hardware(corpus, monkeypatch, tmp_path):
    """Exercise report/limits/parity code with explicit CPU stand-ins only."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(backend, "inspect", lambda: {"device": "fake-cpu-control"})
    monkeypatch.setattr(backend, "policy", lambda: nullcontext())
    monkeypatch.setattr(
        srnet_training,
        "_learn",
        lambda **kw: (srnet.network().eval(), [{"updates": 1, "mean_pair_loss": 0.7}]),
    )
    tensor_to, module_to = torch.Tensor.to, torch.nn.Module.to
    monkeypatch.setattr(
        torch.Tensor,
        "to",
        lambda self, *a, **kw: (
            tensor_to(self, "cpu") if a == ("cuda:0",) else tensor_to(self, *a, **kw)
        ),
    )
    monkeypatch.setattr(
        torch.nn.Module,
        "to",
        lambda self, *a, **kw: (
            module_to(self, "cpu") if a == ("cuda:0",) else module_to(self, *a, **kw)
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 0)
    report = probe.probe(tmp_path / "emulated.json")
    assert report["generated_optimizer_updates"] == 1 and not report["real_data_used"]
    assert (
        not report["independent_math_oracle"] and report["accuracy_qualification"] == "unavailable"
    )
    assert str(tmp_path) not in json.dumps(report)
    monkeypatch.setattr(srnet, "float_logits", lambda *a: np.full((1, 2), np.nan))
    with pytest.raises(ValueError, match="parity"):
        probe.probe(tmp_path / "bad-parity.json")
    monkeypatch.setattr(srnet, "float_logits", lambda *a: np.zeros((1, 2)))
    monkeypatch.setattr(np, "allclose", lambda *a, **kw: True)
    snapshots = iter([{"initial": 1}, {"changed": 2}])
    monkeypatch.setattr(probe, "sources", lambda: next(snapshots))
    with pytest.raises(ValueError, match="sources changed"):
        probe.probe(tmp_path / "bad-sources.json")


@pytest.mark.parametrize(
    "response",
    [
        b'{"status":"unavailable"}',
        b'{"status":"failed"}',
        b"[]",
        b"not json",
        b"x" * 65537,
        b'{"status":"completed","report_sha256":"bad"}',
    ],
)
def test_probe_worker_failures_not_success(tmp_path, monkeypatch, response):
    def fake(*a, **kw):
        kw["stdout"].write(response)
        return subprocess.CompletedProcess(a[0], 2)

    monkeypatch.setattr(probe.subprocess, "run", fake)
    if b"unavailable" in response:
        result = probe.run_job(tmp_path / "out")
        assert result["status"] == "unavailable" and not result["real_model_trained"]
        assert not result["hardware_probe_completed"]
    else:
        with pytest.raises(RuntimeError):
            probe.run_job(tmp_path / "out")


def test_probe_worker_success_checksum_limits_and_timeout(tmp_path, monkeypatch, capsys):
    import resource

    limits = []
    monkeypatch.setattr(resource, "setrlimit", lambda k, v: limits.append((k, v)))
    monkeypatch.setattr(backend, "host_bound", lambda: backend.HOST_BYTES)
    monkeypatch.setattr(probe, "probe", lambda out: save(out, {"status": "emulated-only"}))
    out = tmp_path / "probe.json"
    assert probe.main(["--worker", "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    assert (resource.RLIMIT_CPU, (120, 121)) in limits
    assert (resource.RLIMIT_FSIZE, (1024**2, 1024**2)) in limits
    assert all(k != resource.RLIMIT_AS for k, _ in limits)

    def success(*a, **kw):
        kw["stdout"].write(
            json.dumps({"status": "completed", "report_sha256": digest(out)}).encode()
        )
        return subprocess.CompletedProcess(a[0], 0)

    monkeypatch.setattr(probe.subprocess, "run", success)
    # A worker response cannot point to an unrelated output.
    with pytest.raises(RuntimeError):
        probe.run_job(tmp_path / "target")

    def complete(*a, **kw):
        target = Path(a[0][-1])
        checksum = save(target, {"status": "completed", "execution": "emulated-only"})
        kw["stdout"].write(json.dumps({"status": "completed", "report_sha256": checksum}).encode())
        return subprocess.CompletedProcess(a[0], 0)

    monkeypatch.setattr(probe.subprocess, "run", complete)
    assert probe.main(["--out", str(tmp_path / "success")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"

    def timeout(*a, **kw):
        raise subprocess.TimeoutExpired(a[0], 90)

    monkeypatch.setattr(probe.subprocess, "run", timeout)
    assert probe.main(["--out", str(tmp_path / "timeout")]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "failed"

    def unavailable():
        raise backend.CUDAUnavailable("bounded_cgroup_v2_required")

    monkeypatch.setattr(backend, "host_bound", unavailable)
    assert probe.main(["--worker", "--out", str(tmp_path / "absent")]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "unavailable"


def test_live_cuda_probe_or_explicitly_unavailable(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("actual CUDA hardware unavailable; not passed")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        backend.inspect()
    except backend.CUDAUnavailable as exc:
        pytest.skip(exc.reason + "; not passed")
    report = probe.probe(tmp_path / "real-generated-cuda.json")
    assert report["status"] == "completed" and report["generated_optimizer_updates"] == 1
    assert not report["real_data_used"] and report["execution"]["device"] == "cuda:0"
