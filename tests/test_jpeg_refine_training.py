"""Explicit generated CPU kernels, CUDA availability, endpoint immutability and adapters."""

import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from core import jpeg_refine as c
from core import jpeg_refine_preflight as preflight
from core import jpeg_refine_schedule as s
from core import jpeg_refine_state as state
from core import jpeg_refine_training as training
from core import jpeg_study_optimizer as rule
from core import srnet_cuda
from core.jpeg_timing import write_document
from steganography import research_jpeg_refine as frontend
from tests.test_jpeg_refine import EXECUTION, SHA, SOURCES, config, numeric, rows
from tests.test_jpeg_refine import tiny as tiny  # noqa: F401 - shared independent tiny topology


@pytest.fixture
def reader(monkeypatch):
    class Reader:
        view_count = 5

        def __init__(self, root, checksum, role, *, deadline):
            self.samples = rows(role, 8)
            self.deadline = deadline

        def batch(self, indices, views):
            self.deadline()
            return np.broadcast_to(
                (indices + views).astype("f4")[:, None, None, None], (len(indices), 1, 256, 256)
            ).copy()

        def verify(self):
            self.deadline()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(training.data, "Reader", Reader)
    return Reader


def test_actual_cpu_kernel_interval_and_restored_state_equal_uninterrupted(tiny, reader, tmp_path):
    torch = pytest.importorskip("torch")

    value = config(tmp_path / "data")
    value["root"] = tmp_path / "data"
    first = training.train(
        value,
        tmp_path / "first",
        ("warmup", 20261012, 512),
        SHA,
        EXECUTION,
        device="cpu",
        factory=tiny,
    )
    assert first["status"] == "checkpointed" and first["new_updates"] == 512
    assert not first["real_gpu_segment"] and first["peak_gpu_allocated_bytes"] is None
    parent = {"path": tmp_path / "first" / "snapshot.npz", "sha256": first["checkpoint_sha256"]}
    second = training.train(
        value,
        tmp_path / "second",
        ("warmup", 20261012, 1024),
        SHA,
        EXECUTION,
        parent=parent,
        device="cpu",
        factory=tiny,
    )
    snapshot = state.load(tmp_path / "second" / "snapshot.npz", second["checkpoint_sha256"], SHA)
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(20261012)
        model = tiny().train()
        optimizer = rule.optimizer(model, reference=True, corrected_l2=True)
        values = s.schedule(rows(count=8), 20261012, "warmup")
        with reader(None, None, "fit", deadline=lambda: None) as data:
            for index in range(1024):
                indices, angles, views = (v[index] for v in values)
                pixels = training.study.augment(data.batch(indices, views), angles)
                training.update(model, optimizer, pixels, indices, "cpu")
        captured = state.capture(model, optimizer, snapshot["metadata"])
        for key, value in captured["arrays"].items():
            np.testing.assert_array_equal(value, snapshot["arrays"][key])
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize(
    "fault", ["task", "parent", "runtime", "views", "timeout", "cancel", "symlink", "overwrite"]
)
def test_kernel_guards_before_updates(tiny, reader, tmp_path, fault):
    task, parent, execution = ("warmup", 20261012, 512), None, EXECUTION
    out = tmp_path / "out"
    value = config(tmp_path / "data")

    def clock():
        return 0.0

    if fault == "task":
        task = ("paired", 20261012, 512)
    elif fault == "parent":
        parent = {}
    elif fault == "runtime":
        execution = {**EXECUTION, "library_versions": {}}
    elif fault == "views":
        reader.view_count = 1
    elif fault == "timeout":
        ticks = iter([0.0, 1801.0])

        def clock():
            return next(ticks)
    elif fault == "cancel":
        (tmp_path / "CANCEL").touch()
    elif fault == "symlink":
        (tmp_path / "CANCEL").symlink_to(tmp_path / "missing")
    else:
        out.mkdir()
    with pytest.raises((ValueError, FileExistsError, TimeoutError, InterruptedError)):
        training.train(
            value, out, task, SHA, execution, parent=parent, device="cpu", factory=tiny, clock=clock
        )


@pytest.mark.parametrize("fault", ["shape", "logits", "objective", "gradient", "state"])
def test_training_numeric_guards(tiny, monkeypatch, fault):
    torch = pytest.importorskip("torch")

    model = tiny()
    optimizer = rule.optimizer(model, reference=True)
    pixels = np.ones((16, 1, 8, 8), dtype="f4")
    if fault in ("shape", "logits"):
        monkeypatch.setattr(
            model,
            "forward",
            lambda x: (
                torch.ones((16, 3)) if fault == "shape" else torch.full((16, 2), float("nan"))
            ),
        )
    elif fault == "objective":
        monkeypatch.setattr(rule, "convolution_penalty", lambda m: torch.tensor(float("nan")))
    elif fault == "gradient":
        monkeypatch.setattr(training.srnet_training, "_finite_gradients", lambda *args: False)
    else:
        monkeypatch.setattr(training.study, "_finite_state", lambda *args: False)
    with pytest.raises(ValueError):
        training.update(model, optimizer, pixels, np.arange(16), "cpu")


@pytest.mark.parametrize(
    "arm,stop,roles", [("warmup", 2048, {"development"}), ("paired", 8192, {"development", "fit"})]
)
def test_endpoint_evaluation_never_constructs_optimizer_or_mutates(
    tiny, reader, tmp_path, monkeypatch, arm, stop, roles
):
    snapshot = numeric(arm=arm, stop=stop, parent=SHA, warmup=SHA if arm != "warmup" else None)
    path = tmp_path / "snapshot.npz"
    checksum = state.save(snapshot, path)
    monkeypatch.setattr(
        rule, "optimizer", lambda *args, **kwargs: pytest.fail("evaluation optimizer")
    )
    result = training.evaluate(
        config(tmp_path),
        {"path": path, "sha256": checksum, "binding": SHA},
        device="cpu",
        factory=tiny,
    )
    assert result["model_sha256_before"] == result["model_sha256_after"]
    assert {k for k in result if k in {"development", "fit"}} == roles
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum


def test_evaluation_endpoint_and_mutation_guards(tiny, reader, tmp_path, monkeypatch):
    path = tmp_path / "partial.npz"
    sha = state.save(numeric(), path)
    with pytest.raises(ValueError):
        training.evaluate(
            config(tmp_path),
            {"path": path, "sha256": sha, "binding": SHA},
            device="cpu",
            factory=tiny,
        )
    path = tmp_path / "final.npz"
    sha = state.save(numeric(stop=2048, parent=SHA), path)

    def mutate(model, data, device):
        with __import__("torch").no_grad():
            next(model.parameters()).add_(1)
        return [0.5] * len(data.samples)

    monkeypatch.setattr(training.study, "evaluate", mutate)
    with pytest.raises(ValueError, match="mutated"):
        training.evaluate(
            config(tmp_path),
            {"path": path, "sha256": sha, "binding": SHA},
            device="cpu",
            factory=tiny,
        )


def test_generated_preflight_cpu_save_resume(tiny):
    result = preflight.probe(device="cpu", factory=tiny)
    assert result["generated_parity"] and result["save_resume_parity"]
    assert result["generated_optimizer_updates"] == 14 and not result["real_data_used"]
    assert result["max_update_seconds"] > 0 and result["startup_checkpoint_seconds"] > 0


@pytest.mark.parametrize("fault", ["loss", "state", "resume", "optimizer"])
def test_preflight_fail_closed(tiny, monkeypatch, fault):
    original = training.update
    calls = 0

    def update(model, optimizer, pixels, indices, device):
        nonlocal calls
        calls += 1
        result = original(model, optimizer, pixels, indices, device)
        if fault == "loss" and calls == 2:
            return (100.0, 100.0)
        if fault == "state" and calls == 2:
            with __import__("torch").no_grad():
                next(model.parameters()).add_(10)
        if fault == "optimizer" and calls == 12:
            next(iter(optimizer.state.values()))["step"].add_(1)
        return result

    monkeypatch.setattr(training, "update", update)
    if fault == "resume":
        original_restore = state.restore

        def restore(snapshot, model, optimizer, *args, **kwargs):
            original_restore(snapshot, model, optimizer, *args, **kwargs)
            with __import__("torch").no_grad():
                next(model.parameters()).add_(1)

        monkeypatch.setattr(state, "restore", restore)
    with pytest.raises(ValueError):
        preflight.probe(device="cpu", factory=tiny)


def test_actual_cuda_generated_preflight_requires_owned_bounded_device():
    try:
        execution = srnet_cuda.inspect()
    except (srnet_cuda.CUDAUnavailable, ImportError) as exc:
        pytest.skip(f"actual CUDA unavailable: {getattr(exc, 'reason', 'optional_dependency')}")
    if "RTX 5060" not in execution["gpu_name"]:
        pytest.skip("owned RTX 5060 unavailable")
    with srnet_cuda.policy():
        assert preflight.probe()["save_resume_parity"]


def test_source_closure_and_physical_gpu_inspection(monkeypatch):
    torch = pytest.importorskip("torch")

    assert len(frontend.sources()) == len(frontend.NAMES)
    monkeypatch.setattr(srnet_cuda, "inspect", lambda: {**EXECUTION, "gpu_name": "RTX 5060"})
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="fixture")
    )
    expected = {"name": "RTX 5060", "uuid_sha256": hashlib.sha256(b"fixture").hexdigest()}
    result = frontend.inspect(expected)
    assert result["python_version"] and result["library_versions"]["numpy"]
    with pytest.raises(srnet_cuda.CUDAUnavailable, match="mismatch"):
        frontend.inspect({**expected, "uuid_sha256": SHA})
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace())
    with pytest.raises(srnet_cuda.CUDAUnavailable, match="identity_unavailable"):
        frontend.inspect(expected)


def test_adapter_cli_and_worker_routing(tiny, tmp_path, monkeypatch, capsys):
    import cli

    value = config(tmp_path / "data")
    monkeypatch.setattr(frontend, "sources", lambda: SOURCES)
    monkeypatch.setattr(frontend, "inspect", lambda _: EXECUTION)
    monkeypatch.setattr(frontend.resource, "setrlimit", lambda *args: None)
    monkeypatch.setattr(
        frontend.training, "train", lambda *args, **kwargs: {"status": "checkpointed"}
    )
    result = frontend.worker(
        "train",
        {
            "config": value,
            "parent": None,
            "task": ["warmup", 20261012, 512],
            "binding": SHA,
            "execution": EXECUTION,
        },
        tmp_path,
    )
    assert result["status"] == "checkpointed" and result["source_sha256"] == SOURCES
    monkeypatch.setattr(frontend, "run", lambda *args: {"status": "completed", "reason": None})
    args = SimpleNamespace(config="unused", out="fresh")
    assert cli.cmd_research_jpeg_refine(args) == 0
    assert (
        cli.main(["--quiet", "research", "jpeg-refine", "--config", "unused", "--out", "fresh"])
        == 0
    )
    assert frontend.main(["--config", "unused", "--out", "fresh"]) == 0
    assert '"completed"' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        frontend.main(["--out", "fresh"])
    monkeypatch.setattr(
        frontend, "run", lambda *args: {"status": "unavailable", "reason": "fixture"}
    )
    assert cli.cmd_research_jpeg_refine(args) == 2
    assert frontend.main(["--config", "unused", "--out", "fresh"]) == 2


@pytest.mark.parametrize(
    "exception,reason",
    [
        (ImportError, "optional_dependency_unavailable"),
        (TimeoutError, "worker_interrupted_or_timeout"),
        (ValueError, "worker_contract_failed"),
        (RuntimeError, "worker_runtime_failed"),
    ],
)
def test_worker_failure_metadata_is_sanitized(tmp_path, monkeypatch, exception, reason):
    if exception is RuntimeError:
        pytest.importorskip("torch")
    monkeypatch.setattr(frontend, "sources", lambda: SOURCES)
    monkeypatch.setattr(
        frontend, "inspect", lambda _: (_ for _ in ()).throw(exception("private-path"))
    )
    monkeypatch.setattr(frontend.resource, "setrlimit", lambda *args: None)
    result = frontend.worker("audit", {"config": config(tmp_path)}, tmp_path)
    assert result == {"status": "unavailable", "reason": reason}
    assert "private-path" not in json.dumps(result)


def test_prepared_manifest_audit_validation(tmp_path, monkeypatch):
    value = config(tmp_path)
    manifest = {
        "schema_version": c.views.SCHEMA,
        "originals_per_source": {"fit": 256, "development": 64},
        "samples": {"fit": rows(), "development": rows("development", 64)},
    }
    monkeypatch.setattr(data := c.data, "validate_manifest", lambda value: None)
    value["manifest_sha256"] = write_document(tmp_path / "manifest.json", manifest)
    proof = {
        "schema_version": "jpeg-study-independent-audit-v2",
        "status": "completed",
        "manifest_sha256": value["manifest_sha256"],
        "representation": c.views.contract(),
        "idct_rows": {r: len(v) for r, v in manifest["samples"].items()},
        "idct_views": {r: 5 * len(v) for r, v in manifest["samples"].items()},
    }
    value["audit_sha256"] = write_document(tmp_path / "audit.json", proof)
    value["root"] = tmp_path
    assert c.prepared(value) == manifest
    for key, bad in (("idct_rows", {}), ("status", "unavailable"), ("representation", {})):
        failed = copy.deepcopy(proof)
        failed[key] = bad
        path = tmp_path / f"{key}.json"
        altered = {**value, "audit": str(path), "audit_sha256": write_document(path, failed)}
        with pytest.raises(ValueError):
            c.prepared(altered)
    assert data is c.data
