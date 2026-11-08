"""Generated CPU SRNet QA and emulated CUDA policy; no physical GPU result."""

import copy
import dataclasses
import hashlib
import os
import random
import resource
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from core import colab_artifacts as a
from core import colab_execution as c
from core import spatial_colab_profile as p
from core import spatial_native_data as data
from core import spatial_native_pilot as pilot
from core import spatial_native_training as engine
from steganography import research_colab_profile as cli

torch = pytest.importorskip("torch")
ROOT = Path(__file__).parent.parent
EXPECTED = {"python": "3.12.13", "numpy": "2.0.2", "pillow": "11.3.0",
            "torch": "2.11.0+cu128", "cuda": "12.8", "gpu": "Tesla T4", "driver": "580.82.07"}


def limits():
    return c.Limits(240, 0, 0, 256 * a.MIB, 128 * a.MIB,
                    closing_seconds=10)


def payload():
    split = a.document(ROOT / pilot.PROPOSAL)["split"]  # Public metadata only.
    return {"schema": p.PROFILE_SCHEMA, "split": split, "recipe": copy.deepcopy(p.RECIPE),
            "expected_environment": dict(EXPECTED)}


@pytest.fixture
def metadata(tmp_path):
    folder = tmp_path / "metadata-only"
    folder.mkdir()
    raw = a.read(ROOT / "benchmarks/png-bmp-native-cache-manifest-20261028.json", data.MAX_MANIFEST)
    (folder / "cache.json").write_bytes(raw)
    cache = data.load(folder, a.digest(raw))
    return cache, engine.split_document(payload()["split"])


def test_frozen_proposal_and_metadata_graph_without_pixels(metadata):
    cache, split = metadata
    assert p.contract(payload(), limits()) == split
    p.cache_binding(cache, split)
    assert [path.name for path in cache.root.iterdir()] == ["cache.json"]
    assert all(row.role == "unassigned" for row in cache.rows)


@pytest.mark.parametrize("fault", ["schema", "recipe", "counts", "split", "environment",
                                   "driver", "limits"])
def test_altered_source_data_environment_or_limits_rejected(fault):
    value, bound = payload(), limits()
    if fault == "schema":
        value["schema"] = "old"
    elif fault == "recipe":
        value["recipe"]["native_optimizer_updates"] = 1
    elif fault == "counts":
        value["split"]["train"].pop()
    elif fault == "split":
        value["split"]["seed"] += 1
    elif fault == "environment":
        value["expected_environment"]["torch"] = "cpu"
    elif fault == "driver":
        value["expected_environment"]["driver"] = "/private/path"
    else:
        bound = dataclasses.replace(bound, address_bytes=1024**3)
    with pytest.raises(ValueError):
        p.contract(value, bound)


def test_wrong_graph_rejected(metadata):
    cache, split = metadata
    rows = tuple(dataclasses.replace(row, group=row.id, author=row.id) for row in cache.rows)
    with pytest.raises(ValueError, match="ancestry"):
        p.cache_binding(data.Cache(cache.root, rows, cache.manifest_sha256), split)


def test_driver_query_is_bounded_file_read(tmp_path, monkeypatch):
    path = tmp_path / "driver"
    path.write_text("NVRM version: NVIDIA UNIX Open Kernel Module  580.82.07  Fixture\n")
    assert p.driver_version(path) == "580.82.07"
    original = p.driver_version
    monkeypatch.setattr(p, "driver_version", lambda: "580.82.07")
    assert p.environment()["driver"] == "580.82.07"
    path.write_text("bad")
    with pytest.raises(ValueError):
        original(path)


@pytest.fixture
def cuda(monkeypatch):
    state = {"fraction": 0.9, "deterministic": False, "warn": True}
    module = NS(__version__="emulated", version=NS(cuda="12.8"),
                cuda=NS(is_available=lambda: True, get_device_capability=lambda _: (7, 5),
                        get_arch_list=lambda: ["sm_75"], memory_reserved=lambda _: 0,
                        get_device_properties=lambda _: NS(total_memory=16 * 1024**3),
                        get_per_process_memory_fraction=lambda _: state["fraction"],
                        set_per_process_memory_fraction=lambda n, _: state.update(fraction=n),
                        device=lambda _: nullcontext()),
                backends=NS(cuda=NS(matmul=NS(fp32_precision="none")),
                            cudnn=NS(conv=NS(fp32_precision="tf32"), benchmark=True,
                                     deterministic=False)),
                are_deterministic_algorithms_enabled=lambda: state["deterministic"],
                is_deterministic_algorithms_warn_only_enabled=lambda: state["warn"],
                use_deterministic_algorithms=lambda v, warn_only: state.update(
                    deterministic=v, warn=warn_only), autocast=lambda **kw: nullcontext())
    monkeypatch.setitem(sys.modules, "torch", module)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return module, state


def test_cuda_policy_scoped_precision_allocator_and_failure_restoration(cuda):
    module, state = cuda
    before = dict(state)
    for fail in (False, True):
        try:
            with p.cuda_policy() as info:
                assert info["hard_cgroup_ram_or_swap_enforced"] is False
                assert module.backends.cuda.matmul.fp32_precision == "ieee"
                assert module.backends.cudnn.conv.fp32_precision == "ieee"
                assert state == {"fraction": 0.25, "deterministic": True, "warn": False}
                if fail:
                    raise ValueError("generated exception")
        except ValueError:
            assert fail
        assert state == before
        assert module.backends.cuda.matmul.fp32_precision == "none"
        assert module.backends.cudnn.conv.fp32_precision == "tf32"


@pytest.mark.parametrize("fault", ["missing", "cpu", "workspace", "precision", "arch", "vram",
                                   "reserved"])
def test_cuda_prerequisites_fail_closed(cuda, monkeypatch, fault):
    module, state = cuda
    before = dict(state)
    if fault == "missing":
        module.cuda.is_available = lambda: False
    elif fault == "cpu":
        module.version.cuda = None
    elif fault == "workspace":
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
    elif fault == "precision":
        del module.backends.cudnn.conv.fp32_precision
    elif fault == "arch":
        module.cuda.get_arch_list = lambda: ["sm_90"]
    elif fault == "vram":
        module.cuda.get_device_properties = lambda _: NS(total_memory=1024**3)
    else:
        module.cuda.memory_reserved = lambda _: 5 * 1024**3
    with pytest.raises((RuntimeError, ValueError)), p.cuda_policy():
        pass
    assert state == before


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    root = tmp_path_factory.mktemp("colab-rgb-generated")
    inputs = root / "inputs"
    inputs.mkdir()
    cache, split = pilot.generated_cache(inputs, "RGB", 1)
    result = p.measure(cache, split, root / "measurement", "b" * 64, device="cpu", seconds=220)
    return root, cache, split, result


def test_actual_generated_cpu_gradients_steps_and_restores(measured):
    root, cache, split, result = measured
    assert result["schema"] == "spatial-colab-native-measurement-v1"
    assert not result["physical_cuda"] and result["device"] == "cpu"
    assert result["native_optimizer_updates"] == 0 and result["generated_optimizer_updates"] == 6
    assert result["native_evaluation_presentations"] == 4
    assert all(cell["native_parameters_unchanged"] for cell in result["cells"])
    for index, cell in enumerate(result["cells"]):
        checkpoint = root / "measurement" / f"generated-{index}.numeric"
        assert checkpoint.stat().st_size == cell["checkpoint_bytes"]
        assert checkpoint.stat().st_size <= 128 * a.MIB
        with checkpoint.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == cell["checkpoint_sha256"]
    projected = pilot.projection({**result, "schema": "spatial-rgb-native-measurement-v1"})
    assert projected["approval_required"] and not projected["native_accuracy_inferred"]


def test_measurement_device_deadline_and_overwrite(measured, tmp_path):
    root, cache, split, _ = measured
    for device, seconds in (("cuda:1", 1), ("cpu", 0), ("cpu", float("nan"))):
        with pytest.raises(ValueError):
            p.measure(cache, split, tmp_path / "bad", "b" * 64, device=device, seconds=seconds)
    with pytest.raises(FileExistsError):
        p.measure(cache, split, root / "measurement", "b" * 64, device="cpu")


def test_native_permission_rejected_before_intent_or_cache_access(tmp_path):
    root = tmp_path / "root"
    (root / "core").mkdir(parents=True)
    (root / "core" / "fixture.py").write_text("# generated fixture\n")
    protocol = root / "protocol.json"
    job = c.freeze(root, protocol, "native_profile", payload(), limits())
    assert job["native_execution_available"]
    assert c.verify(root, protocol)[0] == job
    with pytest.raises(ValueError, match="permission"):
        c.run(root, protocol, root / "output")
    assert not (root / ".colab-process-attempts").exists()
    assert not (root / "output").exists()
    with pytest.raises(ValueError):
        c.freeze(root, root / "fit.json", "native_fit", payload(), limits())


def test_permission_schema_hash_phase_and_resource_ack(tmp_path):
    path = tmp_path / "fixture-permission.json"
    fixture = {"schema": "colab-profile-permission-v1", "user_approved": True,
               "contract_sha256": "a" * 64, "kind": "native_profile",
               "resource_acknowledgement": p.RESOURCE_ACK}
    path.write_bytes(a.encode(fixture))
    c.permission({"kind": "native_profile"}, "a" * 64, path)
    for key, value in (("user_approved", False), ("user_approved", 1),
                       ("contract_sha256", "b" * 64),
                       ("kind", "native_fit"), ("resource_acknowledgement", "cgroup_enforced")):
        changed = dict(fixture)
        changed[key] = value
        path.write_bytes(a.encode(changed))
        with pytest.raises(ValueError):
            c.permission({"kind": "native_profile"}, "a" * 64, path)


def test_worker_entry_hard_limits_and_environment_before_model(metadata, monkeypatch, tmp_path):
    cache, split = metadata
    job = {"payload": payload(), "limits": dataclasses.asdict(limits())}
    monkeypatch.setenv("STEGANOGRAPHY_COLAB_CACHE", str(cache.root))
    monkeypatch.setattr(p.resource, "getrlimit", lambda key: (-1, -1))
    with pytest.raises(ValueError, match="limits"):
        p.execute(job, "a" * 64, tmp_path)
    bound = {resource.RLIMIT_DATA: 8 * 1024**3, resource.RLIMIT_CPU: 240,
             resource.RLIMIT_FSIZE: 128 * a.MIB}
    monkeypatch.setattr(p.resource, "getrlimit", lambda key: (bound[key], bound[key]))
    monkeypatch.setattr(p, "environment", lambda: {**EXPECTED, "torch": "wrong"})
    with pytest.raises(ValueError, match="environment"):
        p.execute(job, "a" * 64, tmp_path)


def test_cli_freeze_does_not_create_permission_or_execute(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "steganography" / "cli.py"))
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "fixture.py").write_text("# fixture\n")
    env = tmp_path / "environment.json"
    env.write_bytes(a.encode(EXPECTED))
    protocol = tmp_path / "protocol.json"
    args = ["--protocol", str(protocol)]
    assert cli.main(["freeze", *args, "--environment", str(env), "--proposal",
                     str(ROOT / pilot.PROPOSAL)]) == 0
    assert cli.main(["profile", *args, "--out", str(tmp_path / "output")]) == 2
    assert cli.main(["audit", *args]) == 2
    assert cli.main(["freeze", *args]) == 2
    assert cli.main(["freeze", *args, "--permission", str(env)]) == 2
    assert "profile_only_permission_missing" in capsys.readouterr().out


def test_cuda_session_uses_same_engine_and_restores_actual_cpu_rng(metadata, monkeypatch):
    cache, split = metadata
    plan = engine.Plan(split, 5, 6, 1, 8, "a" * 64)
    before = random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    old_fork = torch.random.fork_rng
    monkeypatch.setattr(torch.random, "fork_rng", lambda **kw: old_fork(devices=[]))
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda _: None)
    monkeypatch.setattr(p, "cuda_policy", lambda: nullcontext())
    marker = object()
    monkeypatch.setattr(p, "_ProfileSession", lambda *args: marker)
    with p.session(cache, plan, device="cuda:0", max_seconds=20) as value:
        assert value is marker and torch.get_num_threads() == 2
        random.random()  # noqa: S311 -- explicit RNG restoration QA
        np.random.rand()
        torch.rand(3)
    assert random.getstate() == before[0]
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert torch.equal(torch.random.get_rng_state(), before[2])
    assert torch.get_num_threads() == threads
    for device, seconds in (("cuda:1", 1), ("cuda:0", 0)):
        with pytest.raises(ValueError), p.session(cache, plan, device=device, max_seconds=seconds):
            pass


def test_worker_entry_success_uses_bound_metadata_without_opening_real_pixels(
    metadata, monkeypatch, tmp_path, measured,
):
    cache, _ = metadata
    fixture_result = copy.deepcopy(measured[-1])
    job = {"payload": payload(), "limits": dataclasses.asdict(limits())}
    monkeypatch.setenv("STEGANOGRAPHY_COLAB_CACHE", str(cache.root))
    maximum = {resource.RLIMIT_DATA: 8 * 1024**3, resource.RLIMIT_CPU: 240,
               resource.RLIMIT_FSIZE: 128 * a.MIB}
    monkeypatch.setattr(p.resource, "getrlimit", lambda key: (maximum[key], maximum[key]))
    monkeypatch.setattr(p, "environment", lambda: {**EXPECTED, "cudnn": 90100})
    monkeypatch.setattr(p, "measure", lambda *args, **kwargs: fixture_result)
    result = p.execute(job, "a" * 64, tmp_path)
    assert result["native_optimizer_updates"] == 0
    assert result["generated_optimizer_updates"] == 6
    assert not result["native_fit_authorized"] and not result["future_reserved_opened"]
    assert not result["hard_cgroup_ram_or_swap_enforced"]
    assert result["environment"]["cudnn"] == 90100
    assert (tmp_path / "environment-before-model.json").exists()
    assert len(list(cache.root.iterdir())) == 1  # Only the public metadata manifest exists.


def test_nonfinite_profile_and_changed_native_parameters_are_rejected(
    measured, monkeypatch, tmp_path,
):
    _, cache, split, _ = measured
    original = torch.isfinite
    monkeypatch.setattr(torch, "isfinite", lambda value: torch.tensor(False))
    with pytest.raises(ValueError, match="nonfinite"):
        p.measure(cache, split, tmp_path / "nonfinite", "b" * 64, device="cpu")
    monkeypatch.setattr(torch, "isfinite", original)
    counter = iter(["before", "changed"])
    monkeypatch.setattr(pilot, "parameters", lambda _: next(counter))
    with pytest.raises(ValueError, match="changed parameters"):
        p.measure(cache, split, tmp_path / "changed", "b" * 64, device="cpu")


def test_restored_generated_consumption_mismatch_rejected(measured, monkeypatch, tmp_path):
    _, cache, split, _ = measured
    original = engine._Session.resume

    def corrupt(self, path, sha):
        original(self, path, sha)
        self.updates -= 1

    monkeypatch.setattr(engine._Session, "resume", corrupt)
    with pytest.raises(ValueError, match="restore mismatch"):
        p.measure(cache, split, tmp_path / "restore", "b" * 64, device="cpu")


def test_parent_dispatch_environment_with_generated_stub_only(metadata, monkeypatch, tmp_path):
    cache, _ = metadata
    root = tmp_path / "source"
    (root / "core").mkdir(parents=True)
    (root / "core" / "fixture.py").write_text("# owned fixture\n")
    protocol = root / "protocol.json"
    c.freeze(root, protocol, "native_profile", payload(), limits())
    binding = a.digest(protocol.read_bytes())
    fixture_permission = root / "fixture-only-permission.json"
    fixture_permission.write_bytes(a.encode({
        "schema": "colab-profile-permission-v1", "user_approved": True,
        "contract_sha256": binding, "kind": "native_profile",
        "resource_acknowledgement": p.RESOURCE_ACK,
    }))
    with pytest.raises(ValueError, match="cache"):
        c.run(root, protocol, root / "missing-cache", permission_path=fixture_permission)
    stub = (
        "import sys,json,hashlib,os;from pathlib import Path;"
        "root,protocol,out,parent=sys.argv[1:];out=Path(out);(out/'work').mkdir();"
        "assert os.environ['CUDA_VISIBLE_DEVICES']=='0';"
        "assert os.environ['STEGANOGRAPHY_COLAB_PERMISSION'];"
        "assert os.environ['STEGANOGRAPHY_COLAB_CACHE'];"
        "r={'contract_sha256':hashlib.sha256(Path(protocol).read_bytes()).hexdigest(),"
        "'emulated_dispatch_only':True};"
        "(out/'work'/'worker-result.json').write_text(json.dumps(r))"
    )
    monkeypatch.setattr(c, "BOOTSTRAP", stub)
    out = root / "output"
    result = c.run(root, protocol, out, permission_path=fixture_permission, cache=cache.root)
    assert result["status"] == "unavailable"  # A stub cannot pass physical/numeric acceptance.
    assert result["worker_reaped"]
    assert c.audit(root, protocol, out) == result


def test_registered_worker_dispatch_is_emulated_not_physical(metadata, monkeypatch, tmp_path):
    root = tmp_path / "source"
    (root / "core").mkdir(parents=True)
    (root / "core" / "fixture.py").write_text("# fixture\n")
    protocol = root / "protocol.json"
    c.freeze(root, protocol, "native_profile", payload(), limits())
    binding = a.digest(protocol.read_bytes())
    permit = root / "fixture-only-permission.json"
    permit.write_bytes(a.encode({"schema": "colab-profile-permission-v1", "user_approved": True,
                                "kind": "native_profile", "contract_sha256": binding,
                                "resource_acknowledgement": p.RESOURCE_ACK}))
    monkeypatch.setenv("STEGANOGRAPHY_COLAB_PERMISSION", str(permit))
    monkeypatch.setattr(c, "__file__", str(root / "core" / "colab_execution.py"))
    monkeypatch.setattr(c.ctypes, "CDLL", lambda *args, **kwargs: NS(prctl=lambda *args: 0))
    monkeypatch.setattr(c.signal, "signal", lambda *args: None)
    monkeypatch.setattr(c.signal, "setitimer", lambda *args: None)
    monkeypatch.setattr(p, "execute", lambda *args: {"contract_sha256": binding, "emulated": True})
    out = root / "output"
    out.mkdir()
    assert c.worker_main([str(protocol), str(out), str(os.getppid())]) == 0
    assert a.document(out / "work" / "worker-result.json")["emulated"]


def test_generated_qa_may_not_consume_native_permission_or_cache(tmp_path):
    root = tmp_path / "root"
    (root / "core").mkdir(parents=True)
    (root / "core" / "fixture.py").touch()
    protocol = root / "protocol.json"
    fixed = dataclasses.replace(limits(), data_bytes=128 * a.MIB,
                                resident_watch_bytes=8 * 1024**3)
    c.freeze(root, protocol, "generated_qa", {"action": "success", "value": 0}, fixed)
    with pytest.raises(ValueError, match="permission"):
        c.run(root, protocol, root / "output", permission_path=Path("unused"))
    with pytest.raises(ValueError, match="cache"):
        c.run(root, protocol, root / "output", cache=Path("unused"))


def test_cli_profile_and_audit_delegate_without_training(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(c, "run", lambda *args, **kwargs: {"status": "completed"})
    monkeypatch.setattr(c, "audit", lambda *args, **kwargs: {"status": "completed"})
    common = ["--protocol", "fixture", "--out", str(tmp_path)]
    assert cli.main(["profile", *common]) == 0
    assert cli.main(["audit", *common]) == 0
    assert "completed" in capsys.readouterr().out


def test_native_profile_view_rejects_steps_checkpoints_and_nontrain_access(measured, tmp_path):
    _, cache, split, _ = measured
    plan = engine.Plan(split, 5, 6, 1, 2, "b" * 64)
    with p.session(cache, plan, max_seconds=30) as value:
        before = pilot.parameters(value.model)
        with pytest.raises(ValueError, match="optimizer"):
            value.step()
        with pytest.raises(ValueError, match="generated"):
            value.checkpoint(tmp_path / "forbidden.numeric")
        for role, scores in (("development", False), ("future_reserved", False), ("train", True)):
            with pytest.raises(ValueError, match="train timing"):
                value.scores(role, record_scores=scores)
        assert value.updates == 0 and not value.optimizer.state
        assert pilot.parameters(value.model) == before
    assert not (tmp_path / "forbidden.numeric").exists()


def test_real_namespace_cannot_request_generated_optimizer(metadata):
    cache, split = metadata
    plan = engine.Plan(split, 5, 6, 1, 8, "b" * 64)
    with pytest.raises(ValueError, match="generated profile"), p.session(
        cache, plan, max_seconds=20, generated_optimizer=True,
    ):
        pass


def test_nonfinite_timing_restores_model_mode(measured, monkeypatch):
    _, cache, split, _ = measured
    plan = engine.Plan(split, 5, 6, 1, 2, "b" * 64)
    with p.session(cache, plan, max_seconds=20) as value:
        monkeypatch.setattr(np, "isfinite", lambda _: np.asarray(False))
        with pytest.raises(ValueError, match="nonfinite"):
            value.scores("train")
        assert value.model.training


def test_independent_numeric_and_step_replay_from_generated_cpu_states(measured):
    root, _, _, result = measured
    report = p.audit_measurement(root / "measurement", result, "b" * 64,
                                 engine.capacity.environment())
    assert report["numeric_checkpoints_verified"] == 2 and report["commits_verified"] == 6
    assert report["generated_optimizer_updates_verified"] == 6
    assert all(row["arrays_verified"] > 400 for row in report["cells"])
    assert report["native_pixels_opened_by_auditor"] is False


def test_tampered_generated_commit_rejected_independently(measured):
    root, _, _, result = measured
    path = root / "measurement" / "001-commit.json"
    raw = path.read_bytes()
    value = a.document(path)
    value["result"]["optimizer_updates"] = 2
    path.write_bytes(a.encode(value))
    try:
        with pytest.raises(ValueError, match="commit"):
            p.audit_measurement(root / "measurement", result, "b" * 64,
                                 engine.capacity.environment())
    finally:
        path.write_bytes(raw)


def test_prospective_environment_freeze_cannot_execute_locally(tmp_path):
    root = tmp_path / "source"
    (root / "core").mkdir(parents=True)
    (root / "core" / "fixture.py").touch()
    target = {"python": "3.12.13", "implementation": "CPython", "platform": "linux"}
    protocol = root / "future.json"
    value = c.freeze(root, protocol, "native_profile", payload(), limits(),
                     target_environment=target)
    assert value["environment"] == target
    if c.environment() != target:
        with pytest.raises(ValueError, match="environment"):
            c.verify(root, protocol)
    for kind, environment in (("generated_qa", target), ("native_profile", {})):
        bound = limits() if kind == "native_profile" else dataclasses.replace(
            limits(), data_bytes=128 * a.MIB, resident_watch_bytes=8 * 1024**3,
        )
        with pytest.raises(ValueError, match="prospective"):
            c.freeze(root, root / "bad.json", kind, payload(), bound,
                     target_environment=environment)


def test_cli_cache_copy_requires_exact_permission_and_unassigned_metadata(
    metadata, tmp_path, monkeypatch, capsys,
):
    cache, _ = metadata
    root = tmp_path / "root"
    (root / "core").mkdir(parents=True)
    (root / "core" / "fixture.py").touch()
    monkeypatch.setattr(cli, "__file__", str(root / "steganography" / "cli.py"))
    protocol = root / "protocol.json"
    c.freeze(root, protocol, "native_profile", payload(), limits())
    binding = a.digest(protocol.read_bytes())
    permit = root / "fixture-only-permission.json"
    permit.write_bytes(a.encode({"schema": "colab-profile-permission-v1", "user_approved": True,
                                "kind": "native_profile", "contract_sha256": binding,
                                "resource_acknowledgement": p.RESOURCE_ACK}))
    common = ["cache-package", "--protocol", str(protocol), "--out", str(root / "packet.json")]
    assert cli.main(common) == 2
    assert cli.main([*common, "--cache", str(cache.root)]) == 2
    seen = []

    def copy_fixture(actual_cache, actual_split, target):
        assert all(row.role == "unassigned" for row in actual_cache.rows)
        assert len(actual_split.train) == 64
        seen.append(target)
        return "c" * 64

    monkeypatch.setattr(cli.handoff, "cache_package", copy_fixture)
    assert cli.main([*common, "--cache", str(cache.root), "--permission", str(permit)]) == 0
    assert len(seen) == 1 and "packaged_train_only" in capsys.readouterr().out


def test_cli_rejects_approved_or_reassigned_proposal(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "core").mkdir(parents=True)
    (root / "core" / "fixture.py").touch()
    monkeypatch.setattr(cli, "__file__", str(root / "steganography" / "cli.py"))
    proposal = a.document(ROOT / pilot.PROPOSAL)
    proposal["native_roles_applied"] = True
    path = root / "bad-proposal.json"
    path.write_bytes(a.encode(proposal))
    env = root / "env.json"
    env.write_bytes(a.encode(EXPECTED))
    assert cli.main(["freeze", "--protocol", str(root / "out.json"), "--proposal", str(path),
                     "--environment", str(env)]) == 2


def test_parent_closing_orchestration_with_explicit_emulation(metadata, monkeypatch, tmp_path):
    cache, _ = metadata
    root = tmp_path / "root"
    (root / "core").mkdir(parents=True)
    (root / "core" / "fixture.py").write_text("# generated dispatch fixture\n")
    protocol = root / "protocol.json"
    c.freeze(root, protocol, "native_profile", payload(), limits())
    binding = a.digest(protocol.read_bytes())
    permit = root / "fixture-only-permission.json"
    permit.write_bytes(a.encode({"schema": "colab-profile-permission-v1", "user_approved": True,
                                "kind": "native_profile", "contract_sha256": binding,
                                "resource_acknowledgement": p.RESOURCE_ACK}))
    stub = (
        "import sys,json,hashlib;from pathlib import Path;"
        "root,protocol,out,parent=sys.argv[1:];out=Path(out);"
        "(out/'work'/'profile').mkdir(parents=True);"
        "job=json.loads(Path(protocol).read_text());"
        "r={'contract_sha256':hashlib.sha256(Path(protocol).read_bytes()).hexdigest(),"
        "'physical_cuda':True,'split':job['payload']['split'],'environment':{},"
        "'emulated_dispatch_only':True};"
        "(out/'work'/'worker-result.json').write_text(json.dumps(r))"
    )
    monkeypatch.setattr(c, "BOOTSTRAP", stub)
    seen = []

    def emulated_numeric_audit(folder, result, actual_binding, observed):
        assert result["emulated_dispatch_only"] and actual_binding == binding
        seen.append(folder)
        a.put(folder.parent / "independent-numeric-audit.json", a.encode({"emulation_only": True}))

    monkeypatch.setattr(p, "audit_measurement", emulated_numeric_audit)
    output = root / "output"
    result = c.run(root, protocol, output, permission_path=permit, cache=cache.root)
    assert result["status"] == "completed" and result["worker"]["emulated_dispatch_only"]
    assert seen and "verified-profile.zip" in result["output_files"]
    assert c.audit(root, protocol, output) == result
