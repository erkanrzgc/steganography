"""Generated CPU and explicit emulation only; no physical profile or native fitting."""

import copy
import dataclasses
import json
import resource
from pathlib import Path

import numpy as np
import pytest

from core import colab_artifacts as a
from core import colab_execution as c
from core import spatial_colab_fit as f
from core import spatial_colab_handoff as h
from core import spatial_colab_profile as p
from core import spatial_native_data as data
from core import spatial_native_training as engine
from steganography import research_colab_train as cli

torch = pytest.importorskip("torch")
ROOT = Path(__file__).parent.parent
DEBIT = {
    "schema": "colab-fit-external-debit-v1",
    "seconds": 3.0,
    "output_bytes": 1000,
    "complete": True,
}
ENV = {
    "python": "3.12.13",
    "numpy": "2.0.2",
    "pillow": "11.3.0",
    "torch": "2.11.0+cu128",
    "cuda": "12.8",
    "cudnn": 90100,
    "gpu": "Tesla T4",
    "driver": "580.82.07",
}


@pytest.fixture
def proposed(tmp_path, monkeypatch):
    root = tmp_path / "source"
    (root / "core").mkdir(parents=True)
    (root / "core/fixture.py").write_text("# generated fixture only\n")
    out = tmp_path / "emulated-profile"
    out.mkdir()
    (out / "closed.json").write_text("{}\n")
    split = a.document(ROOT / p.pilot.PROPOSAL)["split"]
    cells = [
        {
            "seed": seed,
            "native_forward_backward_seconds": [0.1, 0.1, 0.1],
            "generated_step_seconds": [0.1, 0.1, 0.1],
            "native_train_evaluation": {"max_pair_seconds": 0.001},
            "native_initialization_seconds": 0.01,
            "generated_initialization_seconds": 0.01,
            "checkpoint_write_seconds": 0.01,
            "checkpoint_restore_seconds": 0.01,
            "native_optimizer_updates": 0,
            "generated_optimizer_updates": 3,
            "native_parameters_unchanged": True,
            "checkpoint_bytes": 58 * a.MIB,
        }
        for seed in p.pilot.SEEDS
    ]
    observed = {
        "status": "completed",
        "kind": "native_profile",
        "contract_sha256": "a" * 64,
        "charged_seconds": 10.0,
        "output_files": {"generated.numeric": {"bytes": 116 * a.MIB}},
        "worker": {
            "schema": "spatial-colab-native-measurement-v1",
            "split": split,
            "recipe": p.RECIPE,
            "physical_cuda": True,
            "environment": ENV,
            "native_optimizer_updates": 0,
            "generated_optimizer_updates": 6,
            "cells": cells,
        },
    }
    monkeypatch.setattr(c, "audit", lambda *args: copy.deepcopy(observed))
    payload, limits = f.proposal(root, out, DEBIT)
    return root, out, payload, limits, observed


def fixture_permission(root, protocol, payload):
    path = root / "fixture-only-permission.json"
    path.write_bytes(
        a.encode(
            {
                "schema": "colab-fit-permission-v1",
                "user_approved": True,
                "kind": "native_fit",
                "contract_sha256": a.digest(protocol.read_bytes()),
                "profile_receipt_sha256": payload["profile_receipt_sha256"],
                "resource_acknowledgement": p.RESOURCE_ACK,
            }
        )
    )
    return path


def test_source_bound_proposal_has_no_permission_or_pixel_reads(proposed):
    root, out, payload, limits, _ = proposed
    assert f.contract(payload, limits).mode == "RGB"
    assert limits.data_bytes == limits.resident_watch_bytes == 0
    assert limits.seconds == 1787
    assert (
        payload["projection_seconds"]
        > p.pilot.projection(
            {**proposed[-1]["worker"], "schema": "spatial-rgb-native-measurement-v1"}
        )["fit_projection_seconds"]
    )
    target = root / "prospective-fit.json"
    f.freeze(root, out, DEBIT, target)
    job, binding, _ = c.source_contract(root, target)
    assert job["kind"] == "native_fit"
    assert not list(root.glob("*permission*"))
    assert job["environment"]["python"] == "3.12.13"
    f.bind_profile(root, job, out)
    job["payload"]["prior_seconds"] += 1
    with pytest.raises(ValueError, match="binding"):
        f.bind_profile(root, job, out)


@pytest.mark.parametrize(
    "change",
    [
        "recipe",
        "fields",
        "environment",
        "hash",
        "time",
        "output",
        "projection",
        "checkpoint",
        "limits",
        "debit",
    ],
)
def test_tampered_contracts_rejected(proposed, change):
    _, _, value, limits, _ = proposed
    value = copy.deepcopy(value)
    if change == "recipe":
        value["recipe"]["epochs"] = 21
    elif change == "fields":
        value["unexpected"] = 1
    elif change == "environment":
        value["environment"]["cudnn"] = None
    elif change == "hash":
        value["profile_receipt_sha256"] = "bad"
    elif change == "time":
        value["prior_seconds"] = float("nan")
    elif change == "output":
        value["prior_output_bytes"] = True
    elif change == "projection":
        value["projection_seconds"] = 1800
    elif change == "checkpoint":
        value["max_checkpoint_bytes"] = 128 * a.MIB
    elif change == "limits":
        limits = dataclasses.replace(
            limits, data_bytes=8 * 1024**3, resident_watch_bytes=8 * 1024**3
        )
    else:
        value["setup_debit"]["complete"] = False
    with pytest.raises(ValueError):
        f.contract(value, limits)


@pytest.mark.parametrize(
    "value",
    [
        None,
        {"seconds": 1},
        {**DEBIT, "complete": 1},
        {**DEBIT, "seconds": -1},
        {**DEBIT, "output_bytes": True},
    ],
)
def test_external_debit_not_invented(value):
    with pytest.raises((TypeError, ValueError)):
        f.setup_debit(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "unavailable"),
        ("kind", "generated_qa"),
        ("physical_cuda", False),
        ("native_optimizer_updates", 1),
        ("generated_optimizer_updates", 5),
        ("recipe", {}),
    ],
)
def test_incomplete_or_nonphysical_profile_never_allows_fit(proposed, monkeypatch, field, value):
    root, out, _, _, observed = proposed
    observed = copy.deepcopy(observed)
    (observed if field in ("status", "kind") else observed["worker"])[field] = value
    monkeypatch.setattr(c, "audit", lambda *args: observed)
    with pytest.raises(ValueError, match="physical"):
        f.accepted_profile(root, out)


@pytest.mark.parametrize(
    "fault", ["missing", "profile_permission", "boolean", "profile_hash", "contract"]
)
def test_separate_exact_fit_permission_required(proposed, fault):
    root, _, payload, limits, _ = proposed
    protocol = root / "job.json"
    c.freeze(root, protocol, "native_fit", payload, limits)
    job, binding, _ = c.verify(root, protocol)
    path = fixture_permission(root, protocol, payload)
    if fault == "missing":
        path = None
    else:
        value = a.document(path)
        if fault == "profile_permission":
            value["schema"] = "colab-profile-permission-v1"
        elif fault == "boolean":
            value["user_approved"] = 1
        elif fault == "profile_hash":
            value["profile_receipt_sha256"] = "0" * 64
        else:
            value["contract_sha256"] = "0" * 64
        path.write_bytes(a.encode(value))
    with pytest.raises(ValueError, match="permission"):
        c.permission(job, binding, path)


@pytest.mark.parametrize(
    "pairs,expected",
    [([[0, 1], [0, 1]], 1), ([[1, 0]], 0), ([[0.5, 0.5]], 0.5), ([[0, 0.5], [0.5, 1]], 0.875)],
)
def test_rank_auc_matches_independent_tie_math(pairs, expected):
    assert f.auc(pairs) == expected


@pytest.mark.parametrize(
    "pairs", [[], [1, 2], [[0, 1, 2]], [[0, float("nan")]], [[0, 1.1]], [[0, 1]] * 129]
)
def test_auc_bounds(pairs):
    with pytest.raises(ValueError):
        f.auc(pairs)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("colab-fit-generated")
    inputs = root / "inputs"
    inputs.mkdir()
    cache, split = p.pilot.generated_cache(inputs, "RGB", 1)
    out = root / "fit"
    result = f.train(cache, split, out, "b" * 64, device="cpu", seconds=220, epochs=6)
    return root, cache, split, out, result


def test_real_cpu_training_telemetry_and_numeric_replay(trained):
    root, cache, split, out, result = trained
    assert result["physical_cuda"] is False
    assert result["fit_updates"] == 12 and result["presentations"] == 24
    assert all(len(row["epochs"]) == 6 and len(row["checkpoints"]) == 2 for row in result["cells"])
    report = f.audit_training(out, result, "b" * 64, engine.capacity.environment())
    assert report["commits_verified"] == 12 and report["numeric_states_verified"] == 4
    assert report["aucs_verified"] == 20 and report["native_pixels_opened"] is False
    for cell in result["cells"]:
        assert np.isfinite(cell["losses"]).all()
    assert len(list(out.glob("*-intent.json"))) == len(list(out.glob("*-commit.json"))) == 12
    assert a.document(out / "progress.json")["epoch"] == 6
    assert all(row.role == "unassigned" for row in cache.rows)
    assert (
        not result["qualified"]
        and not result["deployment"]
        and not result["future_reserved_opened"]
    )


@pytest.mark.parametrize(
    "fault",
    [
        "binding",
        "cells",
        "exposure",
        "epoch",
        "auc",
        "passed",
        "checkpoints",
        "name",
        "numeric",
        "status",
        "updates",
        "reserved",
        "final_dev",
    ],
)
def test_independent_auditor_rejects_tampering(trained, fault):
    _, _, _, out, original = trained
    value = copy.deepcopy(original)
    cell = value["cells"][0]
    if fault == "binding":
        value["contract_sha256"] = "c" * 64
    elif fault == "cells":
        value["cells"].reverse()
    elif fault == "exposure":
        cell["updates"] += 1
    elif fault == "epoch":
        cell["epochs"][0]["epoch"] += 1
    elif fault == "auc":
        cell["initial"]["train"]["auc"] = 0.123456
    elif fault == "passed":
        cell["passed"] = not cell["passed"]
    elif fault == "checkpoints":
        cell["checkpoints"].pop()
    elif fault == "name":
        cell["checkpoints"][0]["name"] = "../bad.numeric"
    elif fault == "numeric":
        cell["checkpoints"][0]["sha256"] = "0" * 64
    elif fault == "status":
        value["status"] = "passed"
    elif fault == "updates":
        value["fit_updates"] += 1
    elif fault == "reserved":
        value["future_reserved_opened"] = True
    else:
        cell["final"]["development"] = {"scores": [[0, 1]], "auc": 1.0, "originals": 1}
    with pytest.raises(ValueError):
        f.audit_training(out, value, "b" * 64, engine.capacity.environment())


@pytest.mark.parametrize("fault", ["intent", "cursor", "loss", "epoch_loss"])
def test_persisted_ledger_tampering_rejected(trained, fault):
    _, _, _, out, result = trained
    path = out / ("001-intent.json" if fault == "intent" else "001-commit.json")
    if fault == "epoch_loss":
        path = out / f"seed-{p.pilot.SEEDS[0]}-epoch-01.json"
    raw = path.read_bytes()
    value = json.loads(raw)
    if fault == "intent":
        value["cell"] = 1
    elif fault == "cursor":
        value["result"]["cursor"] += 1
    elif fault == "loss":
        value["result"]["loss"] += 1
    else:
        value["loss"] += 1
    path.write_bytes(a.encode(value))
    try:
        with pytest.raises(ValueError):
            f.audit_training(out, result, "b" * 64, engine.capacity.environment())
    finally:
        path.write_bytes(raw)


def test_generated_checkpoint_resume_and_closed_roles(trained, tmp_path):
    _, cache, split, out, result = trained
    plan = engine.Plan(split, p.pilot.SEEDS[0], f.RECIPE["payload_seed"], 6, 8, "b" * 64)
    with f.session(cache, plan, max_seconds=120) as session:
        checkpoint = result["cells"][0]["checkpoints"][0]
        session.resume(out / checkpoint["name"], checkpoint["sha256"])
        assert session.updates == 5
        assert session.step()["optimizer_updates"] == 6
        with pytest.raises(ValueError, match="scoring"):
            session.scores("future_reserved")
        before = session.model.training
        assert session.scores("development")["originals"] == 1
        assert session.model.training == before
    for device, seconds in [("cuda:1", 10), ("cpu", 0)]:
        with pytest.raises(ValueError), f.session(cache, plan, device=device, max_seconds=seconds):
            pass
    with pytest.raises(ValueError):
        f.train(cache, split, tmp_path / "bad", "b" * 64, device="cpu", epochs=0)
    with pytest.raises(ValueError):
        f.train(cache, split, tmp_path / "bad", "b" * 64, device="cuda:0", epochs=1)
    with pytest.raises(ValueError, match="checkpoint"):
        f.train(
            cache,
            split,
            tmp_path / "bound",
            "b" * 64,
            device="cpu",
            epochs=1,
            max_checkpoint_bytes=1,
        )


def test_worker_environment_and_limits_precede_model(proposed, tmp_path, monkeypatch):
    _, _, payload, limits, _ = proposed
    job = {"payload": payload, "limits": dataclasses.asdict(limits)}
    monkeypatch.setattr(f.profile, "environment", lambda: dict(ENV))
    monkeypatch.setattr(f.resource, "getrlimit", lambda key: (1, 1))
    monkeypatch.setenv("STEGANOGRAPHY_COLAB_CACHE", "fixture")
    monkeypatch.setattr(f.data, "load", lambda *args: object())
    monkeypatch.setattr(f, "train", lambda *args, **kwargs: {"physical_cuda": True})
    result = f.execute(job, "b" * 64, tmp_path)
    assert result["recipe"] == f.RECIPE and result["environment"] == ENV
    monkeypatch.setattr(f.profile, "environment", lambda: {**ENV, "driver": "changed"})
    with pytest.raises(ValueError, match="environment"):
        f.execute(job, "b" * 64, tmp_path)
    monkeypatch.setattr(f.resource, "getrlimit", lambda key: (resource.RLIM_INFINITY,) * 2)
    with pytest.raises(ValueError, match="limits"):
        f.execute(job, "b" * 64, tmp_path)


def test_cli_proposal_and_errors(proposed, monkeypatch, capsys):
    root, out, payload, limits, _ = proposed
    monkeypatch.setattr(cli, "__file__", str(root / "steganography/fixture.py"))
    debit = root / "debit.json"
    debit.write_bytes(a.encode(DEBIT))
    protocol = root / "fit.json"
    args = [
        "freeze-fit",
        "--protocol",
        str(protocol),
        "--profile-out",
        str(out),
        "--setup-debit",
        str(debit),
    ]
    assert cli.main(args) == 0
    assert "permission_missing" in capsys.readouterr().out
    assert cli.main(["freeze-fit", "--protocol", str(protocol)]) == 2
    assert cli.main(["fit", "--protocol", str(protocol)]) == 2
    common = ["--protocol", str(protocol), "--out", str(root / "output")]
    assert cli.main(["development-package", *common]) == 2
    assert cli.main(["development-package", *common, "--cache", "fixture"]) == 2
    permit = fixture_permission(root, protocol, payload)
    monkeypatch.setattr(cli.service.data, "load", lambda *args: object())
    seen = []
    monkeypatch.setattr(h, "development_package", lambda *args: seen.append(args) or "d" * 64)
    assert (
        cli.main(
            ["development-package", *common, "--cache", "fixture", "--permission", str(permit)]
        )
        == 0
    )
    assert len(seen) == 1
    monkeypatch.setattr(c, "run", lambda *args, **kwargs: {"status": "completed"})
    assert cli.main(["fit", *common, "--profile-out", str(out)]) == 0
    assert cli.main(["audit", *common]) == 0


@pytest.mark.parametrize(
    "fault", ["schema", "scope", "entry", "base64", "hash", "symlink", "overwrite"]
)
def test_development_transport_never_opens_reserved_pixels(tmp_path, fault):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    cache, split = p.pilot.generated_cache(inputs, "RGB", 1)
    for source_id in split.future_reserved:
        path = cache.root / (source_id + ".u8")
        path.rename(path.with_suffix(".closed"))
        path.symlink_to("/closed-reserved-pixels")
    packet = tmp_path / "development.json"
    h.development_package(cache, split, packet)
    train = tmp_path / "train.json"
    sha = h.cache_package(cache, split, train)
    copied = h.install_cache(train, sha, split, tmp_path / "copied")
    value = a.document(packet, h.CACHE_LIMIT)
    name = split.development[0]
    if fault == "schema":
        value["schema"] = "old"
    elif fault == "scope":
        value["files"][split.future_reserved[0]] = value["files"].pop(name)
    elif fault == "entry":
        value["files"][name]["base64"] = "bad"
    elif fault == "base64":
        value["files"][name]["base64"] = "!" * 262144
    elif fault == "hash":
        value["files"][name]["sha256"] = "0" * 64
    elif fault == "symlink":
        (copied.root / (name + ".u8")).symlink_to("/closed-pixels")
    else:
        (copied.root / (name + ".u8")).touch()
    packet.write_bytes(a.encode(value))
    with pytest.raises((ValueError, FileExistsError)):
        h.install_development(copied, split, packet, a.digest(packet.read_bytes()))
    assert not (copied.root / (split.future_reserved[0] + ".u8")).exists()


def test_development_roundtrip_and_package_failure(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    cache, split = p.pilot.generated_cache(inputs, "RGB", 1)
    packet = tmp_path / "development.json"
    sha = h.development_package(cache, split, packet)
    train = tmp_path / "train.json"
    train_sha = h.cache_package(cache, split, train)
    copied = h.install_cache(train, train_sha, split, tmp_path / "copied")
    with pytest.raises(ValueError, match="hash"):
        h.install_development(copied, split, packet, "0" * 64)
    h.install_development(copied, split, packet, sha)
    assert (copied.root / (split.development[0] + ".u8")).read_bytes() == (
        cache.root / (split.development[0] + ".u8")
    ).read_bytes()
    path = cache.root / (split.development[0] + ".u8")
    path.write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash"):
        h.development_package(cache, split, tmp_path / "bad")


def test_report_archive_excludes_numeric_but_binds_all_metadata(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    (out / "model.numeric").write_bytes(b"generated-numeric-fixture")
    (out / "fit.json").write_text('{"fixture":true}')
    limits = c.Limits(10, 0, 0, 8 * a.MIB, a.MIB, closing_seconds=1)
    sha = h.evidence_package(out, limits, "b" * 64, include_numeric=False)
    checked = h.verify_evidence(out / "verified-training.zip", sha, "b" * 64)
    assert set(checked["files"]) == {"fit.json"}
    assert (out / "model.numeric").exists()


def test_native_fit_controller_dispatch_large_closing_and_live_progress(
    proposed,
    monkeypatch,
    capsys,
):
    root, prior, payload, limits, _ = proposed
    protocol = root / "job.json"
    c.freeze(root, protocol, "native_fit", payload, limits)
    permit = fixture_permission(root, protocol, payload)
    cache_dir = root / "metadata"
    cache_dir.mkdir()
    raw = a.read(ROOT / "benchmarks/png-bmp-native-cache-manifest-20261028.json", data.MAX_MANIFEST)
    (cache_dir / "cache.json").write_bytes(raw)
    monkeypatch.setattr(f, "bind_profile", lambda *args: None)
    with pytest.raises(ValueError, match="profile directory"):
        c.run(root, protocol, root / "missing", cache=cache_dir, permission_path=permit)
    stub = """import json,hashlib,sys,time
from pathlib import Path
root,protocol,out,parent=sys.argv[1:];out=Path(out);work=out/'work';work.mkdir()
fit=work/'fit';fit.mkdir();job=json.loads(Path(protocol).read_text());binding=hashlib.sha256(Path(protocol).read_bytes()).hexdigest()
progress={'contract_sha256':binding,'seed':2026102301,'epoch':1,'loss':.5,'optimizer_updates':16,'development':{'auc':.6}}
(fit/'progress.json').write_text(json.dumps(progress));time.sleep(.15)
for i in range(600):(fit/('fixture-%04d.json'%i)).write_text('{}')
result={'contract_sha256':binding,'physical_cuda':True,'split':job['payload']['split'],'recipe':job['payload']['recipe'],'environment':job['payload']['environment'],'epochs':20,'fit_updates':640,'presentations':5120,'emulated_only':True}
(work/'worker-result.json').write_text(json.dumps(result))
"""
    monkeypatch.setattr(c, "BOOTSTRAP", stub)
    monkeypatch.setattr(f, "audit_training", lambda *args: {"emulation_only": True})
    output = root / "output"
    result = c.run(
        root, protocol, output, cache=cache_dir, profile_out=prior, permission_path=permit
    )
    assert result["status"] == "completed" and result["worker"]["emulated_only"]
    assert (output / "closed.json").stat().st_size > 60000
    assert "development_auc" in capsys.readouterr().out
    changed = copy.deepcopy(payload)
    changed["setup_debit"]["seconds"] += 1
    changed["prior_seconds"] += 1
    second_protocol = root / "second-job.json"
    second_limits = dataclasses.replace(limits, seconds=limits.seconds - 1)
    c.freeze(root, second_protocol, "native_fit", changed, second_limits)
    second_permit = fixture_permission(root, second_protocol, changed)
    with pytest.raises(FileExistsError):
        c.run(
            root,
            second_protocol,
            root / "second-output",
            cache=cache_dir,
            profile_out=prior,
            permission_path=second_permit,
        )
    # Restore the real reader: the emulated profile fixture patched audit only.
    monkeypatch.undo()
    assert c.audit(root, protocol, output)["status"] == "completed"
    assert "verified-training.zip" in result["output_files"]


def test_training_notebook_default_is_source_only(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "training_builder", ROOT / "scripts/build-colab-training-notebook.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    out = tmp_path / "delivery"
    binding = builder.build(ROOT, out)
    notebook = a.document(out / builder.NOTEBOOK_NAME)
    codes = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    code = "\n".join(codes)
    assert binding["permission_created"] is False and binding["native_fit_execution"] is False
    for name in [
        "PREPARE_FIT_PROPOSAL",
        "VERIFY_APPROVED_FIT",
        "RUN_APPROVED_FIT",
        "DOWNLOAD_VERIFIED_TRAINING",
    ]:
        assert name + " = False" in code
    assert builder.SOURCE_NAME in code
    for part in codes:
        compile(part, "delivered-cell", "exec")
    assert "controller.run" in code and "handoff.install_development" in code


def test_fit_worker_dispatch_is_emulated_only(proposed, tmp_path, monkeypatch):
    root, prior, payload, limits, _ = proposed
    protocol = root / "job.json"
    c.freeze(root, protocol, "native_fit", payload, limits)
    permit = fixture_permission(root, protocol, payload)
    monkeypatch.setenv("STEGANOGRAPHY_COLAB_PERMISSION", str(permit))
    monkeypatch.setenv("STEGANOGRAPHY_COLAB_PROFILE", str(prior))
    monkeypatch.setattr(c, "__file__", str(root / "core/colab_execution.py"))
    from types import SimpleNamespace

    monkeypatch.setattr(
        c.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(prctl=lambda *args: 0)
    )
    monkeypatch.setattr(c.signal, "signal", lambda *args: None)
    monkeypatch.setattr(c.signal, "setitimer", lambda *args: None)
    monkeypatch.setattr(f, "bind_profile", lambda *args: None)
    monkeypatch.setattr(f, "execute", lambda *args: {"emulation_only": True})
    out = root / "output"
    out.mkdir()
    import os

    assert c.worker_main([str(protocol), str(out), str(os.getppid())]) == 0
    assert a.document(out / "work/worker-result.json")["emulation_only"] is True
    with pytest.raises(ValueError, match="only"):
        pjob = root / "generated.json"
        c.freeze(
            root,
            pjob,
            "generated_qa",
            {"action": "success", "value": 0},
            c.Limits(2, 128 * a.MIB, 8 * 1024**3, 8 * a.MIB, a.MIB, closing_seconds=0.1),
        )
        c.run(root, pjob, root / "wrong-context", profile_out=prior)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_independent_auc_never_accepts_nonfinite_scalar(trained, value):
    _, _, _, out, original = trained
    result = copy.deepcopy(original)
    result["cells"][0]["initial"]["train"]["auc"] = value
    with pytest.raises(ValueError):
        f.audit_training(out, result, "b" * 64, engine.capacity.environment())


def test_numeric_debit_rechecked_after_hash_validation(trained, monkeypatch):
    _, _, _, out, result = trained
    original = f.numeric.load

    def mismatched(*args, **kwargs):
        meta, arrays = original(*args, **kwargs)
        meta["consumption"]["cursor"] += 1
        return meta, arrays

    monkeypatch.setattr(f.numeric, "load", mismatched)
    with pytest.raises(ValueError, match="debit"):
        f.audit_training(out, result, "b" * 64, engine.capacity.environment())


def test_archive_member_count_cannot_be_unbounded(tmp_path):
    with pytest.raises(ValueError, match="ceiling"):
        h.verify_evidence(tmp_path / "missing", "b" * 64, "b" * 64, max_members=1000000)


def test_live_progress_rename_is_tolerated_but_closing_remains_strict(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    def disappeared(**kwargs):
        assert kwargs == {"follow_symlinks": False}
        raise FileNotFoundError("owned temporary progress entry was atomically renamed")

    entry = SimpleNamespace(path=str(tmp_path / "progress.next.json"), stat=disappeared)
    with monkeypatch.context() as scoped:
        scoped.setattr(c.os, "scandir", lambda folder: nullcontext([entry]))
        limits = c.Limits(2, 0, 0, 8 * a.MIB, a.MIB, closing_seconds=.1)
        assert c.files(tmp_path, limits) == (0, {})
        with pytest.raises(FileNotFoundError):
            c.files(tmp_path, limits, hashes=True)


def test_delivered_training_defaults_execute_without_models_or_media(tmp_path):
    import importlib.util
    import subprocess
    import sys

    spec = importlib.util.spec_from_file_location(
        "builder", ROOT / "scripts/build-colab-training-notebook.py"
    )
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    out = tmp_path / "delivery"
    builder.build(ROOT, out)
    notebook = a.document(out / builder.NOTEBOOK_NAME)
    sources = [
        "".join(cell["source"]).replace(
            "/content/spatial-colab-native-training-source-20261107", str(tmp_path / "installed")
        )
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    path = tmp_path / "cells.json"
    path.write_text(json.dumps(sources))
    script = """import json,sys,types
from pathlib import Path
packet=Path(sys.argv[1]).read_bytes();calls=[]
files=types.SimpleNamespace(upload=lambda:calls.append('source-only') or {sys.argv[3]:packet})
google=types.ModuleType('google');colab=types.ModuleType('google.colab');colab.files=files
google.colab=colab;sys.modules['google']=google;sys.modules['google.colab']=colab
ns={}
for source in json.loads(Path(sys.argv[2]).read_text()):exec(source,ns)
assert calls==['source-only'] and 'torch' not in sys.modules
flags=['RUN_APPROVED_PROFILE','PREPARE_FIT_PROPOSAL','VERIFY_APPROVED_FIT',
       'RUN_APPROVED_FIT','DOWNLOAD_VERIFIED_TRAINING']
assert all(ns[key] is False for key in flags)
assert not (ns['source_root']/'.colab-process-attempts').exists()
print('source-only-no-media-no-model-no-attempt')
"""
    result = subprocess.run(  # noqa: S603 -- owned source-only notebook fixture
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            script,
            str(out / builder.SOURCE_NAME),
            str(path),
            builder.SOURCE_NAME,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "source-only-no-media-no-model-no-attempt" in result.stdout
