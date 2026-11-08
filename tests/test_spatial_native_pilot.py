"""Generated CPU coverage for native measurement, frozen proposals and consent."""

import copy
import dataclasses
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from core import colab_artifacts as a
from core import spatial_native_data as d
from core import spatial_native_pilot as p
from core import spatial_native_supervisor as sup
from core import spatial_native_training as e
from steganography import research_spatial_native as cli

torch = pytest.importorskip("torch")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(a.encode(value))


def metadata(tmp_path):
    rows = []
    for i in range(105):
        rows.append(
            d.Source(
                f"source-{i}",
                "RGB",
                f"group-{i}",
                f"Author {i}",
                f"https://commons.wikimedia.org/wiki/File:Fixture_{i}.png",
                a.digest(f"original-{i}".encode()),
                a.digest(f"pixels-{i}".encode()),
                "CC0-1.0",
                d.native.LICENSES["CC0-1.0"],
                "0" * 64,
            )
        )
    cache = d.Cache(tmp_path, tuple(rows), "a" * 64)
    split = e.partition(cache, "RGB", (64, 20, 21), 0)
    return cache, split


def contract(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    cache, split = metadata(tmp_path)
    save(
        root / p.PROPOSAL,
        {
            "status": "proposal_only",
            "split": dataclasses.asdict(split),
            "models": {"total_fit_updates": 640},
        },
    )
    (root / p.DOCUMENT).parent.mkdir(parents=True, exist_ok=True)
    (root / p.DOCUMENT).write_text("Fixture protocol.\n")
    for name in (
        "cli.py",
        "config.py",
        "registry.py",
        "tests/test_spatial_native_training.py",
        "tests/test_spatial_native_pilot.py",
        p.ACTOR,
        p.DEPLOYMENT,
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
    env = {**e.capacity.environment(), "driver": "fixture"}
    protocol = root / "protocol.json"
    p.freeze(root, protocol, env)
    return root, protocol, cache, split, env


def sample_profile(split, binding="b" * 64):
    cells = []
    for seed in p.SEEDS:
        cells.append(
            {
                "seed": seed,
                "native_forward_backward_seconds": [1.0, 1.0, 1.0],
                "generated_step_seconds": [1.0, 1.0, 1.0],
                "native_train_evaluation": {"max_pair_seconds": 1.0},
                "native_initialization_seconds": 1.0,
                "generated_initialization_seconds": 1.0,
                "checkpoint_write_seconds": 1.0,
                "checkpoint_restore_seconds": 1.0,
                "native_optimizer_updates": 0,
                "generated_optimizer_updates": 3,
                "native_parameters_unchanged": True,
                "checkpoint_bytes": 100,
            }
        )
    return {
        "schema": "spatial-rgb-native-measurement-v1",
        "cells": cells,
        "generated_optimizer_updates": 6,
        "native_optimizer_updates": 0,
    }


def approval(split, binding="b" * 64, profile_sha="c" * 64):
    return {
        "schema": "spatial-rgb-native-approval-v1",
        "user_approved": True,
        "protocol_sha256": binding,
        "profile_sha256": profile_sha,
        "split_sha256": a.digest(a.encode(dataclasses.asdict(split))),
        "max_cumulative_seconds": 1800,
        "max_cumulative_output_bytes": 768 * a.MIB,
    }


def accepted_profile(binding="b" * 64):
    return {
        "status": "profile_ready",
        "protocol_sha256": binding,
        "physical_cuda": True,
        "charged_seconds": 10.0,
        "logical_output_bytes": 100,
        "projection": {"fit_projection_seconds": 100.0},
        "max_checkpoint_bytes": a.MIB,
    }


def test_source_freeze_verify_immutable_fields_and_environment(tmp_path):
    root, protocol, cache, split, env = contract(tmp_path)
    config, binding, parsed = p.verify(root, protocol, cache)
    assert parsed == split and binding == p.checksum(protocol)
    with pytest.raises(FileExistsError):
        p.freeze(root, protocol, env)
    with pytest.raises(ValueError, match="environment"):
        p.freeze(root, root / "bad.json", {})
    changed = copy.deepcopy(config)
    changed["settings"]["epochs"] = 19
    save(protocol, changed)
    with pytest.raises(ValueError, match="protocol"):
        p.verify(root, protocol, cache)
    save(protocol, config)
    (root / "cli.py").write_text("changed")
    with pytest.raises(ValueError, match="source"):
        p.verify(root, protocol, cache)
    (root / "cli.py").write_text("")
    changed = copy.deepcopy(config)
    changed["proposal_sha256"] = "0" * 64
    save(protocol, changed)
    with pytest.raises(ValueError, match="proposal hash"):
        p.verify(root, protocol, cache)
    changed = copy.deepcopy(config)
    changed["split"]["seed"] = 1
    save(protocol, changed)
    with pytest.raises(ValueError, match="split mismatch"):
        p.verify(root, protocol, cache)
    proposal = json.loads((root / p.PROPOSAL).read_text())
    proposal["status"] = "approved"
    save(root / p.PROPOSAL, proposal)
    with pytest.raises(ValueError, match="proposal mismatch"):
        p.freeze(root, root / "bad-proposal.json", env)


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    root = tmp_path_factory.mktemp("native-measure-generated")
    inputs = root / "inputs"
    inputs.mkdir()
    cache, split = p.generated_cache(inputs, "L", 1)
    result = p.measure(cache, split, root / "measurement", "b" * 64, device="cpu", seconds=120)
    return root, cache, split, result


def test_complete_cpu_measurement_is_generated_qa_only(measured):
    root, cache, split, result = measured
    assert result["device"] == "cpu" and not result["physical_cuda"]
    assert result["native_optimizer_updates"] == 0 and result["generated_optimizer_updates"] == 6
    assert (
        result["native_backward_presentations"] == 12
        and result["native_evaluation_presentations"] == 4
    )
    assert all(c["native_parameters_unchanged"] for c in result["cells"])
    assert len(list((root / "measurement").glob("*-intent.json"))) == 6
    assert len(list((root / "measurement").glob("*-commit.json"))) == 6
    assert len(list((root / "measurement").glob("*.numeric"))) == 2
    estimated = p.projection(result)
    assert estimated["fit_projection_seconds"] > 0 and not estimated["native_accuracy_inferred"]
    assert result["cuda_peak"] == {"allocated_bytes": None, "reserved_bytes": None}
    for device, seconds in [("wrong", 1), ("cpu", 0), ("cpu", float("nan")), ("cpu", True)]:
        with pytest.raises(ValueError, match="device/deadline"):
            p.measure(cache, split, root / "bad", "b" * 64, device=device, seconds=seconds)
    with pytest.raises(TimeoutError):
        p.remaining(time.monotonic() - 1)


@pytest.mark.parametrize("fault", ["cells", "counts", "timing", "parameters"])
def test_projection_rejects_incomplete_or_corrupt_evidence(measured, fault):
    result = copy.deepcopy(measured[3])
    if fault == "cells":
        result["cells"] = result["cells"][:1]
    elif fault == "counts":
        result["native_optimizer_updates"] = 1
    elif fault == "timing":
        result["cells"][0]["generated_step_seconds"][1] = float("nan")
    else:
        result["cells"][0]["native_parameters_unchanged"] = False
    with pytest.raises(ValueError):
        p.projection(result)


def test_gpu_only_helpers_with_emulated_counters(monkeypatch):

    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("release"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2)
    p.sync("cpu")
    p.sync("cuda:0")
    p.release("cuda:0")
    assert p.memory("cuda:0") == {"allocated_bytes": 1, "reserved_bytes": 2}
    assert calls == ["sync", "release"]


def test_measurement_refuses_nonfinite_native_gradients(tmp_path, monkeypatch):

    source = tmp_path / "source"
    source.mkdir()
    cache, split = p.generated_cache(source, "L", 1)
    monkeypatch.setattr(
        torch.nn.functional,
        "cross_entropy",
        lambda *_: torch.tensor(float("nan"), requires_grad=True),
    )
    with pytest.raises(ValueError, match="nonfinite"):
        p.measure(cache, split, tmp_path / "out", "b" * 64, device="cpu", seconds=120)


def test_approval_is_bound_to_exact_profile_split_and_limits(tmp_path):
    _, split = metadata(tmp_path)
    good = approval(split)
    profile = accepted_profile()
    assert p.approved_limits(good, profile, "c" * 64, "b" * 64, split) == (
        1790.0,
        768 * a.MIB - 100,
    )
    for changed in (
        None,
        {},
        good | {"user_approved": False},
        good | {"protocol_sha256": "0" * 64},
        good | {"split_sha256": "0" * 64},
    ):
        with pytest.raises(ValueError, match="approval"):
            p.approved_limits(changed, profile, "c" * 64, "b" * 64, split)
    for edits in (
        {"max_cumulative_seconds": True},
        {"max_cumulative_seconds": 1801},
        {"max_cumulative_seconds": float("nan")},
        {"max_cumulative_seconds": 100},
        {"max_cumulative_output_bytes": 99},
        {"max_cumulative_output_bytes": 769 * a.MIB},
    ):
        with pytest.raises(ValueError, match="ineligible"):
            p.approved_limits(good | edits, profile, "c" * 64, "b" * 64, split)
    for changed in (profile | {"physical_cuda": False}, profile | {"status": "unavailable"}):
        with pytest.raises(ValueError, match="ineligible"):
            p.approved_limits(good, changed, "c" * 64, "b" * 64, split)


def test_full_fit_adapter_with_emulated_steps_and_closed_reserved(tmp_path, monkeypatch):
    root, protocol, cache, split, _ = contract(tmp_path)
    binding = p.checksum(protocol)
    profile_path = tmp_path / "profile.json"
    save(profile_path, accepted_profile(binding))
    consent = approval(split, binding, p.checksum(profile_path))

    class Session:
        def __init__(self, plan):
            self.model = object()
            self.updates = self.cursor = self.presentations = 0

        def scores(self, role):
            n = len(getattr(split, role))
            return {"scores": np.tile([0.1, 0.9], (n, 1)), "auc": 1.0}

        def step(self):
            self.updates += 1
            self.cursor += 4
            self.presentations += 8
            return {"loss": 1 / (1 + self.updates), "optimizer_updates": self.updates}

        def checkpoint(self, path):
            path.write_bytes(b"generated test receipt")
            return p.checksum(path)

    @contextmanager
    def session(cache, plan, **kwargs):
        yield Session(plan)

    monkeypatch.setattr(p.engine, "session", session)
    monkeypatch.setattr(p.historical, "model_sha", lambda _: "d" * 64)
    monkeypatch.setattr(p, "release", lambda _: None)
    result = p.fit(root, protocol, cache, tmp_path / "fit", consent, profile_path)
    assert result["fit_updates"] == 640 and result["presentations"] == 5120
    assert (
        result["status"] == "passed_exploratory_native_gate"
        and not result["future_reserved_opened"]
    )
    assert sum(len(c["checkpoints"]) for c in result["cells"]) == 8
    assert len(list((tmp_path / "fit").glob("*-commit.json"))) == 640
    with pytest.raises(ValueError, match="approval"):
        p.fit(root, protocol, cache, tmp_path / "bad", None, profile_path)


def test_inventory_hash_bounds_and_live_observer(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "empty-directory").mkdir()
    (out / "numeric").write_bytes(b"x" * 10)
    assert sup.streaming_hash(out / "numeric") == p.checksum(out / "numeric")
    save(out / "ready.json", {"protocol_sha256": "b" * 64, "pid": 123, "phase": "profile"})
    monkeypatch.setattr(
        sup,
        "kernel",
        lambda pid=None: {
            "memory_max_bytes": 8 * 1024**3,
            "swap_max_bytes": 0,
            "memory_peak_bytes": 100,
            "memory_current_bytes": 50,
        },
    )
    total, files = sup.inventory(out)
    assert total > 10 and len(files) == 2
    result = sup.observe(out, "b" * 64)
    assert result["kernel"]["memory_peak_bytes"] == 100
    with pytest.raises(FileExistsError):
        sup.observe(out, "b" * 64)
    with pytest.raises(ValueError, match="binding"):
        sup.observe(out, "c" * 64)
    link = out / "link"
    link.symlink_to(out / "numeric")
    with pytest.raises(ValueError, match="symlink"):
        sup.inventory(out)


def test_supervisor_phase_and_marker_guards(tmp_path, monkeypatch):
    root, protocol, cache, split, env = contract(tmp_path)
    binding = p.checksum(protocol)
    monkeypatch.setattr(
        sup,
        "kernel",
        lambda pid=None: {
            "memory_max_bytes": 8 * 1024**3,
            "swap_max_bytes": 0,
            "memory_peak_bytes": 100,
            "memory_current_bytes": 50,
        },
    )
    monkeypatch.setattr(p.historical, "environment", lambda: env)

    def measure(cache, split, out, *args, **kwargs):
        out.mkdir()
        save(out / "measurement.json", sample_profile(split))
        return sample_profile(split)

    monkeypatch.setattr(p, "measure", measure)
    original_document = a.document

    def document(path, *args):
        if path.name == "observer.json" and not original_exists(path):
            save(
                path,
                {
                    "protocol_sha256": binding,
                    "pid": __import__("os").getpid(),
                    "kernel": {
                        "memory_max_bytes": 8 * 1024**3,
                        "swap_max_bytes": 0,
                        "memory_peak_bytes": 100,
                    },
                    "logical_output_bytes": 1000,
                },
            )
        return original_document(path, *args)

    original_exists = Path.exists
    monkeypatch.setattr(
        Path, "exists", lambda path: True if path.name == "observer.json" else original_exists(path)
    )
    monkeypatch.setattr(a, "document", document)
    result = sup.execute(root, protocol, cache, tmp_path / "out", phase="profile")
    assert result["status"] == "profile_ready" and result["kernel"]["memory_peak_bytes"] == 100
    with pytest.raises(FileExistsError):
        sup.execute(root, protocol, cache, tmp_path / "out2", phase="profile")
    with pytest.raises(ValueError, match="phase"):
        sup.execute(root, protocol, cache, tmp_path / "out3", phase="wrong")
    with pytest.raises(ValueError, match="approval"):
        sup.execute(root, protocol, cache, tmp_path / "out3", phase="fit")


def test_cli_refuses_implicit_or_mixed_execution(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli.resource, "setrlimit", lambda *_: None)
    protocol = tmp_path / "protocol.json"
    save(protocol, {})
    for extra in ([], ["--profile", "--fit"], ["--profile", "--out", str(tmp_path / "out")]):
        assert cli.main(["--protocol", str(protocol), *extra]) == 2
    assert "unavailable" in capsys.readouterr().out
    monkeypatch.setattr(cli.supervisor, "observe", lambda *_: {})
    assert cli.main(["--protocol", str(protocol), "--observe", "--out", str(tmp_path / "out")]) == 0
    monkeypatch.setattr(cli.service, "freeze", lambda *_: {})
    env = tmp_path / "env.json"
    save(env, {})
    assert cli.main(["--protocol", str(protocol), "--freeze-environment", str(env)]) == 0
    assert cli.main(["--protocol", str(protocol), "--freeze-environment", str(env), "--fit"]) == 2


def test_kernel_paths_limits_and_finite_counters(monkeypatch):
    values = {
        "cgroup": b"0::/user.slice/native-test\n",
        "memory.max": b"8589934592\n",
        "memory.swap.max": b"0\n",
        "memory.peak": b"1048576\n",
        "memory.current": b"1000\n",
    }
    monkeypatch.setattr(a, "read", lambda path, limit: values[path.name])
    assert sup.kernel(123)["memory_peak_bytes"] == 1048576
    for bad in (b"wrong\n", b"0::relative\n", b"0::/../escape\n", b"0::/" + b"x/" * 33):
        values["cgroup"] = bad
        with pytest.raises(ValueError, match="cgroup"):
            sup.kernel()
    values["cgroup"] = b"0::/user.slice/native-test\n"
    for bad in (b"max\n", b"-1\n"):
        values["memory.max"] = bad
        with pytest.raises(ValueError, match="counter"):
            sup.kernel()
    values["memory.max"] = b"17179869184\n"
    with pytest.raises(ValueError, match="envelope"):
        sup.kernel()
    values["memory.max"] = b"8589934592\n"
    values["memory.swap.max"] = b"1\n"
    with pytest.raises(ValueError, match="envelope"):
        sup.kernel()


def test_output_type_size_and_live_identity_fail_closed(tmp_path, monkeypatch):
    import os

    out = tmp_path / "out"
    out.mkdir()
    os.mkfifo(out / "fifo")
    with pytest.raises(ValueError, match="inventory"):
        sup.inventory(out)
    with pytest.raises(ValueError, match="regular"):
        sup.streaming_hash(out / "fifo")
    (out / "fifo").unlink()
    huge = out / "huge"
    with huge.open("wb") as stream:
        stream.truncate(128 * a.MIB + 1)
    with pytest.raises(ValueError, match="inventory"):
        sup.inventory(out)
    with pytest.raises(ValueError, match="read limit"):
        sup.streaming_hash(huge)
    huge.unlink()
    save(out / "ready.json", {"protocol_sha256": "b" * 64, "pid": 123, "phase": "profile"})
    monkeypatch.setattr(
        sup,
        "kernel",
        lambda pid=None: {"memory_max_bytes": 100, "swap_max_bytes": 0, "memory_peak_bytes": 101},
    )
    with pytest.raises(ValueError, match="resource"):
        sup.observe(out, "b" * 64)
    save(out / "ready.json", {"protocol_sha256": "b" * 64, "pid": False, "phase": "profile"})
    with pytest.raises(ValueError, match="binding"):
        sup.observe(out, "b" * 64)


def test_cli_dispatches_only_explicit_bound_phases(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.resource, "setrlimit", lambda *_: None)
    protocol = tmp_path / "protocol.json"
    save(protocol, {"split": {"manifest_sha256": "a" * 64}})
    monkeypatch.setattr(cli.data, "load", lambda *_: object())
    calls = []
    monkeypatch.setattr(
        cli.supervisor,
        "execute",
        lambda *args, **kwargs: calls.append(kwargs) or {"status": "profile_ready"},
    )
    assert (
        cli.main(
            [
                "--protocol",
                str(protocol),
                "--profile",
                "--cache",
                "metadata",
                "--out",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    assert calls[0]["phase"] == "profile"
    monkeypatch.setattr(cli.supervisor, "finalize", lambda *args: calls.append(args) or {})
    assert (
        cli.main(
            [
                "--protocol",
                str(protocol),
                "--cache",
                "metadata",
                "--out",
                str(tmp_path / "out"),
                "--finalize",
                "--kernel-start-us",
                "1",
                "--kernel-exit-us",
                "2",
            ]
        )
        == 0
    )


@pytest.mark.parametrize("fault", ["none", "environment", "output", "wait", "observer", "late"])
def test_supervisor_live_closeout_adversarial_cases(tmp_path, monkeypatch, fault):
    import os

    root, protocol, cache, split, env = contract(tmp_path)
    binding = p.checksum(protocol)
    live = {"memory_max_bytes": 8 * 1024**3, "swap_max_bytes": 0, "memory_peak_bytes": 100}
    monkeypatch.setattr(sup, "kernel", lambda pid=None: live)
    monkeypatch.setattr(p.historical, "environment", lambda: {} if fault == "environment" else env)
    profile_path, consent_path = tmp_path / "prior.json", tmp_path / "consent.json"
    save(profile_path, accepted_profile(binding))
    save(consent_path, approval(split, binding, p.checksum(profile_path)))
    out = tmp_path / "fit"

    def fit(*args):
        out.mkdir()
        payload = {"status": "failed_learning_gate"}
        save(out / "fit.json", payload)
        return payload

    monkeypatch.setattr(p, "fit", fit)
    original_inventory = sup.inventory
    if fault == "output":
        monkeypatch.setattr(sup, "inventory", lambda _: (p.CUMULATIVE_OUTPUT_CEILING, {}))
    original_sleep = sup.time.sleep

    def closing(_):
        if fault == "wait":
            monkeypatch.setattr(p, "remaining", lambda _: 0.5)
            return
        save(
            out / "observer.json",
            {
                "protocol_sha256": binding,
                "pid": os.getpid(),
                "kernel": live | ({"swap_max_bytes": 1} if fault == "observer" else {}),
                "logical_output_bytes": 1000,
                "files": original_inventory(out)[1],
            },
        )
        if fault == "late":
            before = sup.time.monotonic()
            monkeypatch.setattr(sup.time, "monotonic", lambda: before + 1801)

    monkeypatch.setattr(sup.time, "sleep", closing)
    arguments = dict(phase="fit", approval_path=consent_path, profile_path=profile_path)
    if fault == "none":
        assert (
            sup.execute(root, protocol, cache, out, **arguments)["status"] == "failed_learning_gate"
        )
    else:
        with pytest.raises((ValueError, TimeoutError)):
            sup.execute(root, protocol, cache, out, **arguments)
    assert (tmp_path / "fit-attempt.json").exists()
    monkeypatch.setattr(sup.time, "sleep", original_sleep)


@pytest.mark.parametrize("fault", ["none", "time", "limit", "binding", "changed", "payload", "fit"])
def test_independent_kernel_time_and_immutable_closeout(tmp_path, monkeypatch, fault):
    root, protocol, cache, split, _ = contract(tmp_path)
    binding = p.checksum(protocol)
    out = tmp_path / "output"
    out.mkdir()
    save(out / "measurement.json", sample_profile(split))
    observer = {"protocol_sha256": binding, "files": sup.inventory(out)[1]}
    save(out / "observer.json", observer)
    worker = accepted_profile(binding) | {
        "phase_seconds_limit": 240,
        "phase_output_limit_bytes": p.PROFILE_OUTPUT,
        "phase": "fit" if fault == "fit" else "profile",
        "observer_sha256": p.checksum(out / "observer.json"),
        "payload_sha256": p.checksum(out / "measurement.json"),
    }
    if fault == "binding":
        worker["protocol_sha256"] = "0" * 64
    if fault == "payload":
        worker["payload_sha256"] = "0" * 64
    save(out / "result.json", worker)
    if fault == "changed":
        save(out / "measurement.json", {"changed": True})
    start, stop = (
        (2, 1) if fault == "time" else (1_000_000, 300_000_000 if fault == "limit" else 3_000_000)
    )
    if fault in ("none", "fit"):
        result = sup.finalize(root, protocol, cache, out, start, stop)
        assert result["charged_seconds"] >= 2 and result["output_files_verified"] == 3
        if fault == "none":
            assert result["max_checkpoint_bytes"] == 100 + a.MIB
        with pytest.raises(FileExistsError):
            sup.finalize(root, protocol, cache, out, start, stop)
    else:
        with pytest.raises(ValueError):
            sup.finalize(root, protocol, cache, out, start, stop)
