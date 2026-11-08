"""Generated metadata fixtures; no Colab allocation, CUDA call or media access."""

import dataclasses
import importlib.metadata
import json
import subprocess
from types import SimpleNamespace

import pytest

from core import colab_artifacts as a
from core import spatial_colab_readiness as r


def roots(tmp_path, identity="0::/scope\n", version="v2"):
    proc, control = tmp_path / "proc", tmp_path / "cgroup"
    (proc / "123").mkdir(parents=True)
    (proc / "123" / "cgroup").write_text(identity)
    folder = control / "memory" / "scope" if version == "v1" else control / "scope"
    folder.mkdir(parents=True)
    for name, value in zip(r.COUNTERS[version], ("8192", "0", "4096", "2048"), strict=True):
        (folder / name).write_text(value)
    return proc, control, folder


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_resources_observe_metadata_and_keep_immutable(tmp_path, version):
    identity = "7:cpu:/ignored\n9:memory:/scope\n" if version == "v1" else "0::/scope\n"
    proc, control, folder = roots(tmp_path, identity, version)
    result = r.resources(proc_root=proc, cgroup_root=control, pid=123)
    assert result.status == "observed_metadata_only" and result.version == version
    assert result.counters[0][1] == 8192
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = "passed"
    (folder / r.COUNTERS[version][0]).write_text("max")
    assert r.resources(proc_root=proc, cgroup_root=control, pid=123).counters[0][1] is None
    (folder / r.COUNTERS[version][0]).write_text(str(2**63))
    assert r.resources(proc_root=proc, cgroup_root=control, pid=123).counters[0][1] is None


@pytest.mark.parametrize("identity,reason", [
    ("0::/../../private-parent\n", "parent_identity"),
    ("0::relative\n", "malformed_identity"),
    ("bad-line\n", "malformed_identity"),
    ("x::/scope\n", "malformed_identity"),
    ("0::/a\x00b\n", "malformed_identity"),
    ("0::/" + "a/" * 33 + "\n", "malformed_identity"),
    ("1:cpu:/ignored\n", "ambiguous_or_missing_controller"),
    ("0::/scope\n0::/scope\n", "ambiguous_or_missing_controller"),
])
def test_unsafe_and_ambiguous_identities_never_become_host_paths(tmp_path, identity, reason):
    proc, control, _ = roots(tmp_path, identity)
    result = r.resources(proc_root=proc, cgroup_root=control, pid=123)
    assert result.status == "unavailable" and result.reason == reason
    assert "private-parent" not in json.dumps(dataclasses.asdict(result))


@pytest.mark.parametrize("value", ["-1", "1.0", "١", "9" * 33, ""])
def test_bounded_finite_counter_parsing(tmp_path, value):
    proc, control, folder = roots(tmp_path)
    (folder / "memory.max").write_text(value)
    result = r.resources(proc_root=proc, cgroup_root=control, pid=123)
    assert result.status == "unavailable"


def test_resources_reject_symlink_missing_and_bad_encoding(tmp_path):
    proc, control, folder = roots(tmp_path)
    (folder / "memory.max").unlink()
    assert r.resources(proc_root=proc, cgroup_root=control, pid=123).status == "unavailable"
    outside = tmp_path / "outside"
    outside.write_text("8192")
    (folder / "memory.max").symlink_to(outside)
    assert r.resources(proc_root=proc, cgroup_root=control, pid=123).status == "unavailable"
    (proc / "123" / "cgroup").write_bytes(b"\xff")
    assert r.resources(proc_root=proc, cgroup_root=control, pid=123).status == "unavailable"


@pytest.mark.parametrize("pid", [0, -1, True, "123", 2**31])
def test_explicit_process_identity(pid):
    with pytest.raises(ValueError, match="process identity"):
        r.resources(pid=pid)


def fake_query(monkeypatch, raw=b"Tesla T4, 15360, 580.82.07\n", status=0, error=None):
    monkeypatch.setattr(r.shutil, "which", lambda name: "/usr/bin/" + name)
    calls = []

    def run(arguments, **kwargs):
        calls.append(arguments)
        assert arguments[1:4] == ["--fsize=8192:8192", "--cpu=5:5", "--"]
        assert kwargs["timeout"] == 10
        if error:
            raise error
        kwargs["stdout"].write(raw)
        return SimpleNamespace(returncode=status)

    monkeypatch.setattr(r.subprocess, "run", run)
    return calls


def test_fixed_bounded_gpu_query(tmp_path, monkeypatch):
    calls = fake_query(monkeypatch)
    result = r.gpu(workdir=tmp_path)
    assert result == r.GpuMetadata("visible_metadata_only", "not_kernel_execution",
                                  "Tesla T4", 15360, "580.82.07")
    assert len(calls) == 1


@pytest.mark.parametrize("raw,status,reason", [
    (b"x" * 8193, 0, "query_failed_or_oversized"),
    (b"private error", 1, "query_failed_or_oversized"),
    (b"", 0, "single_gpu_required"),
    (b"a\nb\n", 0, "single_gpu_required"),
    (b"a, x, 1.0", 0, "invalid_query_format"),
    (b"a, 0, 1.0", 0, "invalid_query_format"),
    (b"a, 524289, 1.0", 0, "invalid_query_format"),
    (b"a, " + b"9" * 5000 + b", 1.0", 0, "invalid_query_format"),
    (b"a, 10, bad", 0, "invalid_query_format"),
    (b"a, 10", 0, "invalid_query_format"),
    (b"a\x1b, 10, 1.0", 0, "invalid_query_format"),
    (b"\xff", 0, "query_unavailable"),
])
def test_gpu_query_failures_are_bounded_and_redacted(tmp_path, monkeypatch, raw, status, reason):
    fake_query(monkeypatch, raw, status)
    result = r.gpu(workdir=tmp_path)
    assert result.status == "unavailable" and result.reason == reason
    assert "private error" not in json.dumps(dataclasses.asdict(result))


@pytest.mark.parametrize("error,reason", [
    (subprocess.TimeoutExpired("fixture", 10), "query_timeout"),
    (OSError("private path"), "query_unavailable"),
])
def test_gpu_timeout_and_missing_tools(tmp_path, monkeypatch, error, reason):
    fake_query(monkeypatch, error=error)
    assert r.gpu(workdir=tmp_path).reason == reason
    monkeypatch.setattr(r.shutil, "which", lambda name: None)
    assert r.gpu().reason == "missing_query_tool"


def test_gpu_output_directory_symlink_is_rejected(tmp_path):
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        r.gpu(workdir=link)


def pinned_packages(monkeypatch, gpu=False):
    packages = {"numpy": "2.0.2", "Pillow": "11.3.0",
                "torch": "2.11.0+cu128" if gpu else "2.11.0+cpu"}

    def version(name):
        if name not in packages:
            raise importlib.metadata.PackageNotFoundError(name)
        return packages[name]

    monkeypatch.setattr(r.importlib.metadata, "version", version)
    monkeypatch.setattr(r.platform, "python_version", lambda: "3.12.13")
    return packages


def test_colab_parent_identity_does_not_skip_card_query(tmp_path, monkeypatch):
    proc, control, _ = roots(tmp_path, "0::/../../generated-outside\n")
    observed = r.resources(proc_root=proc, cgroup_root=control, pid=123)
    pinned_packages(monkeypatch, gpu=True)
    monkeypatch.setattr(r, "resources", lambda: observed)
    calls = fake_query(monkeypatch)
    result = r.inventory(check_gpu=True, workdir=tmp_path)
    assert result["status"] == "partial_resources_unavailable"
    assert result["resources"]["reason"] == "parent_identity"
    assert result["gpu"]["name"] == "Tesla T4" and len(calls) == 1
    assert result["resource_enforcement"] == "unverified"
    assert result["optimizer_updates"] == result["tensor_operations"] == 0


def test_cpu_default_environment_and_gpu_statuses(monkeypatch):
    packages = pinned_packages(monkeypatch)
    monkeypatch.setattr(
        r, "resources", lambda: r.ResourceMetadata("observed_metadata_only", "fixture")
    )
    result = r.inventory()
    assert result["status"] == "metadata_ready_only" and result["gpu"]["status"] == "not_requested"
    packages["torch"] = "2.10.0+cpu"
    assert r.inventory()["status"] == "unavailable_environment"
    packages["torch"] = "2.11.0+cu128"
    monkeypatch.setattr(r, "gpu", lambda **kwargs: r.GpuMetadata("unavailable", "fixture"))
    assert r.inventory(check_gpu=True)["status"] == "unavailable_gpu"


def test_record_hash_exclusive_attempt_and_deadline(tmp_path, monkeypatch):
    pinned_packages(monkeypatch)
    monkeypatch.setattr(r, "resources", lambda: r.ResourceMetadata("unavailable", "fixture"))
    out = tmp_path / "out"
    result = r.record(out)
    assert a.digest((out / "result.json").read_bytes()) == result["receipt_sha256"]
    assert a.document(out / "attempt.json")["check_gpu"] is False
    before = (out / "result.json").read_bytes()
    with pytest.raises(FileExistsError):
        r.record(out)
    assert (out / "result.json").read_bytes() == before
    times = iter((1.0, 62.0))
    monkeypatch.setattr(r.time, "monotonic", lambda: next(times))
    assert r.record(tmp_path / "late")["status"] == "unavailable_deadline"


@pytest.mark.parametrize("choice", [0, 1, "false", None])
def test_gpu_query_requires_boolean_and_creates_no_attempt(tmp_path, choice):
    with pytest.raises(ValueError, match="boolean"):
        r.record(tmp_path / "out", check_gpu=choice)
    with pytest.raises(ValueError, match="boolean"):
        r.inventory(check_gpu=choice)
    assert not (tmp_path / "out").exists()
