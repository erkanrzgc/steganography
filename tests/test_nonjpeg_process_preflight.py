"""Own-process/kernel regression and immutable-runner amendment contracts."""

import copy
import json
import os
import runpy
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

from core import nonjpeg_comparison as old
from core import nonjpeg_process_preflight as revision
from steganography import research_nonjpeg_process as cli
from tests.test_nonjpeg_comparison import dump
from tests.test_nonjpeg_comparison import matrix as original_matrix


@pytest.fixture
def amended(tmp_path, monkeypatch):
    matrix = original_matrix.__wrapped__(tmp_path, monkeypatch)
    parent_path = Path(matrix.config["protocol"])
    failed_path = matrix.root / "failed.json"
    failed = {"status": "unavailable", "fit": {
        "phase": "resource_preflight", "real_data_optimizer_updates": 0}}
    failed_sha = dump(failed_path, failed)
    monkeypatch.setattr(revision, "PARENT", parent_path)
    monkeypatch.setattr(revision, "FAILED", failed_path)
    protocol = copy.deepcopy(matrix.protocol)
    protocol.update({
        "source_sha256": revision.sources(),
        "parent_protocol_sha256": old.sha(parent_path.read_bytes()),
        "failed_results_sha256": failed_sha,
        "budget_bytes": revision.OUTPUT_BYTES,
        "accounting": {"prior_seconds_debit": 21, "prior_output_bytes_debit": 65536,
                       "remaining_seconds": 1779,
                       "remaining_output_bytes": revision.OUTPUT_BYTES},
    })
    config = {**matrix.config, "protocol": str(matrix.root / "amended.json")}
    config["protocol_sha256"] = dump(Path(config["protocol"]), protocol)
    monkeypatch.setattr(revision, "cpu_bound", lambda: matrix.protocol["runtime"])
    monkeypatch.setattr(old, "wav_reference", lambda rows, *a: np.zeros(len(rows)))
    matrix.amended_config, matrix.amended_protocol = config, protocol
    return matrix


def test_complete_generated_amendment_keeps_original_module_isolated(amended):
    bound, sources = old.cpu_bound, old.sources
    old_code = old.execute.__code__
    report = revision.execute(amended.amended_config, amended.root / "out")
    assert report["status"] == "completed"
    assert len(report["trials"]) == 14 and sum(len(t["cells"]) for t in report["trials"]) == 480
    assert report["source_sha256"] == revision.sources()
    assert old.cpu_bound is bound and old.sources is sources and old.execute.__code__ is old_code
    assert report["seconds"] >= 21
    accounting = json.loads((amended.root / "out/external-accounting.json").read_bytes())
    assert accounting["real_prior_updates"] == 0 and accounting["prior_seconds_debit"] == 21
    assert report["real_data_optimizer_updates"] == sum(
        t["cv"]["all_optimizer_updates"] for t in report["trials"])


@pytest.mark.parametrize("fault", ["parent_hash", "failed_hash", "prior_time", "prior_bytes",
                                  "remaining_time", "remaining_bytes", "recipe", "runtime",
                                  "budget", "nonzero_prior", "wrong_phase", "prior_status",
                                  "parent_sources"])
def test_amendment_never_rebinds_history_or_resets_budget(amended, fault):
    protocol = amended.amended_protocol
    if fault == "parent_hash":
        protocol["parent_protocol_sha256"] = "0" * 64
    elif fault == "failed_hash":
        protocol["failed_results_sha256"] = "0" * 64
    elif fault in ("prior_time", "prior_bytes", "remaining_time", "remaining_bytes"):
        key = {"prior_time": "prior_seconds_debit", "prior_bytes": "prior_output_bytes_debit",
               "remaining_time": "remaining_seconds", "remaining_bytes": "remaining_output_bytes"}
        protocol["accounting"][key[fault]] += 1
    elif fault in ("recipe", "runtime"):
        protocol[fault] = {"changed": True}
    elif fault == "budget":
        protocol["budget_bytes"] += 1
    elif fault == "parent_sources":
        parent, _ = old.read(revision.PARENT)
        parent["source_sha256"] = {}
        protocol["parent_protocol_sha256"] = dump(revision.PARENT, parent)
    else:
        failed, _ = old.read(revision.FAILED)
        if fault == "nonzero_prior":
            failed["fit"]["real_data_optimizer_updates"] = 1
        elif fault == "wrong_phase":
            failed["fit"]["phase"] = "fit_and_measure"
        else:
            failed["status"] = "completed"
        protocol["failed_results_sha256"] = dump(revision.FAILED, failed)
    amended.amended_config["protocol_sha256"] = dump(
        Path(amended.amended_config["protocol"]), protocol)
    with pytest.raises(ValueError, match="amendment binding"):
        revision.execute(amended.amended_config, amended.root / "out")
    assert not (amended.root / "out").exists()


def test_prior_debit_stops_work_without_touching_original_globals(amended):
    original = old.cpu_bound
    clock = iter([0, 1780, 1781])
    result = revision.execute(amended.amended_config, amended.root / "out",
                              clock=lambda: next(clock))
    assert result["reason"] == "time_budget_exhausted"
    assert result["real_data_optimizer_updates"] == 0
    assert old.cpu_bound is original


def test_current_numeric_pid_is_selected_without_resolving_alias(monkeypatch, tmp_path):
    called = []
    monkeypatch.setattr(revision.os, "getpid", lambda: 12345)
    monkeypatch.setattr(old, "cpu_bound", lambda **kw: called.append(kw) or {"ok": True})
    assert revision.cpu_bound(root=tmp_path) == {"ok": True}
    assert called == [{"proc": Path("/proc/12345/cgroup"), "root": tmp_path}]
    revision.cpu_bound(proc=tmp_path / "identity", root=tmp_path)
    assert called[-1]["proc"] == tmp_path / "identity"


def test_real_proc_alias_and_untrusted_symlinks_are_still_rejected(tmp_path):
    if not Path("/proc/self/cgroup").exists():
        pytest.skip("actual procfs unavailable")
    with pytest.raises(ValueError, match="symlink input"):
        old.regular_read(Path("/proc/self/cgroup"), 4096)
    assert old.regular_read(Path("/proc") / str(os.getpid()) / "cgroup", 4096)
    real = tmp_path / "real"
    real.write_bytes(b"bounded")
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="symlink input"):
        old.regular_read(link, 64)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="nonregular input"):
        old.regular_read(fifo, 64)


def test_source_initializers_are_bounded_regular_files_including_empty(tmp_path):
    empty = tmp_path / "empty.py"
    empty.touch()
    assert revision.source_hash(empty) == old.sha(b"")
    link = tmp_path / "link.py"
    link.symlink_to(empty)
    with pytest.raises(ValueError, match="source symlink"):
        revision.source_hash(link)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="nonregular source"):
        revision.source_hash(fifo)
    large = tmp_path / "large.py"
    large.write_bytes(b"x" * (1024**2 + 1))
    with pytest.raises(ValueError, match="source byte bound"):
        revision.source_hash(large)


def test_reserved_prior_output_applies_to_every_replay_write(tmp_path, monkeypatch):
    original = old.OUTPUT_BYTES
    monkeypatch.setattr(revision, "OUTPUT_BYTES", 64)
    (tmp_path / "existing").write_bytes(b"x" * 64)
    with pytest.raises(ValueError, match="cumulative output limit"):
        revision.write(tmp_path / "replay.json", {"ok": True})
    assert not (tmp_path / "replay.json").exists() and original == old.OUTPUT_BYTES


@pytest.mark.parametrize("threads", [1, 2])
def test_actual_owned_kernel_service_checks_current_process(threads):
    runner, controller = shutil.which("systemd-run"), shutil.which("systemctl")
    if not runner or not controller or not os.environ.get("XDG_RUNTIME_DIR"):
        pytest.skip("actual bounded user service unavailable")
    ready = subprocess.run(  # noqa: S603 -- resolved local systemd executable, fixed arguments
        [controller, "--user", "show-environment"], capture_output=True, timeout=10)
    if ready.returncode:
        pytest.skip("actual user systemd bus unavailable")
    command = [runner, "--user", "--quiet", "--wait", "--pipe", "--collect",
               "--unit=nonjpeg-preflight-test-" + uuid.uuid4().hex,
               "--property=MemoryMax=256M", "--property=MemorySwapMax=0",
               "--property=CPUQuota=100%", "--property=RuntimeMaxSec=30",
               "--property=Restart=no", "--setenv=OMP_NUM_THREADS=" + str(threads),
               "--setenv=OPENBLAS_NUM_THREADS=1", "--working-directory=" + str(old.ROOT),
               sys.executable, "-c",
               "import json; from core.nonjpeg_process_preflight import cpu_bound; "
               "print(json.dumps(cpu_bound()))"]
    result = subprocess.run(  # noqa: S603 -- owned generated probe, no carrier/tool output inputs
        command, capture_output=True, text=True, timeout=40)
    if threads == 2:
        assert result.returncode != 0 and "bounded single-thread CPU service required" in (
            result.stderr)
    else:
        assert result.returncode == 0, "actual bounded kernel preflight failed"
        record = json.loads(result.stdout)
        assert record["memory_max_bytes"] <= 256 * 1024**2
        assert 0 < record["cpu_quota"] <= 1 and record["swap_max_bytes"] == 0
        assert record["gpu_used"] is False


def test_cli_thin_adapter_and_entrypoint(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli.resource, "setrlimit", lambda *a: None)
    monkeypatch.setattr(cli.documents, "read", lambda *a: ({}, "a" * 64))
    monkeypatch.setattr(cli.service, "execute", lambda *a: {"status": "completed"})
    args = ["--config", "private", "--out", str(tmp_path)]
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
    monkeypatch.setattr(cli.service, "execute", lambda *a: {"status": "unavailable"})
    assert cli.main(args) == 2
    monkeypatch.setattr(sys, "argv", [cli.__file__, *args])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(cli.__file__, run_name="__main__")
    assert stopped.value.code == 2
