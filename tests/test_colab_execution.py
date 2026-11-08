"""Generated process QA: real Linux limits, interruption and immutable evidence."""

import copy
import json
import os
import runpy
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from core import colab_artifacts as a
from core import colab_execution as c


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "source"
    (root / "core").mkdir(parents=True)
    for name in ("colab_execution.py", "colab_artifacts.py"):
        shutil.copyfile(Path(c.__file__).parent / name, root / "core" / name)
    (root / "core" / "__init__.py").touch()
    return root


def limits(**changes):
    values = dict(seconds=8, closing_seconds=0.5, data_bytes=128 * a.MIB,
                  resident_watch_bytes=8 * 1024**3, output_bytes=8 * a.MIB,
                  file_bytes=a.MIB, address_bytes=256 * a.MIB)
    return c.Limits(**(values | changes))


def job(root, action="success", value=0, **changes):
    path = root / "protocol.json"
    c.freeze(root, path, "generated_qa", {"action": action, "value": value}, limits(**changes))
    return path


@pytest.mark.parametrize("change", [
    {"seconds": float("nan")}, {"seconds": True}, {"seconds": 1801},
    {"closing_seconds": 0}, {"closing_seconds": 8}, {"data_bytes": 1},
    {"data_bytes": 17 * 1024**3}, {"resident_watch_bytes": 1},
    {"output_bytes": 1}, {"file_bytes": 1}, {"address_bytes": 1},
    {"file_bytes": 129 * a.MIB}, {"address_bytes": True}, {"output_bytes": 1.0},
])
def test_limits_reject_nonfinite_or_unbounded(change):
    with pytest.raises(ValueError):
        limits(**change)


def test_source_contract_is_exclusive_complete_and_environment_bound(source):
    protocol = job(source)
    config, sha, parsed = c.verify(source, protocol)
    assert config["native_execution_available"] is False
    assert parsed == limits() and sha == a.digest(protocol.read_bytes())
    with pytest.raises(FileExistsError):
        job(source)
    changed = copy.deepcopy(config)
    changed["environment"]["python"] = "wrong"
    protocol.write_bytes(a.encode(changed))
    with pytest.raises(ValueError, match="environment"):
        c.verify(source, protocol)
    protocol.write_bytes(a.encode(config))
    (source / "core" / "unreviewed.py").write_text("# Added source\n")
    with pytest.raises(ValueError, match="source"):
        c.verify(source, protocol)


@pytest.mark.parametrize("field,value", [
    ("schema", "old"), ("policy", "old"), ("kind", "native_fit"),
    ("automatic_retry", True), ("native_execution_available", True),
    ("payload", []), ("environment", []),
])
def test_malformed_contract(source, field, value):
    protocol = job(source)
    document = a.document(protocol)
    document[field] = value
    protocol.write_bytes(a.encode(document))
    with pytest.raises(ValueError, match="schema"):
        c.verify(source, protocol)


def test_untrusted_or_empty_source_and_native_kind_rejected(source):
    with pytest.raises(ValueError):
        c.source_hashes(source / "missing")
    with pytest.raises(ValueError):
        c.freeze(source, source / "bad.json", "native_profile", {}, limits())
    path = source / "core" / "link.py"
    path.symlink_to(source / "core" / "colab_execution.py")
    with pytest.raises(ValueError, match="symlink"):
        c.source_hashes(source)


def test_real_worker_kernel_limits_and_complete_audit(source):
    protocol = job(source)
    out = source / "output"
    result = c.run(source, protocol, out)
    assert result["status"] == "completed", result
    assert result["worker_reaped"] and result["worker_exit"] == 0
    assert result["hard_resident_cgroup_limit"] == "not_enforced"
    assert result["hard_swap_cgroup_limit"] == "not_enforced"
    assert result["worker"]["optimizer_updates"] == 0
    kernel = result["worker"]["kernel_limits"]
    assert kernel["data"] == [128 * a.MIB] * 2
    assert kernel["file"] == [a.MIB] * 2
    assert kernel["address"] == [256 * a.MIB] * 2
    assert kernel["core"] == [0, 0]
    assert 0 < result["kernel_worker_peak_resident_bytes"] <= 8 * 1024**3
    assert result["charged_seconds"] < limits().seconds
    assert c.audit(source, protocol, out) == result
    with pytest.raises(FileExistsError):
        c.run(source, protocol, source / "another-output")
    with pytest.raises(FileExistsError):
        c.run(source, protocol, out)
    assert not (source / "another-output").exists()


@pytest.mark.parametrize("action,value,changes", [
    ("sleep", 5, {"seconds": 1.5, "closing_seconds": 0.2}),
    ("memory", 64 * a.MIB, {"data_bytes": 16 * a.MIB}),
    ("memory", 64 * a.MIB, {"resident_watch_bytes": 16 * a.MIB}),
    ("output", 2 * a.MIB, {"file_bytes": 65536}),
    ("output", 2 * a.MIB, {"output_bytes": 256 * 1024, "file_bytes": 256 * 1024}),
    ("symlink", 0, {}), ("crash", 0, {}),
])
def test_real_failures_preserved_without_retry(source, action, value, changes):
    protocol = job(source, action, value, **changes)
    out = source / "output"
    result = c.run(source, protocol, out)
    assert result["status"] == "unavailable", result
    assert result["worker_reaped"]
    assert c.audit(source, protocol, out) == result
    assert (out / "worker.log").exists()
    with pytest.raises(FileExistsError):
        c.run(source, protocol, source / "different-output")


def test_interrupt_kills_and_reaps_exact_worker(source, monkeypatch):
    protocol = job(source, "sleep", 5)
    original = c.resident
    seen = []

    def interrupt(pid):
        seen.append(pid)
        original(pid)
        raise KeyboardInterrupt

    monkeypatch.setattr(c, "resident", interrupt)
    result = c.run(source, protocol, source / "output")
    assert result["failure"] == "interrupted" and result["worker_reaped"]
    assert result["worker_exit"] == -signal.SIGKILL
    assert seen and not Path(f"/proc/{seen[0]}").exists()


def test_spawn_failure_still_consumes_attempt(source, monkeypatch):
    protocol = job(source)

    def fail(*args, **kwargs):
        raise OSError("private host path and token must not be reported")

    monkeypatch.setattr(c.subprocess, "Popen", fail)
    result = c.run(source, protocol, source / "output")
    assert result["failure"] == "OSError" and not result["worker_reaped"]
    assert "private host" not in json.dumps(result)
    with pytest.raises(FileExistsError):
        c.run(source, protocol, source / "other")


@pytest.mark.parametrize("change", ["log", "receipt", "anchor", "invalidate"])
def test_modified_or_invalidated_closing_rejected(source, change):
    protocol = job(source)
    out = source / "output"
    result = c.run(source, protocol, out)
    assert result["status"] == "completed"
    if change == "log":
        (out / "worker.log").write_text("tampered")
    elif change == "receipt":
        document = a.document(out / "closed.json")
        document["charged_seconds"] = 0
        (out / "closed.json").write_bytes(a.encode(document))
    elif change == "anchor":
        anchor = source / ".colab-process-attempts" / (result["contract_sha256"] + ".closed.json")
        anchor.write_text("{}")
    else:
        (out / "unavailable-finalization.json").write_text("{}")
    with pytest.raises(ValueError):
        c.audit(source, protocol, out)


def test_missing_kernel_limiter_is_not_an_attempt(source, monkeypatch):
    protocol = job(source)
    monkeypatch.setattr(c.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError):
        c.run(source, protocol, source / "output")
    assert not (source / ".colab-process-attempts").exists()


def test_inaccessible_observer_is_unavailable_and_reaped(source, monkeypatch):
    protocol = job(source, "sleep", 2)

    def missing(pid):
        raise ValueError("no counters")

    monkeypatch.setattr(c, "resident", missing)
    result = c.run(source, protocol, source / "output")
    assert result["failure"] == "ValueError"
    assert result["resident_observer_gaps"] == 3 and result["worker_reaped"]


@pytest.mark.parametrize("action", ["success", "sleep", "cpu", "memory", "output", "symlink"])
def test_registered_worker_in_process_uses_only_generated_fixtures(source, monkeypatch, action):
    protocol = job(source, action, 0)
    out = source / "inline"
    out.mkdir()

    class Libc:
        def prctl(self, *args):
            return 0

    monkeypatch.setattr(c.ctypes, "CDLL", lambda *args, **kwargs: Libc())
    monkeypatch.setattr(c.signal, "signal", lambda *args: None)
    monkeypatch.setattr(c.signal, "setitimer", lambda *args: None)
    monkeypatch.setattr(c, "__file__", str(source / "core" / "colab_execution.py"))
    assert c.worker_main([str(protocol), str(out), str(os.getppid())]) == 0
    result = a.document(out / "work" / "worker-result.json")
    assert result["generated_only"] and result["optimizer_updates"] == 0


def test_worker_crash_reports_type_only(source, monkeypatch, capsys):
    protocol = job(source, "crash")
    out = source / "inline"
    out.mkdir()

    class Libc:
        def prctl(self, *args):
            return 0

    monkeypatch.setattr(c.ctypes, "CDLL", lambda *args, **kwargs: Libc())
    monkeypatch.setattr(c.signal, "signal", lambda *args: None)
    monkeypatch.setattr(c.signal, "setitimer", lambda *args: None)
    monkeypatch.setattr(c, "__file__", str(source / "core" / "colab_execution.py"))
    assert c.worker_main([str(protocol), str(out), str(os.getppid())]) == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "unavailable", "failure_type": "RuntimeError",
    }


def test_cli_freeze_run_audit_and_errors(source, monkeypatch, capsys):
    from steganography import research_colab_execution as cli

    monkeypatch.setattr(cli, "__file__", str(source / "steganography" / "cli.py"))
    protocol, out = source / "protocol.json", source / "output"
    common = ["--protocol", str(protocol)]
    assert cli.main(["freeze", *common]) == 0
    assert cli.main(["run", *common, "--out", str(out)]) == 0
    assert cli.main(["audit", *common, "--out", str(out)]) == 0
    assert cli.main(["run", *common]) == 2
    assert cli.main(["freeze", *common, "--out", str(out)]) == 2
    assert cli.main(["freeze", *common]) == 2
    assert "private" not in capsys.readouterr().out


def test_cli_unavailable_return_code(source, monkeypatch):
    from steganography import research_colab_execution as cli

    monkeypatch.setattr(cli, "__file__", str(source / "steganography" / "cli.py"))
    protocol = job(source, "crash")
    assert cli.main(["run", "--protocol", str(protocol), "--out", str(source / "output")]) == 2


@pytest.mark.parametrize("payload", [
    {}, {"action": "exec", "value": 0}, {"action": "success", "value": True},
    {"action": "success", "value": float("nan")}, {"action": "success", "value": -1},
    {"action": "success", "value": 129 * a.MIB},
])
def test_no_arbitrary_command_or_unbounded_qa(payload, tmp_path):
    with pytest.raises(ValueError):
        c._qa(payload, tmp_path, "a" * 64)


def test_files_count_depth_and_nonregular_bounds(tmp_path):
    folder = tmp_path / "deep"
    folder.mkdir()
    current = folder
    for _ in range(9):
        current = current / "x"
        current.mkdir()
    with pytest.raises(ValueError, match="depth"):
        c.files(folder, limits())
    fifo = tmp_path / "fifo"
    fifo.mkdir()
    os.mkfifo(fifo / "pipe")
    with pytest.raises(ValueError, match="type"):
        c.files(fifo, limits())


def test_worker_arguments_parent_and_proc_counters(monkeypatch):
    with pytest.raises(ValueError):
        c.worker_main([])
    class Libc:
        def prctl(self, *args):
            return -1

    monkeypatch.setattr(c.ctypes, "CDLL", lambda *args, **kwargs: Libc())
    assert c.worker_main(["unused", "unused", str(os.getppid())]) == 2
    assert c.resident(os.getpid())["VmRSS"] > 0
    monkeypatch.setattr(c.a, "read", lambda *args: b"State:\tZ (zombie)\n")
    assert c.resident(1) is None
    for raw in (b"x", b"State:\tS\nVmRSS:\t-1 kB\n", b"State:\tS\nVmRSS: 5 MB\n"):
        monkeypatch.setattr(c.a, "read", lambda *args, raw=raw: raw)
        with pytest.raises(ValueError):
            c.resident(1)


@pytest.mark.parametrize("error", [FileNotFoundError, ProcessLookupError])
def test_proc_disappearance_is_not_an_invalid_live_counter(monkeypatch, error):
    def vanished(*args):
        raise error("the exact child disappeared during proc read")

    monkeypatch.setattr(c.a, "read", vanished)
    assert c.resident(123) is None


def test_kernel_timer_and_parent_death_kill_worker(source):
    protocol = job(source, "sleep", 20, seconds=2, closing_seconds=0.2)
    out = source / "standalone"
    out.mkdir()
    child = subprocess.Popen([  # noqa: S603 -- fixed generated source-bound worker
        sys.executable, "-I", "-B", "-c", c.BOOTSTRAP, str(source),
        str(protocol), str(out), str(os.getpid()),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert child.wait(timeout=5) == -signal.SIGALRM
    assert (out / "work").exists()

    # A separate supervisor disappears: PR_SET_PDEATHSIG kills its fixed worker.
    parent_code = (
        "import os,subprocess,sys,time;"
        "p=subprocess.Popen(sys.argv[1:]+[str(os.getpid())],"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);"
        "print(p.pid,flush=True);time.sleep(30)"
    )
    abandoned = source / "abandoned"
    abandoned.mkdir()
    parent = subprocess.Popen([  # noqa: S603 -- fixed generated parent-death fixture
        sys.executable, "-I", "-c", parent_code, sys.executable, "-I", "-B", "-c",
        c.BOOTSTRAP, str(source), str(protocol), str(abandoned),
    ], stdout=subprocess.PIPE, text=True)
    assert parent.stdout is not None
    pid = int(parent.stdout.readline())
    deadline = time.monotonic() + 1.5
    while not (abandoned / "work").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert (abandoned / "work").exists()
    parent.kill()
    parent.wait(timeout=5)
    deadline = time.monotonic() + 3

    def stopped():
        try:
            return c.resident(pid) is None
        except FileNotFoundError:
            return True
        except ValueError:
            return False  # The exiting task can temporarily have no mm counters.

    while not stopped() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert stopped()


def test_excessive_closing_invalidates_receipt(source, monkeypatch):
    protocol = job(source, seconds=5, closing_seconds=0.05)
    original = c.a.put

    def slow_close(path, *args):
        result = original(path, *args)
        if path.name == "closed.json":
            time.sleep(0.1)
        return result

    monkeypatch.setattr(c.a, "put", slow_close)
    out = source / "output"
    with pytest.raises(TimeoutError, match="invalidated"):
        c.run(source, protocol, out)
    assert (out / "unavailable-finalization.json").exists()
    with pytest.raises(ValueError, match="invalidated"):
        c.audit(source, protocol, out)


def test_contract_change_during_run_is_unavailable(source, monkeypatch):
    protocol = job(source, "sleep", 0.1)
    original = c.resident

    def change(pid):
        value = a.document(protocol)
        value["payload"]["value"] = 0.2
        protocol.write_bytes(a.encode(value))
        return original(pid)

    monkeypatch.setattr(c, "resident", change)
    result = c.run(source, protocol, source / "output")
    assert result["status"] == "unavailable"
    assert result["failure"] == "unavailable_closing_audit"


def test_generated_cpu_and_output_are_finite(tmp_path):
    for action, value in (("cpu", 0.001), ("memory", 1024), ("output", 1)):
        c._qa({"action": action, "value": value}, tmp_path, "a" * 64)
    assert (tmp_path / "generated.bin").stat().st_size == 65536


def test_notebook_delivery_hashes_bootstrap_and_isolated_roundtrip(tmp_path):
    root = Path(c.__file__).parent.parent
    builder = runpy.run_path(str(root / "scripts" / "build-colab-process-notebook.py"))["build"]
    first, second = tmp_path / "delivery-1", tmp_path / "delivery-2"
    binding = builder(root, first)
    assert builder(root, second) == binding
    name = "png_bmp_colab_process_qa_20261106.ipynb"
    assert (first / name).read_bytes() == (second / name).read_bytes()
    nb = a.document(first / name)
    assert "accelerator" not in nb["metadata"]
    code = "".join(nb["cells"][1]["source"])
    compile(code, "owned-bootstrap", "exec")
    compile("".join(nb["cells"][2]["source"]), "owned-generated-run", "exec")
    namespace = {}
    exec(code.split('if any(name == "core"')[0], namespace)  # noqa: S102 -- owned builder output
    raw = (first / "sources.json").read_bytes()
    delivered = namespace["install_sources"](raw, tmp_path / "delivered")
    assert a.digest(raw) == binding["source_package_sha256"]
    for relative, sha in binding["source_sha256"].items():
        assert a.digest((delivered / relative).read_bytes()) == sha
    with pytest.raises(FileExistsError):
        namespace["install_sources"](raw, delivered)
    with pytest.raises(ValueError, match="hash"):
        namespace["install_sources"](raw + b"x", tmp_path / "bad")
    link = tmp_path / "linked"
    link.symlink_to(delivered, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        namespace["install_sources"](raw, link)

    script = (
        "import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);"
        "from core import colab_execution as c;"
        "root=Path(sys.argv[1]);p=root/'qa.json';out=root/'result';"
        "c.freeze(root,p,'generated_qa',{'action':'success','value':0},"
        "c.Limits(10,134217728,8589934592,8388608,1048576,268435456,1));"
        "r=c.run(root,p,out);assert c.audit(root,p,out)==r;"
        "assert r['status']=='completed',r;assert 'torch' not in sys.modules;print('verified')"
    )
    completed = subprocess.run(  # noqa: S603 -- fixed owned generated CPU fixture
        [sys.executable, "-I", "-B", "-c", script, str(delivered)],
        capture_output=True, text=True, timeout=15, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "verified"
