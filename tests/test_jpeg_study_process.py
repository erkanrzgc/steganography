"""Real isolated limit exits and bounded, secret-free worker diagnostics."""

import json
import os
import signal
import sys
import time
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from core.jpeg_study_process import MAX_REPORT, WorkerExit, WorkerFailure, run


def command(body):
    return [sys.executable, "-c", "import resource;"
            "resource.setrlimit(resource.RLIMIT_CORE,(0,0));import signal;"
            "signal.signal(signal.SIGXFSZ,signal.SIG_DFL);" + body]


@pytest.mark.parametrize("number,reason", [
    (signal.SIGXCPU, "worker_cpu_limit"),
    (signal.SIGXFSZ, "worker_file_size_limit"),
    (signal.SIGKILL, "worker_signalled"),
    (signal.SIGTERM, "worker_signalled"),
    (signal.SIGRTMIN + 1, "worker_signalled"),
])
def test_real_signal_exit_is_portable_and_never_assumes_kill_means_oom(tmp_path, number, reason):
    with pytest.raises(WorkerFailure) as caught:
        run(command(f"import os;os.kill(os.getpid(),{int(number)})"), cwd=tmp_path, timeout=5)
    failure = caught.value
    assert isinstance(failure, ValueError)
    assert failure.unavailable() == {
        "status": "unavailable", "reason": reason,
        "worker_execution": {"returncode": -number, "signal": number,
                             "signal_name": getattr(number, "name", None), "stdout_bytes": 0},
    }
    assert str(tmp_path) not in json.dumps(failure.unavailable())


def test_kernel_cpu_limit_is_distinct_from_wall_timeout(tmp_path):
    with pytest.raises(WorkerFailure) as caught:
        run(command("resource.setrlimit(resource.RLIMIT_CPU,(1,2));exec('while True: pass')"),
            cwd=tmp_path, timeout=6)
    assert caught.value.reason == "worker_cpu_limit"
    assert caught.value.execution.returncode == -signal.SIGXCPU


def test_nonzero_empty_success_and_stdin_never_copy_secret_output(tmp_path):
    body = "import sys;print('/private/password=do-not-copy');" \
           "print('api_key=do-not-copy',file=sys.stderr);sys.exit(7)"
    with pytest.raises(WorkerFailure) as caught:
        run(command(body), cwd=tmp_path, timeout=5)
    record = caught.value.unavailable()
    assert record["reason"] == "worker_exit_nonzero"
    assert record["worker_execution"]["returncode"] == 7
    assert record["worker_execution"]["signal"] is None
    assert record["worker_execution"]["signal_name"] is None
    assert record["worker_execution"]["stdout_bytes"] > 0
    assert all(secret not in json.dumps(record) + str(caught.value)
               for secret in ("/private/", "password", "api_key", "do-not-copy"))
    with pytest.raises(WorkerFailure) as empty:
        run(command("pass"), cwd=tmp_path, timeout=5)
    assert empty.value.reason == "worker_empty_output"
    assert run(command("import sys;sys.stdout.buffer.write(sys.stdin.buffer.read())"),
               cwd=tmp_path, timeout=5, payload=b"ok") == b"ok"


@pytest.mark.parametrize("body", [
    "import sys;sys.stdout.write('x'*300000);sys.stdout.flush()",
    "import sys,time;sys.stdout.write('x'*300000);sys.stdout.flush();time.sleep(30)",
])
def test_output_cap_after_exit_and_while_running(tmp_path, body):
    with pytest.raises(WorkerFailure) as caught:
        run(command(body), cwd=tmp_path, timeout=5)
    assert caught.value.reason == "worker_output_limit"
    assert caught.value.execution.stdout_bytes > MAX_REPORT


def test_worker_exit_record_is_immutable():
    execution = WorkerExit(-signal.SIGXCPU, 0)
    with pytest.raises(FrozenInstanceError):
        execution.stdout_bytes = 1
    record = execution.record()
    record["returncode"] = 0
    assert execution.returncode == -signal.SIGXCPU


def test_signal_failure_kills_owned_descendant_without_executing_artifacts(tmp_path):
    marker = tmp_path / "child.pid"
    body = ("import os,subprocess,sys,time;from pathlib import Path;"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
            f"Path({str(marker)!r}).write_text(str(p.pid));"
            f"os.kill(os.getpid(),{int(signal.SIGXCPU)})")
    with pytest.raises(WorkerFailure, match="cpu_limit"):
        run(command(body), cwd=tmp_path, timeout=5)
    pid = int(marker.read_text())
    until = time.monotonic() + 2
    while time.monotonic() < until:
        status = Path(f"/proc/{pid}/stat")
        if not status.exists() or status.read_text().split(")", 1)[1].split()[0] == "Z":
            break
        time.sleep(.02)
    else:
        pytest.fail("owned descendant survived worker failure")
    # It may briefly remain as a zombie until the host init reaps the orphan.
    assert pid != os.getpid()
