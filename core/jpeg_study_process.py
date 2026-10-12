"""Bounded owned research workers and portable numeric failure evidence."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

MAX_REPORT = 256 * 1024
MAX_SIMULATION_REPORT = 768 * 1024


@dataclass(frozen=True)
class WorkerExit:
    returncode: int
    stdout_bytes: int

    def record(self):
        number = -self.returncode if self.returncode < 0 else None
        name = None
        if number is not None:
            with suppress(ValueError):
                name = signal.Signals(number).name
        return {"returncode": self.returncode, "signal": number,
                "signal_name": name,
                "stdout_bytes": self.stdout_bytes}


class WorkerFailure(ValueError):
    def __init__(self, execution: WorkerExit, *, output_limit=False):
        self.execution = execution
        if output_limit:
            self.reason = "worker_output_limit"
        elif execution.returncode == -signal.SIGXCPU:
            self.reason = "worker_cpu_limit"
        elif execution.returncode == -signal.SIGXFSZ:
            self.reason = "worker_file_size_limit"
        elif execution.returncode < 0:
            # SIGKILL alone is not evidence of OOM or a CPU hard-limit event.
            self.reason = "worker_signalled"
        elif execution.returncode:
            self.reason = "worker_exit_nonzero"
        else:
            self.reason = "worker_empty_output"
        super().__init__("study worker failed: " + self.reason)

    def unavailable(self):
        return {"status": "unavailable", "reason": self.reason,
                "worker_execution": self.execution.record()}


def run(command, *, cwd: Path, timeout, cancelled=lambda: False,
        payload=None, maximum=MAX_REPORT):
    """Retain exit/signal evidence, never raw command, output or stderr secrets."""
    if payload is not None and len(payload) > 1024**2:
        raise ValueError("study child input exceeds bounds")
    if maximum not in (MAX_REPORT, MAX_SIMULATION_REPORT):
        raise ValueError("study child output limit invalid")
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as input_file:
        if payload is not None:
            input_file.write(payload)
            input_file.seek(0)
        process = subprocess.Popen(  # noqa: S603 - fixed module and arguments, no shell
            command, cwd=cwd, stdin=input_file if payload is not None else subprocess.DEVNULL,
            stdout=output, stderr=subprocess.DEVNULL, start_new_session=True,
            env={**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                 "NUMBA_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        )
        started, overflow = time.monotonic(), False
        try:
            while process.poll() is None:
                if cancelled():
                    raise InterruptedError("study cancelled")
                if time.monotonic() - started >= timeout:
                    raise TimeoutError("study worker deadline exceeded")
                if os.fstat(output.fileno()).st_size > maximum:
                    overflow = True
                    break
                time.sleep(0.1)
        finally:
            # Kill descendants even if their parent has already exited.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        size = os.fstat(output.fileno()).st_size
        if overflow or size > maximum or process.returncode or not size:
            raise WorkerFailure(WorkerExit(process.returncode, size),
                                output_limit=overflow or size > maximum)
        output.seek(0)
        raw = output.read(maximum + 1)
        if len(raw) > maximum:
            raise WorkerFailure(WorkerExit(process.returncode, len(raw)), output_limit=True)
        return raw
