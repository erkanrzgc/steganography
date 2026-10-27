"""Independent one-attempt wall/output/cgroup supervision for generated spatial work."""

from __future__ import annotations

import json
import os
import resource
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

from core import colab_artifacts as a
from core import spatial_control as control
from core.wav_control_runner import cgroup, process_started


def inventory(out: Path) -> int:
    total, count = 0, 0
    for path in out.rglob("*"):
        a.safe(path)
        try:
            info = path.stat()
        except FileNotFoundError:
            if path.name == "progress.next.json":
                continue
            raise
        count += 1
        if count > 7000 or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise ValueError("spatial output entry/type limit")
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total


def stop(process) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=5)
        return
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def peaks(folder: Path) -> dict:
    return {
        "memory_peak_bytes": int(a.read(folder / "memory.peak", 64)),
        "swap_peak_bytes": int(a.read(folder / "memory.swap.current", 64)),
        "swap_peak_basis": "kernel_memory_swap_max_zero",
        "memory_max_bytes": int(a.read(folder / "memory.max", 64)),
        "memory_swap_max_bytes": int(a.read(folder / "memory.swap.max", 64)),
    }


def execute(root: Path, protocol: Path, out: Path) -> dict:
    started = process_started()
    expires = started + control.SECONDS
    contract, binding, _ = control.load(root, protocol)
    if root.name != "worktree" or out != root.parent / "output":
        raise ValueError("spatial control requires its fresh dedicated worktree/output")
    a.safe(out)
    a.put(
        root.parent / "attempt.json",
        a.encode(
            {
                "protocol_sha256": binding,
                "automatic_retry": False,
                "real_fit_authorized": False,
                "supervisor_pid": os.getpid(),
                "started_monotonic": started,
                "deadline_monotonic": expires,
            }
        ),
    )
    a.directory(out)
    result = {
        "status": "unavailable",
        "reason": "preflight_or_worker_incomplete",
        "generated_only": True,
        "real_fit_authorized": False,
        "protocol_sha256": binding,
    }
    process = None
    try:
        folder = cgroup()
        if control.environment() != contract["environment"]:
            raise ValueError("spatial control environment mismatch")
        env = {
            **os.environ,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": str(root),
        }
        command = [
            sys.executable,
            str(root / "steganography/research_spatial_control.py"),
            "--protocol",
            str(protocol),
            "--out",
            str(out),
            "--worker-deadline",
            str(expires - control.RESERVE_SECONDS),
        ]
        with (out / "stdout.log").open("xb") as stdout, (out / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(  # noqa: S603 — source-bound local worker only.
                command, cwd=root, env=env, start_new_session=True, stdout=stdout, stderr=stderr
            )
            with (out / "parent-observations.jsonl").open("xb") as observations:
                try:
                    while process.poll() is None:
                        elapsed = time.monotonic() - started
                        size = inventory(out)
                        observations.write(
                            (
                                json.dumps(
                                    {
                                        "monotonic": time.monotonic(),
                                        "elapsed_seconds": elapsed,
                                        "bytes": size,
                                        "worker_progress_sha256": control.checksum(
                                            out / "progress.json"
                                        )
                                        if (out / "progress.json").is_file()
                                        else None,
                                    },
                                    sort_keys=True,
                                    allow_nan=False,
                                )
                                + "\n"
                            ).encode()
                        )
                        observations.flush()
                        os.fsync(observations.fileno())
                        if time.monotonic() >= expires or size > control.MAX_OUTPUT - 2 * a.MIB:
                            stop(process)
                            break
                        time.sleep(0.5)
                finally:
                    if process.poll() is None:
                        stop(process)
        observed = {
            "wall_seconds": time.monotonic() - started,
            "output_bytes": inventory(out),
            **peaks(folder),
        }
        if process.returncode == 0 and (out / "worker.json").is_file():
            worker = a.document(out / "worker.json", 4 * a.MIB)
            result = control.decide(root, protocol, out, worker, observed)
            result["worker_status"] = worker.get("status")
            result["worker_reason"] = worker.get("reason")
        else:
            result.update(control.ledger(out, strict=False))
            result["resources"] = observed
        result["child_exit_code"] = process.returncode
        observed.update(peaks(folder))
        observed["output_bytes"] = inventory(out)
        observed["wall_seconds"] = time.monotonic() - started
        if (
            observed["wall_seconds"] > control.SECONDS
            or observed["output_bytes"] > control.MAX_OUTPUT - 2 * a.MIB
            or observed["memory_peak_bytes"] > control.SETTINGS["host_bytes"]
            or observed["swap_peak_bytes"] != 0
        ):
            result["status"], result["reason"] = "unavailable", "closing_resource_limit"
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as error:
        result["reason"] = type(error).__name__
        result.update(control.ledger(out, strict=False))
    finally:
        if process is not None and process.poll() is None:
            stop(process)
    a.put(out / "decision.json", a.encode(result), 4 * a.MIB)
    return result


def worker(root: Path, protocol: Path, out: Path, expires: float) -> None:
    _, binding, _ = control.load(root, protocol)
    attempt = a.document(root.parent / "attempt.json")
    if (
        out != root.parent / "output"
        or attempt["protocol_sha256"] != binding
        or os.getppid() != attempt["supervisor_pid"]
        or not 0 < expires <= attempt["deadline_monotonic"]
        or time.monotonic() >= expires
    ):
        raise ValueError("spatial worker requires live bound supervisor")
    cgroup()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * a.MIB, 128 * a.MIB))
    control.worker(root, protocol, out, expires)
