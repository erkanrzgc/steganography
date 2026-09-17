"""Central, resource-bounded process runner for optional forensic tools."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from core.ctf_types import ToolExecution


class ToolRunner:
    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_output: int = 1024 * 1024,
        memory_bytes: int = 768 * 1024 * 1024,
    ) -> None:
        if timeout <= 0 or max_output <= 0 or memory_bytes <= 0:
            raise ValueError("tool limits must be positive")
        self.timeout = timeout
        self.max_output = max_output
        self.memory_bytes = memory_bytes
        self._versions: dict[str, str | None] = {}

    def run(
        self,
        tool: str,
        args: Sequence[str],
        *,
        cwd: Path,
        secret_values: Sequence[str] = (),
        timeout: float | None = None,
        should_cancel: Callable[[], bool] | None = None,
        _include_version: bool = True,
    ) -> ToolExecution:
        started = time.perf_counter()
        executable = shutil.which(tool)
        redacted = tuple(
            "[REDACTED]" if value and value in set(secret_values) else value
            for value in (tool, *map(str, args))
        )
        if executable is None:
            return ToolExecution(
                tool,
                None,
                "unavailable",
                _elapsed(started),
                None,
                redacted,
                error=f"{tool} is not installed",
            )
        cwd = Path(cwd)
        if not cwd.is_dir() or cwd.is_symlink():
            raise ValueError("tool working directory must be a regular directory")
        cwd = cwd.resolve()
        if should_cancel and should_cancel():
            return ToolExecution(
                tool,
                self.version(tool),
                "cancelled",
                _elapsed(started),
                None,
                redacted,
            )

        limit = timeout if timeout is not None else self.timeout
        if limit <= 0:
            raise ValueError("tool timeout must be positive")
        process = subprocess.Popen(  # noqa: S603 - argv executable is resolved, never a shell
            [executable, *map(str, args)],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={"PATH": os.environ.get("PATH", ""), "LANG": "C", "LC_ALL": "C"},
            start_new_session=True,
            preexec_fn=self._limits(limit) if os.name == "posix" else None,
        )
        deadline = time.monotonic() + limit
        captured = bytearray()

        def drain() -> None:
            if process.stdout is None:
                return
            while chunk := process.stdout.read(65536):
                remaining = self.max_output + 1 - len(captured)
                if remaining > 0:
                    captured.extend(chunk[:remaining])

        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        status = "completed"
        error: str | None = None
        while process.poll() is None:
            if should_cancel and should_cancel():
                status = "cancelled"
                error = "cancelled"
                _terminate(process)
                break
            if time.monotonic() >= deadline:
                status = "timed_out"
                error = f"tool exceeded {limit:g} seconds"
                _terminate(process)
                break
            time.sleep(0.02)
        # A tool may exit while descendants still hold stdout open.
        # Kill its process group before waiting for the bounded reader.
        _terminate(process)
        process.wait()
        reader.join(timeout=1)
        output_bytes = bytes(captured)
        clipped = len(output_bytes) > self.max_output
        output = output_bytes[: self.max_output].decode("utf-8", errors="replace")
        output = output.replace(str(cwd), ".")
        for secret in secret_values:
            if secret:
                output = output.replace(secret, "[REDACTED]")
        if clipped:
            output += "\n[output truncated]"
        if status == "completed" and process.returncode:
            status = "failed"
            error = f"tool exited with status {process.returncode}"
        return ToolExecution(
            tool=tool,
            version=self.version(tool) if _include_version else None,
            status=status,  # type: ignore[arg-type]
            duration_ms=_elapsed(started),
            exit_code=process.returncode,
            command=redacted,
            output=output,
            error=error,
        )

    def version(self, tool: str) -> str | None:
        if tool in self._versions:
            return self._versions[tool]
        executable = shutil.which(tool)
        if executable is None:
            value = None
        else:
            try:
                version_flag = {"exiftool": "-ver", "pngcheck": "-V", "ffmpeg": "-version"}
                completed = self.run(
                    tool,
                    [version_flag.get(tool, "--version")],
                    cwd=Path.cwd(),
                    timeout=2,
                    _include_version=False,
                )
                value = completed.output[:256].splitlines()[0]
            except (OSError, subprocess.SubprocessError, IndexError):
                value = None
        self._versions[tool] = value
        return value

    def _limits(self, timeout: float):
        memory_bytes = self.memory_bytes
        max_output = self.max_output

        def apply() -> None:
            import resource

            cpu = max(1, int(timeout) + 1)
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
            resource.setrlimit(resource.RLIMIT_FSIZE, (max_output, max_output))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

        return apply


def _terminate(process: subprocess.Popen[bytes]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _elapsed(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
