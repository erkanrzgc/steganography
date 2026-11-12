"""Resource-bounded adapters for optional command-line forensic tools."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from abc import abstractmethod
from pathlib import Path

from core.analyzer import Analyzer
from core.result import AnalysisResult, Signal

_TIMEOUT_SECONDS = 20
_MAX_OUTPUT = 1024 * 1024
_MEMORY_BYTES = 512 * 1024 * 1024


class _ExternalAnalyzer(Analyzer):
    executable = ""
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def command(self, src: Path) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def interpret(self, output: str, returncode: int) -> tuple[Signal, ...]:
        raise NotImplementedError

    def analyze(self, src: Path) -> AnalysisResult:
        if src.suffix.lower() not in self.extensions:
            return AnalysisResult(self.name, 0, (), None, status="unsupported")
        executable = shutil.which(self.executable)
        if executable is None:
            return AnalysisResult(
                self.name,
                0,
                (),
                f"{self.executable} is not installed",
                status="unavailable",
            )
        command = self.command(src)
        command[0] = executable
        try:
            with tempfile.TemporaryFile() as output_file:
                completed = subprocess.run(  # noqa: S603 - fixed executable and argv only
                    command,
                    stdout=output_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=_TIMEOUT_SECONDS,
                    text=False,
                    stdin=subprocess.DEVNULL,
                    env={
                        "PATH": os.environ.get("PATH", ""),
                        "LANG": "C",
                        "LC_ALL": "C",
                    },
                    preexec_fn=_limits if os.name == "posix" else None,
                )
                output_file.seek(0)
                output_bytes = output_file.read(_MAX_OUTPUT)
        except subprocess.TimeoutExpired:
            return AnalysisResult(
                self.name, 0, (), None, status="error", error="subprocess timeout"
            )
        output = output_bytes.decode("utf-8", errors="replace")
        signals = self.interpret(output, completed.returncode)
        return AnalysisResult(
            self.name,
            max((signal.score for signal in signals), default=0),
            signals,
            None,
        )


class ZstegAdapter(_ExternalAnalyzer):
    name = "zsteg"
    executable = "zsteg"
    extensions = (".png", ".bmp")

    def command(self, src: Path) -> list[str]:
        return [self.executable, "--all", str(src)]

    def interpret(self, output: str, returncode: int) -> tuple[Signal, ...]:
        lines = [line for line in re.split(r"[\r\n]+", output) if _zsteg_candidate(line)]
        if not lines:
            return ()
        return (
            Signal(
                "zsteg_candidate",
                70,
                f"zsteg reported {len(lines)} candidate bit streams",
                category="external_zsteg",
                evidence="strong",
            ),
        )


# zsteg prints short random "text" and libmagic guesses for every natural image,
# and the high bit planes of smooth images read as long printable text. Only
# low-plane (b1/b2) or appended-data results that look like real content count.
_ZSTEG_ROW = re.compile(r"^\s*(?P<name>\S+)\s+\.\.\s+(?P<kind>text|file):\s*(?P<value>.*)$")
_ZSTEG_LOW = re.compile(r"^(b[12],[a-z]+,(lsb|msb)(,[a-zA-Z]+)?|extradata:\d+)$")
_ZSTEG_TEXT_ROWS = re.compile(r"^(chunk:\d+:\w+|meta\b.*)$")
_ZSTEG_FLAG = re.compile(r"[A-Za-z0-9_]{2,}\{[^{}\s]{2,}\}")
_ZSTEG_MIN_TEXT = 20
_ZSTEG_MIN_DISTINCT = 8
_ZSTEG_FILE_TYPES = ("zip", "gzip", "bzip2", "xz compressed", "7-zip", "rar archive",
                     "png image", "jpeg image", "gif image", "pdf document", "elf ", "pe32")


def _zsteg_candidate(line: str) -> bool:
    row = _ZSTEG_ROW.match(line)
    if not row:
        return False
    value = row["value"].strip()
    low = bool(_ZSTEG_LOW.match(row["name"]))
    if (low or _ZSTEG_TEXT_ROWS.match(row["name"])) and _ZSTEG_FLAG.search(value):
        return True
    if not low:
        return False
    if row["kind"] == "file":
        return any(kind in value.lower() for kind in _ZSTEG_FILE_TYPES)
    text = value[1:-1] if len(value) >= 2 and value[0] == value[-1] == '"' else ""
    if "repeated" in value or len(text) < _ZSTEG_MIN_TEXT:
        return False
    # Smooth regions give long but repetitive runs such as "jUUUUUUU...".
    common = max(text.count(c) for c in set(text))
    return len(set(text)) >= _ZSTEG_MIN_DISTINCT and common <= len(text) / 2


class StegseekAdapter(_ExternalAnalyzer):
    name = "stegseek"
    executable = "stegseek"
    extensions = (".jpg", ".jpeg", ".bmp", ".wav", ".au")

    def command(self, src: Path) -> list[str]:
        return [self.executable, "--seed", str(src)]

    def interpret(self, output: str, returncode: int) -> tuple[Signal, ...]:
        lowered = output.lower()
        if returncode == 0 and ("seed:" in lowered or "embedded" in lowered):
            return (
                Signal(
                    "steghide_seed_recovered",
                    95,
                    "Stegseek recovered a Steghide embedding seed",
                    category="external_stegseek",
                    evidence="strong",
                ),
            )
        return ()


class ExifToolAdapter(_ExternalAnalyzer):
    name = "exiftool"
    executable = "exiftool"
    extensions = (
        ".png",
        ".bmp",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".wav",
        ".pdf",
    )

    def command(self, src: Path) -> list[str]:
        return [self.executable, "-validate", "-warning", "-error", str(src)]

    def interpret(self, output: str, returncode: int) -> tuple[Signal, ...]:
        warnings = [line for line in output.splitlines() if "warning" in line.lower()]
        errors = [line for line in output.splitlines() if "error" in line.lower()]
        if not warnings and not errors:
            return ()
        score = min(60, 15 + len(warnings) * 5 + len(errors) * 10)
        return (
            Signal(
                "metadata_validation",
                score,
                f"ExifTool validation: {len(warnings)} warnings, {len(errors)} errors",
                category="metadata_structure",
                evidence="heuristic",
            ),
        )


def _limits() -> None:
    """Apply limits in the child immediately before exec (POSIX only)."""
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (_MEMORY_BYTES, _MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU, (_TIMEOUT_SECONDS, _TIMEOUT_SECONDS + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (_MAX_OUTPUT, _MAX_OUTPUT))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
