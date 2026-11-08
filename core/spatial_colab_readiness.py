"""Read-only Colab readiness; hardware visibility never establishes resource control."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import resource
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from core import colab_artifacts as a

MAX_SECONDS = 60
MAX_OUTPUT_BYTES = 65536
PACKAGES = ("numpy", "Pillow", "torch", "onnx", "onnxruntime")
EXPECTED = {"python": "3.12.13", "numpy": "2.0.2", "Pillow": "11.3.0"}
COUNTERS = {
    "v2": ("memory.max", "memory.swap.max", "memory.peak", "memory.current"),
    "v1": (
        "memory.limit_in_bytes", "memory.memsw.limit_in_bytes",
        "memory.max_usage_in_bytes", "memory.usage_in_bytes",
    ),
}


@dataclass(frozen=True)
class ResourceMetadata:
    status: str
    reason: str
    version: str | None = None
    counters: tuple[tuple[str, int | None], ...] = ()


@dataclass(frozen=True)
class GpuMetadata:
    status: str
    reason: str
    name: str | None = None
    vram_mib: int | None = None
    driver: str | None = None


def resources(
    *, proc_root: Path = Path("/proc"), cgroup_root: Path = Path("/sys/fs/cgroup"),
    pid: int | None = None,
) -> ResourceMetadata:
    """Refuse inaccessible namespace identities; never normalize parent segments."""
    process = os.getpid() if pid is None else pid
    if type(process) is not int or not 0 < process < 2**31:
        raise ValueError("positive bounded process identity required")
    try:
        raw = a.read(proc_root / str(process) / "cgroup", 4096).decode()
        candidates = []
        for line in raw.splitlines():
            fields = line.split(":", 2)
            if len(fields) != 3 or not fields[0].isascii() or not fields[0].isdecimal():
                return ResourceMetadata("unavailable", "malformed_identity")
            hierarchy, controllers, relative = fields
            version = "v1" if "memory" in controllers.split(",") else "v2"
            if version == "v2" and (hierarchy != "0" or controllers):
                continue
            path = PurePosixPath(relative)
            if ".." in path.parts:
                return ResourceMetadata("unavailable", "parent_identity", version)
            if not path.is_absolute() or len(path.parts) > 32 or "\x00" in relative:
                return ResourceMetadata("unavailable", "malformed_identity", version)
            folder = cgroup_root / "memory" if version == "v1" else cgroup_root
            candidates.append((version, folder.joinpath(*path.parts[1:])))
        if len(candidates) != 1:
            return ResourceMetadata("unavailable", "ambiguous_or_missing_controller")
        version, folder = candidates[0]
        counters = []
        for name in COUNTERS[version]:
            value = a.read(folder / name, 32).decode().strip()
            if value == "max":
                number = None
            elif value.isascii() and value.isdecimal() and len(value) <= 20:
                number = int(value)
                number = number if number < 2**60 else None
            else:
                return ResourceMetadata("unavailable", "malformed_counter", version)
            counters.append((name, number))
        return ResourceMetadata(
            "observed_metadata_only", "not_enforcement", version, tuple(counters)
        )
    except (OSError, UnicodeError, ValueError):
        return ResourceMetadata("unavailable", "inaccessible_or_unsafe_metadata")


def gpu(*, workdir: Path | None = None) -> GpuMetadata:
    """One fixed, bounded NVIDIA query; no Torch import or tensor operation."""
    if workdir is not None:
        a.safe(workdir)
    limiter, binary = shutil.which("prlimit"), shutil.which("nvidia-smi")
    if limiter is None or binary is None:
        return GpuMetadata("unavailable", "missing_query_tool")
    arguments = [
        limiter, "--fsize=8192:8192", "--cpu=5:5", "--", binary,
        "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits",
    ]
    try:
        with tempfile.TemporaryFile(dir=workdir) as capture:
            process = subprocess.run(  # noqa: S603 -- fixed system query and kernel limits
                arguments, stdout=capture, stderr=subprocess.STDOUT, timeout=10, check=False,
            )
            capture.seek(0)
            raw = capture.read(8193)
        if process.returncode or len(raw) > 8192:
            return GpuMetadata("unavailable", "query_failed_or_oversized")
        lines = raw.decode("ascii").strip().splitlines()
        if len(lines) != 1:
            return GpuMetadata("unavailable", "single_gpu_required")
        fields = [text.strip() for text in lines[0].split(",")]
        if (
            len(fields) != 3 or not 0 < len(fields[0]) <= 128
            or not all(32 <= ord(c) < 127 for c in fields[0])
            or not fields[1].isdigit() or len(fields[1]) > 6
            or not 0 < int(fields[1]) <= 524288
            or re.fullmatch(r"[0-9]{1,4}(?:\.[0-9]{1,4}){1,3}", fields[2]) is None
        ):
            return GpuMetadata("unavailable", "invalid_query_format")
        return GpuMetadata("visible_metadata_only", "not_kernel_execution", fields[0],
                           int(fields[1]), fields[2])
    except subprocess.TimeoutExpired:
        return GpuMetadata("unavailable", "query_timeout")
    except (OSError, UnicodeError):
        return GpuMetadata("unavailable", "query_unavailable")


def inventory(*, check_gpu: bool = False, workdir: Path | None = None) -> dict:
    """Observe independent components; resource failure does not skip the GPU query."""
    if type(check_gpu) is not bool:
        raise ValueError("explicit boolean GPU query choice required")
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    observed = {"python": platform.python_version(), **packages}
    expected = {**EXPECTED, "torch": "2.11.0+cu128" if check_gpu else "2.11.0+cpu"}
    matches = all(observed.get(k) == v for k, v in expected.items())
    host = resources()
    card = gpu(workdir=workdir) if check_gpu else GpuMetadata("not_requested", "cpu_default")
    if not matches:
        status = "unavailable_environment"
    elif check_gpu and card.status != "visible_metadata_only":
        status = "unavailable_gpu"
    elif host.status != "observed_metadata_only":
        status = "partial_resources_unavailable"
    else:
        status = "metadata_ready_only"
    return {
        "schema": "spatial-colab-readiness-v3", "status": status,
        "python": observed["python"], "packages": packages, "package_metadata_matches": matches,
        "resources": asdict(host), "gpu": asdict(card), "resource_enforcement": "unverified",
        "process_limits": {
            name: list(resource.getrlimit(key))
            for name, key in (("address_space", resource.RLIMIT_AS),
                              ("cpu_seconds", resource.RLIMIT_CPU),
                              ("file_bytes", resource.RLIMIT_FSIZE))
        },
        "models_created": 0, "optimizer_updates": 0, "tensor_operations": 0,
        "torch_imported_by_service": False, "profile_attempt_started": False,
    }


def record(out: Path, *, check_gpu: bool = False) -> dict:
    """Exclusive metadata evidence, without allocating a runtime or authorizing fitting."""
    if type(check_gpu) is not bool:
        raise ValueError("explicit boolean GPU query choice required")
    started = time.monotonic()
    a.directory(out)
    a.put(out / "attempt.json", a.encode({
        "schema": "spatial-colab-readiness-attempt-v3", "check_gpu": check_gpu,
        "max_seconds": MAX_SECONDS, "retry": False,
    }))
    result = inventory(check_gpu=check_gpu, workdir=out)
    result["seconds"] = time.monotonic() - started
    if result["seconds"] > MAX_SECONDS:
        result["status"] = "unavailable_deadline"
    result["receipt_sha256"] = a.put(out / "result.json", a.encode(result), MAX_OUTPUT_BYTES)
    return result
