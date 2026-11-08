"""One native pilot phase and live closing observer; never reuse old attempts."""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path, PurePosixPath

from core import colab_artifacts as a
from core import spatial_native_data as data
from core import spatial_native_pilot as service


def kernel(pid: int | None = None) -> dict:
    proc = Path("/proc") / str(pid or os.getpid())
    raw = a.read(proc / "cgroup", 4096).decode()
    lines = [s[3:] for s in raw.splitlines() if s.startswith("0::")]
    if len(lines) != 1:
        raise ValueError("native cgroup identity")
    relative = PurePosixPath(lines[0])
    if not relative.is_absolute() or ".." in relative.parts or len(relative.parts) > 32:
        raise ValueError("native cgroup path")
    folder = Path("/sys/fs/cgroup").joinpath(*relative.parts[1:])

    def number(name):
        value = a.read(folder / name, 32).decode().strip()
        if not value.isascii() or not value.isdecimal():
            raise ValueError("finite native cgroup counter")
        return int(value)

    result = {
        k: number(v)
        for k, v in {
            "memory_max_bytes": "memory.max",
            "swap_max_bytes": "memory.swap.max",
            "memory_peak_bytes": "memory.peak",
            "memory_current_bytes": "memory.current",
        }.items()
    }
    if not 0 < result["memory_max_bytes"] <= 8 * 1024**3 or result["swap_max_bytes"] != 0:
        raise ValueError("native kernel RAM/swap envelope")
    return result


def inventory(out: Path) -> tuple[int, dict[str, dict]]:
    result: dict[str, dict] = {}
    total = 0
    for p in sorted(out.rglob("*")):
        a.safe(p)
        if p.is_dir():
            continue
        info = p.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > 128 * a.MIB or len(result) >= 4096:
            raise ValueError("bounded native output inventory")
        result[str(p.relative_to(out))] = {
            "bytes": info.st_size,
            "sha256": service.checksum(p) if info.st_size <= 4 * a.MIB else streaming_hash(p),
        }
        total += info.st_size
    return total, result


def streaming_hash(path: Path) -> str:
    import hashlib

    a.safe(path)
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("native output regular file")
        size = 0
        while chunk := stream.read(a.MIB):
            size += len(chunk)
            if size > 128 * a.MIB:
                raise ValueError("native output read limit")
            digest.update(chunk)
    return digest.hexdigest()


def observe(out: Path, binding: str) -> dict:
    """External bounded-byte observer; the worker stays alive until its receipt."""
    a.identity(binding)
    ready = a.document(out / "ready.json", 4 * a.MIB)
    if (
        ready.get("protocol_sha256") != binding
        or type(ready.get("pid")) is not int
        or ready["pid"] <= 0
    ):
        raise ValueError("native live observer binding")
    live = kernel(ready["pid"])
    count, files = inventory(out)
    limit = service.PROFILE_OUTPUT if ready["phase"] == "profile" else ready["output_limit_bytes"]
    if count + 65536 > limit or live["memory_peak_bytes"] > live["memory_max_bytes"]:
        raise ValueError("native live closing resource bound")
    value = {
        "schema": "spatial-native-live-observer-v1",
        "protocol_sha256": binding,
        "pid": ready["pid"],
        "kernel": live,
        "logical_output_bytes": count + 65536,
        "files": files,
        "observed_monotonic": time.monotonic(),
    }
    a.put(out / "observer.json", a.encode(value), 4 * a.MIB)
    return value


def execute(
    root: Path,
    protocol: Path,
    cache: data.Cache,
    out: Path,
    *,
    phase: str,
    approval_path: Path | None = None,
    profile_path: Path | None = None,
) -> dict:
    config, binding, split = service.verify(root, protocol, cache)
    if phase not in ("profile", "fit"):
        raise ValueError("explicit native pilot phase")
    approval = None
    if phase == "fit":
        if approval_path is None or profile_path is None:
            raise ValueError("native fit approval inputs")
        approval = a.document(approval_path)
        prior = a.document(profile_path, 4 * a.MIB)
        seconds, byte_limit = service.approved_limits(
            approval, prior, service.checksum(profile_path), binding, split
        )
    else:
        seconds, byte_limit = service.PROFILE_SECONDS, service.PROFILE_OUTPUT
    marker = out.parent / f"{phase}-attempt.json"
    a.put(
        marker,
        a.encode(
            {
                "schema": "spatial-native-attempt-v1",
                "phase": phase,
                "protocol_sha256": binding,
                "pid": os.getpid(),
                "monotonic": time.monotonic(),
                "retry": False,
            }
        ),
    )
    started = time.monotonic()
    deadline = started + seconds
    kernel()
    if service.historical.environment() != config["environment"]:
        raise ValueError("native pilot environment mismatch")
    if phase == "profile":
        payload = service.measure(
            cache, split, out, binding, seconds=min(220, service.remaining(deadline) - 20)
        )
        projected = service.projection(payload)
    else:
        if approval is None or profile_path is None:
            raise ValueError("native fit approval inputs")
        payload = service.fit(root, protocol, cache, out, approval, profile_path)
        projected = None
    service.verify(root, protocol, cache)
    count, _ = inventory(out)
    if count + 65536 > byte_limit:
        raise ValueError("native pilot output bound")
    ready = {
        "schema": "spatial-native-observer-ready-v1",
        "protocol_sha256": binding,
        "phase": phase,
        "pid": os.getpid(),
        "output_limit_bytes": byte_limit,
    }
    a.put(out / "ready.json", a.encode(ready))
    while not (out / "observer.json").exists():
        if service.remaining(deadline) <= 1:
            raise TimeoutError("native live observer deadline")
        time.sleep(0.1)
    observer = a.document(out / "observer.json", 4 * a.MIB)
    if (
        observer.get("protocol_sha256") != binding
        or observer.get("pid") != os.getpid()
        or observer["kernel"]["swap_max_bytes"] != 0
        or not 0
        < observer["kernel"]["memory_peak_bytes"]
        <= observer["kernel"]["memory_max_bytes"]
        <= 8 * 1024**3
    ):
        raise ValueError("native closing observer receipt")
    result = {
        "schema": "spatial-rgb-native-profile-result-v1"
        if phase == "profile"
        else "spatial-rgb-native-fit-result-v1",
        "status": "profile_ready" if phase == "profile" else payload["status"],
        "protocol_sha256": binding,
        "phase": phase,
        "physical_cuda": True,
        "phase_seconds_limit": seconds,
        "phase_output_limit_bytes": byte_limit,
        "charged_seconds": time.monotonic() - started + 2,
        "logical_output_bytes": observer["logical_output_bytes"],
        "kernel": observer["kernel"],
        "observer_sha256": service.checksum(out / "observer.json"),
        "projection": projected,
        "payload_sha256": service.checksum(
            out / ("measurement.json" if phase == "profile" else "fit.json")
        ),
    }
    if result["charged_seconds"] > seconds:
        raise TimeoutError("native closing total deadline")
    a.put(out / "result.json", a.encode(result), 4 * a.MIB)
    return result


def finalize(
    root: Path, protocol: Path, cache: data.Cache, out: Path, start_us: int, exit_us: int
) -> dict:
    """Independent post-exit time debit, using the retained live-kernel observer."""
    _, binding, _ = service.verify(root, protocol, cache)
    worker = a.document(out / "result.json", 4 * a.MIB)
    observer = a.document(out / "observer.json", 4 * a.MIB)
    if (
        worker.get("protocol_sha256") != binding
        or observer.get("protocol_sha256") != binding
        or worker["observer_sha256"] != service.checksum(out / "observer.json")
        or type(start_us) is not int
        or type(exit_us) is not int
        or not 0 < start_us < exit_us
    ):
        raise ValueError("native post-exit binding/time evidence")
    limit = worker["phase_seconds_limit"]
    byte_limit = worker["phase_output_limit_bytes"]
    if (
        not 0 < limit <= service.FIT_SECONDS_CEILING
        or not 0 < byte_limit <= service.CUMULATIVE_OUTPUT_CEILING
    ):
        raise ValueError("native post-exit allowance")
    elapsed = (exit_us - start_us) / 1e6
    charged = max(worker["charged_seconds"], elapsed)
    total, files = inventory(out)
    if charged > limit or total + 65536 > byte_limit:
        raise ValueError("native post-exit cumulative resource bound")
    for name, record in observer["files"].items():
        if name not in files or files[name] != record:
            raise ValueError("native post-exit immutable output changed")
    result = {
        **worker,
        "charged_seconds": charged,
        "kernel_execution_seconds": elapsed,
        "logical_output_bytes": total + 65536,
        "output_files_verified": len(files),
        "peak_source": "live cgroup memory.peak after work and checkpoint verification",
        "post_exit_peak_required": False,
    }
    if worker["phase"] == "profile":
        measured = a.document(out / "measurement.json", 4 * a.MIB)
        if service.checksum(out / "measurement.json") != worker["payload_sha256"]:
            raise ValueError("native measurement payload hash")
        result["max_checkpoint_bytes"] = (
            max(c["checkpoint_bytes"] for c in measured["cells"]) + a.MIB
        )
        result["candidate_fit_eligible"] = (
            charged + result["projection"]["fit_projection_seconds"] < 1800
            and total + 65536 + 8 * result["max_checkpoint_bytes"] + 8 * a.MIB
            < service.CUMULATIVE_OUTPUT_CEILING
        )
    a.put(out / "accepted.json", a.encode(result), 4 * a.MIB)
    return result
