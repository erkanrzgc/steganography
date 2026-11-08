"""Source-bound Colab process controller; no runtime allocation or implicit fitting."""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import signal
import stat
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from core import colab_artifacts as a

SCHEMA = "colab-process-contract-v1"
NATIVE_SCHEMA = "colab-native-profile-process-contract-v2"
POLICY = "hard-data-limit-and-resident-watch-v1"
NATIVE_POLICY = "provider-managed-ram-with-headroom-watch-v1"
KINDS = ("generated_qa", "native_profile", "native_fit")
BOOTSTRAP = (
    "import sys;sys.path.insert(0,sys.argv.pop(1));"
    "from core.colab_execution import worker_main;raise SystemExit(worker_main())"
)


@dataclass(frozen=True)
class Limits:
    seconds: float
    data_bytes: int
    resident_watch_bytes: int
    output_bytes: int
    file_bytes: int
    address_bytes: int = 0
    closing_seconds: float = 5.0

    def __post_init__(self):
        for value in (self.seconds, self.closing_seconds):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("finite process time limits required")
        if not 0.2 <= self.seconds <= 1800 or not 0.05 <= self.closing_seconds < self.seconds:
            raise ValueError("bounded process/closing time required")
        values = (self.data_bytes, self.resident_watch_bytes, self.output_bytes, self.file_bytes)
        if any(type(v) is not int for v in (*values, self.address_bytes)):
            raise ValueError("integer process limits required")
        automatic_memory = self.data_bytes == self.resident_watch_bytes == 0
        if not automatic_memory and not 16 * a.MIB <= self.data_bytes <= 16 * 1024**3:
            raise ValueError("bounded data segment required")
        if not automatic_memory and not 16 * a.MIB <= self.resident_watch_bytes <= 16 * 1024**3:
            raise ValueError("bounded resident watch required")
        if not 128 * 1024 <= self.output_bytes <= 768 * a.MIB:
            raise ValueError("bounded cumulative output required")
        if not 65536 <= self.file_bytes <= min(self.output_bytes, 128 * a.MIB):
            raise ValueError("bounded per-file output required")
        if self.address_bytes and not 32 * a.MIB <= self.address_bytes <= 1024**4:
            raise ValueError("bounded address space required")


def source_names(root: Path) -> tuple[str, ...]:
    return tuple(sorted(
        str(p.relative_to(root))
        for folder in ("core", "steganography")
        for p in (root / folder).rglob("*.py")
    ))


def source_hashes(root: Path) -> dict[str, str]:
    names = source_names(root)
    if not names or len(names) > 512:
        raise ValueError("bounded reviewed source closure required")
    result = {}
    for name in names:
        path = root / name
        a.safe(path)
        result[name] = a.digest(a.read(path, a.MIB) if path.stat().st_size else b"")
    return result


def environment() -> dict:
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": sys.platform}


def freeze(root: Path, path: Path, kind: str, payload: dict, limits: Limits, *,
           target_environment: dict | None = None) -> dict:
    if kind not in KINDS or not isinstance(payload, dict) or type(limits) is not Limits:
        raise ValueError("explicit registered process contract required")
    if kind == "generated_qa" and not limits.data_bytes:
        raise ValueError("generated QA requires its explicit fixed memory limits")
    if kind != "generated_qa":
        from core import spatial_colab_fit, spatial_colab_profile

        service = spatial_colab_fit if kind == "native_fit" else spatial_colab_profile
        service.contract(payload, limits)
    if target_environment is not None and (kind == "generated_qa" or target_environment != {
        "python": "3.12.13", "implementation": "CPython", "platform": "linux",
    }):
        raise ValueError("only an explicit prospective Colab interpreter may be frozen")
    value = {
        "schema": SCHEMA if kind == "generated_qa" else NATIVE_SCHEMA,
        "kind": kind, "policy": POLICY if kind == "generated_qa" else NATIVE_POLICY,
        "limits": asdict(limits), "payload": payload,
        "environment": environment() if target_environment is None else target_environment,
        "source_sha256": source_hashes(root), "automatic_retry": False,
        "native_execution_available": kind != "generated_qa",
    }
    a.put(path, a.encode(value), 4 * a.MIB)
    return value


def source_contract(root: Path, path: Path) -> tuple[dict, str, Limits]:
    raw = a.read(path, 4 * a.MIB)
    value = json.loads(raw)
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "kind", "policy", "limits", "payload", "environment",
                          "source_sha256", "automatic_retry",
                          "native_execution_available"}
        or value["kind"] not in KINDS
        or value["schema"] != (SCHEMA if value["kind"] == "generated_qa" else NATIVE_SCHEMA)
        or value["policy"] != (POLICY if value["kind"] == "generated_qa" else NATIVE_POLICY)
        or value["automatic_retry"] is not False
        or value["native_execution_available"] is not (value["kind"] != "generated_qa")
        or not isinstance(value["payload"], dict) or not isinstance(value["environment"], dict)
    ):
        raise ValueError("process contract schema mismatch")
    limits = Limits(**value["limits"])
    if value["kind"] == "generated_qa" and not limits.data_bytes:
        raise ValueError("generated QA requires fixed memory limits")
    if value["source_sha256"] != source_hashes(root):
        raise ValueError("process source binding mismatch")
    if value["kind"] != "generated_qa":
        from core import spatial_colab_fit, spatial_colab_profile

        service = spatial_colab_fit if value["kind"] == "native_fit" else spatial_colab_profile
        service.contract(value["payload"], limits)
    return value, a.digest(raw), limits


def verify(root: Path, path: Path) -> tuple[dict, str, Limits]:
    value, binding, limits = source_contract(root, path)
    if value["environment"] != environment():
        raise ValueError("process environment binding mismatch")
    return value, binding, limits


def permission(job: dict, binding: str, path: Path | None) -> None:
    if job["kind"] == "generated_qa":
        if path is not None:
            raise ValueError("generated QA does not consume native permission")
        return
    if job["kind"] == "native_fit":
        from core.spatial_colab_fit import approval

        approval(job, binding, path)
        return
    from core.spatial_colab_profile import RESOURCE_ACK

    value = a.document(path) if path is not None else None
    if value is None or value.get("user_approved") is not True or value != {
        "schema": "colab-profile-permission-v1", "user_approved": True,
        "contract_sha256": binding, "kind": "native_profile",
        "resource_acknowledgement": RESOURCE_ACK,
    }:
        raise ValueError("explicit source/resource-bound Colab profile permission required")


def files(out: Path, limits: Limits, *, hashes: bool = False,
          closing_bytes: int = 65536) -> tuple[int, dict]:
    total, result, stack = 0, {}, [(out, 0)]
    entries = 0
    while stack:
        folder, depth = stack.pop()
        if depth > 8:
            raise ValueError("process output depth")
        with os.scandir(folder) as listing:
            for entry in listing:
                entries += 1
                if entries > 4096:
                    raise ValueError("process output file count")
                path = Path(entry.path)
                try:
                    info = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    if hashes:
                        raise  # Closing inventory must never accept disappearing evidence.
                    continue  # A live atomic progress rename can remove its temporary entry.
                if stat.S_ISDIR(info.st_mode):
                    stack.append((path, depth + 1))
                    continue
                if not stat.S_ISREG(info.st_mode) or info.st_size > limits.file_bytes:
                    raise ValueError("process output type/size")
                total += info.st_size
                if total + closing_bytes > limits.output_bytes:
                    raise ValueError("process cumulative output")
                item: dict[str, Any] = {"bytes": info.st_size}
                if hashes:
                    a.safe(path)
                    digest = hashlib.sha256()
                    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    with os.fdopen(fd, "rb") as stream:
                        count = 0
                        while chunk := stream.read(a.MIB):
                            count += len(chunk)
                            if count > limits.file_bytes:
                                raise ValueError("process output changed during audit")
                            digest.update(chunk)
                    if count != info.st_size:
                        raise ValueError("process output changed during audit")
                    item["sha256"] = digest.hexdigest()
                result[str(path.relative_to(out))] = item
    return total, result


def resident(pid: int) -> dict | None:
    try:
        raw = a.read(Path("/proc") / str(pid) / "status", 32768).decode()
    except (FileNotFoundError, ProcessLookupError):
        return None  # The exact child is disappearing; wait4 must still reap/charge it.
    states = [line.split() for line in raw.splitlines() if line.startswith("State:")]
    if len(states) != 1 or len(states[0]) < 2:
        raise ValueError("process state unavailable")
    if states[0][1] in ("Z", "X"):
        return None  # wait4 supplies the final peak for a just-exited worker.
    values = {}
    for key in ("VmRSS", "VmHWM", "VmSwap"):
        lines = [line.split() for line in raw.splitlines() if line.startswith(key + ":")]
        if len(lines) != 1 or len(lines[0]) != 3 or lines[0][2] != "kB":
            raise ValueError("process resident counter unavailable")
        value = lines[0][1]
        if not value.isascii() or not value.isdecimal() or len(value) > 16:
            raise ValueError("process resident counter malformed")
        values[key] = int(value) * 1024
    return values


def _kill(pid: int) -> None:
    with suppress(ProcessLookupError):
        os.killpg(pid, signal.SIGKILL)


def run(root: Path, protocol: Path, out: Path, *, permission_path: Path | None = None,
        cache: Path | None = None, profile_out: Path | None = None) -> dict:
    """One registered worker; interruption preserves every output and consumes the intent."""
    started = time.monotonic()
    job, binding, limits = verify(root, protocol)
    permission(job, binding, permission_path)
    receipt_limit = 4 * a.MIB if job["kind"] == "native_fit" else 60000
    if job["kind"] == "native_fit":
        from core import spatial_colab_fit

        if profile_out is None:
            raise ValueError("accepted Colab profile directory required")
        spatial_colab_fit.bind_profile(root, job, profile_out)
    elif profile_out is not None:
        raise ValueError("profile input belongs to native fitting only")
    initial_memory = None
    if job["kind"] != "generated_qa":
        if cache is None:
            raise ValueError("explicit native cache required")
        from core import spatial_colab_profile as profile

        profile.cache_binding(profile.data.load(cache, job["payload"]["split"]["manifest_sha256"]),
                              profile.engine.split_document(job["payload"]["split"]))
        from core import colab_memory

        initial_memory = colab_memory.observe()
    elif cache is not None:
        raise ValueError("generated process may not open a native cache")
    a.safe(out)
    if out.exists():
        raise FileExistsError("process output already exists")
    limiter = shutil.which("prlimit")
    if limiter is None or sys.platform != "linux":
        raise RuntimeError("Linux kernel process limits required")
    ledger = root / ".colab-process-attempts"
    a.safe(ledger)
    ledger.mkdir(mode=0o700, exist_ok=True)
    if job["kind"] == "native_fit":
        profile_key = job["payload"]["profile_receipt_sha256"]
        a.put(ledger / ("fit-profile-" + profile_key + ".intent.json"), a.encode({
            "profile_receipt_sha256": profile_key, "contract_sha256": binding,
            "retry": False,
        }))
    a.put(ledger / (binding + ".intent.json"), a.encode({
        "schema": "colab-process-attempt-v1", "contract_sha256": binding,
        "kind": job["kind"], "retry": False,
    }))
    a.directory(out)
    a.put(out / "contract.json", a.encode(job), 4 * a.MIB)
    if initial_memory is not None:
        a.put(out / "runtime-memory-before-worker.json", a.encode(initial_memory))
    arguments = [
        limiter,
        f"--fsize={limits.file_bytes}:{limits.file_bytes}",
        f"--cpu={max(1, math.ceil(limits.seconds))}:{max(1, math.ceil(limits.seconds))}",
        "--core=0:0",
    ]
    if limits.data_bytes:
        arguments.append(f"--data={limits.data_bytes}:{limits.data_bytes}")
    if limits.address_bytes:
        arguments.append(f"--as={limits.address_bytes}:{limits.address_bytes}")
    arguments += ["--", sys.executable, "-I", "-B", "-c", BOOTSTRAP, str(root.absolute()),
                  str(protocol.absolute()), str(out.absolute()), str(os.getpid())]
    env = {key: os.environ[key] for key in ("PATH", "LD_LIBRARY_PATH", "LANG", "LC_ALL")
           if key in os.environ}
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"})
    if job["kind"] == "generated_qa":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["STEGANOGRAPHY_COLAB_CACHE"] = str(cache.absolute())  # type: ignore[union-attr]
        env["STEGANOGRAPHY_COLAB_PERMISSION"] = str(permission_path.absolute())  # type: ignore[union-attr]
    if profile_out is not None:
        env["STEGANOGRAPHY_COLAB_PROFILE"] = str(profile_out.absolute())
    child = None
    usage = None
    failure = None
    status = None
    peak_observed = 0
    observer_gaps = consecutive_gaps = 0
    last_progress = None
    try:
        with (out / "worker.log").open("xb") as log:
            if initial_memory is not None:
                colab_memory.check(initial_memory)
            child = subprocess.Popen(  # noqa: S603 -- fixed, source-verified worker bootstrap
                arguments, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
            )
            deadline = started + limits.seconds - limits.closing_seconds
            while True:
                pid, state, observed = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    status, usage = os.waitstatus_to_exitcode(state), observed
                    child.returncode = status
                    break
                try:
                    snapshot = resident(child.pid)
                    if snapshot is not None:
                        peak_observed = max(peak_observed, snapshot["VmHWM"])
                    if snapshot is not None and (
                        (limits.resident_watch_bytes
                         and peak_observed > limits.resident_watch_bytes) or snapshot["VmSwap"]
                    ):
                        raise MemoryError("process resident/swap watch")
                    consecutive_gaps = 0
                except FileNotFoundError:
                    pass
                except ValueError:
                    # exec/exit can briefly expose a task with no memory map.
                    observer_gaps += 1
                    consecutive_gaps += 1
                    if consecutive_gaps >= 3:
                        raise ValueError("process resident observer unavailable") from None
                if initial_memory is not None:
                    colab_memory.check(colab_memory.observe(), initial=initial_memory)
                if job["kind"] == "native_fit":
                    progress_path = out / "work" / "fit" / "progress.json"
                    if progress_path.exists():
                        progress = a.document(progress_path, 65536)
                        if progress["contract_sha256"] != binding:
                            raise ValueError("training progress binding mismatch")
                        marker = progress["seed"], progress["epoch"]
                        if marker != last_progress:
                            print(spatial_colab_fit.jsonline(progress), flush=True)
                            last_progress = marker
                files(out, limits, closing_bytes=receipt_limit)
                if time.monotonic() >= deadline:
                    raise TimeoutError("complete process deadline")
                time.sleep(min(0.05, max(0.001, deadline - time.monotonic())))
            if status:
                failure = "worker_failed"
    except BaseException as error:
        failure = (
            "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit))
            else type(error).__name__
        )
    finally:
        if child is not None and usage is None:
            _kill(child.pid)
            _, state, usage = os.wait4(child.pid, 0)
            status = os.waitstatus_to_exitcode(state)
            child.returncode = status
    try:
        if verify(root, protocol)[1] != binding:
            raise ValueError("process contract changed during execution")
        total, inventory = files(out, limits, hashes=True, closing_bytes=receipt_limit)
        peak = int(usage.ru_maxrss * 1024) if usage is not None else None
        if peak is None or (limits.resident_watch_bytes and peak > limits.resident_watch_bytes):
            failure = failure or "unavailable_resident_peak"
        worker = a.document(out / "work" / "worker-result.json", 4 * a.MIB)
        if worker.get("contract_sha256") != binding:
            raise ValueError("worker result binding")
        if job["kind"] == "native_profile" and failure is None:
            from core import spatial_colab_profile
            from core.spatial_colab_handoff import evidence_package

            if worker["split"] != job["payload"]["split"] or worker["physical_cuda"] is not True:
                raise ValueError("complete physical native profile binding required")
            spatial_colab_profile.audit_measurement(out / "work" / "profile", worker, binding,
                                                    worker["environment"])
            evidence_package(out, limits, binding)
            total, inventory = files(out, limits, hashes=True)
            if verify(root, protocol)[1] != binding:
                raise ValueError("profile source changed while closing evidence")
        if job["kind"] == "native_fit" and failure is None:
            from core.spatial_colab_handoff import evidence_package

            if (worker["split"] != job["payload"]["split"] or worker["physical_cuda"] is not True
                    or worker["recipe"] != job["payload"]["recipe"]
                    or worker["environment"] != job["payload"]["environment"]
                    or worker["epochs"] != 20 or worker["fit_updates"] != 640
                    or worker["presentations"] != 5120):
                raise ValueError("complete physical fitting result required")
            spatial_colab_fit.audit_training(out / "work" / "fit", worker, binding,
                                            worker["environment"])
            evidence_package(out, limits, binding, include_numeric=False)
            total, inventory = files(out, limits, hashes=True, closing_bytes=receipt_limit)
            if verify(root, protocol)[1] != binding:
                raise ValueError("fitting source changed during closing")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        total, inventory, worker = None, {}, None
        failure = failure or "unavailable_closing_audit"
        peak = int(usage.ru_maxrss * 1024) if usage is not None else None
    # Conservatively debit a full closing reserve in addition to all observed work.
    elapsed = time.monotonic() - started + limits.closing_seconds
    if elapsed > limits.seconds:
        failure = failure or "unavailable_total_deadline"
    result = {
        "schema": "colab-process-closed-v1",
        "status": "completed" if failure is None else "unavailable",
        "failure": failure, "kind": job["kind"], "contract_sha256": binding,
        "policy": job["policy"], "hard_resident_cgroup_limit": "not_enforced",
        "hard_swap_cgroup_limit": "not_enforced", "limits": asdict(limits),
        "charged_seconds": elapsed, "worker_exit": status, "worker_reaped": usage is not None,
        "kernel_worker_peak_resident_bytes": peak, "live_observed_peak_bytes": peak_observed,
        "kernel_peak_includes_spawn": True, "resident_observer_gaps": observer_gaps,
        "kernel_worker_cpu_seconds": usage.ru_utime + usage.ru_stime if usage else None,
        "logical_output_bytes_before_closing": total, "output_files": inventory,
        "worker": worker, "automatic_retry": False,
        "runtime_memory": initial_memory,
    }
    if len(a.encode(result)) > receipt_limit:
        result.update(status="unavailable", failure="unavailable_receipt_size", output_files={})
    receipt = a.put(out / "closed.json", a.encode(result), receipt_limit)
    a.put(ledger / (binding + ".closed.json"), a.encode({
        "contract_sha256": binding, "status": result["status"],
        "receipt_sha256": receipt,
    }))
    if time.monotonic() - started > min(elapsed, limits.seconds):
        a.put(out / "unavailable-finalization.json", a.encode({
            "status": "unavailable", "failure": "closing_reserve_exceeded",
            "contract_sha256": binding,
        }))
        raise TimeoutError("closing exceeded its conservative debit; receipt invalidated")
    return result


def audit(root: Path, protocol: Path, out: Path) -> dict:
    """Verify the exclusive ledger and every closed output; never resume a worker."""
    job, binding, limits = verify(root, protocol)
    if (out / "unavailable-finalization.json").exists():
        raise ValueError("closing receipt invalidated")
    receipt_limit = 4 * a.MIB if job["kind"] == "native_fit" else 60000
    raw = a.read(out / "closed.json", receipt_limit)
    result = json.loads(raw)
    anchor = a.document(root / ".colab-process-attempts" / (binding + ".closed.json"))
    if anchor != {"contract_sha256": binding, "status": result["status"],
                  "receipt_sha256": a.digest(raw)}:
        raise ValueError("closing ledger binding mismatch")
    if (result["contract_sha256"] != binding or result["kind"] != job["kind"]
            or result["limits"] != asdict(limits) or result["policy"] != job["policy"]):
        raise ValueError("closing process binding mismatch")
    if result["status"] != "completed":
        return result  # Retain unavailable evidence without relabelling a failed audit.
    total, inventory = files(out, limits, hashes=True, closing_bytes=receipt_limit)
    inventory.pop("closed.json")
    if (inventory != result["output_files"] or total > limits.output_bytes
            or not 0 < result["charged_seconds"] <= limits.seconds
            or result["worker_exit"] != 0 or result["worker_reaped"] is not True
            or result["kernel_worker_peak_resident_bytes"] <= 0
            or (limits.resident_watch_bytes
                and result["kernel_worker_peak_resident_bytes"] > limits.resident_watch_bytes)):
        raise ValueError("closed evidence changed or exceeds limits")
    return result


def _qa(payload: dict, out: Path, binding: str) -> dict:
    if set(payload) != {"action", "value"} or payload["action"] not in {
        "success", "sleep", "cpu", "memory", "output", "symlink", "crash",
    } or type(payload["value"]) not in (int, float) or not math.isfinite(payload["value"]):
        raise ValueError("generated QA action required")
    if not 0 <= payload["value"] <= 128 * a.MIB:
        raise ValueError("bounded generated QA value")
    action, value = payload["action"], payload["value"]
    if action == "sleep":
        time.sleep(value)
    elif action == "cpu":
        until = time.monotonic() + min(value, 30)
        while time.monotonic() < until:
            pass
    elif action == "memory":
        buffer = bytearray(int(value))
        buffer[:] = b"x" * len(buffer)
    elif action == "output":
        with (out / "generated.bin").open("xb") as stream:
            for _ in range(math.ceil(value / 65536)):
                stream.write(b"x" * 65536)
    elif action == "symlink":
        (out / "generated-link").symlink_to("/does-not-exist")
    elif action == "crash":
        raise RuntimeError("generated failure")
    return {"schema": "colab-generated-worker-v1", "contract_sha256": binding,
            "generated_only": True, "optimizer_updates": 0, "action": action,
            "kernel_limits": {name: list(resource.getrlimit(key)) for name, key in (
                ("data", resource.RLIMIT_DATA), ("file", resource.RLIMIT_FSIZE),
                ("cpu", resource.RLIMIT_CPU), ("core", resource.RLIMIT_CORE),
                ("address", resource.RLIMIT_AS),
            )}}


def worker_main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        raise ValueError("fixed process worker arguments required")
    protocol, out, parent = Path(args[0]), Path(args[1]), int(args[2])
    root = Path(__file__).parent.parent
    # Kernel termination survives a notebook interrupt or disappearance of its parent.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
        return 2
    try:
        job, binding, limits = verify(root, protocol)
        permission_path = os.environ.get("STEGANOGRAPHY_COLAB_PERMISSION")
        permission(job, binding, Path(permission_path) if permission_path else None)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        signal.setitimer(signal.ITIMER_REAL, limits.seconds - limits.closing_seconds)
        a.directory(out / "work")
        if job["kind"] == "generated_qa":
            result = _qa(job["payload"], out / "work", binding)
        elif job["kind"] == "native_profile":
            from core import spatial_colab_profile

            result = spatial_colab_profile.execute(job, binding, out / "work")
        else:
            from core import spatial_colab_fit

            prior = Path(os.environ["STEGANOGRAPHY_COLAB_PROFILE"])
            spatial_colab_fit.bind_profile(root, job, prior)
            result = spatial_colab_fit.execute(job, binding, out / "work")
        a.put(out / "work" / "worker-result.json", a.encode(result), 4 * a.MIB)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, MemoryError, TimeoutError,
            KeyError, TypeError) as error:
        print(json.dumps({"status": "unavailable", "failure_type": type(error).__name__}))
        return 2
