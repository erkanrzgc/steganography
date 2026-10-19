"""Bounded own-process preflight revision without altering frozen fitting code."""

from __future__ import annotations

import os
import stat
import time
from pathlib import Path
from types import FunctionType

from core import nonjpeg_comparison as old

ROOT = old.ROOT
PARENT = ROOT / "benchmarks/nonjpeg-linear-protocol-20261015.json"
FAILED = ROOT / "benchmarks/nonjpeg-linear-results-20261015.json"
PRIOR_SECONDS = 21
PRIOR_BYTES = 64 * 1024
OUTPUT_BYTES = old.OUTPUT_BYTES - PRIOR_BYTES
NAMES = (*old.NAMES, "core/filetype.py", "core/__init__.py", "modules/__init__.py",
         "core/nonjpeg_process_preflight.py", "steganography/research_nonjpeg_process.py",
         "docs/NONJPEG_PROCESS_PREFLIGHT_PROTOCOL.md",
         "benchmarks/nonjpeg-linear-protocol-20261015.json",
         "benchmarks/nonjpeg-linear-results-20261015.json",
         "benchmarks/nonjpeg-linear-preflight-diagnosis-20261015.json")


def source_hash(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("source symlink")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("nonregular source")
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read(1024**2 + 1)
    if len(raw) > 1024**2:
        raise ValueError("source byte bound")
    return old.sha(raw)  # Empty package initializers are valid source files.


def sources():
    return {name: source_hash(ROOT / name) for name in NAMES}


def cpu_bound(*, proc=None, root=Path("/sys/fs/cgroup")):
    # /proc/self is a trusted kernel alias, but the carrier reader rejects every
    # symlink. Address our numeric PID directly; never resolve an input symlink.
    identity = Path("/proc") / str(os.getpid()) / "cgroup" if proc is None else proc
    return old.cpu_bound(proc=identity, root=root)


class ChargedClock:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.first = True

    def __call__(self):
        now = self.clock()
        if self.first:
            self.first = False
            return now - PRIOR_SECONDS
        return now


def write(path, value):
    bounded = FunctionType(old.write.__code__, {**old.write.__globals__,
                                               "OUTPUT_BYTES": OUTPUT_BYTES})
    return bounded(path, value)


def execute(config, out, *, clock=time.monotonic):
    protocol, _ = old.read(Path(config["protocol"]), config["protocol_sha256"])
    parent, parent_sha = old.read(PARENT)
    failed, failed_sha = old.read(FAILED)
    if (
        protocol.get("parent_protocol_sha256") != parent_sha
        or protocol.get("failed_results_sha256") != failed_sha
        or parent.get("source_sha256") != old.sources()
        or failed.get("status") != "unavailable"
        or failed.get("fit", {}).get("phase") != "resource_preflight"
        or failed.get("fit", {}).get("real_data_optimizer_updates") != 0
        or protocol.get("accounting") != {
            "prior_seconds_debit": PRIOR_SECONDS, "prior_output_bytes_debit": PRIOR_BYTES,
            "remaining_seconds": old.SECONDS - PRIOR_SECONDS,
            "remaining_output_bytes": OUTPUT_BYTES,
        }
        or any(protocol.get(k) != parent.get(k) for k in (
            "readiness_sha256", "corpus_sha256", "audit_sha256", "seeds", "epochs",
            "budget_seconds", "runtime", "recipe", "expected_models",
            "expected_originals_per_source", "wav_reference",
        ))
        or protocol.get("budget_bytes") != OUTPUT_BYTES
    ):
        raise ValueError("process preflight amendment binding mismatch")

    # Reuse the exact frozen function bodies with private dependency bindings.
    # Neither this module nor concurrent callers mutate the original globals.
    def inspect():
        result = cpu_bound()
        write(out / "external-accounting.json", {
            "schema_version": "nonjpeg-process-accounting-v1",
            **protocol["accounting"], "parent_protocol_sha256": parent_sha,
            "failed_results_sha256": failed_sha, "real_prior_updates": 0,
        })
        return result

    run = FunctionType(old.execute.__code__, {
        **old.execute.__globals__, "cpu_bound": inspect, "sources": sources,
        "OUTPUT_BYTES": OUTPUT_BYTES, "write": write,
    })
    return run(config, out, clock=ChargedClock(clock))
