"""Observed runtime headroom, without an inherited laptop-sized RAM ceiling."""

from __future__ import annotations

from pathlib import Path

from core import colab_artifacts as a
from core import spatial_colab_readiness as metadata

POLICY = "provider-managed-ram-with-headroom-watch-v1"


def observe(*, meminfo: Path = Path("/proc/meminfo")) -> dict:
    raw = a.read(meminfo, 65536).decode("ascii")
    values = {}
    for name in ("MemTotal", "MemAvailable"):
        rows = [line.split() for line in raw.splitlines() if line.startswith(name + ":")]
        if (len(rows) != 1 or len(rows[0]) != 3 or rows[0][2] != "kB"
                or not rows[0][1].isascii() or not rows[0][1].isdecimal()
                or len(rows[0][1]) > 16):
            raise ValueError("runtime memory observation unavailable")
        values[name] = int(rows[0][1]) * 1024
    if not 0 < values["MemAvailable"] <= values["MemTotal"] <= 1024**5:
        raise ValueError("runtime memory counters inconsistent")
    capacity, available = values["MemTotal"], values["MemAvailable"]
    scope = "kernel_reported_capacity_not_a_verified_tenant_limit"
    limits = metadata.resources()
    counters = dict(limits.counters)
    return {"schema": "colab-runtime-memory-observation-v1", "policy": POLICY,
            "capacity_bytes": capacity, "available_bytes": available,
            "headroom_bytes": capacity // 10, "scope": scope,
            "inherits_runtime_limits": True, "custom_data_or_rss_ceiling_bytes": None,
            "observed_cgroup_counters": counters, "tenant_limit_verified": False}


def check(observed: dict, *, initial: dict | None = None) -> None:
    if initial is not None and (observed["capacity_bytes"] != initial["capacity_bytes"]
                                or observed["scope"] != initial["scope"]):
        raise ValueError("runtime memory observation changed")
    if observed["available_bytes"] <= observed["headroom_bytes"]:
        raise MemoryError("runtime memory headroom depleted")
