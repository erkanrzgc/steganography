"""Fresh-process POSIX resource guard for the optional external-tool adapters.

The parent invokes this trusted file with an isolated interpreter. Limits are
applied after interpreter startup, avoiding Python callbacks between fork and
exec in a multithreaded analysis process. No shell or carrier is executed.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence


def apply_limits(memory: int, cpu: int, output: int) -> None:
    """Apply a validated, bounded policy in the fresh launcher process."""
    if (any(type(value) is not int for value in (memory, cpu, output))
            or not 1 <= memory <= 8 * 1024**3 or not 1 <= cpu <= 3600
            or not 1 <= output <= 16 * 1024**2):
        raise ValueError("invalid resource policy")
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (output, output))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def main(argv: Sequence[str] | None = None) -> int:
    """Replace the launcher with the parent's resolved executable and exact argv."""
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if (not 5 <= len(args) <= 256 or args[3] != "--" or not args[4]
                or any("\0" in arg for arg in args)
                or sum(len(arg.encode("utf-8")) for arg in args) > 128 * 1024):
            raise ValueError("invalid launcher arguments")
        memory, cpu, output = (int(value) for value in args[:3])
        apply_limits(memory, cpu, output)
        os.execv(args[4], args[4:])  # noqa: S606 - parent's fixed resolved tool, no shell
        return 0
    except (ValueError, OSError):
        print("external tool launcher unavailable", file=sys.stderr)
        return 126


if __name__ == "__main__":  # pragma: no cover - exercised by real subprocess checks
    raise SystemExit(main())
