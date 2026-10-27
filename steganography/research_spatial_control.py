"""Explicit one-attempt generated spatial control; no native fitting or downloads."""

import argparse
import resource
from pathlib import Path

from core import spatial_control as service
from core import spatial_control_supervisor as supervisor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--freeze-environment", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker-deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = Path(__file__).parent.parent.absolute()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * service.a.MIB, 128 * service.a.MIB))
    try:
        if args.freeze_environment is not None:
            if args.execute or args.out is not None or args.worker_deadline is not None:
                raise ValueError("freeze cannot execute")
            service.freeze(root, args.protocol, service.a.document(args.freeze_environment))
            return 0
        if args.worker_deadline is not None:
            if args.out is None or args.execute:
                raise ValueError("worker arguments")
            supervisor.worker(root, args.protocol, args.out, args.worker_deadline)
            return 0
        if not args.execute or args.out is None:
            raise ValueError("explicit generated execution switch required")
        result = supervisor.execute(root, args.protocol, args.out)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError, TimeoutError):
        print("unavailable: spatial generated control gate failed")
        return 2
    print(result["status"])
    return 0 if result["status"] == "passed_generated_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
