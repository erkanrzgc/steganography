"""Explicit generated-only WAV runner; no real media, downloads or checkpoint."""

import argparse
import resource
from pathlib import Path

from core import wav_control_runner as service


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--worker-deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (service.MAX_JSON, service.MAX_JSON))
    root = Path(__file__).parent.parent.absolute()
    try:
        if args.worker_deadline is not None:
            service.worker(root, args.protocol, args.out, args.worker_deadline)
            return 0
        decision = service.execute(root, args.protocol, args.out)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError):
        import traceback

        traceback.print_exc()  # Private bounded stderr; portable verdict has no host path.
        print("unavailable")
        return 2
    print(decision["status"])
    return 0 if decision["status"] == "passed_generated_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
