"""Explicit fresh generated CPU pair-control profile/fit adapter."""

import argparse
import resource
from pathlib import Path

from core import wav_pair_control as service


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("profile", "fit"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--binding", type=Path)
    args = parser.parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (service.MIB, service.MIB))
    root = Path(__file__).parent.parent.absolute()
    try:
        if args.phase == "profile":
            result = service.profile(root, args.protocol, args.cache)
        else:
            if args.binding is None:
                raise ValueError("fit binding required")
            result = service.fit(root, args.protocol, args.cache, args.binding)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError):
        import traceback

        traceback.print_exc()  # Private bounded stderr only.
        print("unavailable")
        return 2
    print(result.get("decision", result["status"]))
    return 0 if result.get("decision", "passed_generated_only") == "passed_generated_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
