"""Explicit CPU AudioMNIST pair-head preparation and separately approved fitting."""

import argparse
import resource
from pathlib import Path

from core import wav_pair_native as service


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("profile", "fit"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--media", type=Path)
    parser.add_argument("--approval", type=Path)
    args = parser.parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (service.base.MIB, service.base.MIB))
    root = Path(__file__).parent.parent.absolute()
    try:
        if args.phase == "profile":
            if args.media is None:
                raise ValueError("native profile media required")
            result = service.profile(root, args.protocol, args.media, args.cache, args.qa)
        else:
            if args.approval is None:
                raise ValueError("native explicit approval required")
            result = service.fit(root, args.protocol, args.cache, args.qa, args.approval)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError, SyntaxError):
        print("unavailable: native preparation/approval/runtime gate failed")
        return 2
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
