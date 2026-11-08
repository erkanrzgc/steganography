"""Explicit source-bound native RGB pilot measurement and approval-bound fitting."""

import argparse
import resource
import traceback
from pathlib import Path

from core import colab_artifacts as a
from core import spatial_native_data as data
from core import spatial_native_pilot as service
from core import spatial_native_supervisor as supervisor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--freeze-environment", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--observe", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--kernel-start-us", type=int)
    parser.add_argument("--kernel-exit-us", type=int)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--profile-result", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).parent.parent.absolute()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * a.MIB, 128 * a.MIB))
    try:
        if args.freeze_environment is not None:
            if (
                args.profile
                or args.fit
                or args.observe
                or args.finalize
                or args.out is not None
                or args.approval is not None
            ):
                raise ValueError("freeze cannot execute")
            service.freeze(root, args.protocol, a.document(args.freeze_environment))
            return 0
        if sum((args.profile, args.fit, args.observe, args.finalize)) != 1 or args.out is None:
            raise ValueError("one explicit native phase required")
        if args.observe:
            supervisor.observe(args.out, service.checksum(args.protocol))
            return 0
        if args.cache is None:
            raise ValueError("bound cache required")
        config = a.document(args.protocol, 4 * a.MIB)
        cache = data.load(args.cache, config["split"]["manifest_sha256"])
        if args.finalize:
            supervisor.finalize(
                root, args.protocol, cache, args.out, args.kernel_start_us, args.kernel_exit_us
            )
            return 0
        result = supervisor.execute(
            root,
            args.protocol,
            cache,
            args.out,
            phase="profile" if args.profile else "fit",
            approval_path=args.approval,
            profile_path=args.profile_result,
        )
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError, TimeoutError):
        traceback.print_exc()
        print("unavailable: native pilot phase failed")
        return 2
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
