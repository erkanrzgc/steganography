"""Colab fit proposal, permission-gated development transport, execution and audit."""

import argparse
import json
from pathlib import Path

from core import colab_artifacts as a
from core import colab_execution as controller
from core import spatial_colab_fit as service
from core import spatial_colab_handoff as handoff


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze-fit", "development-package", "fit", "audit"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--profile-out", type=Path)
    parser.add_argument("--setup-debit", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--permission", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cumulative-amendment", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).parent.parent
    try:
        if args.phase == "freeze-fit":
            if (args.profile_out is None or args.setup_debit is None or args.cache is not None
                    or args.permission is not None or args.out is not None):
                raise ValueError("measured profile and complete external debit required")
            service.freeze(root, args.profile_out, a.document(args.setup_debit), args.protocol,
                           amended=args.cumulative_amendment)
            result = {"status": "frozen_fit_proposal_permission_missing"}
        else:
            if args.out is None or args.setup_debit is not None or args.cumulative_amendment:
                raise ValueError("explicit output and one phase required")
            job, binding, limits = controller.source_contract(root, args.protocol)
            if job["kind"] != "native_fit":
                raise ValueError("native fitting contract required")
            if args.phase == "development-package":
                if args.cache is None:
                    raise ValueError("explicit existing native cache required")
                controller.permission(job, binding, args.permission)
                split = service.contract(job["payload"], limits)
                cache = service.data.load(args.cache, split.manifest_sha256)
                sha = handoff.development_package(cache, split, args.out)
                result = {"status": "packaged_development_only", "sha256": sha}
            elif args.phase == "fit":
                result = controller.run(root, args.protocol, args.out, cache=args.cache,
                                        profile_out=args.profile_out,
                                        permission_path=args.permission)
            else:
                result = controller.audit(root, args.protocol, args.out)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as error:
        print(json.dumps({"status": "unavailable", "failure_type": type(error).__name__}))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] != "unavailable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
