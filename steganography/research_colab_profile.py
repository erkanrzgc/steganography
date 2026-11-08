"""Explicit Colab native cost-profile preparation; no fitting command."""

import argparse
import json
from pathlib import Path

from core import colab_artifacts as a
from core import colab_execution as process
from core import spatial_colab_handoff as handoff
from core import spatial_colab_profile as service


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "cache-package", "profile", "audit"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--permission", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).parent.parent
    result: dict
    try:
        if args.phase == "freeze":
            if (args.environment is None or args.proposal is None or args.out is not None
                    or args.permission is not None or args.cache is not None):
                raise ValueError("metadata-only profile freeze inputs required")
            proposal = a.document(args.proposal)
            if (proposal["status"] != "proposal_only"
                    or proposal["native_roles_applied"] is not False):
                raise ValueError("unassigned metadata-only proposal required")
            payload = {"schema": service.PROFILE_SCHEMA, "split": proposal["split"],
                       "recipe": service.RECIPE,
                       "expected_environment": a.document(args.environment)}
            limits = process.Limits(240, 0, 0, 256 * a.MIB, 128 * a.MIB,
                                    closing_seconds=10)
            process.freeze(root, args.protocol, "native_profile", payload, limits,
                           target_environment={"python": "3.12.13", "implementation": "CPython",
                                               "platform": "linux"})
            result = {"status": "frozen_profile_only_permission_missing"}
        else:
            if args.out is None or args.proposal is not None or args.environment is not None:
                raise ValueError("explicit profile output and one phase required")
            if args.phase == "cache-package":
                job, binding, limits = process.source_contract(root, args.protocol)
                if job["kind"] != "native_profile" or args.cache is None:
                    raise ValueError("bound native cache transport inputs required")
                process.permission(job, binding, args.permission)
                split = service.contract(job["payload"], limits)
                cache = service.data.load(args.cache, split.manifest_sha256)
                service.cache_binding(cache, split)
                checksum = handoff.cache_package(cache, split, args.out)
                result = {"status": "packaged_train_only", "sha256": checksum,
                          "native_roles_rewritten": False}
            elif args.phase == "profile":
                result = process.run(root, args.protocol, args.out,
                                     permission_path=args.permission, cache=args.cache)
            else:
                result = process.audit(root, args.protocol, args.out)
    except (OSError, ValueError, RuntimeError, ImportError, KeyError, TypeError) as error:
        print(json.dumps({"status": "unavailable", "failure_type": type(error).__name__}))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] != "unavailable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
