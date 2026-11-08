"""Explicit generated-only Colab process QA; contains no native fitting command."""

import argparse
import json
from pathlib import Path

from core import colab_artifacts as a
from core import colab_execution as service


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run", "audit"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).parent.parent
    try:
        if args.phase == "freeze":
            if args.out is not None:
                raise ValueError("freeze does not execute")
            service.freeze(root, args.protocol, "generated_qa", {"action": "success", "value": 0},
                           service.Limits(10, 128 * a.MIB, 8 * 1024**3, 8 * a.MIB, a.MIB,
                                          address_bytes=256 * a.MIB, closing_seconds=1))
            result = {"status": "frozen_generated_qa_only"}
        else:
            if args.out is None:
                raise ValueError("explicit output required")
            call = service.run if args.phase == "run" else service.audit
            result = call(root, args.protocol, args.out)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(json.dumps({"status": "unavailable", "failure_type": type(error).__name__}))
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] != "unavailable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
