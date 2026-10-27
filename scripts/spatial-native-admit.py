"""Explicit reviewed native-source acquisition/preparation; no fitting."""

import argparse
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import colab_artifacts as artifacts  # noqa: E402
from core import spatial_commons_native as service  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validate", "acquire", "prepare"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--originals", type=Path)
    parser.add_argument("--prior-ledger", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * artifacts.MIB, 64 * artifacts.MIB))
    try:
        value = artifacts.document(args.manifest)
        service.manifest(value)
        if args.phase == "validate":
            result = {"status": "ready", "media_opened": False}
        elif args.out is None:
            raise ValueError("output required")
        elif args.phase == "acquire":
            if args.prior_ledger is None:
                raise ValueError("prior transfer ledger required")
            raw = artifacts.read(args.prior_ledger, artifacts.MIB)
            prior = artifacts.document(args.prior_ledger)
            result = service.acquire(
                value,
                args.out,
                allow_download=args.allow_download,
                prior_ledger={
                    "sha256": artifacts.digest(raw),
                    "historical_bytes": prior["historical_bytes"],
                },
            )
        else:
            if args.originals is None:
                raise ValueError("originals required")
            result = service.prepare(value, args.originals, args.out)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        print("unavailable: source, artifact or environment gate failed; preserve receipts")
        return 2
    print(artifacts.encode(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
