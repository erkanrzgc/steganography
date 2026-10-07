"""Explicit frozen optimizer-and-exposure-matched microbatch learning control."""

import argparse
from pathlib import Path

from steganography.research_features import read_document
from steganography.research_srnet_widebatch import run as run_context

PROTOCOL_SHA = "e521568b94919a5225bf5c13fa2b9e2bc4f04e72852a58a1074dc68bcb085d11"
BASELINE_SHA = "ae828ea18337fcff82f40506717c7fe5f33d664e380537632a4ac1ca4dcf9427"


def run(config, out):
    return run_context(config, out, _accumulation=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (7600, 7601))
        resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2, 32 * 1024**2))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        config, _ = read_document(args.config)
        report = run(config, args.out)
    except Exception:
        print("failed/unavailable: no complete usable accumulation control")
        return 2
    print("completed: train-only accumulation control; detector qualification unavailable")
    return (
        0
        if all(
            a["learning_objectives_passed"] and a["audit"]["numerical_gates_passed"]
            for a in report["arms"]
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
