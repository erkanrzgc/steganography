"""Read-only full-exposure generated WAV result checker; never starts fitting."""

import argparse
import json
from pathlib import Path

from core import wav_control_gate as service


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        protocol, digest = service.read_protocol(Path(__file__).parent.parent, args.protocol)
        receipt, _ = service.read_document(args.receipt)
        result = service.evaluate(protocol, digest, receipt)
    except (OSError, ValueError, KeyError, TypeError):
        result = {"status": "unavailable", "reason": "invalid_bound_control_document",
                  "real_fit_authorized": False}
    print(json.dumps(result, allow_nan=False))
    return 0 if result["status"] == "passed_generated_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
