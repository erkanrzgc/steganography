"""Explicit frozen-manifest CTF evaluation; never download or generate challenges."""

import argparse
import json
from pathlib import Path

from core.ctf_evaluation import evaluate_suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hardware", required=True, help="Non-secret reference hardware label")
    args = parser.parse_args()
    result = evaluate_suite(args.manifest, args.manifest_sha256, args.out, hardware=args.hardware)
    print(json.dumps({key: result[key] for key in
                      ("case_count", "completed_exact_rate", "blind_gate", "qualification")}))
    return 0 if all(row["status"] == "completed" for row in result["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
