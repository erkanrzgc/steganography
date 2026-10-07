#!/usr/bin/env python3
"""Read-only generated SRNet control audit; launch with an external timeout."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.srnet_positive_audit import audit  # noqa: E402
from steganography.research_features import read_document  # noqa: E402
from steganography.research_jpeg import write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report, checksum = read_document(args.result / "control.json")
        result = audit(report, args.result, ROOT)
        result["control_report_sha256"] = checksum
        write_json(args.out, result)
    except Exception:
        print("failed/unavailable: no verified generated-control audit")
        return 2
    print("completed: generated-control numerical replay; detector qualification unavailable")
    return 0 if result["numerical_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
