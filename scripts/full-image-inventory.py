"""Explicit build/runtime inventory gate; output is portable and secret-free."""

import argparse
import json
from pathlib import Path

from core.tool_inventory import read_lock, verify_inventory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    report = verify_inventory(read_lock(args.lock), cwd=Path.cwd())
    if args.baseline:
        if args.baseline.is_symlink() or not args.baseline.is_file():
            raise ValueError("inventory baseline must be regular")
        with args.baseline.open("rb") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            raise ValueError("inventory baseline exceeds byte limit")
        baseline = json.loads(raw)
        if report != baseline:
            report["status"] = "unavailable"
            report["reason"] = "installed inventory differs from build baseline"
    if args.out:
        if args.out.is_symlink() or any(p.is_symlink() for p in args.out.parents):
            raise ValueError("inventory output uses symlink")
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    print(json.dumps(report))
    return 0 if report["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
