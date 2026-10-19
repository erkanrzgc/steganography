"""Explicit native-media pilot acquisition entry point (no fitting)."""

import argparse
from pathlib import Path

from core.research_acquisition import acquire


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--unicode-only", action="store_true")
    args = parser.parse_args()
    result = acquire(args.out, unicode_only=args.unicode_only)
    print({k: v["count"] for k, v in result["families"].items()})


if __name__ == "__main__":
    main()
