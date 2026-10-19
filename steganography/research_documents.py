"""Explicit local document-pilot preparation; never downloads or fits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.document_corpus import prepare, regular_read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(json.loads(regular_read(args.config, 128 * 1024)), args.out)
    print(report["status"])


if __name__ == "__main__":
    main()
