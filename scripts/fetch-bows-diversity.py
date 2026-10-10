"""Explicit bounded third-origin acquisition, not model training."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.bows_dataset import MAX_ARCHIVE, acquire, read  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reserved-manifest", type=Path, action="append", required=True)
    parser.add_argument("--native-layout-archive", type=Path)
    args = parser.parse_args()
    report = acquire(
        args.out,
        args.reserved_manifest,
        archive=read(args.native_layout_archive, MAX_ARCHIVE)
        if args.native_layout_archive
        else None,
        native=args.native_layout_archive is not None,
    )
    print(f"{report['status']}: {len(report['samples'])} originals; accuracy unavailable")


if __name__ == "__main__":
    main()
