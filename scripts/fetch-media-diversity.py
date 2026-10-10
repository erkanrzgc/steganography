"""Explicit fixed-origin RGB image and environmental audio acquisition."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.media_dataset import SOURCES, acquire  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=tuple(SOURCES))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--reserved-manifest", type=Path, action="append", required=True)
    parser.add_argument("--esc-original-group-retry", action="store_true")
    args = parser.parse_args()
    try:
        result = acquire(
            args.dataset,
            args.out,
            archive_path=args.archive,
            reserved_paths=args.reserved_manifest,
            esc_group_retry=args.esc_original_group_retry,
        )
    except Exception as exc:
        parser.exit(
            2,
            f"Acquisition failed ({type(exc).__name__}); "
            "partial files retained, no success claim\n",
        )
    print(f"{args.dataset}: {result['originals']} originals; no model trained")


if __name__ == "__main__":
    main()
