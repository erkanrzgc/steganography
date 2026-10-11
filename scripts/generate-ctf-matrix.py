"""Generate explicit small independent-encoder DEVELOPMENT fixtures, never blind challenges."""

import argparse
from pathlib import Path

from steganography.benchmarking.ctf_matrix import generate_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--external", action="store_true", help="Require full-image encoder tools")
    args = parser.parse_args()
    document = generate_matrix(args.out, external=args.external)
    print(f"Generated {len(document['cases'])} development fixtures; no blind qualification.")


if __name__ == "__main__":
    main()
