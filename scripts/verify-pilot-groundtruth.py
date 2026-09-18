"""Independently extract every pilot stego with its upstream generator."""

import argparse
import json
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from steganography.benchmarking.pilot import PASSWORD, checked_path, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.root / "manifest.json").read_text())
    samples = [s for s in manifest["samples"] if s["label"] == "stego"]

    def verify(sample):
        if sample["method"] not in {"steghide", "openstego"}:
            return False
        path = checked_path(args.root, sample["path"])
        expected = checked_path(args.root, f"expected/{path.stem.split('-')[0]}.bin")
        executable = shutil.which(sample["method"])
        if executable is None:
            return False
        with tempfile.TemporaryDirectory(prefix="verify-stego-") as directory:
            output = Path(directory) / "payload.bin"
            if sample["method"] == "steghide":
                command = [
                    executable,
                    "extract",
                    "-sf",
                    str(path),
                    "-p",
                    PASSWORD,
                    "-xf",
                    str(output),
                ]
            else:
                command = [
                    executable,
                    "extract",
                    "-sf",
                    str(path),
                    "-xd",
                    directory,
                    "-xf",
                    output.name,
                ]
            try:
                subprocess.run(  # noqa: S603 - fixed independent extractor and fixture args
                    command,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    timeout=30,
                )
            except (OSError, subprocess.SubprocessError):
                return False
            return (
                digest(path.read_bytes()) == sample["sha256"]
                and output.is_file()
                and output.read_bytes() == expected.read_bytes()
            )

    with ThreadPoolExecutor(max_workers=4) as executor:
        outcomes = list(executor.map(verify, samples))
    print(
        json.dumps(
            {
                "expected": len(samples),
                "exact_matches": sum(outcomes),
                "failed": len(samples) - sum(outcomes),
                "manifest_sha256": digest((args.root / "manifest.json").read_bytes()),
            },
            indent=2,
        )
    )
    raise SystemExit(0 if outcomes and all(outcomes) else 1)


if __name__ == "__main__":
    main()
