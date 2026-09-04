#!/usr/bin/env python3
"""Reject a retired identifier in tracked files and supplied artifacts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

RETIRED = bytes((99, 121, 98, 101, 114, 109, 52, 102, 105, 97))


def _tracked_files() -> list[Path]:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required for the tracked-file identity check")
    result = subprocess.run(  # noqa: S603 - resolved executable, fixed arguments
        [git, "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [Path(value) for value in result.stdout.decode().split("\0") if value]


def _matches(path: Path) -> list[str]:
    matches: list[str] = []
    if RETIRED in path.as_posix().lower().encode():
        matches.append(str(path))
    if not path.is_file():
        return matches
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                name_match = RETIRED in member.filename.lower().encode()
                content_match = RETIRED in archive.read(member).lower()
                if name_match or content_match:
                    matches.append(f"{path}:{member.filename}")
        return matches
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                stream = archive.extractfile(member) if member.isfile() else None
                content = stream.read().lower() if stream else b""
                if RETIRED in member.name.lower().encode() or RETIRED in content:
                    matches.append(f"{path}:{member.name}")
        return matches
    if RETIRED in path.read_bytes().lower():
        matches.append(str(path))
    return matches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = _tracked_files()
    for supplied in args.paths:
        paths.extend(supplied.rglob("*") if supplied.is_dir() else [supplied])
    matches = sorted({match for path in paths for match in _matches(path)})
    if matches:
        print("retired identifier found:")
        print("\n".join(f"  {match}" for match in matches))
        return 1
    print(f"identity check passed ({len(paths)} files inspected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
