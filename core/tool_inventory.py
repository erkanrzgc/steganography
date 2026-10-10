"""Explicit full-image inventory verification; no downloads or execution of evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from core.tools import ToolRunner

MAX_FILE_BYTES = 64 * 1024 * 1024


def digest_file(path: Path) -> str:
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("inventory file must be bounded and regular")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        total = 0
        while chunk := stream.read(65536):
            total += len(chunk)
            if total > MAX_FILE_BYTES:
                raise ValueError("inventory file exceeds byte limit")
            digest.update(chunk)
    return digest.hexdigest()


def read_lock(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
        raise ValueError("invalid bounded inventory lock")
    with path.open("rb") as stream:
        raw = stream.read(65537)
    if len(raw) > 65536:
        raise ValueError("inventory lock exceeds byte limit")
    lock = json.loads(raw)
    if not isinstance(lock, dict) or lock.get("schema_version") != "full-image-tools-v1":
        raise ValueError("invalid inventory schema")
    names: set[str] = set()
    for key in ("tools", "gems"):
        rows = lock.get(key)
        if not isinstance(rows, list) or not 1 <= len(rows) <= 32:
            raise ValueError("invalid inventory rows")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("invalid inventory row")
            for field in ("name", "version"):
                if not isinstance(row.get(field), str) or not re.fullmatch(
                    r"[A-Za-z0-9][A-Za-z0-9_.:+-]{0,99}", row[field]
                ):
                    raise ValueError("invalid inventory identity")
            if row["name"] in names:
                raise ValueError("duplicate inventory identity")
            names.add(row["name"])
            if not isinstance(row.get("license"), str) or not re.fullmatch(
                r"[A-Za-z0-9 .()+-]{1,150}", row["license"]
            ):
                raise ValueError("invalid license identity")
            if "sha256" in row and (
                not isinstance(row["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            ):
                raise ValueError("invalid inventory checksum")
            if "package" in row and (
                not isinstance(row["package"], str)
                or not re.fullmatch(r"[a-z0-9][a-z0-9+.-]{0,99}", row["package"])
            ):
                raise ValueError("invalid package identity")
            if "gem" in row and row["gem"] != row["name"]:
                raise ValueError("invalid gem identity")
    return lock


def digest_tree(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("invalid gem tree")
    digest = hashlib.sha256()
    paths: list[Path] = []
    directories = [root]
    count = total = 0
    while directories:
        with os.scandir(directories.pop()) as entries:
            for entry in entries:
                count += 1
                if count > 2048 or entry.is_symlink():
                    raise ValueError("gem tree exceeds limits or contains symlink")
                if entry.is_dir(follow_symlinks=False):
                    directories.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    paths.append(Path(entry.path))
                else:
                    raise ValueError("gem tree contains non-regular file")
    for path in sorted(paths):
        total += path.stat().st_size
        if total > MAX_FILE_BYTES:
            raise ValueError("gem tree exceeds byte limit")
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(digest_file(path).encode())
    return digest.hexdigest()


def verify_inventory(lock: dict[str, Any], *, cwd: Path) -> dict[str, Any]:
    runner = ToolRunner(timeout=10, max_output=16384)
    rows = []
    for spec in [*lock["tools"], *lock["gems"]]:
        name = spec["name"]
        is_tool = spec in lock["tools"]
        executable = shutil.which(name) if is_tool else None
        row: dict[str, Any] = {"name": name, "expected_version": spec["version"]}
        try:
            if is_tool and executable is None:
                raise ValueError("executable unavailable")
            if "package" in spec:
                result = runner.run(
                    "dpkg-query", ["-W", "-f=${Version}", spec["package"]],
                    cwd=cwd, _include_version=False,
                )
                license_path = Path("/usr/share/doc") / spec["package"] / "copyright"
                row["license_sha256"] = digest_file(license_path)
            elif "gem" in spec or not is_tool:
                # Names are validated by read_lock, never arbitrary Ruby input.
                code = (
                    's=Gem::Specification.find_by_name(ARGV.fetch(0)); '
                    'puts s.version; puts s.licenses.join(" OR "); puts s.full_gem_path'
                )
                result = runner.run("ruby", ["-e", code, name], cwd=cwd, _include_version=False)
                declaration = result.output.splitlines()[1]
                if declaration != spec["license"]:
                    raise ValueError("gem license declaration mismatch")
                row["license_declaration"] = declaration
                row["installed_tree_sha256"] = digest_tree(Path(result.output.splitlines()[2]))
            else:
                result = runner.run(name, ["--version"], cwd=cwd, _include_version=False)
                row["license_sha256"] = digest_file(Path("/usr/share/doc/stegseek/COPYING"))
            if result.status != "completed":
                raise ValueError("version probe unavailable or failed")
            observed = result.output.strip().splitlines()[0]
            if name == "stegseek":
                if not re.search(r"\bStegSeek " + re.escape(spec["version"]) + r"\b", observed):
                    raise ValueError("version mismatch")
                observed = spec["version"]
            if observed != spec["version"]:
                raise ValueError("version mismatch")
            row.update(version=observed, status="verified")
            if executable:
                row["executable_sha256"] = digest_file(Path(executable))
            if name == "openstego":
                row["jar_sha256"] = digest_file(Path("/usr/share/openstego/lib/openstego.jar"))
            if "license" in spec:
                row["license"] = spec["license"]
            if "sha256" in spec:
                row["distribution_sha256"] = spec["sha256"]
        except (OSError, ValueError, IndexError):
            row.update(status="unavailable", reason="version/license/identity verification failed")
        rows.append(row)
    return {
        "schema_version": "full-image-inventory-v1",
        "status": "verified" if all(row["status"] == "verified" for row in rows) else "unavailable",
        "entries": rows,
        "transitive_dependencies_fully_locked": False,
    }
