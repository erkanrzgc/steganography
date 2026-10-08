"""Metadata-only whole-lineage layout for bounded JPEG preparation blocks."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from pathlib import PurePosixPath

BLOCK_ORIGINALS = 128
METHODS = ("JUNIWARD", "UERD")


def identity(value):
    if not isinstance(value, str) or not re.fullmatch("[a-f0-9]{64}", value):
        raise ValueError("invalid JPEG scale identity")
    return value


def safe_name(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("unsafe scale source path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("unsafe scale source path")


def layout(alaska, boss, quarantine, reserved):
    """Preserve complete source groups and roles; never select by model score."""
    if (
        not isinstance(alaska, list)
        or not 4 <= len(alaska) <= 4000
        or not isinstance(boss, list)
        or not 2 <= len(boss) <= 1000
        or not isinstance(quarantine, set)
        or not isinstance(reserved, set)
    ):
        raise ValueError("JPEG scale source limits")
    for key in quarantine | reserved:
        identity(key)
    groups: dict[str, list] = defaultdict(list)
    seen: dict[str, str] = {}
    for row in alaska:
        if not isinstance(row, dict):
            raise ValueError("invalid ALASKA scale row")
        safe_name(row.get("path"))
        digest, lineage = identity(row.get("sha256")), identity(row.get("lineage"))
        if (
            row.get("source_group") != "ALASKA2"
            or row.get("format") != "JPEG"
            or row.get("split") not in ("train", "validation")
            or row.get("label") not in ("cover", "stego")
            or {digest, lineage} & reserved
            or seen.setdefault(digest, lineage) != lineage
        ):
            raise ValueError("ALASKA scale roles or reserved overlap")
        groups[lineage].append(row)
    if not quarantine <= groups.keys():
        raise ValueError("unknown quarantine lineage")
    usable = []
    for lineage, rows in groups.items():
        covers = [r for r in rows if r["label"] == "cover"]
        if (
            len(rows) != 4
            or len(covers) != 1
            or covers[0]["sha256"] != lineage
            or covers[0].get("method") is not None
            or {r.get("method") for r in rows} != {None, "JMiPOD", *METHODS}
            or len({r["split"] for r in rows}) != 1
            or any(r["label"] != ("cover" if r.get("method") is None else "stego") for r in rows)
        ):
            raise ValueError("incomplete ALASKA scale lineage")
        unchanged = any(r["label"] == "stego" and r["sha256"] == lineage for r in rows)
        if unchanged != (lineage in quarantine):
            raise ValueError("quarantine does not match unchanged ALASKA pairs")
        if lineage not in quarantine:
            if len({r["sha256"] for r in rows}) != 4:
                raise ValueError("duplicate ALASKA method bytes")
            usable.append(lineage)
    if not usable:
        raise ValueError("no eligible ALASKA scale lineage")
    boss_seen: set[str] = set()
    for row in boss:
        if not isinstance(row, dict):
            raise ValueError("invalid BOSS scale row")
        safe_name(row.get("path"))
        digest = identity(row.get("sha256"))
        if (
            row.get("source_group") != "BOSSbase-1.01"
            or row.get("label") != "cover"
            or row.get("method") is not None
            or row.get("split") not in ("train", "validation")
            or row.get("lineage") != digest
            or digest in boss_seen | reserved | set(seen)
        ):
            raise ValueError("BOSS scale roles or duplicate/reserved identity")
        boss_seen.add(digest)
    ordered = {
        "ALASKA2": sorted(
            usable, key=lambda key: hashlib.sha256(("scale:20261008:" + key).encode()).digest()
        ),
        "BOSSbase-1.01": [
            r["sha256"]
            for r in sorted(
                boss,
                key=lambda r: hashlib.sha256(
                    ("jpeg-context:20261006:" + r["sha256"]).encode()
                ).digest(),
            )
        ],
    }
    blocks = []
    for source, keys in ordered.items():
        for offset in range(0, len(keys), BLOCK_ORIGINALS):
            lineages = keys[offset : offset + BLOCK_ORIGINALS]
            if source == "BOSSbase-1.01" and len(lineages) < 2:
                raise ValueError("BOSS block needs at least two originals; no silent truncation")
            blocks.append({"source_group": source, "offset": offset, "lineages": lineages})
    return blocks
