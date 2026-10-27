"""Separately reviewed native Commons admission; preserves the frozen profile."""

from __future__ import annotations

import hashlib
import io
import re
import struct
import urllib.parse
from pathlib import Path

import numpy as np
from PIL import Image

from core import colab_artifacts as artifacts
from core import spatial_colab_data as data

LICENSES = {
    **data.LICENSES,
    "CC-BY-2.0": "https://creativecommons.org/licenses/by/2.0/",
    "CC-BY-3.0": "https://creativecommons.org/licenses/by/3.0/",
}
MAX_TRANSFER = 64 * artifacts.MIB
MAX_ORIGINALS = 16
FIELDS = {
    "id",
    "page",
    "original_url",
    "author",
    "license",
    "license_url",
    "original_sha1",
    "original_bytes",
    "ancestry_group",
    "ancestry_review",
    "role",
    "media_status",
    "native_mode",
    "page_revision",
    "evidence_sha256",
}


def manifest(value: dict) -> list[dict]:
    if (
        set(value) != {"schema", "sources"}
        or value["schema"] != "spatial-commons-reviewed-v2"
        or not isinstance(value["sources"], list)
        or not 1 <= len(value["sources"]) <= MAX_ORIGINALS
    ):
        raise ValueError("reviewed source manifest schema")
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    author_groups: dict[str, str] = {}
    page_groups: dict[str, str] = {}
    for row in value["sources"]:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("reviewed source fields")
        for key in ("id", "ancestry_group"):
            if not isinstance(row[key], str) or not re.fullmatch(r"[a-z0-9-]{1,80}", row[key]):
                raise ValueError("source identity")
        for key, host, prefix in (
            ("page", "commons.wikimedia.org", "/wiki/File:"),
            ("original_url", "upload.wikimedia.org", "/wikipedia/commons/"),
        ):
            url = urllib.parse.urlsplit(row[key])
            if (
                url.scheme != "https"
                or url.netloc != host
                or not url.path.startswith(prefix)
                or "/thumb/" in url.path
                or url.query
                or url.fragment
            ):
                raise ValueError("source URL allowlist")
        if (
            row["license"] not in LICENSES
            or row["license_url"] != LICENSES[row["license"]]
            or not isinstance(row["author"], str)
            or not 1 <= len(row["author"]) <= 1024
            or row["author"] != row["author"].strip()
            or row["native_mode"] not in {"L", "RGB"}
            or row["ancestry_review"] != "conservative-author-group-reviewed"
            or row["role"] != "unassigned"
            or row["media_status"] != "not_downloaded"
            or type(row["original_bytes"]) is not int
            or not 1 <= row["original_bytes"] <= data.MAX_FILE
            or type(row["page_revision"]) is not int
            or row["page_revision"] <= 0
        ):
            raise ValueError("license/native/provenance gate")
        artifacts.identity(row["original_sha1"], 40)
        artifacts.identity(row["evidence_sha256"])
        if row["id"] in seen_ids or row["original_sha1"] in seen_hashes:
            raise ValueError("duplicate original")
        for groups, key in ((author_groups, row["author"].casefold()), (page_groups, row["page"])):
            if key in groups and groups[key] != row["ancestry_group"]:
                raise ValueError("common-source group leakage")
            groups[key] = row["ancestry_group"]
        seen_ids.add(row["id"])
        seen_hashes.add(row["original_sha1"])
    if sum(row["original_bytes"] + 1 for row in value["sources"]) > MAX_TRANSFER:
        raise ValueError("total transfer bound")
    return value["sources"]


def native_pixels(raw: bytes, mode: str) -> np.ndarray:
    # Pillow can downcast 16-bit truecolour PNG to RGB8. Check IHDR first.
    if len(raw) < 33 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[8:16] != b"\x00\x00\x00\rIHDR":
        raise ValueError("native PNG header required")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", raw[16:29]
    )
    if (
        mode not in {"L", "RGB"}
        or depth != 8
        or color != (0 if mode == "L" else 2)
        or compression != 0
        or filtering != 0
        or interlace not in {0, 1}
        or min(width, height) < 256
        or width * height > data.MAX_PIXELS
    ):
        raise ValueError("native PNG mode/depth/extent")
    return data.crop(raw)


def acquire(value: dict, out: Path, *, allow_download: bool, prior_ledger: dict) -> dict:
    rows = manifest(value)
    if allow_download is not True:
        raise ValueError("explicit acquisition command required")
    if set(prior_ledger) != {"sha256", "historical_bytes"} or (
        prior_ledger["historical_bytes"] is not None
        and (
            type(prior_ledger["historical_bytes"]) is not int
            or prior_ledger["historical_bytes"] < 0
        )
    ):
        raise ValueError("prior transfer ledger")
    artifacts.identity(prior_ledger["sha256"])
    artifacts.directory(out)
    reservation = {
        "schema": "spatial-native-acquisition-v2",
        "prior_ledger": prior_ledger,
        "manifest_sha256": artifacts.digest(artifacts.encode(value)),
        "reserved_transfer_bytes": sum(row["original_bytes"] + 1 for row in rows),
        "historical_total_complete": prior_ledger["historical_bytes"] is not None,
        "status": "unavailable",
        "retry": False,
    }
    artifacts.put(out / "reservation.json", artifacts.encode(reservation))
    acquired = []
    for row in rows:
        raw = data.fetch(row["original_url"], row["original_bytes"])
        if (
            len(raw) != row["original_bytes"]
            or hashlib.sha1(raw, usedforsecurity=False).hexdigest() != row["original_sha1"]
        ):
            raise ValueError("original hash/extent mismatch")
        checksum = artifacts.put(out / (row["id"] + ".original"), raw, data.MAX_FILE)
        native_pixels(raw, row["native_mode"])
        acquired.append({"id": row["id"], "sha256": checksum, "bytes": len(raw)})
    result = {**reservation, "status": "ready", "acquired": acquired}
    artifacts.put(out / "acquired.json", artifacts.encode(result))
    return result


def prepare(value: dict, originals: Path, out: Path) -> dict:
    rows = manifest(value)
    artifacts.directory(out)
    records = []
    seen: set[str] = set()
    total = 0
    for row in rows:
        raw = artifacts.read(originals / (row["id"] + ".original"), data.MAX_FILE)
        if (
            len(raw) != row["original_bytes"]
            or hashlib.sha1(raw, usedforsecurity=False).hexdigest() != row["original_sha1"]
        ):
            raise ValueError("original hash/extent mismatch")
        pixels = native_pixels(raw, row["native_mode"])
        pixel_hash = artifacts.digest(pixels.tobytes())
        if pixel_hash in seen:
            raise ValueError("duplicate crop; review common ancestry")
        seen.add(pixel_hash)
        payload = bytes.fromhex(artifacts.digest((row["id"] + ":generated-control").encode()))
        stego = data.embed(pixels, payload)
        for extension in ("png", "bmp"):
            for kind, values in (("cover", pixels), ("lsb-control", stego)):
                buffer = io.BytesIO()
                Image.fromarray(values).save(buffer, format=extension.upper())
                encoded = buffer.getvalue()
                decoded = data.crop(encoded)
                if not np.array_equal(decoded, values) or data.recover(decoded, len(payload)) != (
                    payload if kind == "lsb-control" else data.recover(values, len(payload))
                ):
                    raise ValueError("independent pixel/payload verification")
                total += len(encoded)
                if total > data.MAX_OUTPUT:
                    raise ValueError("prepared output bound")
                name = f"{row['id']}-{kind}.{extension}"
                checksum = artifacts.put(out / name, encoded, data.MAX_FILE)
                records.append(
                    {
                        "name": name,
                        "sha256": checksum,
                        "source_id": row["id"],
                        "group": row["ancestry_group"],
                        "role": "unassigned",
                        "original_sha256": artifacts.digest(raw),
                        "native_mode": row["native_mode"],
                        "crop": [0, 0, 256, 256],
                        "kind": kind,
                        "payload_verified": kind == "lsb-control",
                        "license": row["license"],
                        "license_url": row["license_url"],
                        "author": row["author"],
                        "source_page": row["page"],
                        "changes": (
                            "Top-left 256x256 crop; PNG/BMP encoding; lsb-control adds "
                            "a generated 32-byte plane-0 payload."
                        ),
                    }
                )
    result = {
        "schema": "spatial-native-prepared-v2",
        "status": "ready",
        "artifacts": records,
        "manifest_sha256": artifacts.digest(artifacts.encode(value)),
        "output_bytes": total,
        "representative_benchmark": False,
        "cover_cleanliness_verified": False,
        "optimizer_updates": 0,
    }
    artifacts.put(out / "prepared.json", artifacts.encode(result))
    return result
