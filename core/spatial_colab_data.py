"""Opt-in Commons acquisition and native, unassigned PNG/BMP preparation."""

from __future__ import annotations

import hashlib
import io
import re
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from core import colab_artifacts as artifacts

MAX_FILE = 32 * artifacts.MIB
MAX_PIXELS = 16_000_000
MAX_SOURCES = 128
MAX_OUTPUT = 256 * artifacts.MIB
LICENSES = {
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
}


def manifest(value: dict) -> list[dict]:
    if (
        set(value) != {"schema", "sources"}
        or value["schema"] != "spatial-commons-v1"
        or not isinstance(value["sources"], list)
        or not 1 <= len(value["sources"]) <= MAX_SOURCES
    ):
        raise ValueError("Commons manifest schema")
    names: set[str] = set()
    hashes: set[str] = set()
    pages: dict[str, str] = {}
    for row in value["sources"]:
        if not isinstance(row, dict) or set(row) != {
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
        }:
            raise ValueError("Commons source fields")
        for field in ("id", "ancestry_group"):
            if not isinstance(row[field], str) or not re.fullmatch(r"[a-z0-9-]{1,80}", row[field]):
                raise ValueError("unsafe source identity")
        page = urllib.parse.urlsplit(row["page"])
        media = urllib.parse.urlsplit(row["original_url"])
        if (
            page.scheme != "https"
            or page.netloc != "commons.wikimedia.org"
            or not page.path.startswith("/wiki/File:")
            or page.query
            or page.fragment
            or media.scheme != "https"
            or media.netloc != "upload.wikimedia.org"
            or not media.path.startswith("/wikipedia/commons/")
            or "/thumb/" in media.path
            or media.query
            or media.fragment
            or row["license"] not in LICENSES
            or row["license_url"] != LICENSES[row["license"]]
            or not isinstance(row["author"], str)
            or not 1 <= len(row["author"]) <= 1024
            or row["role"] != "unassigned"
            or row["media_status"] != "not_downloaded"
            or row["ancestry_review"] not in {"pending", "whole-source-group-reviewed"}
            or type(row["original_bytes"]) is not int
            or not 1 <= row["original_bytes"] <= MAX_FILE
        ):
            raise ValueError("Commons license/provenance/role gate")
        artifacts.identity(row["original_sha1"], 40)
        if row["id"] in names or row["original_sha1"] in hashes:
            raise ValueError("duplicate original")
        if page.path in pages and pages[page.path] != row["ancestry_group"]:
            raise ValueError("source ancestry leakage")
        pages[page.path] = row["ancestry_group"]
        names.add(row["id"])
        hashes.add(row["original_sha1"])
    return value["sources"]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("acquisition redirect rejected")


def fetch(url: str, limit: int) -> bytes:
    address = urllib.parse.urlsplit(url)
    if (
        address.scheme != "https"
        or address.netloc != "upload.wikimedia.org"
        or not address.path.startswith("/wikipedia/commons/")
        or "/thumb/" in address.path
        or address.query
        or address.fragment
        or type(limit) is not int
        or not 1 <= limit <= MAX_FILE
    ):
        raise ValueError("download URL/limit allowlist")
    request = urllib.request.Request(  # noqa: S310 — explicit HTTPS host allowlist above.
        url, headers={"User-Agent": "steganography-dfir/0.6 research"}
    )
    opener = urllib.request.build_opener(_NoRedirect())
    with opener.open(request, timeout=30) as response:  # noqa: S310 — HTTPS host allowlist.
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("transfer limit")
    return raw


def acquire(value: dict, out: Path, *, allow_download: bool, prior_ledger: dict) -> dict:
    """Charge upper bounds before I/O. An unknown historical total stays unknown."""
    rows = manifest(value)
    if allow_download is not True:
        raise ValueError("explicit download command required")
    if any(r["ancestry_review"] != "whole-source-group-reviewed" for r in rows):
        raise ValueError("ancestry review required")
    if (
        set(prior_ledger) != {"sha256", "historical_bytes"}
        or prior_ledger["historical_bytes"] is not None
        and (
            type(prior_ledger["historical_bytes"]) is not int
            or prior_ledger["historical_bytes"] < 0
        )
    ):
        raise ValueError("prior transfer ledger required")
    artifacts.identity(prior_ledger["sha256"])
    total = sum(r["original_bytes"] for r in rows)
    if total + len(rows) > MAX_OUTPUT:
        raise ValueError("acquisition total byte limit")
    artifacts.directory(out)
    ledger = {
        "schema": "spatial-acquisition-v1",
        "status": "unavailable",
        "prior_ledger": prior_ledger,
        "reserved_transfer_bytes": total + len(rows),
        "historical_total_complete": prior_ledger["historical_bytes"] is not None,
        "manifest_sha256": artifacts.digest(artifacts.encode(value)),
    }
    artifacts.put(out / "reservation.json", artifacts.encode(ledger))
    acquired = []
    for row in rows:
        raw = fetch(row["original_url"], row["original_bytes"])
        if (
            len(raw) != row["original_bytes"]
            or hashlib.sha1(raw, usedforsecurity=False).hexdigest() != row["original_sha1"]
        ):
            raise ValueError("original hash/size mismatch")
        checksum = artifacts.put(out / (row["id"] + ".original"), raw, MAX_FILE)
        acquired.append({"id": row["id"], "sha256": checksum, "bytes": len(raw)})
    result = {**ledger, "status": "ready", "acquired": acquired}
    artifacts.put(out / "acquired.json", artifacts.encode(result))
    return result


def crop(raw: bytes) -> np.ndarray:
    if not raw or len(raw) > MAX_FILE:
        raise ValueError("carrier byte limit")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            if (
                image.format not in {"PNG", "BMP"}
                or image.mode not in {"L", "RGB"}
                or getattr(image, "n_frames", 1) != 1
                or min(width, height) < 256
                or width * height > MAX_PIXELS
                or image.info.get("transparency") is not None
                or image.getexif().get(274, 1) != 1
            ):
                raise ValueError("native single L/RGB PNG/BMP required without orientation/alpha")
            # Fixed top-left crop, before embedding; never resize or convert modes.
            return np.asarray(image.crop((0, 0, 256, 256)), dtype=np.uint8).copy()


def embed(pixels: np.ndarray, payload: bytes) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    if len(bits) > pixels.size:
        raise ValueError("payload capacity")
    result = pixels.copy()
    flat = result.reshape(-1)
    flat[: len(bits)] = (flat[: len(bits)] & 254) | bits
    return result


def recover(pixels: np.ndarray, length: int) -> bytes:
    """Scalar decoder independent of vectorized packing/embedding."""
    if type(length) is not int or not 0 <= length <= pixels.size // 8:
        raise ValueError("decoder length bound")
    result = bytearray()
    flat = pixels.reshape(-1)
    for start in range(0, length * 8, 8):
        value = 0
        for offset in range(8):
            value = value * 2 + (int(flat[start + offset]) % 2)
        result.append(value)
    return bytes(result)


def prepare(value: dict, originals: Path, out: Path) -> dict:
    rows = manifest(value)
    if any(r["ancestry_review"] != "whole-source-group-reviewed" for r in rows):
        raise ValueError("ancestry review required")
    artifacts.directory(out)
    records = []
    seen: set[str] = set()
    total = 0
    for row in rows:
        raw = artifacts.read(originals / (row["id"] + ".original"), MAX_FILE)
        if (
            len(raw) != row["original_bytes"]
            or hashlib.sha1(raw, usedforsecurity=False).hexdigest() != row["original_sha1"]
        ):
            raise ValueError("original hash/size mismatch")
        pixels = crop(raw)
        pixel_hash = artifacts.digest(pixels.tobytes())
        if pixel_hash in seen:
            raise ValueError("duplicate decoded crop; merge common ancestry before admission")
        seen.add(pixel_hash)
        payload = bytes.fromhex(artifacts.digest((row["id"] + ":generated-control").encode()))
        stego = embed(pixels, payload)
        for extension in ("png", "bmp"):
            for kind, values in (("cover", pixels), ("lsb-control", stego)):
                buffer = io.BytesIO()
                Image.fromarray(values).save(buffer, format=extension.upper())
                encoded = buffer.getvalue()
                decoded = crop(encoded)
                if not np.array_equal(decoded, values) or (
                    kind == "lsb-control" and recover(decoded, len(payload)) != payload
                ):
                    raise ValueError("independent payload/pixel check")
                total += len(encoded)
                if total > MAX_OUTPUT:
                    raise ValueError("preparation output limit")
                name = f"{row['id']}-{kind}.{extension}"
                checksum = artifacts.put(out / name, encoded, MAX_FILE)
                records.append(
                    {
                        "name": name,
                        "sha256": checksum,
                        "group": row["ancestry_group"],
                        "role": "unassigned",
                        "native_mode": "L" if pixels.ndim == 2 else "RGB",
                        "original_sha256": artifacts.digest(raw),
                        "source_id": row["id"],
                        "crop": [0, 0, 256, 256],
                        "kind": kind,
                        "payload_verified": kind == "lsb-control",
                    }
                )
    result = {
        "schema": "spatial-prepared-v1",
        "status": "ready",
        "artifacts": records,
        "manifest_sha256": artifacts.digest(artifacts.encode(value)),
        "output_bytes": total,
        "representative_benchmark": False,
        "optimizer_updates": 0,
    }
    artifacts.put(out / "prepared.json", artifacts.encode(result))
    return result
