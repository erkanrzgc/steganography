"""Explicit bounded BOWS2 original acquisition; never runs or extracts archive paths."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import tarfile
import time
import urllib.request
from pathlib import PurePosixPath

from PIL import Image

PAGE = "https://dud.inf.tu-dresden.de/~westfeld/rsp/rsp.html"
URL = "https://dud.inf.tu-dresden.de/~westfeld/rsp/bows2-1g.tar.gz"
MAX_ARCHIVE = 200 * 1024**2
MAX_EXPANDED = 320 * 1024**2
MAX_FILE = 300 * 1024
COUNT = 1000
NATIVE_SHA256 = "3e26b0faf740f7cd6d3a67df5ac86dba1631b8bcc69f17302498c785e240bd49"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fresh(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("dataset paths cannot use symlinks")
    if path.exists():
        raise FileExistsError("dataset output already exists")


def read(path, limit):
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("dataset input must be a nonsymlink regular file")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("dataset input exceeds byte limit")
    return raw


def download(url, limit, deadline):
    if url not in {URL, PAGE}:
        raise ValueError("unexpected dataset origin")
    # Two fixed public HTTPS endpoints; no caller-selected scheme or credentials.
    with urllib.request.build_opener(NoRedirect).open(url, timeout=15) as response:  # noqa: S310
        if response.status != 200 or int(response.headers.get("Content-Length", "0")) > limit:
            raise ValueError("dataset response status/length outside limits")
        result = io.BytesIO()
        while True:
            deadline()
            chunk = response.read(min(65536, limit + 1 - result.tell()))
            if not chunk:
                return result.getvalue()
            result.write(chunk)
            if result.tell() > limit:
                raise ValueError("dataset download exceeds byte limit")


def identities(paths):
    if not paths:
        raise ValueError("prior acquisition manifests are required")
    reserved, manifests = set(), []
    for path in paths:
        raw = read(path, 8 * 1024**2)
        document = json.loads(raw)
        if not isinstance(document, dict) or not document.get("samples"):
            raise ValueError("reserved manifest must contain original identities")
        for row in document["samples"]:
            for key in ("sha256", "lineage"):
                digest = row.get(key)
                if digest is None and key == "lineage":
                    continue
                if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                    raise ValueError("invalid reserved identity")
                reserved.add(digest)
        manifests.append(hashlib.sha256(raw).hexdigest())
    return reserved, sorted(manifests)


def inspect(raw, reserved, deadline, *, native=False):
    if native and hashlib.sha256(raw).hexdigest() != NATIVE_SHA256:
        raise ValueError("native-layout archive checksum mismatch")
    if len(raw) > MAX_ARCHIVE:
        raise ValueError("compressed archive exceeds limit")
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        expanded = stream.read(MAX_EXPANDED + 1)
    if len(expanded) > MAX_EXPANDED:
        raise ValueError("expanded archive exceeds limit")
    deadline()
    rows, names, hashes, pixels = [], set(), set(), set()
    # Only plain bounded USTAR records. Parse headers before tarfile can consume
    # unbounded PAX/long-name/sparse extension structures or instantiate members.
    offset, members = 0, 0
    while offset + 512 <= len(expanded) and expanded[offset : offset + 512] != bytes(512):
        header = expanded[offset : offset + 512]
        if header[156:157] not in {b"0", b"\0", b"5"} or header[124] & 128:
            raise ValueError("archive links/extensions/sparse records forbidden")
        size = int(header[124:136].strip(b"\0 ") or b"0", 8)
        members += 1
        if members > 2000 or not 0 <= size <= MAX_FILE or offset + 512 + size > len(expanded):
            raise ValueError("TAR member count/size outside limits")
        offset += 512 + ((size + 511) // 512) * 512
    if offset + 1024 > len(expanded) or any(expanded[offset:]):
        raise ValueError("invalid or concatenated TAR ending")
    with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as archive:
        for member in archive:
            deadline()
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in member.name
                or member.name in names
                or member.sparse is not None
            ):
                raise ValueError("unsafe/duplicate archive path")
            names.add(member.name)
            if member.isdir() and member.size == 0:
                continue
            if (
                not member.isfile()
                or path.suffix.lower() != ".pgm"
                or not 0 < member.size <= MAX_FILE
            ):
                raise ValueError("unexpected archive member")
            image_stream = archive.extractfile(member)
            if image_stream is None:
                raise ValueError("missing archive image")
            with image_stream:
                data = image_stream.read(MAX_FILE + 1)
            if len(data) != member.size:
                raise ValueError("truncated archive image")
            digest = hashlib.sha256(data).hexdigest()
            if digest in reserved or digest in hashes:
                raise ValueError("duplicate/prior original identity")
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "PPM" or image.mode != "L" or image.size != (512, 512):
                    raise ValueError("unexpected BOWS2 image geometry/format")
                pixel_sha = hashlib.sha256(image.tobytes()).hexdigest()
            if pixel_sha in pixels:
                raise ValueError("duplicate decoded original")
            hashes.add(digest)
            pixels.add(pixel_sha)
            rows.append((member.name, digest, pixel_sha, data))
    if len(rows) != (1001 if native else COUNT):
        raise ValueError("incomplete BOWS2 original count")
    if native and {row[0] for row in rows} != ({f"{i}.pgm" for i in range(1, 1001)} | {"3661.pgm"}):
        raise ValueError("native-layout member identities mismatch")
    return sorted(rows)


def acquire(out, reserved_paths, *, archive=None, page=None, native=False):
    fresh(out)
    started = time.monotonic()

    def deadline():
        if time.monotonic() - started > 600:
            raise ValueError("BOWS2 acquisition deadline exceeded")

    reserved, bindings = identities(reserved_paths)
    page = download(PAGE, 65536, deadline) if page is None else page
    if len(page) > 65536 or b"bows2-1g.tar.gz" not in page:
        raise ValueError("source evaluation context missing")
    if native and archive is None:
        raise ValueError("native-layout import requires the explicitly cached archive")
    raw = download(URL, MAX_ARCHIVE, deadline) if archive is None else archive
    rows = inspect(raw, reserved, deadline, native=native)
    fresh(out)
    out.mkdir(parents=True)
    for name, data in (("archive.tar.gz", raw), ("source-page.html", page)):
        with (out / name).open("xb") as stream:
            stream.write(data)
    samples = []
    for index, (member, digest, pixel_sha, data) in enumerate(rows):
        deadline()
        target = f"{index:04d}.pgm"
        with (out / target).open("xb") as stream:
            stream.write(data)
        split_digest = hashlib.sha256(("bows2:20261010:" + digest).encode()).hexdigest()
        fraction = int(split_digest[:16], 16) / 2**64
        samples.append(
            {
                "path": target,
                "upstream_member": member,
                "sha256": digest,
                "lineage": digest,
                "pixel_sha256": pixel_sha,
                "bytes": len(data),
                "source_group": "BOWS2",
                "split": "train" if fraction < 0.8 else "validation",
                "camera": None,
                "device": None,
                "cover_cleanliness_verified": False,
            }
        )
    deadline()
    report = {
        "schema_version": "bows2-original-acquisition-v1",
        "status": "completed",
        "source_url": URL,
        "source_page_url": PAGE,
        "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "archive_bytes": len(raw),
        "source_page_sha256": hashlib.sha256(page).hexdigest(),
        "reserved_manifest_sha256": bindings,
        "upstream_sha256_verified": False,
        "license": "no_explicit_redistribution_license_verified_local_only",
        "samples": samples,
        "seconds": time.monotonic() - started,
        "accuracy_qualification": "unavailable",
        "model_trained": False,
        "archive_layout": "verified-native-1001-v1" if native else "advertised-1000-v1",
    }
    with (out / "source.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report
