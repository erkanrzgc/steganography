"""Explicit ALASKA2 holdout acquisition using bounded HTTPS ZIP range reads.

No full-archive fallback, extraction to archive-controlled paths, or credential
downloads. Requires locally configured Kaggle credentials and accepted rules.
"""

import argparse
import base64
import hashlib
import io
import json
import random
import re
import stat
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

SOURCE_URL = "https://www.kaggle.com/c/alaska2-image-steganalysis/data"
API_URL = "https://www.kaggle.com/api/v1/competitions/data/download-all/alaska2-image-steganalysis"
FAMILIES = ("Cover", "JMiPOD", "JUNIWARD", "UERD")
SEED = 20261004
MAX_FILE = 2 * 1024 * 1024
MAX_METADATA = 64 * 1024 * 1024
MAX_TOTAL = 1024 * 1024 * 1024
MAX_MEMBERS = 310_000
MAX_SECONDS = 1800
MEMBER_NAME = re.compile(r"(Cover|JMiPOD|JUNIWARD|UERD)/([0-9]{5}\.jpg)\Z")


class AcquisitionError(ValueError):
    """Only safe, fixed messages; never include credential or signed URLs."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def open_request(request, timeout=30):
    return urllib.request.build_opener(NoRedirect).open(request, timeout=timeout)


def validate_archive_url(url):
    parsed = urllib.parse.urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc not in ("storage.googleapis.com", "storage.googleapis.com:443")
        or parsed.fragment
    ):
        raise AcquisitionError("unexpected signed archive origin")


def no_symlinks(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise AcquisitionError("symlink paths are not accepted")


def bounded_read(path, limit):
    no_symlinks(path)
    if not path.is_file() or path.stat().st_size > limit:
        raise AcquisitionError("local input is not a bounded regular file")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise AcquisitionError("local input exceeds byte limit")
    return data


def archive_url(credential_path):
    credentials = json.loads(bounded_read(credential_path, 16 * 1024))
    if not all(isinstance(credentials.get(k), str) and credentials[k] for k in ("username", "key")):
        raise AcquisitionError("invalid Kaggle credential document")
    if credential_path.stat().st_mode & 0o077:
        raise AcquisitionError("Kaggle credential must have private permissions")
    token = base64.b64encode((credentials["username"] + ":" + credentials["key"]).encode()).decode()
    # Fixed official HTTPS endpoint; automatic redirects are disabled.
    request = urllib.request.Request(API_URL, headers={"Authorization": "Basic " + token})  # noqa: S310
    try:
        with open_request(request):
            raise AcquisitionError("expected a signed archive redirect; body not downloaded")
    except urllib.error.HTTPError as exc:
        location = exc.headers.get("Location", "")
        code = exc.code
        exc.close()
        if code != 302:
            raise AcquisitionError(f"Kaggle archive access denied: HTTP {code}") from None
    validate_archive_url(location)
    return location


class RemoteZip(io.RawIOBase):
    """Seekable metadata reader plus thread-safe, globally bounded range requests."""

    def __init__(self, url, *, validate_origin=None):
        # Anonymous research acquisitions can explicitly supply their fixed-origin
        # validator; the authenticated ALASKA2 default stays storage-only.
        (validate_archive_url if validate_origin is None else validate_origin)(url)
        self._url = url
        self.deadline = time.monotonic() + MAX_SECONDS
        self.lock = threading.Lock()
        self.requested_bytes = 0
        self.position = 0
        self.requests = 0
        with open_request(urllib.request.Request(url, method="HEAD")) as response:  # noqa: S310
            if response.status != 200:
                raise AcquisitionError("unexpected archive HEAD response")
            self.size = int(response.headers.get("Content-Length", "0"))
            self.etag = response.headers.get("ETag", "")
        if not 22 <= self.size <= 64 * 1024**3 or not self.etag:
            raise AcquisitionError("missing or invalid immutable archive metadata")

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise AcquisitionError("invalid seek mode")
        position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if not 0 <= position <= self.size:
            raise AcquisitionError("invalid archive seek")
        self.position = position
        return position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        data = self.range(self.position, size)
        self.position += len(data)
        return data

    def range(self, start, size):
        if not 0 <= size <= MAX_METADATA or not 0 <= start <= start + size <= self.size:
            raise AcquisitionError("range exceeds bounds")
        if not size:
            return b""
        end = start + size - 1
        for attempt in range(3):
            with self.lock:
                remaining = self.deadline - time.monotonic()
                if remaining <= 0 or self.requests >= 12_100:
                    raise AcquisitionError("acquisition deadline/request limit exceeded")
                if self.requested_bytes + size > MAX_TOTAL:
                    raise AcquisitionError("aggregate network byte budget exceeded")
                self.requested_bytes += size
                self.requests += 1
            request = urllib.request.Request(  # noqa: S310 -- origin validated at construction
                self._url, headers={"Range": f"bytes={start}-{end}", "If-Match": self.etag}
            )
            try:
                with open_request(request, timeout=min(30, remaining)) as response:
                    if response.status != 206:
                        raise AcquisitionError("server refused bounded range; body not downloaded")
                    if (
                        response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}"
                        or response.headers.get("ETag") != self.etag
                        or int(response.headers.get("Content-Length", "0")) != size
                    ):
                        raise AcquisitionError("archive changed or mismatched range headers")
                    data = response.read(size + 1)
                if len(data) != size:
                    raise AcquisitionError("incomplete archive range")
                if time.monotonic() > self.deadline:
                    raise AcquisitionError("acquisition deadline exceeded")
                return data
            except urllib.error.HTTPError as exc:
                code = exc.code
                exc.close()
                if code not in (429, 500, 502, 503, 504) or attempt == 2:
                    raise AcquisitionError(f"archive request failed: HTTP {code}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 2:
                    raise AcquisitionError("archive network request failed") from None
            time.sleep(2**attempt)
        raise AcquisitionError("archive retries exhausted")  # pragma: no cover


def catalog_members(remote):
    # This acquisition contract accepts single-disk ZIP/ZIP64 with no comment.
    # Check directory bytes AND entry count before zipfile allocates ZipInfo objects.
    end = struct.unpack("<4s4H2IH", remote.range(remote.size - 22, 22))
    signature, disk, directory_disk, disk_count, count, size, offset, comment = end
    boundary = remote.size - 22
    if signature != b"PK\x05\x06" or disk or directory_disk or comment or disk_count != count:
        raise AcquisitionError("unsupported archive end record")
    if count == 0xFFFF or size == 0xFFFFFFFF or offset == 0xFFFFFFFF:
        locator = struct.unpack("<4sIQI", remote.range(remote.size - 42, 20))
        magic, zip_disk, record_offset, disks = locator
        if (
            magic != b"PK\x06\x07"
            or zip_disk
            or disks != 1
            or record_offset + 56 > remote.size - 42
        ):
            raise AcquisitionError("invalid ZIP64 locator")
        record = struct.unpack("<4sQ2H2I4Q", remote.range(record_offset, 56))
        magic, length, _, _, disk, directory_disk, disk_count, count, size, offset = record
        if magic != b"PK\x06\x06" or length != 44 or disk or directory_disk or disk_count != count:
            raise AcquisitionError("unsupported ZIP64 directory")
        boundary = record_offset
    if not 0 < count <= MAX_MEMBERS or not 0 < size <= MAX_METADATA or offset + size != boundary:
        raise AcquisitionError("archive directory count/size/offset limit exceeded")
    with zipfile.ZipFile(remote) as archive:
        members = archive.infolist()
    if len(members) != count:
        raise AcquisitionError("archive directory entry count mismatch")
    return members


def select_members(members, count, *, seed=SEED, excluded_names=()):
    if not 1 <= count <= 1000 or len(members) > MAX_MEMBERS:
        raise AcquisitionError("sample count or archive member limit exceeded")
    groups: dict[str, dict[str, zipfile.ZipInfo]] = {}
    for member in members:
        match = MEMBER_NAME.fullmatch(member.filename)
        if not match:
            continue
        family, name = match.groups()
        group = groups.setdefault(name, {})
        if family in group:
            raise AcquisitionError("duplicate archive member")
        group[family] = member
    complete = sorted(
        name
        for name, group in groups.items()
        if set(group) == set(FAMILIES) and name not in excluded_names
    )
    if len(complete) < count:
        raise AcquisitionError("insufficient complete cover/stego groups")
    # Reproducible sampling, not a cryptographic primitive.
    identities = sorted(random.Random(seed).sample(complete, count))  # noqa: S311
    selected = [groups[name][family] for name in identities for family in FAMILIES]
    for member in selected:
        mode = member.external_attr >> 16
        if (
            stat.S_IFMT(mode) not in (0, stat.S_IFREG)
            or member.flag_bits & 1
            or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
            or not 0 < member.file_size <= MAX_FILE
            or not 0 < member.compress_size <= MAX_FILE
        ):
            raise AcquisitionError("unsafe selected ZIP member")
    if sum(m.file_size for m in selected) > MAX_TOTAL:
        raise AcquisitionError("selected output exceeds byte budget")
    return selected, len(complete)


def member_bytes(remote, member):
    # A single bounded read covers the local header, name/extra and compressed
    # body. Refuse unusually large headers instead of widening the request.
    size = min(member.compress_size + 4096, remote.size - member.header_offset)
    raw = remote.range(member.header_offset, size)
    if len(raw) < 30:
        raise AcquisitionError("truncated ZIP local header")
    header = struct.unpack_from("<4s5H3I2H", raw)
    signature, _, flags, method, _, _, crc, compressed, expanded, namesize, extra = header
    start = 30 + namesize + extra
    encoding = "utf-8" if flags & 0x800 else "cp437"
    if (
        signature != b"PK\x03\x04"
        or flags != member.flag_bits
        or method != member.compress_type
        or start > 4096
        or start + member.compress_size > len(raw)
        or raw[30 : 30 + namesize].decode(encoding) != member.filename
        or (not flags & 8 and crc != member.CRC)
        or (not flags & 8 and compressed not in (member.compress_size, 0xFFFFFFFF))
        or (not flags & 8 and expanded not in (member.file_size, 0xFFFFFFFF))
    ):
        raise AcquisitionError("ZIP local header does not match catalog")
    data = raw[start : start + member.compress_size]
    if method == zipfile.ZIP_DEFLATED:
        decoder = zlib.decompressobj(-15)
        data = decoder.decompress(data, member.file_size + 1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise AcquisitionError("invalid or oversized compressed member")
    if len(data) != member.file_size or zlib.crc32(data) != member.CRC:
        raise AcquisitionError("member size/CRC mismatch")
    return data


def validate_image(data, member, reserved):
    if len(data) != member.file_size or zlib.crc32(data) != member.CRC:
        raise AcquisitionError("existing member size/CRC mismatch")
    digest = hashlib.sha256(data).hexdigest()
    if digest in reserved:
        raise AcquisitionError("sample overlaps a reserved corpus")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG" or not 0 < image.width * image.height <= 4_000_000:
            raise AcquisitionError("unexpected image format/dimensions")
        width, height = image.size
        image.load()
    family = member.filename.split("/")[0]
    return {
        "path": member.filename,
        "sha256": digest,
        "size": len(data),
        "upstream_crc32": f"{member.CRC:08x}",
        "format": "JPEG",
        "width": width,
        "height": height,
        "label": "cover" if family == "Cover" else "stego",
        "method": None if family == "Cover" else family,
        "source_group": "ALASKA2",
        "camera": None,
        "device": None,
        "app": None,
        "payload_rate": None,
        "quality_factor": None,
        "split": "test",
    }


def write_json(path, value):
    no_symlinks(path)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def acquire(
    out,
    credential,
    reserved_paths,
    *,
    count=1000,
    resume=False,
    purpose="evaluation",
    seed=SEED,
):
    if purpose not in {"evaluation", "development"}:
        raise AcquisitionError("invalid acquisition purpose")
    no_symlinks(out)
    if (out / "source.json").exists():
        raise AcquisitionError("completed acquisition exists; refusing overwrite")
    reserved: set[str] = set()
    reserved_documents = []
    excluded_names = set()
    for path in reserved_paths:
        raw = bounded_read(path, MAX_METADATA)
        for sample in json.loads(raw)["samples"]:
            reserved.update(sample.get(k, "") for k in ("sha256", "lineage"))
            match = MEMBER_NAME.fullmatch(str(sample.get("path", "")))
            if sample.get("source_group") == "ALASKA2" and match:
                excluded_names.add(match.group(2))
        reserved_documents.append(hashlib.sha256(raw).hexdigest())
    if resume and not out.is_dir():
        raise AcquisitionError("resume requires an existing acquisition directory")
    if not resume:
        out.mkdir(parents=True, mode=0o700, exist_ok=False)
    remote = RemoteZip(archive_url(credential))
    members = catalog_members(remote)
    selected, eligible = select_members(members, count, seed=seed, excluded_names=excluded_names)
    archive_count = len(members)
    selection = {
        "schema_version": "alaska2-selection-v1",
        "source_url": SOURCE_URL,
        "seed": seed,
        "algorithm": "random.Random(seed).sample(sorted_complete_basenames, count)",
        "purpose": (
            "frozen evaluation only; never train or calibrate on these samples"
            if purpose == "evaluation"
            else "development only; not an independent test source"
        ),
        "archive_bytes": remote.size,
        "archive_etag": remote.etag,
        "archive_members": archive_count,
        "eligible_lineages": eligible,
        "reserved_manifest_sha256": reserved_documents,
        "selected_bytes": sum(m.file_size for m in selected),
        "members": [
            {
                "path": m.filename,
                "size": m.file_size,
                "crc32": f"{m.CRC:08x}",
                "compressed_size": m.compress_size,
                "header_offset": m.header_offset,
            }
            for m in selected
        ],
    }
    if purpose == "development":
        selection["excluded_reserved_lineages"] = len(excluded_names)
        selection["split_policy"] = (
            "whole-lineage SHA256(seed:basename), train < 0.8 else validation"
        )
    plan_path = out / "selection.json"
    if resume:
        if json.loads(bounded_read(plan_path, MAX_METADATA)) != selection:
            raise AcquisitionError("resume selection/archive/reserved provenance changed")
    else:
        write_json(plan_path, selection)
    for family in FAMILIES:
        no_symlinks(out / family)
        (out / family).mkdir(mode=0o700, exist_ok=resume)
    print(
        f"selected {count} lineages / {len(selected)} files / {selection['selected_bytes']} bytes",
        flush=True,
    )

    stopped = threading.Event()

    def fetch_one(member):
        target = out / member.filename
        no_symlinks(target)
        existed = target.exists()
        if existed and not resume:
            raise AcquisitionError("existing output cannot be overwritten")
        data = bounded_read(target, MAX_FILE) if existed else member_bytes(remote, member)
        record = validate_image(data, member, reserved)
        if not existed:
            with target.open("xb") as stream:
                stream.write(data)
        return record

    def fetch(member):
        if stopped.is_set():
            raise AcquisitionError("acquisition stopped after an earlier failure")
        try:
            return fetch_one(member)
        except Exception:
            stopped.set()
            raise

    records = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        try:
            for index, record in enumerate(executor.map(fetch, selected), 1):
                records.append(record)
                if index % 100 == 0:
                    print(f"verified {index}/{len(selected)} files", flush=True)
        except BaseException:
            stopped.set()
            raise
    for index in range(0, len(records), 4):
        cover = records[index]
        for record in records[index : index + 4]:
            if (record["width"], record["height"]) != (cover["width"], cover["height"]):
                raise AcquisitionError("cover/stego dimensions differ")
            record["lineage"] = cover["sha256"]
            if purpose == "development":
                name = Path(cover["path"]).name
                fraction = (
                    int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big")
                    / 2**64
                )
                record["split"] = "train" if fraction < 0.8 else "validation"
            if (
                hashlib.sha256(bounded_read(out / record["path"], MAX_FILE)).hexdigest()
                != record["sha256"]
            ):
                raise AcquisitionError("written file SHA-256 mismatch")
    manifest = {
        "schema_version": "alaska2-acquisition-v1",
        "source_group": "ALASKA2",
        "source_url": SOURCE_URL,
        "license": "Subject to Competition Rules; local research only; do not redistribute",
        "purpose": purpose,
        "selection_sha256": hashlib.sha256(bounded_read(plan_path, MAX_METADATA)).hexdigest(),
        "selection_seed": seed,
        "upstream_sha256_verified": False,
        "upstream_crc32_verified": True,
        "reserved_sha256_overlap": 0,
        "reserved_manifest_sha256": reserved_documents,
        "total_bytes": sum(r["size"] for r in records),
        "network_requested_bytes_upper_bound": remote.requested_bytes,
        "network_range_requests": remote.requests,
        "unique_cover_hashes": len({r["sha256"] for r in records if r["label"] == "cover"}),
        "samples": records,
    }
    write_json(out / "source.json", manifest)
    print(
        f"complete: {len(records)} files; CRC, JPEG decode and SHA-256 verified locally", flush=True
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--credential", type=Path, default=Path.home() / ".kaggle/kaggle.json")
    parser.add_argument("--reserved-manifest", action="append", type=Path, required=True)
    parser.add_argument("--purpose", choices=("evaluation", "development"), default="evaluation")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--count", type=int, default=1000, choices=range(1, 1001), metavar="1..1000"
    )
    parser.add_argument(
        "--resume", action="store_true", help="validate and reuse only an incomplete selection"
    )
    args = parser.parse_args(argv)
    try:
        acquire(
            args.out,
            args.credential,
            args.reserved_manifest,
            count=args.count,
            resume=args.resume,
            purpose=args.purpose,
            seed=args.seed,
        )
    except Exception as exc:
        # Library/network exceptions may include secret signed URLs or host paths.
        detail = str(exc) if isinstance(exc, AcquisitionError) else type(exc).__name__
        print(
            f"acquisition failed: {detail}; incomplete output retained, no success manifest written"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
