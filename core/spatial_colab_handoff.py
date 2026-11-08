"""Offline reviewed source delivery and train-only copies; never upload or acquire."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import struct
import zipfile
from pathlib import Path, PurePosixPath

from core import colab_artifacts as a
from core import colab_execution as process
from core import spatial_native_data as data
from core import spatial_native_training as engine

BASE_COMMIT = "2f41bfe2e95febcaab80f80ab4875c72073c3a32"
SOURCE_LIMIT = 4 * a.MIB
CACHE_LIMIT = 32 * a.MIB


def source_package(root: Path, target: Path) -> str:
    sources = process.source_hashes(root)
    records = {}
    for name in (*sources, "LICENSE"):
        path = root / name
        a.safe(path)
        raw = a.read(path, a.MIB) if path.stat().st_size else b""
        records[name] = {"sha256": a.digest(raw), "utf8": raw.decode()}
    packet = {"schema": "colab-native-source-package-v1", "base_commit": BASE_COMMIT,
              "files": records}
    return a.put(target, a.encode(packet), SOURCE_LIMIT)


def install_sources(packet: Path, checksum: str, out: Path) -> dict:
    a.identity(checksum)
    raw = a.read(packet, SOURCE_LIMIT)
    if a.digest(raw) != checksum:
        raise ValueError("reviewed source packet hash mismatch")
    value = json.loads(raw)
    if (set(value) != {"schema", "base_commit", "files"}
            or value["schema"] != "colab-native-source-package-v1"
            or value["base_commit"] != BASE_COMMIT or not isinstance(value["files"], dict)
            or not 3 <= len(value["files"]) <= 513 or "LICENSE" not in value["files"]):
        raise ValueError("reviewed source packet schema")
    prepared = {}
    for name, row in value["files"].items():
        if name != "LICENSE" and re.fullmatch(
            r"(?:core|steganography)/(?:[a-zA-Z0-9_]+/){0,3}[a-zA-Z0-9_]+\.py", name,
        ) is None:
            raise ValueError("reviewed source path")
        if (not isinstance(row, dict) or set(row) != {"sha256", "utf8"}
                or not isinstance(row["utf8"], str)):
            raise ValueError("reviewed source entry")
        content = row["utf8"].encode()
        if len(content) > a.MIB or a.digest(content) != a.identity(row["sha256"]):
            raise ValueError("reviewed source file hash/extent")
        prepared[name] = content
    a.directory(out)
    for name, content in prepared.items():
        target = out / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        a.safe(target)
        with target.open("xb") as stream:
            stream.write(content)
    observed = process.source_hashes(out)
    expected = {n: row["sha256"] for n, row in value["files"].items() if n.endswith(".py")}
    if observed != expected:
        raise ValueError("installed source closure mismatch")
    return {"source_package_sha256": checksum, "source_sha256": observed}


def cache_package(cache: data.Cache, split: engine.Split, target: Path) -> str:
    """Explicit copying command only; reserved/development pixel paths are never opened."""
    split.validate(cache)
    manifest = a.read(cache.root / "cache.json", data.MAX_MANIFEST)
    if a.digest(manifest) != cache.manifest_sha256:
        raise ValueError("cache manifest changed")
    lookup = {row.id: row for row in cache.rows}
    records = {}
    for source_id in split.train:
        row = lookup[source_id]
        size = 256 * 256 * (1 if row.mode == "L" else 3)
        raw = a.read(cache.root / (source_id + ".u8"), size)
        if len(raw) != size or a.digest(raw) != row.crop_sha256:
            raise ValueError("train cache copy hash mismatch")
        records[source_id] = {"sha256": row.crop_sha256,
                              "base64": base64.b64encode(raw).decode()}
    packet = {"schema": "colab-train-cache-package-v1", "manifest_utf8": manifest.decode(),
              "manifest_sha256": cache.manifest_sha256, "train_ids": list(split.train),
              "files": records}
    return a.put(target, a.encode(packet), CACHE_LIMIT)


def install_cache(packet: Path, checksum: str, split: engine.Split, out: Path) -> data.Cache:
    a.identity(checksum)
    raw = a.read(packet, CACHE_LIMIT)
    if a.digest(raw) != checksum:
        raise ValueError("train cache packet hash mismatch")
    value = json.loads(raw)
    if (set(value) != {"schema", "manifest_utf8", "manifest_sha256", "train_ids", "files"}
            or value["schema"] != "colab-train-cache-package-v1"
            or value["manifest_sha256"] != split.manifest_sha256
            or value["train_ids"] != list(split.train) or not isinstance(value["files"], dict)
            or set(value["files"]) != set(split.train)
            or not isinstance(value["manifest_utf8"], str)):
        raise ValueError("train-only cache packet schema/split")
    manifest = value["manifest_utf8"].encode()
    if len(manifest) > data.MAX_MANIFEST or a.digest(manifest) != split.manifest_sha256:
        raise ValueError("train cache metadata binding")
    a.directory(out)
    a.put(out / "cache.json", manifest, data.MAX_MANIFEST)
    cache = data.load(out, split.manifest_sha256)
    split.validate(cache)
    lookup = {row.id: row for row in cache.rows}
    for source_id, record in value["files"].items():
        if (not isinstance(record, dict) or set(record) != {"sha256", "base64"}
                or not isinstance(record["base64"], str) or len(record["base64"]) > 262144):
            raise ValueError("train cache entry extent")
        content = base64.b64decode(record["base64"], validate=True)
        row = lookup[source_id]
        if (len(content) != 256 * 256 * (1 if row.mode == "L" else 3)
                or a.digest(content) != row.crop_sha256 or record["sha256"] != row.crop_sha256):
            raise ValueError("train cache entry hash/extent")
        a.put(out / (source_id + ".u8"), content, 196608)
    return cache


def stream_hash(path: Path, limit: int) -> str:
    a.safe(path)
    digest, count = hashlib.sha256(), 0
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("regular evidence file required")
        while chunk := stream.read(a.MIB):
            count += len(chunk)
            if count > limit:
                raise ValueError("evidence file byte limit")
            digest.update(chunk)
    return digest.hexdigest()


def evidence_package(out: Path, limits: process.Limits, binding: str, *,
                     include_numeric: bool = True) -> str:
    """Private verified artifacts, before closing; no actual native pixels are in out."""
    a.identity(binding)
    total, inventory = process.files(out, limits, hashes=True)
    if not include_numeric:
        inventory = {n: v for n, v in inventory.items() if not n.endswith(".numeric")}
    selected_bytes = sum(v["bytes"] for v in inventory.values())
    max_members = 513 if include_numeric else 4097
    manifest = a.encode({"schema": "colab-profile-evidence-v1", "contract_sha256": binding,
                         "files": inventory})
    overhead = len(manifest) + sum(128 + 2 * len(n.encode()) for n in inventory) + 1024
    if (len(inventory) >= max_members or len(manifest) > a.MIB
            or selected_bytes + overhead > limits.file_bytes
            or total + selected_bytes + overhead + (65536 if include_numeric else 4 * a.MIB)
            > limits.output_bytes):
        raise ValueError("evidence archive reservation unavailable")
    target = out / ("verified-profile.zip" if include_numeric else "verified-training.zip")
    a.safe(target)
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        archive.writestr("payload-manifest.json", manifest)
        for name in sorted(inventory):
            path = out / name
            if stream_hash(path, limits.file_bytes) != inventory[name]["sha256"]:
                raise ValueError("evidence changed before packing")
            archive.write(path, arcname=name)
    checksum = stream_hash(target, limits.file_bytes)
    verify_evidence(target, checksum, binding, limits.file_bytes, max_members=max_members)
    return checksum


def development_package(cache: data.Cache, split: engine.Split, target: Path) -> str:
    """Later fit permission is checked by the caller before opening development pixels."""
    split.validate(cache)
    lookup = {row.id: row for row in cache.rows}
    records = {}
    for source_id in split.development:
        row = lookup[source_id]
        content = a.read(cache.root / (source_id + ".u8"), 196608)
        if row.mode != "RGB" or len(content) != 196608 or a.digest(content) != row.crop_sha256:
            raise ValueError("development cache extent/hash mismatch")
        records[source_id] = {"sha256": row.crop_sha256,
                              "base64": base64.b64encode(content).decode()}
    return a.put(target, a.encode({"schema": "colab-development-cache-package-v1",
                                   "manifest_sha256": cache.manifest_sha256,
                                   "development_ids": list(split.development), "files": records}),
                 CACHE_LIMIT)


def install_development(cache: data.Cache, split: engine.Split, packet: Path,
                         checksum: str) -> None:
    split.validate(cache)
    raw = a.read(packet, CACHE_LIMIT)
    if a.digest(raw) != a.identity(checksum):
        raise ValueError("development packet hash mismatch")
    value = json.loads(raw)
    if (set(value) != {"schema", "manifest_sha256", "development_ids", "files"}
            or value["schema"] != "colab-development-cache-package-v1"
            or value["manifest_sha256"] != cache.manifest_sha256
            or value["development_ids"] != list(split.development)
            or not isinstance(value["files"], dict)
            or set(value["files"]) != set(split.development)):
        raise ValueError("development packet scope mismatch")
    lookup = {row.id: row for row in cache.rows}
    prepared = {}
    for name, row in value["files"].items():
        if (not isinstance(row, dict) or set(row) != {"sha256", "base64"}
                or not isinstance(row["base64"], str) or len(row["base64"]) != 262144):
            raise ValueError("development entry extent")
        content = base64.b64decode(row["base64"], validate=True)
        expected = lookup[name]
        if (expected.mode != "RGB" or len(content) != 196608
                or a.digest(content) != expected.crop_sha256
                or row["sha256"] != expected.crop_sha256):
            raise ValueError("development entry hash")
        path = cache.root / (name + ".u8")
        a.safe(path)
        if path.exists():
            raise FileExistsError("development file already present")
        prepared[path] = content
    for path, content in prepared.items():
        a.put(path, content, 196608)


def verify_evidence(path: Path, checksum: str, binding: str, limit: int = 128 * a.MIB, *,
                     max_members: int = 513) -> dict:
    """Stream-verify data only. Never extract or execute a member."""
    try:
        if type(max_members) is not int or max_members not in (513, 4097):
            raise ValueError("reviewed evidence member ceiling required")
        return _verify_evidence(path, checksum, binding, limit, max_members)
    except zipfile.BadZipFile:
        raise ValueError("invalid evidence archive") from None


def _verify_evidence(path: Path, checksum: str, binding: str, limit: int,
                      max_members: int) -> dict:
    a.identity(checksum)
    a.identity(binding)
    if stream_hash(path, limit) != checksum:
        raise ValueError("evidence archive hash mismatch")
    # Reject excessive central directories before ZipFile allocates its member table.
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        if size < 22:
            raise ValueError("evidence archive trailer")
        stream.seek(-22, os.SEEK_END)
        signature, disk, start_disk, count_disk, count, directory_size, offset, comment = (
            struct.unpack("<4s4H2LH", stream.read(22))
        )
    if (signature != b"PK\x05\x06" or disk or start_disk or count_disk != count
            or not 2 <= count <= max_members or directory_size > a.MIB or comment
            or offset + directory_size != size - 22):
        raise ValueError("bounded single-disk evidence directory required")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if (not 2 <= len(infos) <= max_members or len(set(names)) != len(names)
                or "payload-manifest.json" not in names
                or sum(info.file_size for info in infos) > limit):
            raise ValueError("evidence archive inventory")
        for info in infos:
            relative = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if (relative.is_absolute() or ".." in relative.parts or info.is_dir()
                    or len(relative.parts) > 10 or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                    or info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1
                    or info.compress_size != info.file_size
                    or not 0 <= info.file_size <= limit):
                raise ValueError("evidence archive member")
        entry = archive.getinfo("payload-manifest.json")
        if entry.file_size > a.MIB:
            raise ValueError("evidence manifest extent")
        manifest = json.loads(archive.read(entry))
        if (set(manifest) != {"schema", "contract_sha256", "files"}
                or manifest["schema"] != "colab-profile-evidence-v1"
                or manifest["contract_sha256"] != binding
                or set(manifest["files"]) != set(names) - {"payload-manifest.json"}):
            raise ValueError("evidence manifest binding")
        for name, row in manifest["files"].items():
            if set(row) != {"bytes", "sha256"} or archive.getinfo(name).file_size != row["bytes"]:
                raise ValueError("evidence member size")
            digest, count = hashlib.sha256(), 0
            with archive.open(name) as stream:
                while chunk := stream.read(a.MIB):
                    count += len(chunk)
                    if count > limit:
                        raise ValueError("evidence member extent")
                    digest.update(chunk)
            if digest.hexdigest() != row["sha256"]:
                raise ValueError("evidence member hash")
    return manifest
