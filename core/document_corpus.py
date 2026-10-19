"""Bounded local document pilot; no downloads, fitting or detector scores."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

MAX_FILE = 2 * 1024**2
MAX_OUTPUT = 16 * 1024**2
MAX_DOCUMENTS = 64
MAX_SECONDS = 120
EXTENSIONS = (".pdf", ".txt", ".md", ".zip")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def regular_read(path: Path, limit: int) -> bytes:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("document input symlink")
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
            raise ValueError("document input type/size")
        raw = stream.read(limit + 1)
    if len(raw) != info.st_size or len(raw) > limit:
        raise ValueError("document input changed or oversized")
    return raw


def _token(value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.+:-]{0,127}", value
    ) is None:
        raise ValueError("document public identity")
    return value


def _put(path: Path, raw: bytes) -> None:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("document output symlink")
    with path.open("xb") as stream:
        stream.write(raw)


def inspect(raw: bytes, extension: str) -> dict:
    """Only structural inventory checks, never a clean-file verdict."""
    if extension in (".txt", ".md"):
        text = raw.decode("utf-8")
        if "\0" in text or not text.strip():
            raise ValueError("document UTF-8 text required")
        return {"kind": "utf8_text", "characters": len(text)}
    if extension == ".pdf":
        if not raw.startswith(b"%PDF-") or b"%%EOF" not in raw[-1024:]:
            raise ValueError("document PDF header/end required")
        return {"kind": "pdf", "full_parser_validation": "unavailable"}
    if extension != ".zip":
        raise ValueError("unsupported document extension")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if not 0 < len(members) <= 512 or sum(m.file_size for m in members) > MAX_OUTPUT:
            raise ValueError("document archive member/expanded bound")
        for member in members:
            name = PurePosixPath(member.filename.replace("\\", "/"))
            mode = member.external_attr >> 16
            if (name.is_absolute() or ".." in name.parts or stat.S_ISLNK(mode)
                    or member.flag_bits & 1):
                raise ValueError("document unsafe/encrypted archive member")
    return {"kind": "zip", "members": len(members), "member_extraction": "not_performed"}


def _variants(raw: bytes, extension: str) -> tuple[bytes, bytes, str, str]:
    payload = hashlib.sha256(b"document-pilot-v1:" + raw).digest()
    if extension in (".txt", ".md"):
        bits = f"{int.from_bytes(payload, 'big'):0256b}"
        suffix = "".join("\u200b" if bit == "0" else "\u200c" for bit in bits).encode()
        method = "unicode-suffix-bitpair-v1"
    else:
        suffix = payload
        method = "opaque-append32-v1"
    return raw + suffix, raw + b" " * len(suffix), digest(payload), method


def recover(raw: bytes, original_bytes: int, method: str) -> bytes:
    """Scalar oracle uses known original extent, not autonomous recovery."""
    suffix = raw[original_bytes:]
    if method == "opaque-append32-v1":
        if len(suffix) != 32:
            raise ValueError("document opaque payload extent")
        return suffix
    if method != "unicode-suffix-bitpair-v1":
        raise ValueError("document oracle method")
    text = suffix.decode("utf-8")
    if len(text) != 256 or any(char not in "\u200b\u200c" for char in text):
        raise ValueError("document Unicode payload extent/alphabet")
    result = bytearray(32)
    for position, char in enumerate(text):
        result[position // 8] |= (char == "\u200c") << (7 - position % 8)
    return bytes(result)


def prepare(config: dict, out: Path, *, clock=time.monotonic) -> dict:
    started = clock()
    records = config.get("documents")
    if config.get("schema_version") != "local-document-pilot-config-v1" or (
        not isinstance(records, list) or not 1 <= len(records) <= MAX_DOCUMENTS
    ):
        raise ValueError("document configuration/count")
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("document fresh nonsymlink output required")
    blobs: dict[str, bytes] = {}
    sources: dict[Path, str] = {}
    samples: list[dict[str, Any]] = []
    seen: set[str] = set()

    def deadline():
        if clock() - started >= MAX_SECONDS:
            raise TimeoutError("document preparation deadline")

    for index, record in enumerate(records):
        deadline()
        source = Path(record["input"])
        license_path = Path(record["license_input"])
        if any(p.absolute().is_relative_to(out.absolute()) for p in (source, license_path)):
            raise ValueError("document input inside output")
        raw = regular_read(source, MAX_FILE)
        license_raw = regular_read(license_path, MAX_FILE)
        sha, license_sha = digest(raw), digest(license_raw)
        if sha != record["sha256"] or license_sha != record["license_sha256"] or sha in seen:
            raise ValueError("document checksum/duplicate")
        if record.get("license_review") != "declared_file_scope_reviewed":
            raise ValueError("document file-scope license review required")
        seen.add(sha)
        sources.update({source: sha, license_path: license_sha})
        extension = source.suffix.lower()
        inventory = inspect(raw, extension)
        positive, negative, payload_sha, method = _variants(raw, extension)
        if digest(recover(positive, len(raw), method)) != payload_sha:
            raise ValueError("document scalar payload oracle")
        license_name = "license-" + license_sha + ".txt"
        blobs[license_name] = license_raw
        variants = {}
        for role, contents in (("original", raw), ("controlled_positive", positive),
                               ("whitespace_control", negative)):
            name = f"document-{index:03d}-{role}{extension}"
            blobs[name] = contents
            variants[role] = {"artifact": name, "sha256": digest(contents), "bytes": len(contents)}
        samples.append({"source_id": _token(record["source_id"]),
                        "source_group": _token(record["source_group"]),
                        "license": _token(record["license"]), "license_artifact": license_name,
                        "license_sha256": license_sha, "original_sha256": sha,
                        "extension": extension, "split": "development_unassigned",
                        "cover_certification": "unavailable; natural source is a benign candidate",
                        "inventory": inventory, "method": method, "payload_bytes": 32,
                        "payload_sha256": payload_sha, "variants": variants})
        if sum(map(len, blobs.values())) > MAX_OUTPUT:
            raise ValueError("document aggregate output bound")
    report = {"schema_version": "local-document-pilot-v1", "status": "prepared_unassigned",
              "original_files": len(samples),
              "source_groups": len({s["source_group"] for s in samples}),
              "samples": samples, "training_ready": False, "optimizer_updates": 0,
              "statistical_detection": "unavailable", "downloaded_bytes": 0,
              "test_media_opened": False, "independent_recording_count": "unavailable",
              "autonomous_recovery": "unavailable; oracle needs original extent",
              "artifact_bytes_excluding_manifest": sum(map(len, blobs.values()))}
    encoded = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
    if sum(map(len, blobs.values())) + len(encoded) > MAX_OUTPUT:
        raise ValueError("document manifest/output bound")
    deadline()
    if any(digest(regular_read(path, MAX_FILE)) != sha for path, sha in sources.items()):
        raise ValueError("document source mutation")
    out.mkdir(parents=True, mode=0o700)
    for name, raw in blobs.items():
        deadline()
        _put(out / name, raw)
        if digest(regular_read(out / name, MAX_FILE + 768)) != digest(raw):
            raise ValueError("document output checksum")
    for sample in samples:
        deadline()
        original = sample["variants"]["original"]
        positive = sample["variants"]["controlled_positive"]
        raw = regular_read(out / positive["artifact"], MAX_FILE + 768)
        if digest(raw) != positive["sha256"] or digest(recover(
            raw, original["bytes"], sample["method"]
        )) != sample["payload_sha256"]:
            raise ValueError("document on-disk oracle/checksum")
    deadline()
    if any(digest(regular_read(path, MAX_FILE)) != sha for path, sha in sources.items()):
        raise ValueError("document source mutation after write")
    _put(out / "manifest.json", encoded)
    return report
