"""Explicit pinned FSDD cover acquisition; no stego or accuracy claim."""

import argparse
import hashlib
import io
import json
import re
import stat
import struct
import time
import urllib.request
import wave
import zipfile
from pathlib import Path

COMMIT = "d6938f9bf1545aa66d8489fc9f1385a7abd64282"  # upstream v1.0.10
SOURCE = "https://github.com/Jakobovski/free-spoken-digit-dataset"
URL = f"https://codeload.github.com/Jakobovski/free-spoken-digit-dataset/zip/{COMMIT}"
PREFIX = f"free-spoken-digit-dataset-{COMMIT}/"
MAX_ARCHIVE = 64 * 1024 * 1024
MAX_FILE = 1024 * 1024
MAX_MEMBERS = 4096


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def download():
    started = time.monotonic()
    # One fixed HTTPS endpoint, no credentials, redirects or executable content.
    with urllib.request.build_opener(NoRedirect).open(URL, timeout=15) as response:
        if response.status != 200:
            raise ValueError("unexpected FSDD response")
        if int(response.headers.get("Content-Length", "0")) > MAX_ARCHIVE:
            raise ValueError("FSDD archive exceeds byte budget")
        output = io.BytesIO()
        while True:
            data = response.read(min(65536, MAX_ARCHIVE + 1 - output.tell()))
            if not data:
                break
            output.write(data)
            if output.tell() > MAX_ARCHIVE or time.monotonic() - started > 180:
                raise ValueError("FSDD acquisition byte/time limit exceeded")
        return output.getvalue()


def acquire(out, *, archive=None):
    if any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("symlink output is forbidden")
    if out.exists():
        raise FileExistsError("output exists")
    raw = download() if archive is None else archive
    # Bound central directory allocation before ZipFile constructs ZipInfo objects.
    offset = raw.rfind(b"PK\x05\x06", max(0, len(raw) - 65557))
    if len(raw) > MAX_ARCHIVE or offset < 0 or offset + 22 > len(raw):
        raise ValueError("invalid bounded archive")
    _, disk, directory_disk, disk_count, count, size, start, comment = struct.unpack_from(
        "<4s4H2IH", raw, offset
    )
    if (
        disk
        or directory_disk
        or disk_count != count
        or not 1 <= count <= MAX_MEMBERS
        or size > 2 * 1024 * 1024
        or start + size != offset
        or offset + 22 + comment != len(raw)
    ):
        raise ValueError("archive directory outside limits")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        readmes = [m for m in archive.infolist() if m.filename == PREFIX + "README.md"]
        if (
            len(readmes) != 1
            or not 0 < readmes[0].file_size <= 32768
            or not 0 < readmes[0].compress_size <= 32768
            or readmes[0].flag_bits & 1
            or stat.S_IFMT(readmes[0].external_attr >> 16) not in (0, stat.S_IFREG)
            or readmes[0].compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
        ):
            raise ValueError("missing or unsafe source license evidence")
        with archive.open(readmes[0]) as stream:
            readme = stream.read(32769)
        if (
            len(readme) != readmes[0].file_size
            or b"https://creativecommons.org/licenses/by-sa/4.0/" not in readme
        ):
            raise ValueError("source license evidence changed")
        names = set()
        selected = []
        total = 0
        for member in archive.infolist():
            if not member.filename.endswith(".wav"):
                continue
            match = re.fullmatch(
                re.escape(PREFIX) + r"recordings/([0-9]_([a-z]+)_[0-9]+\.wav)", member.filename
            )
            if (
                not match
                or not 0 < member.file_size <= MAX_FILE
                or not 0 < member.compress_size <= MAX_FILE
                or member.flag_bits & 1
                or stat.S_IFMT(member.external_attr >> 16) not in (0, stat.S_IFREG)
                or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
            ):
                raise ValueError("unsafe or oversized WAV member")
            name, speaker = match.groups()
            if name in names:
                raise ValueError("duplicate WAV member")
            names.add(name)
            total += member.file_size
            if total > MAX_ARCHIVE:
                raise ValueError("expanded corpus exceeds byte limit")
            selected.append((name, speaker, member))
        if not selected:
            raise ValueError("no WAV recordings")
        out.mkdir(parents=True, mode=0o700, exist_ok=False)
        records = []
        for name, speaker, member in sorted(selected):
            with archive.open(member) as stream:
                data = stream.read(MAX_FILE + 1)  # ZIP reader verifies upstream CRC.
            if len(data) != member.file_size:
                raise ValueError("WAV size mismatch")
            with wave.open(io.BytesIO(data)) as audio:
                if (
                    audio.getnchannels() != 1
                    or audio.getsampwidth() != 2
                    or audio.getframerate() != 8000
                    or not 0 < audio.getnframes() <= 80000
                    or audio.getcomptype() != "NONE"
                ):
                    raise ValueError("unexpected bounded PCM format")
                frames = audio.getnframes()
                if len(audio.readframes(frames)) != frames * 2:
                    raise ValueError("truncated PCM samples")
            with (out / name).open("xb") as stream:
                stream.write(data)
            digest = hashlib.sha256(data).hexdigest()
            records.append(
                {
                    "path": name,
                    "sha256": digest,
                    "lineage": digest,
                    "size": len(data),
                    "source_group": "FSDD-v1.0.10",
                    "speaker": speaker,
                    "device": None,
                    "split": None,
                    "label": "cover",
                    "method": None,
                    "format": "WAV",
                    "sample_rate": 8000,
                    "sample_width": 2,
                    "channels": 1,
                    "frames": frames,
                    "upstream_crc32": f"{member.CRC:08x}",
                }
            )
    manifest = {
        "schema_version": "fsdd-acquisition-v1",
        "source_url": SOURCE,
        "source_commit": COMMIT,
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "attribution": "FSDD contributors (Jakobovski/free-spoken-digit-dataset)",
        "license_evidence_url": f"{SOURCE}/blob/{COMMIT}/README.md",
        "license_evidence_sha256": hashlib.sha256(readme).hexdigest(),
        "purpose": "candidate covers only; no stego generation or evaluation yet",
        "split_policy": "unassigned; group by speaker and original recording before embedding",
        "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "upstream_sha256_verified": False,
        "archive_bytes": len(raw),
        "total_bytes": total,
        "samples": records,
    }
    with (out / "source.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = acquire(args.out)
    except Exception as exc:
        parser.exit(2, f"FSDD acquisition failed ({type(exc).__name__}); partial output retained\n")
    print(f"FSDD acquired: {len(manifest['samples'])} recordings; no benchmark performed")


if __name__ == "__main__":
    main()
