"""Explicit bounded DIV2K/ESC-50 acquisition; media, never upstream code."""

from __future__ import annotations

import csv
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
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image

ESC_COMMIT = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
ESC_PREFIX = f"ESC-50-{ESC_COMMIT}/"
DIV_PAGE = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
ESC_SOURCE = "https://github.com/karolpiczak/ESC-50"
SOURCES = {
    "div2k-train": {
        "url": DIV_PAGE + "DIV2K_train_HR.zip",
        "evidence": DIV_PAGE,
        "prefix": "DIV2K_train_HR/",
        "count": 800,
        "first": 1,
        "source": "DIV2K",
        "license": "academic_research_only_original_owner_copyright",
        "archive_limit": 4 * 1024**3,
    },
    "div2k-validation": {
        "url": DIV_PAGE + "DIV2K_valid_HR.zip",
        "evidence": DIV_PAGE,
        "prefix": "DIV2K_valid_HR/",
        "count": 100,
        "first": 801,
        "source": "DIV2K",
        "license": "academic_research_only_original_owner_copyright",
        "archive_limit": 1024**3,
    },
    "esc50": {
        "url": f"https://codeload.github.com/karolpiczak/ESC-50/zip/{ESC_COMMIT}",
        "evidence": f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{ESC_COMMIT}/README.md",
        "prefix": ESC_PREFIX + "audio/",
        "count": 2000,
        "source": "ESC-50",
        "license": "CC-BY-NC-3.0; ESC-10 subset separately CC-BY",
        "archive_limit": 1024**3,
    },
}
MAX_SECONDS = 2400
MAX_FILE = 32 * 1024**2
MAX_EXPANDED = 6 * 1024**3
MAX_DOCUMENT = 1024**2
LEGACY_ORIGINAL_MANIFESTS = frozenset(
    {
        "a2412f69124b3c2cb84907d8f276dd7a2f21a0aacb6eac212a123c889822f832",
        "a81fb50acdf989fe3cb37c6ac88316140e611d5f1da3046962426498ac1a7728",
    }
)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def regular(path):
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError("media input must be regular and non-symlink")


def fetch(url, out, limit, deadline):
    """Fixed-origin caller; stream to exclusive file, not a corpus-sized buffer."""
    with urllib.request.build_opener(NoRedirect).open(url, timeout=15) as response:
        if response.status != 200:
            raise ValueError("unexpected acquisition HTTP response")
        declared = response.headers.get("Content-Length")
        expected = int(declared) if declared is not None else None
        if expected is not None and not 0 <= expected <= limit:
            raise ValueError("download exceeds byte limit")
        size = 0
        digest = hashlib.sha256()
        with out.open("xb") as stream:
            while True:
                deadline()
                chunk = response.read(min(1024**2, limit + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise ValueError("download exceeds byte limit")
                stream.write(chunk)
                digest.update(chunk)
        if not size or (expected is not None and size != expected):
            raise ValueError("download truncated or empty")
    return {"sha256": digest.hexdigest(), "bytes": size}


def zip_preflight(path, limit):
    """Bound directory allocation before ZipFile creates member objects."""
    regular(path)
    size = path.stat().st_size
    if not 22 <= size <= limit:
        raise ValueError("archive byte limit")
    with path.open("rb") as stream:
        stream.seek(size - min(size, 65557))
        tail = stream.read(65557)
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0 or offset + 22 > len(tail):
        raise ValueError("invalid ZIP directory")
    _, disk, d_disk, n_disk, count, directory_size, start, comment = struct.unpack_from(
        "<4s4H2IH", tail, offset
    )
    end = size - len(tail) + offset
    if (
        disk
        or d_disk
        or n_disk != count
        or not 1 <= count <= 5000
        or directory_size > 2 * 1024**2
        or start + directory_size != end
        or offset + 22 + comment != len(tail)
    ):
        raise ValueError("ZIP directory allocation outside limits")


def members(archive, config):
    names, selected, total = set(), {}, 0
    for member in archive.infolist():
        name, path = member.filename, PurePosixPath(member.filename)
        kind = stat.S_IFMT(member.external_attr >> 16)
        total += member.file_size
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or name in names
            or kind not in (0, stat.S_IFREG, stat.S_IFDIR)
            or member.flag_bits & 1
            or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
            or member.file_size > 128 * 1024**2
            or total > MAX_EXPANDED
            or (member.is_dir() and member.file_size != 0)
        ):
            raise ValueError("unsafe or oversized ZIP member")
        names.add(name)
        if name.startswith(config["prefix"]) and not member.is_dir():
            basename = name[len(config["prefix"]) :]
            pattern = (
                r"[0-9]{4}\.png"
                if config["source"] == "DIV2K"
                else r"[1-5]-[0-9]+-[A-Z]-[0-9]+\.wav"
            )
            if not re.fullmatch(pattern, basename) or not 0 < member.file_size <= MAX_FILE:
                raise ValueError("invalid media name/size")
            selected[basename] = member
    if len(selected) != config["count"]:
        raise ValueError("incomplete media count")
    if config["source"] == "DIV2K" and set(selected) != {
        f"{i:04d}.png" for i in range(config["first"], config["first"] + config["count"])
    }:
        raise ValueError("unexpected DIV2K original identities")
    return selected


def esc_metadata(raw):
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    if len(rows) != 2000:
        raise ValueError("ESC metadata count")
    result = {}
    groups: dict[str, str] = {}
    cells: Counter[tuple[str, int]] = Counter()
    for row in rows:
        filename = row.get("filename", "")
        fold, source, target = row.get("fold", ""), row.get("src_file", ""), row.get("target", "")
        if (
            not re.fullmatch(r"[1-5]-[0-9]+-[A-Z]-[0-9]+\.wav", filename)
            or fold not in "12345"
            or len(fold) != 1
            or not source.isdigit()
            or not target.isdigit()
            or not 0 <= int(target) < 50
            or filename != f"{fold}-{source}-{row.get('take')}-{target}.wav"
            or row.get("esc10") not in ("True", "False")
            or not row.get("category")
            or filename in result
            or groups.setdefault(source, fold) != fold
        ):
            raise ValueError("ESC source fragments cross folds or metadata invalid")
        result[filename] = row
        cells[(fold, int(target))] += 1
    if len(cells) != 250 or set(cells.values()) != {8}:
        raise ValueError("ESC fold/class balance mismatch")
    return result


def media_details(data, source):
    if source == "DIV2K":
        with Image.open(io.BytesIO(data)) as image:
            if (
                image.format != "PNG"
                or image.mode != "RGB"
                or getattr(image, "n_frames", 1) != 1
                or min(image.size) < 256
                or image.width * image.height > 8_000_000
            ):
                raise ValueError("DIV2K PNG geometry/mode outside limits")
            decoded = image.tobytes()
            return {
                "width": image.width,
                "height": image.height,
                "decoded_sha256": hashlib.sha256(decoded).hexdigest(),
                "format": "PNG",
            }
    with wave.open(io.BytesIO(data)) as audio:
        if audio.getparams()[:4] != (1, 2, 44100, 220500) or audio.getcomptype() != "NONE":
            raise ValueError("ESC WAV PCM contract")
        pcm = audio.readframes(220500)
        if len(pcm) != 441000:
            raise ValueError("truncated ESC WAV samples")
    return {
        "sample_rate": 44100,
        "sample_width": 2,
        "channels": 1,
        "frames": 220500,
        "decoded_sha256": hashlib.sha256(pcm).hexdigest(),
        "format": "WAV",
    }


def acquire(key: str, out: Path, *, reserved_paths: list[Path], archive_path: Path | None = None):
    if key not in SOURCES:
        raise ValueError("unsupported explicit media source")
    if out.exists() or any(p.is_symlink() for p in (out, *out.parents)):
        raise FileExistsError("output exists or uses symlink")
    config = SOURCES[key]
    started = time.monotonic()

    def deadline():
        if time.monotonic() - started > MAX_SECONDS:
            raise ValueError("media acquisition deadline exceeded")

    if not 1 <= len(reserved_paths) <= 128:
        raise ValueError("prior acquisition manifests required")
    reserved, prior_bindings = set(), []
    for prior_path in reserved_paths:
        regular(prior_path)
        with prior_path.open("rb") as prior_stream:
            prior_raw = prior_stream.read(8 * MAX_DOCUMENT + 1)
        if len(prior_raw) > 8 * MAX_DOCUMENT:
            raise ValueError("prior manifest byte limit")
        prior = json.loads(prior_raw)
        prior_digest = hashlib.sha256(prior_raw).hexdigest()
        samples = prior.get("samples") if isinstance(prior, dict) else None
        if not isinstance(samples, list) or not 1 <= len(samples) <= 15000:
            raise ValueError("prior manifest sample limit")
        for sample in samples:
            for field in ("sha256", "lineage"):
                digest = sample.get(field) if isinstance(sample, dict) else None
                if (
                    field == "lineage"
                    and digest is None
                    and prior_digest in LEGACY_ORIGINAL_MANIFESTS
                ):
                    digest = sample.get("sha256")
                if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
                    raise ValueError("invalid prior original identity")
                reserved.add(digest)
        prior_bindings.append(prior_digest)

    if archive_path is not None:
        zip_preflight(archive_path, config["archive_limit"])
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    proof = fetch(config["evidence"], out / "license-evidence.txt", MAX_DOCUMENT, deadline)
    evidence = (out / "license-evidence.txt").read_bytes()
    expected = (
        b"academic research purpose only"
        if config["source"] == "DIV2K"
        else b"creativecommons.org/licenses/by-nc/3.0/"
    )
    if expected not in evidence:
        raise ValueError("upstream license evidence changed")
    if archive_path is None:
        archive_path = out / "archive.zip"
        binding = fetch(config["url"], archive_path, config["archive_limit"], deadline)
    else:
        archive_digest = hashlib.sha256()
        with archive_path.open("rb") as archive_stream:
            while chunk := archive_stream.read(1024**2):
                deadline()
                archive_digest.update(chunk)
        binding = {"sha256": archive_digest.hexdigest(), "bytes": archive_path.stat().st_size}
    zip_preflight(archive_path, config["archive_limit"])
    rows = []
    encoded: set[str] = set()
    decoded: set[str] = set()
    with zipfile.ZipFile(archive_path) as archive:
        selected = members(archive, config)
        metadata = {}
        if key == "esc50":
            info = archive.getinfo(ESC_PREFIX + "meta/esc50.csv")
            if not 0 < info.file_size <= MAX_DOCUMENT:
                raise ValueError("ESC metadata size limit")
            csv_raw = archive.read(info)
            metadata = esc_metadata(csv_raw)
            if set(metadata) != set(selected):
                raise ValueError("ESC metadata/media membership mismatch")
            with (out / "upstream-metadata.csv").open("xb") as metadata_stream:
                metadata_stream.write(csv_raw)
        for name, member in sorted(selected.items()):
            deadline()
            with archive.open(member) as member_stream:
                data = member_stream.read(MAX_FILE + 1)
            if len(data) != member.file_size:
                raise ValueError("media member size mismatch")
            digest = hashlib.sha256(data).hexdigest()
            details = media_details(data, config["source"])
            if digest in reserved or digest in encoded or details["decoded_sha256"] in decoded:
                raise ValueError("duplicate original bytes or decoded samples")
            encoded.add(digest)
            decoded.add(details["decoded_sha256"])
            meta = metadata.get(name, {})
            split = "train" if key == "div2k-train" else "validation"
            if key == "esc50":
                split = {"1": "train", "2": "train", "3": "train", "4": "validation", "5": "test"}[
                    meta["fold"]
                ]
            row = {
                "path": name,
                "upstream_member": member.filename,
                "sha256": digest,
                "lineage": digest,
                "size": len(data),
                "source_group": config["source"],
                "split": split,
                "label": "cover",
                "method": None,
                "camera": None,
                "device": None,
                **details,
            }
            if meta:
                row.update(
                    upstream_recording_id=meta["src_file"],
                    fold=int(meta["fold"]),
                    category=meta["category"],
                    target=int(meta["target"]),
                    group_key="ESC-50:" + meta["src_file"],
                    esc10=meta["esc10"] == "True",
                )
            with (out / name).open("xb") as stream:
                stream.write(data)
            rows.append(row)
    deadline()
    manifest = {
        "schema_version": "media-diversity-acquisition-v1",
        "status": "completed",
        "dataset": key,
        "source_url": config["url"],
        "source_commit": ESC_COMMIT if metadata else None,
        "license": config["license"],
        "license_evidence_url": config["evidence"],
        "license_evidence_sha256": proof["sha256"],
        "archive_sha256": binding["sha256"],
        "archive_bytes": binding["bytes"],
        "prior_manifest_sha256": sorted(prior_bindings),
        "upstream_sha256_verified": False,
        "originals": len(rows),
        "splits": dict(Counter(r["split"] for r in rows)),
        "samples": rows,
        "total_media_bytes": sum(r["size"] for r in rows),
        "cover_cleanliness_verified": False,
        "camera_scene_independence_verified": False,
        "model_trained": False,
        "accuracy_qualification": "unavailable",
        "publication": "private research only; no originals/derivatives/weights published",
        "seconds": time.monotonic() - started,
    }
    with (out / "source.json").open("x") as manifest_stream:
        json.dump(manifest, manifest_stream, indent=2)
        manifest_stream.write("\n")
    return manifest
