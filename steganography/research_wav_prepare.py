"""Acquire only the frozen train/development AudioMNIST WAV pilot files."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
import urllib.request
import wave
from pathlib import Path
from urllib.parse import urlparse

from core.nonjpeg_data import regular_read

COMMIT = "630d7dab4c040882834de6fa21baf9a60372accd"
DRAFT_SHA256 = "06b0df7a3c8a21b62a9134d5600c12d00808e11b633e70857f0f8bf0fa687244"
BASE = f"https://raw.githubusercontent.com/soerenab/AudioMNIST/{COMMIT}/"
MAX_FILE = 512 * 1024
MAX_TOTAL = 384 * 1024**2
MAX_SECONDS = 1800


def _read_manifest(path: Path) -> dict:
    raw = regular_read(path, 4 * 1024**2)
    if hashlib.sha256(raw).hexdigest() != DRAFT_SHA256:
        raise ValueError("frozen AudioMNIST draft hash mismatch")
    doc = json.loads(raw)
    if (
        doc.get("schema_version") != "wav-next-experiment-draft-v1"
        or doc.get("status") != "draft_not_executable"
        or doc.get("source_commit") != COMMIT
    ):
        raise ValueError("frozen AudioMNIST draft binding mismatch")
    rows = doc.get("proposed_files")
    if not isinstance(rows, list) or len(rows) != 600:
        raise ValueError("expected frozen 600-file train/development selection")
    speaker_roles = doc.get("speaker_roles")
    if (
        not isinstance(speaker_roles, dict)
        or set(speaker_roles) != {"train", "development", "reserved"}
        or len(speaker_roles["train"]) != 20
        or len(speaker_roles["development"]) != 10
        or len(speaker_roles["reserved"]) != 30
        or set(speaker_roles["train"]) & set(speaker_roles["development"])
        or (set(speaker_roles["train"]) | set(speaker_roles["development"]))
        & set(speaker_roles["reserved"])
    ):
        raise ValueError("frozen speaker assignment contract")
    roles: set[str] = set()
    paths: set[str] = set()
    groups: dict[str, str] = {}
    for row in rows:
        rel = row.get("relative_source_path")
        role, group = row.get("role"), row.get("speaker_group")
        if (
            role not in ("train", "development")
            or not isinstance(group, str)
            or row.get("parent_recording_id") is not None
            or not isinstance(rel, str)
            or rel in paths
            or not rel.startswith(f"data/{group}/")
            or group not in speaker_roles[role]
            or not rel.endswith(".wav")
            or any(part in ("", ".", "..") for part in rel.split("/"))
        ):
            raise ValueError("unsafe or inconsistent frozen sample row")
        paths.add(rel)
        roles.add(role)
        if groups.setdefault(group, role) != role:
            raise ValueError("speaker group crosses frozen roles")
    if (
        len([r for r in rows if r["role"] == "train"]) != 400
        or len([r for r in rows if r["role"] == "development"]) != 200
    ):
        raise ValueError("frozen role counts changed")
    if (
        sum(role == "train" for role in groups.values()) != 20
        or sum(role == "development" for role in groups.values()) != 10
    ):
        raise ValueError("frozen speaker role count changed")
    return doc


def _fetch(url: str, deadline: float) -> bytes:
    if urlparse(url).hostname != "raw.githubusercontent.com" or not url.startswith(BASE):
        raise ValueError("source URL is outside pinned official commit")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("acquisition deadline")
    request = urllib.request.Request(url, headers={"User-Agent": "steganography-research/0.6"})  # noqa: S310 - URL is fixed to pinned HTTPS raw GitHub
    with urllib.request.urlopen(request, timeout=min(30, remaining)) as response:  # noqa: S310 - fixed HTTPS GitHub host and commit
        if response.geturl() != url:
            raise ValueError("redirect escaped pinned source host")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > MAX_FILE:
            raise ValueError("per-file download bound")
        raw = response.read(MAX_FILE + 1)
        if len(raw) > MAX_FILE or (length is not None and len(raw) != int(length)):
            raise ValueError("download length bound/mismatch")
        return raw


def acquire(draft: Path, out: Path) -> dict:
    plan = _read_manifest(draft)
    if out.exists() or any(parent.is_symlink() for parent in (out, *out.parents)):
        raise ValueError("output must be a new non-symlink directory")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + MAX_SECONDS
    files: list[dict] = []
    hashes: dict[str, str] = {}
    total = 0
    for row in plan["proposed_files"]:
        if time.monotonic() >= deadline:
            raise TimeoutError("acquisition deadline")
        rel = row["relative_source_path"]
        raw = _fetch(BASE + rel, deadline)
        total += len(raw)
        if total > MAX_TOTAL:
            raise ValueError("separate WAV raw-transfer ceiling exceeded")
        digest = hashlib.sha256(raw).hexdigest()
        if digest in hashes:
            raise ValueError("duplicate source file across selected rows")
        hashes[digest] = rel
        try:
            with wave.open(io.BytesIO(raw), "rb") as wav:
                params = wav.getparams()
                frames = wav.getnframes()
                pcm = wav.readframes(frames)
        except (wave.Error, EOFError) as exc:
            raise ValueError("invalid WAV source") from exc
        if (
            params.nchannels != 1
            or params.sampwidth != 2
            or params.framerate != 48000
            or params.comptype != "NONE"
            or not 160 <= frames <= 220500
            or len(pcm) != frames * 2
        ):
            raise ValueError("native mono PCM16 source contract")
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if any(p.is_symlink() for p in (target, *target.parents)):
            raise ValueError("symlink output path")
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        files.append(
            {
                "relative_source_path": rel,
                "role": row["role"],
                "speaker_group": row["speaker_group"],
                "parent_recording_id": None,
                "sha256": digest,
                "bytes": len(raw),
                "frames": frames,
                "sample_rate_hz": params.framerate,
                "channels": params.nchannels,
                "sample_width_bytes": params.sampwidth,
            }
        )
    result = {
        "schema_version": "wav-audiomnist-acquisition-v1",
        "status": "acquired_native_headers_verified_not_training_ready",
        "source": "AudioMNIST",
        "source_commit": COMMIT,
        "draft_sha256": hashlib.sha256(draft.read_bytes()).hexdigest(),
        "repository_license": "MIT",
        "consent_basis": "paper reports written participant consent before recording",
        "speaker_role_counts": {
            role: sum(f["role"] == role for f in files) for role in ("train", "development")
        },
        "speaker_groups": sorted({f["speaker_group"] for f in files}),
        "raw_bytes": total,
        "raw_transfer_ceiling_bytes": MAX_TOTAL,
        "elapsed_seconds": time.monotonic() - started,
        "reserved_speakers_accessed": False,
        "optimizer_updates": 0,
        "files": files,
    }
    manifest = out / "manifest.json"
    fd = os.open(manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--draft",
        type=Path,
        default=Path("benchmarks/wav-next-experiment-draft-20261016.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = acquire(args.draft, args.out)
    summary = {
        key: result[key]
        for key in ("status", "speaker_role_counts", "raw_bytes", "elapsed_seconds")
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
