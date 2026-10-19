"""Source-bound, streaming controlled WAV rows for the AudioMNIST pilot."""

from __future__ import annotations

import hashlib
import io
import json
import wave
from pathlib import Path
from typing import Any

import numpy as np

from core.nonjpeg_data import regular_read, replace
from core.nonjpeg_features import pcm16_samples

ROLES = {"train", "development"}
RECIPES = tuple(
    (method, rate, plane)
    for method in ("sequential", "scattered")
    for rate in (5, 40)
    for plane in (0, 1)
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _wav_data_offset(raw: bytes) -> int:
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("invalid RIFF/WAVE")
    offset, data_offset, chunks = 12, None, 0
    while offset + 8 <= len(raw):
        name = raw[offset : offset + 4]
        size = int.from_bytes(raw[offset + 4 : offset + 8], "little")
        end = offset + 8 + size + (size & 1)
        chunks += 1
        if end > len(raw) or chunks > 64:
            raise ValueError("invalid WAV chunk extent")
        if name == b"data":
            if data_offset is not None:
                raise ValueError("ambiguous WAV data chunks")
            data_offset = offset + 8
        offset = end
    if offset != len(raw) or data_offset is None:
        raise ValueError("invalid WAV chunk layout")
    return data_offset


def load_manifest(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    raw = regular_read(manifest_path, 16 * 1024**2)
    manifest = json.loads(raw)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "wav-audiomnist-acquisition-v1"
        or manifest.get("source_commit") != "630d7dab4c040882834de6fa21baf9a60372accd"
        or manifest.get("status") != "acquired_native_headers_verified_not_training_ready"
        or manifest.get("reserved_speakers_accessed") is not False
        or manifest.get("optimizer_updates") != 0
        or not isinstance(manifest.get("files"), list)
        or len(manifest["files"]) != 600
    ):
        raise ValueError("AudioMNIST acquisition manifest binding")
    rows = manifest["files"]
    groups: dict[str, str] = {}
    paths, hashes = set(), set()
    for row in rows:
        rel, role, group, digest = (
            row.get("relative_source_path"),
            row.get("role"),
            row.get("speaker_group"),
            row.get("sha256"),
        )
        if (
            role not in ROLES
            or not isinstance(group, str)
            or not isinstance(rel, str)
            or not rel.startswith(f"data/{group}/")
            or rel in paths
            or not isinstance(digest, str)
            or len(digest) != 64
            or digest in hashes
            or row.get("parent_recording_id") is not None
        ):
            raise ValueError("unsafe acquisition row")
        if groups.setdefault(group, role) != role:
            raise ValueError("speaker group crosses roles")
        paths.add(rel)
        hashes.add(digest)
    if (
        sum(row["role"] == "train" for row in rows) != 400
        or sum(row["role"] == "development" for row in rows) != 200
        or sum(role == "train" for role in groups.values()) != 20
        or sum(role == "development" for role in groups.values()) != 10
    ):
        raise ValueError("acquisition role/group count")
    return manifest, rows


def materialize(root: Path, row: dict[str, Any], recipe: tuple[str, int, int] | None) -> bytes:
    """Read one bound original and yield a cover or one deterministic WAV stego."""
    rel = row["relative_source_path"]
    if Path(rel).is_absolute() or any(p in ("", ".", "..") for p in rel.split("/")):
        raise ValueError("unsafe source path")
    raw = regular_read(root / rel, 512 * 1024)
    if sha(raw) != row["sha256"]:
        raise ValueError("AudioMNIST source checksum mismatch")
    samples = pcm16_samples(raw)
    data_offset = _wav_data_offset(raw)
    if data_offset + samples.size * 2 > len(raw):
        raise ValueError("PCM data extent")
    if recipe is None:
        return raw
    method, rate, plane = recipe
    if recipe not in RECIPES:
        raise ValueError("unknown controlled WAV recipe")
    values = np.frombuffer(raw[data_offset : data_offset + samples.size * 2], dtype="<u2").copy()
    lineage = row["sha256"]
    modified, proof = replace(values, lineage, method, rate, "mono", plane)
    context = f"20261012:{lineage}:{method}:{rate}:mono:{plane}"
    payload = hashlib.shake_256(context.encode()).digest(proof["payload_bytes"])
    eligible = np.arange(values.size)
    if method == "scattered":
        import random

        selected = np.asarray(
            random.Random(context).sample(range(values.size), proof["embedded_bits"])  # noqa: S311 - public deterministic order
        )
    else:
        selected = eligible[: proof["embedded_bits"]]
    recovered_bits = ((modified[selected] >> plane) & 1).astype(np.uint8)
    recovered = np.packbits(recovered_bits).tobytes()
    if recovered != payload or sha(recovered) != proof["payload_sha256"]:
        raise ValueError("independent scalar opaque payload oracle failed")
    output = bytearray(raw)
    output[data_offset : data_offset + values.nbytes] = modified.astype("<u2").tobytes()
    result = bytes(output)
    if len(result) != len(raw) or pcm16_samples(result).shape != samples.shape:
        raise ValueError("controlled WAV structural integrity")
    return result


def rows_for_role(rows: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    if role not in ROLES:
        raise ValueError("closed WAV role")
    selected = [row for row in rows if row["role"] == role]
    expected = 400 if role == "train" else 200
    groups = {row["speaker_group"] for row in selected}
    if len(selected) != expected or len(groups) != (20 if role == "train" else 10):
        raise ValueError("WAV role/group count")
    result = []
    for original in selected:
        lineage = original["sha256"]
        result.append(
            {
                "source": original,
                "lineage": lineage,
                "group": original["speaker_group"],
                "label": "cover",
                "recipe": None,
            }
        )
        for recipe in RECIPES:
            result.append(
                {
                    "source": original,
                    "lineage": lineage,
                    "group": original["speaker_group"],
                    "label": "stego",
                    "recipe": recipe,
                }
            )
    return result


def generated_row_smoke() -> dict[str, Any]:
    """Exercise all eight recipes and independent recovery on synthetic PCM only."""
    samples = np.resize(np.array([-32768, 32767, -1, 0, 1], dtype="<i2"), 160)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(samples.tobytes())
    raw = buffer.getvalue()
    digest = sha(raw)
    # The fixture is generated and accepted through the same validator contract.
    count = 0
    for method, rate, plane in RECIPES:
        values = np.frombuffer(raw[_wav_data_offset(raw) :], dtype="<u2").copy()
        changed, proof = replace(values, digest, method, rate, "mono", plane)
        context = f"20261012:{digest}:{method}:{rate}:mono:{plane}"
        expected = hashlib.shake_256(context.encode()).digest(proof["payload_bytes"])
        selected = (
            np.asarray(
                __import__("random")
                .Random(context)
                .sample(  # noqa: S311 - public deterministic order
                    range(values.size), proof["embedded_bits"]
                )
            )
            if method == "scattered"
            else np.arange(proof["embedded_bits"])
        )
        if np.packbits(((changed[selected] >> plane) & 1).astype(np.uint8)).tobytes() != expected:
            raise ValueError("generated WAV scalar oracle")
        count += 1
    return {
        "status": "ready",
        "recipes_checked": count,
        "optimizer_updates": 0,
        "source_media_opened": False,
    }
