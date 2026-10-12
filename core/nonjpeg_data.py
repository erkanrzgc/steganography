"""Bounded, explicit non-JPEG controlled carriers; no inference or training."""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import re
import stat
import sys
import time
import wave
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from PIL import __version__ as pillow_version

SCHEMA = "nonjpeg-controlled-data-v1"
SEED = 20261012
MAX_FILE = 32 * 1024**2
MAX_DOCUMENT = 8 * 1024**2
MAX_OUTPUT = 2 * 1024**3
SPEAKERS = {
    "lucas": "train",
    "nicolas": "train",
    "theo": "validation",
    "yweweler": "validation",
    "george": "test",
    "jackson": "test",
}
SOURCE_SCHEMAS = {
    "media-diversity-acquisition-v1",
    "bows2-original-acquisition-v1",
    "fsdd-acquisition-v1",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def regular_read(path: Path, limit: int) -> bytes:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink input")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("nonregular input")
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read(limit + 1)
    if not raw or len(raw) > limit:
        raise ValueError("input byte bound")
    return raw


def document(path: Path) -> tuple[dict[str, Any], str]:
    raw = regular_read(path, MAX_DOCUMENT)
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("document schema")
    return result, sha(raw)


def basename(name: Any) -> str:
    if (
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]{1,180}", name)
        or name in (".", "..")
    ):
        raise ValueError("unsafe member name")
    return name


def write(path: Path, raw: bytes) -> None:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlink output")
    with path.open("xb") as stream:
        stream.write(raw)


def write_json(path: Path, data: dict[str, Any]) -> None:
    write(path, (json.dumps(data, indent=2, allow_nan=False) + "\n").encode())


def config(path: Path) -> tuple[dict[str, Any], str]:
    cfg, digest = document(path)
    if (
        set(cfg) != {"schema_version", "sources", "counts"}
        or cfg["schema_version"] != "nonjpeg-preparation-config-v1"
        or not isinstance(cfg["sources"], list)
        or not 1 <= len(cfg["sources"]) <= 5
        or not isinstance(cfg["counts"], dict)
        or set(cfg["counts"]) != {"train", "validation"}
    ):
        raise ValueError("config schema")
    for role, maximum in (("train", 32), ("validation", 8)):
        if type(cfg["counts"][role]) is not int or not 1 <= cfg["counts"][role] <= maximum:
            raise ValueError("selection count limit")
    for source in cfg["sources"]:
        if (
            not isinstance(source, dict)
            or set(source) != {"manifest", "sha256"}
            or not isinstance(source["manifest"], str)
            or not isinstance(source["sha256"], str)
            or not re.fullmatch(r"[a-f0-9]{64}", source["sha256"])
        ):
            raise ValueError("source binding schema")
    return cfg, digest


def selection(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    roles: dict[tuple[str, ...], str] = {}
    seen: set[str] = set()
    bindings = []
    total = 0
    for binding in cfg["sources"]:
        path = Path(binding["manifest"])
        source, digest = document(path)
        rows = source.get("samples")
        if (
            digest != binding["sha256"]
            or source.get("schema_version") not in SOURCE_SCHEMAS
            or not isinstance(source.get("license"), str)
            or not 1 <= len(source.get("license", "")) <= 256
            or (
                source.get("schema_version") != "fsdd-acquisition-v1"
                and source.get("status") != "completed"
            )
            or not isinstance(rows, list)
            or not rows
        ):
            raise ValueError("source schema or checksum")
        total += len(rows)
        if total > 15000:
            raise ValueError("original count bound")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("source row schema")
            original = row.get("sha256")
            origin = row.get("source_group")
            role = row.get("split")
            if source["schema_version"] == "fsdd-acquisition-v1":
                role = SPEAKERS.get(str(row.get("speaker", "")))
                if row.get("split") not in (None, role):
                    raise ValueError("FSDD speaker role conflict")
            if (
                not isinstance(original, str)
                or not re.fullmatch(r"[a-f0-9]{64}", original)
                or row.get("lineage") != original
                or original in seen
                or origin not in ("DIV2K", "BOWS2", "FSDD-v1.0.10", "ESC-50")
                or role not in ("train", "validation", "test")
                or row.get("label", "cover") != "cover"
                or row.get("method") is not None
            ):
                raise ValueError("original identity or role")
            expected_schema = {
                "DIV2K": "media-diversity-acquisition-v1",
                "BOWS2": "bows2-original-acquisition-v1",
                "ESC-50": "media-diversity-acquisition-v1",
                "FSDD-v1.0.10": "fsdd-acquisition-v1",
            }[origin]
            if source["schema_version"] != expected_schema:
                raise ValueError("source format contract")
            for field in ("camera", "device", "speaker", "group_key"):
                value = row.get(field)
                if value is not None and (
                    not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value)
                ):
                    raise ValueError("bounded grouping metadata")
            seen.add(original)
            keys: list[tuple[str, ...]] = [("sha", original)]
            for field in ("decoded_sha256", "pixel_sha256"):
                if row.get(field):
                    keys.append(("decoded", str(row[field])))
            for field in ("speaker", "group_key", "camera", "device"):
                if row.get(field):
                    keys.append((field, origin, str(row[field])))
            if origin == "ESC-50" and not row.get("group_key"):
                raise ValueError("missing audio recording group")
            if origin == "FSDD-v1.0.10" and row.get("speaker") not in SPEAKERS:
                raise ValueError("missing speaker group")
            for key in keys:
                if roles.setdefault(key, role) != role:
                    raise ValueError("original group crosses roles")
            basename(row.get("path"))
            size = row.get("size", row.get("bytes"))
            if type(size) is not int or not 0 < size <= MAX_FILE:
                raise ValueError("original size")
            if role != "test":
                candidates.setdefault((origin, role), []).append(
                    {
                        **row,
                        "split": role,
                        "size": size,
                        "root": path.parent,
                        "source_manifest_sha256": digest,
                        "license": source["license"],
                    }
                )
        bindings.append({"sha256": digest, "license": source["license"]})
    origins = {key[0] for key in candidates}
    if not 1 <= len(origins) <= 4:
        raise ValueError("origin count")
    chosen = []
    for origin in sorted(origins):
        for role in ("train", "validation"):
            rows = candidates.get((origin, role), [])
            count = cfg["counts"][role]
            if len(rows) < count:
                raise ValueError("insufficient fixed source role")
            chosen.extend(
                sorted(
                    rows,
                    key=lambda row: (
                        sha(f"{SEED}:{origin}:{row['sha256']}".encode()),
                        row["sha256"],
                    ),
                )[:count]
            )
    return chosen, bindings


def carrier(raw: bytes, origin: str) -> tuple[np.ndarray, dict[str, Any]]:
    if origin in ("DIV2K", "BOWS2"):
        with Image.open(io.BytesIO(raw)) as image:
            mode = "RGB" if origin == "DIV2K" else "L"
            if (
                image.mode != mode
                or image.format != ("PNG" if mode == "RGB" else "PPM")
                or getattr(image, "n_frames", 1) != 1
                or min(image.size) < 256
                or image.width * image.height > 8_000_000
            ):
                raise ValueError("native image contract")
            left, top = (image.width - 256) // 2, (image.height - 256) // 2
            pixels = np.array(image.crop((left, top, left + 256, top + 256)), dtype=np.uint8)
        return pixels, {"kind": "spatial", "mode": mode, "crop": [left, top, 256, 256]}
    if (
        len(raw) < 12
        or raw[:4] != b"RIFF"
        or raw[8:12] != b"WAVE"
        or int.from_bytes(raw[4:8], "little") + 8 != len(raw)
    ):
        raise ValueError("RIFF extent")
    with wave.open(io.BytesIO(raw)) as audio:
        frames = audio.getnframes()
        rate = 8000 if origin == "FSDD-v1.0.10" else 44100
        if (
            audio.getparams()[:3] != (1, 2, rate)
            or audio.getcomptype() != "NONE"
            or not 160 <= frames <= 220500
        ):
            raise ValueError("native PCM contract")
        pcm = audio.readframes(frames)
        if len(pcm) != frames * 2:
            raise ValueError("truncated PCM")
    offset, regions, chunks = 12, [], 0
    while offset + 8 <= len(raw):
        size = int.from_bytes(raw[offset + 4 : offset + 8], "little")
        end = offset + 8 + size + (size & 1)
        chunks += 1
        if end > len(raw) or chunks > 64:
            raise ValueError("RIFF chunk bound")
        if raw[offset : offset + 4] == b"data":
            regions.append((offset + 8, size))
        offset = end
    if (
        offset != len(raw)
        or len(regions) != 1
        or regions[0][1] != len(pcm)
        or raw[regions[0][0] : regions[0][0] + len(pcm)] != pcm
    ):
        raise ValueError("ambiguous PCM region")
    return np.frombuffer(pcm, dtype="<u2").copy(), {
        "kind": "audio",
        "mode": "PCM16",
        "sample_rate": rate,
        "frames": frames,
        "offset": regions[0][0],
    }


def variants(meta: dict[str, Any]) -> list[tuple[str, str | None, int, str | None, int | None]]:
    formats = ("PNG", "BMP") if meta["kind"] == "spatial" else ("WAV",)
    channels = (
        ("r", "g", "b", "all")
        if meta["mode"] == "RGB"
        else ("gray" if meta["mode"] == "L" else "mono",)
    )
    return [(fmt, None, 0, None, None) for fmt in formats] + [
        (fmt, method, rate, channel, plane)
        for fmt in formats
        for method in ("sequential", "scattered")
        for rate in (5, 40)
        for channel in channels
        for plane in (0, 1)
    ]


def replace(
    values: np.ndarray, lineage: str, method: str, rate: int, channel: str, plane: int
) -> tuple[np.ndarray, dict[str, Any]]:
    if (
        method not in ("sequential", "scattered")
        or rate not in (5, 40)
        or plane not in (0, 1)
        or channel not in ("r", "g", "b", "all", "gray", "mono")
        or values.dtype not in (np.dtype("uint8"), np.dtype("<u2"))
        or values.size > 220500
        or values.size < 160
        or (channel in ("r", "g", "b", "all") and values.shape != (256, 256, 3))
        or (channel == "gray" and values.shape != (256, 256))
        or (channel == "mono" and (values.ndim != 1 or values.dtype != np.dtype("<u2")))
    ):
        raise ValueError("replacement recipe contract")
    context = f"{SEED}:{lineage}:{method}:{rate}:{channel}:{plane}"
    flat = values.reshape(-1).copy()
    eligible = (
        np.arange(flat.size)
        if channel in ("all", "gray", "mono")
        else np.arange("rgb".index(channel), flat.size, 3)
    )
    bits = len(eligible) * rate // 100 // 8 * 8
    if not bits:
        raise ValueError("empty payload")
    payload = hashlib.shake_256(context.encode()).digest(bits // 8)
    selected = (
        eligible[:bits]
        if method == "sequential"
        else eligible[np.asarray(random.Random(context).sample(range(len(eligible)), bits))]  # noqa: S311 - public research order
    )
    mask = 1 << plane
    flat[selected] = (flat[selected] & (np.iinfo(flat.dtype).max ^ mask)) | (
        np.unpackbits(np.frombuffer(payload, dtype=np.uint8)).astype(flat.dtype) << plane
    )
    differences = np.bitwise_xor(flat, values.reshape(-1))
    if not np.any(differences) or np.any(differences & (np.iinfo(flat.dtype).max ^ mask)):
        raise ValueError("replacement bit contract")
    return flat.reshape(values.shape), {
        "payload_sha256": sha(payload),
        "payload_bytes": len(payload),
        "eligible_units": len(eligible),
        "embedded_bits": bits,
        "changed_units": int(np.count_nonzero(differences)),
    }


def prepare(
    cfg: dict[str, Any], out: Path, *, config_sha256: str, source_sha256: dict[str, str]
) -> dict[str, Any]:
    started = time.monotonic()
    chosen, bindings = selection(cfg)
    decoded = set()
    originals: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    total = 0
    for original in chosen:
        if time.monotonic() - started > 1800:
            raise TimeoutError("preparation deadline")
        raw = regular_read(original["root"] / original["path"], MAX_FILE)
        if len(raw) != original["size"] or sha(raw) != original["sha256"]:
            raise ValueError("original integrity")
        pixels, meta = carrier(raw, original["source_group"])
        decoded_hash = sha(pixels.tobytes())
        if decoded_hash in decoded:
            raise ValueError("duplicate selected decoded carrier")
        decoded.add(decoded_hash)
        identity = {
            key: original.get(key)
            for key in (
                "sha256",
                "lineage",
                "source_group",
                "split",
                "source_manifest_sha256",
                "license",
                "speaker",
                "group_key",
                "camera",
                "device",
            )
        }
        originals.append({**identity, "carrier_sha256": decoded_hash, **meta})
        for fmt, method, rate, channel, plane in variants(meta):
            proof: dict[str, Any]
            if method is None:
                modified, proof = pixels, {}
            else:
                if channel is None or plane is None:
                    raise ValueError("missing recipe channel or plane")
                modified, proof = replace(pixels, original["lineage"], method, rate, channel, plane)
            if fmt == "WAV":
                offset = meta["offset"]
                data = (
                    raw[:offset] + modified.astype("<u2").tobytes() + raw[offset + pixels.nbytes :]
                )
            else:
                buffer = io.BytesIO()
                Image.fromarray(modified).save(buffer, format=fmt)
                data = buffer.getvalue()
            total += len(data)
            if total > MAX_OUTPUT - 32 * 1024**2 or len(samples) >= 12000:
                raise ValueError("artifact budget")
            name = f"{len(samples):05d}.{fmt.lower()}"
            write(out / name, data)
            samples.append(
                {
                    **identity,
                    "path": name,
                    "sha256": sha(data),
                    "size": len(data),
                    "format": fmt,
                    "method": method,
                    "rate_percent": rate,
                    "channel": channel,
                    "plane": plane,
                    "label": "cover" if method is None else "stego",
                    **proof,
                }
            )
    result = {
        "schema_version": SCHEMA,
        "status": "prepared_not_audited",
        "seed": SEED,
        "config_sha256": config_sha256,
        "source_sha256": source_sha256,
        "execution": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pillow": pillow_version,
            "sample_byte_order": "little",
            "sample_rate_conversion": False,
            "image_resize": False,
        },
        "source_bindings": bindings,
        "originals": originals,
        "samples": samples,
        "original_counts": dict(Counter(f"{r['source_group']}:{r['split']}" for r in originals)),
        "artifact_count": len(samples),
        "artifact_bytes": total,
        "model_trained": False,
        "optimizer_updates": 0,
        "detector_scored": False,
        "accuracy_qualification": "unavailable",
        "deployed": False,
        "camera_independence_verified": False,
        "blind": False,
        "publication": "private controlled derivatives; preserve source licenses",
        "seconds": time.monotonic() - started,
    }
    write_json(out / "manifest.json", result)
    return result
