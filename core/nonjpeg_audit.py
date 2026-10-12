"""Independent scalar corpus audit. Does not import the preparation generator."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import resource
import stat
import sys
import time
import wave
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


def read(path, limit=32 * 1024**2):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("audit symlink")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("audit regular file")
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read(limit + 1)
    if not raw or len(raw) > limit:
        raise ValueError("audit byte bound")
    return raw


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def member(name):
    if (
        not isinstance(name, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]{1,180}", name)
        or name in (".", "..")
    ):
        raise ValueError("audit unsafe member")
    return name


def decode(raw, origin, fmt=None):
    if origin in ("DIV2K", "BOWS2"):
        with Image.open(io.BytesIO(raw)) as image:
            mode = "RGB" if origin == "DIV2K" else "L"
            if (
                image.mode != mode
                or image.width * image.height > 8_000_000
                or min(image.size) < 256
                or getattr(image, "n_frames", 1) != 1
                or image.format != (fmt or ("PNG" if mode == "RGB" else "PPM"))
            ):
                raise ValueError("audit image representation")
            if fmt is None:
                x, y = (image.width - 256) // 2, (image.height - 256) // 2
                return image.crop((x, y, x + 256, y + 256)).tobytes(), (mode, x, y)
            if image.size != (256, 256):
                raise ValueError("audit saved image size")
            return image.tobytes(), (mode,)
    with wave.open(io.BytesIO(raw)) as audio:
        p = audio.getparams()
        if (
            p.nchannels != 1
            or p.sampwidth != 2
            or p.comptype != "NONE"
            or p.framerate != (8000 if origin == "FSDD-v1.0.10" else 44100)
            or not 160 <= p.nframes <= 220500
        ):
            raise ValueError("audit PCM contract")
        pcm = audio.readframes(p.nframes)
        if len(pcm) != p.nframes * 2:
            raise ValueError("audit PCM truncation")
    return pcm, (p.framerate, p.nframes)


def audit(config_path: Path, root: Path, expected_sources: dict[str, str]):
    started = time.monotonic()

    def deadline():
        if time.monotonic() - started > 1800:
            raise TimeoutError("audit deadline")

    config_raw = read(config_path, 8 * 1024**2)
    cfg = json.loads(config_raw)
    if (
        not isinstance(cfg, dict)
        or set(cfg) != {"schema_version", "sources", "counts"}
        or not isinstance(cfg["sources"], list)
        or not 1 <= len(cfg["sources"]) <= 5
        or not isinstance(cfg["counts"], dict)
        or set(cfg["counts"]) != {"train", "validation"}
        or any(
            type(cfg["counts"][role]) is not int or not 1 <= cfg["counts"][role] <= maximum
            for role, maximum in (("train", 32), ("validation", 8))
        )
    ):
        raise ValueError("audit config bounds")
    manifest_raw = read(root / "manifest.json", 8 * 1024**2)
    corpus = json.loads(manifest_raw)
    if (
        cfg.get("schema_version") != "nonjpeg-preparation-config-v1"
        or corpus.get("schema_version") != "nonjpeg-controlled-data-v1"
        or corpus.get("status") != "prepared_not_audited"
        or corpus.get("config_sha256") != digest(config_raw)
        or corpus.get("source_sha256") != expected_sources
        or corpus.get("seed") != 20261012
        or corpus.get("model_trained") is not False
        or corpus.get("optimizer_updates") != 0
        or corpus.get("detector_scored") is not False
        or corpus.get("execution", {}).get("python") != sys.version.split()[0]
    ):
        raise ValueError("audit contract binding")
    speakers = {
        "lucas": "train",
        "nicolas": "train",
        "theo": "validation",
        "yweweler": "validation",
        "george": "test",
        "jackson": "test",
    }
    candidates: dict[tuple[str, str], list[Any]] = {}
    source_bindings: list[dict[str, Any]] = []
    groups: dict[tuple[str, ...], str] = {}
    identities: set[str] = set()
    source_rows = 0
    for binding in cfg["sources"]:
        path = Path(binding["manifest"])
        raw = read(path, 8 * 1024**2)
        source = json.loads(raw)
        if digest(raw) != binding["sha256"]:
            raise ValueError("audit source checksum")
        if not isinstance(source.get("samples"), list) or not source["samples"]:
            raise ValueError("audit original metadata")
        source_rows += len(source["samples"])
        if source_rows > 15000:
            raise ValueError("audit original count bound")
        source_bindings.append({"sha256": digest(raw), "license": source["license"]})
        for row in source["samples"]:
            role = (
                speakers[row["speaker"]]
                if source["schema_version"] == "fsdd-acquisition-v1"
                else row["split"]
            )
            origin, identity = row["source_group"], row["sha256"]
            expected_schema = {
                "DIV2K": "media-diversity-acquisition-v1",
                "BOWS2": "bows2-original-acquisition-v1",
                "ESC-50": "media-diversity-acquisition-v1",
                "FSDD-v1.0.10": "fsdd-acquisition-v1",
            }.get(origin)
            if (
                expected_schema is None
                or source["schema_version"] != expected_schema
                or role not in ("train", "validation", "test")
                or not re.fullmatch(r"[a-f0-9]{64}", identity)
                or row.get("label", "cover") != "cover"
                or row.get("method") is not None
                or (origin == "ESC-50" and not row.get("group_key"))
            ):
                raise ValueError("audit source role contract")
            if identity in identities or row["lineage"] != identity:
                raise ValueError("audit repeated original")
            identities.add(identity)
            keys: list[tuple[str, ...]] = [("sha", identity)]
            for field in ("decoded_sha256", "pixel_sha256"):
                if row.get(field):
                    keys.append(("decoded", row[field]))
            for field in ("speaker", "group_key", "camera", "device"):
                if row.get(field):
                    keys.append((field, origin, row[field]))
            for key in keys:
                if groups.setdefault(key, role) != role:
                    raise ValueError("audit grouping leakage")
            if role != "test":
                candidates.setdefault((origin, role), []).append(
                    (row, path.parent, source_bindings[-1])
                )
    if corpus["source_bindings"] != source_bindings:
        raise ValueError("audit license binding")
    selected: list[Any] = []
    for origin in sorted({key[0] for key in candidates}):
        for role in ("train", "validation"):
            rows = sorted(
                candidates[(origin, role)],
                key=lambda item: (
                    digest(f"20261012:{origin}:{item[0]['sha256']}".encode()),
                    item[0]["sha256"],
                ),
            )
            count = cfg["counts"][role]
            if len(rows) < count:
                raise ValueError("audit fixed selection count")
            selected.extend((row, parent, binding, role) for row, parent, binding in rows[:count])
    if len(corpus["originals"]) != len(selected) or len(corpus["samples"]) > 12000:
        raise ValueError("audit corpus counts")
    samples = iter(corpus["samples"])
    seen_carriers: set[str] = set()
    cells: Counter[str] = Counter()
    total, sample_index, positives = 0, 0, 0
    for original, (row, parent, binding, role) in zip(corpus["originals"], selected, strict=True):
        deadline()
        raw = read(parent / member(row["path"]))
        if digest(raw) != row["sha256"] or len(raw) != row.get("size", row.get("bytes")):
            raise ValueError("audit original integrity")
        base, geometry = decode(raw, row["source_group"])
        if digest(base) in seen_carriers:
            raise ValueError("audit duplicate carrier")
        seen_carriers.add(digest(base))
        metadata = {
            key: row.get(key)
            for key in (
                "sha256",
                "lineage",
                "source_group",
                "speaker",
                "group_key",
                "camera",
                "device",
            )
        }
        metadata.update(
            split=role, source_manifest_sha256=binding["sha256"], license=binding["license"]
        )
        if any(original.get(key) != value for key, value in metadata.items()) or original[
            "carrier_sha256"
        ] != digest(base):
            raise ValueError("audit ancestry metadata")
        image = row["source_group"] in ("DIV2K", "BOWS2")
        formats = ("PNG", "BMP") if image else ("WAV",)
        channels = (
            ("r", "g", "b", "all") if geometry[0] == "RGB" else ("gray" if image else "mono",)
        )
        if image:
            if (
                original.get("crop") != [geometry[1], geometry[2], 256, 256]
                or original.get("mode") != geometry[0]
            ):
                raise ValueError("audit crop metadata")
        elif original.get("sample_rate") != geometry[0] or original.get("frames") != geometry[1]:
            raise ValueError("audit audio metadata")
        recipes = [(fmt, None, 0, None, None) for fmt in formats] + [
            (fmt, method, rate, channel, plane)
            for fmt in formats
            for method in ("sequential", "scattered")
            for rate in (5, 40)
            for channel in channels
            for plane in (0, 1)
        ]
        for fmt, method, rate, channel, plane in recipes:
            deadline()
            sample = next(samples, None)
            if not isinstance(sample, dict):
                raise ValueError("audit missing cell")
            expected = {
                **metadata,
                "format": fmt,
                "method": method,
                "rate_percent": rate,
                "channel": channel,
                "plane": plane,
                "label": "cover" if method is None else "stego",
                "path": f"{sample_index:05d}.{fmt.lower()}",
            }
            expected.pop("sha256")
            if any(sample.get(key) != value for key, value in expected.items()):
                raise ValueError("audit cell metadata/order")
            encoded = read(root / member(sample["path"]))
            if digest(encoded) != sample["sha256"] or len(encoded) != sample["size"]:
                raise ValueError("audit artifact integrity")
            values, _ = decode(encoded, row["source_group"], fmt)
            if len(values) != len(base):
                raise ValueError("audit carrier extent")
            total += len(encoded)
            sample_index += 1
            if total > 2 * 1024**3:
                raise ValueError("audit output budget")
            if not image:
                # Whole original RIFF extent/metadata must remain intact; identify
                # PCM independently, not from the preparation's offset field.
                offset, regions = 12, []
                if raw[:4] != b"RIFF" or int.from_bytes(raw[4:8], "little") + 8 != len(raw):
                    raise ValueError("audit RIFF extent")
                while offset + 8 <= len(raw):
                    size = int.from_bytes(raw[offset + 4 : offset + 8], "little")
                    if raw[offset : offset + 4] == b"data":
                        regions.append((offset + 8, size))
                    offset += 8 + size + (size % 2)
                if offset != len(raw) or len(regions) != 1 or regions[0][1] != len(base):
                    raise ValueError("audit PCM region")
                begin = regions[0][0]
                if (
                    encoded[:begin] != raw[:begin]
                    or encoded[begin + len(base) :] != raw[begin + len(base) :]
                ):
                    raise ValueError("audit RIFF metadata changed")
            if method is None:
                if values != base:
                    raise ValueError("audit cover representation")
                continue
            if channel is None or plane is None:
                raise ValueError("audit missing recipe fields")
            context = f"20261012:{row['lineage']}:{method}:{rate}:{channel}:{plane}"
            unit_width = 1 if image else 2
            units = len(base) // unit_width
            eligible = (
                range(units)
                if channel in ("all", "gray", "mono")
                else range("rgb".index(channel), units, 3)
            )
            bits = len(eligible) * rate // 100 // 8 * 8
            indices = (
                eligible[:bits]
                if method == "sequential"
                else [
                    eligible[i]
                    for i in random.Random(context).sample(range(len(eligible)), bits)  # noqa: S311 - public research order
                ]
            )
            allowed = set(indices)
            recovered = bytearray(bits // 8)
            for i, position in enumerate(indices):
                value = int.from_bytes(
                    values[position * unit_width : (position + 1) * unit_width], "little"
                )
                recovered[i // 8] |= ((value >> plane) & 1) << (7 - i % 8)
            payload = hashlib.shake_256(context.encode()).digest(bits // 8)
            changed = 0
            for position in range(units):
                old = int.from_bytes(
                    base[position * unit_width : (position + 1) * unit_width], "little"
                )
                new = int.from_bytes(
                    values[position * unit_width : (position + 1) * unit_width], "little"
                )
                delta = old ^ new
                if delta and (position not in allowed or delta != (1 << plane)):
                    raise ValueError("audit nonselected bit changed")
                changed += bool(delta)
            if (
                bytes(recovered) != payload
                or sample.get("payload_sha256") != digest(payload)
                or sample.get("payload_bytes") != len(payload)
                or sample.get("embedded_bits") != bits
                or sample.get("eligible_units") != len(eligible)
                or not changed
                or sample.get("changed_units") != changed
            ):
                raise ValueError("audit scalar payload oracle")
            positives += 1
            cells[f"{row['source_group']}:{role}:{fmt}:{method}:{rate}:{channel}:{plane}"] += 1
    if (
        next(samples, None) is not None
        or corpus["artifact_count"] != sample_index
        or corpus["artifact_bytes"] != total
    ):
        raise ValueError("audit extra artifacts/counts")
    original_counts = dict(Counter(f"{row['source_group']}:{role}" for row, _, _, role in selected))
    if corpus["original_counts"] != original_counts:
        raise ValueError("audit original denominator")
    return {
        "schema_version": "nonjpeg-independent-audit-v1",
        "status": "completed",
        "manifest_sha256": digest(manifest_raw),
        "config_sha256": digest(config_raw),
        "source_sha256": expected_sources,
        "execution": corpus["execution"],
        "original_counts": original_counts,
        "artifact_count": sample_index,
        "artifact_bytes": total,
        "independent_scalar_recoveries": positives,
        "cells": dict(cells),
        "test_media_opened": False,
        "model_trained": False,
        "detector_scored": False,
        "accuracy_qualification": "unavailable",
        "blind": False,
        "deployed": False,
        "seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1801))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    result = audit(args.config, args.root, json.loads(read(args.bindings, 1024**2)))
    path = args.root / "independent-audit.json"
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("audit symlink output")
    with path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
