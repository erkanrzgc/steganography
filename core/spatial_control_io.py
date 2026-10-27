"""Bounded generated PNG/BMP transport component, separate from native media."""

from __future__ import annotations

import io
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from core import colab_artifacts as a
from core import spatial_training_data as data

MAX_IMAGE_BYTES = a.MIB
MAX_SCRATCH_BYTES = 16 * a.MIB


def prepare(out: Path, corpus: dict) -> dict:
    a.directory(out)
    rows, total = [], 0
    for manifest in corpus["profile"]:
        spec = data.GeneratedSpec(**manifest["spec"])
        for index, expected in enumerate(manifest["sources"]):
            cover, control, source = data.pair(spec, index)
            if source != expected:
                raise ValueError("generated transport source mismatch")
            for kind, pixels in (("cover", cover), ("control", control)):
                image = Image.fromarray(pixels[:, :, 0] if spec.mode == "L" else pixels)
                for fmt in ("PNG", "BMP"):
                    buffer = io.BytesIO()
                    image.save(buffer, format=fmt)
                    raw = buffer.getvalue()
                    total += len(raw)
                    if total > MAX_SCRATCH_BYTES:
                        raise ValueError("generated transport scratch limit")
                    name = "g-" + source["id"][-64:] + "-" + kind + "." + fmt.lower()
                    checksum = a.put(out / name, raw, MAX_IMAGE_BYTES)
                    rows.append(
                        {
                            "name": name,
                            "sha256": checksum,
                            "kind": kind,
                            "format": fmt,
                            "spec": asdict(spec),
                            "index": index,
                            "pixel_sha256": source[kind + "_sha256"],
                        }
                    )
    result = {
        "schema": "spatial-generated-transport-v1",
        "images": rows,
        "bytes": total,
        "generated_only": True,
        "native_media_opened": False,
    }
    a.put(out / "manifest.json", a.encode(result))
    return result


def read_image(root: Path, row: dict) -> np.ndarray:
    if not isinstance(row["name"], str) or not re.fullmatch(
        r"g-[0-9a-f]{64}-(cover|control)\.(png|bmp)", row["name"]
    ):
        raise ValueError("unsafe generated transport name")
    spec = data.GeneratedSpec(**row["spec"])
    raw = a.read(root / row["name"], MAX_IMAGE_BYTES)
    if a.digest(raw) != row["sha256"]:
        raise ValueError("generated transport file checksum")
    with Image.open(io.BytesIO(raw)) as image:
        if (
            image.format != row["format"]
            or image.mode != spec.mode
            or image.size != (256, 256)
            or getattr(image, "n_frames", 1) != 1
        ):
            raise ValueError("generated transport format/channel/extent")
        image.verify()
    with Image.open(io.BytesIO(raw)) as image:
        if image.getexif().get(274, 1) != 1:
            raise ValueError("generated transport orientation")
        pixels = np.asarray(image).copy()
    if spec.mode == "L":
        pixels = pixels[:, :, None]
    data.pixels(pixels)
    if a.digest(pixels.tobytes()) != row["pixel_sha256"]:
        raise ValueError("generated transport pixel checksum")
    return pixels


def batch(
    root: Path, transport: dict, spec: data.GeneratedSpec, indices: tuple[int, ...], fmt: str
) -> np.ndarray:
    if fmt not in ("PNG", "BMP") or not 1 <= len(indices) <= 4 or len(set(indices)) != len(indices):
        raise ValueError("generated transport batch")
    values = []
    for index in indices:
        for kind in ("cover", "control"):
            rows = [
                r
                for r in transport["images"]
                if r["spec"] == asdict(spec)
                and r["index"] == index
                and r["kind"] == kind
                and r["format"] == fmt
            ]
            if len(rows) != 1:
                raise ValueError("generated transport missing/duplicate image")
            values.append(read_image(root, rows[0]).transpose(2, 0, 1))
    actual = np.stack(values).astype("<f4")
    expected, _ = data.batch(spec, indices)
    if not np.array_equal(actual, expected):
        raise ValueError("generated transport and in-memory batch differ")
    return actual


def verify(root: Path, transport: dict, corpus: dict) -> dict:
    checks = 0
    for manifest in corpus["profile"]:
        spec = data.GeneratedSpec(**manifest["spec"])
        for index, source in enumerate(manifest["sources"]):
            for fmt in ("PNG", "BMP"):
                values = batch(root, transport, spec, (index,), fmt).astype("u1")
                cover, control = values[0].transpose(1, 2, 0), values[1].transpose(1, 2, 0)
                selected = data.positions(cover.size, int(source["id"][-64:][16:24], 16))
                if (
                    a.digest(data.recover(control, selected)) != source["payload_sha256"]
                    or not np.array_equal(cover & 254, control & 254)
                    or np.max(np.abs(cover.astype(int) - control.astype(int))) > 1
                ):
                    raise ValueError("generated transport payload/LSB mismatch")
                checks += 1
    if len(transport["images"]) != 2 * checks:
        raise ValueError("generated transport extra image")
    return {
        "image_checks": 2 * checks,
        "independent_payload_checks": checks,
        "formats": ["PNG", "BMP"],
        "generated_only": True,
    }
