"""Bounded decoded-luminance center crops for explicit research, not a detector."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import __version__ as PILLOW_VERSION

from core.jpeg_features import MAX_IMAGE_BYTES, validate_jpeg

FEATURE_VERSION = "jpeg-center128-luma-u8-v1"
SIDE = 128
PIXELS = SIDE * SIDE
MAX_BATCH = 8


def decoder_contract() -> dict:
    return {"pillow": PILLOW_VERSION, "jpeg_codec": Image.core.jpeglib_version}


def region(data: bytes) -> dict:
    validate_jpeg(data)
    with Image.open(io.BytesIO(data)) as image:
        if image.mode not in {"RGB", "L"} or min(image.size) < SIDE:
            raise ValueError("pixel research needs RGB/grayscale JPEG at least 128x128")
        left, top = (image.width - SIDE) // 2, (image.height - SIDE) // 2
        return {"image_size": list(image.size), "region": [left, top, SIDE, SIDE]}


def decode(data: bytes) -> bytes:
    box = region(data)["region"]
    with Image.open(io.BytesIO(data)) as image:
        # No EXIF rotation, resize, interpolation, augmentation or reference cover.
        image.load()
        crop = image.convert("L").crop((box[0], box[1], box[0] + SIDE, box[1] + SIDE))
        return crop.tobytes()


def pixel_batch(files: list[bytes]) -> np.ndarray:
    if not 1 <= len(files) <= MAX_BATCH:
        raise ValueError("pixel batch outside limits")
    for data in files:
        region(data)
    payload = len(files).to_bytes(4, "little") + b"".join(
        len(data).to_bytes(4, "little") + data for data in files
    )
    with tempfile.TemporaryFile() as output:
        try:
            result = subprocess.run(  # noqa: S603 - fixed module; bounded stdin, no shell
                [sys.executable, "-m", "core.jpeg_pixels"],
                input=payload,
                stdout=output,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
                cwd=Path(__file__).resolve().parents[1],
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("pixel worker exceeded 15 seconds") from exc
        output.seek(0)
        raw = output.read(len(files) * PIXELS + 1)
    if result.returncode != 0 or len(raw) != len(files) * PIXELS:
        raise RuntimeError("pixel worker failed or exceeded output limits")
    return np.frombuffer(raw, dtype="u1").reshape(len(files), 1, SIDE, SIDE)


def main() -> int:
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (15, 16))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_IMAGE_BYTES, MAX_IMAGE_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        stream = sys.stdin.buffer
        header = stream.read(4)
        count = int.from_bytes(header, "little")
        if len(header) != 4 or not 1 <= count <= MAX_BATCH:
            raise ValueError("invalid pixel batch count")
        for _ in range(count):
            header = stream.read(4)
            size = int.from_bytes(header, "little")
            if len(header) != 4 or not 0 < size <= MAX_IMAGE_BYTES:
                raise ValueError("invalid pixel JPEG size")
            data = stream.read(size)
            if len(data) != size:
                raise ValueError("truncated pixel JPEG")
            sys.stdout.buffer.write(decode(data))
        if stream.read(1):
            raise ValueError("trailing pixel batch bytes")
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
