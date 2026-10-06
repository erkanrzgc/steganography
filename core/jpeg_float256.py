"""Bounded unrounded Y-component IDCT, with block-aligned 256 center crops."""

from __future__ import annotations

import importlib.metadata
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from core.jpeg_features import MAX_IMAGE_BYTES, validate_jpeg

FEATURE_VERSION = "jpeg-y-idct-center256-phase0-f32-v1"
SIDE = 256
PIXELS = SIDE * SIDE
MAX_BATCH = 4
MAX_MAGNITUDE = 2**36


def decoder_contract():
    try:
        version = importlib.metadata.version("jpeglib")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("unrounded JPEG preparation requires the optional dct extra") from exc
    return {
        "jpeglib": version,
        "libjpeg": "6b",
        "numpy": np.__version__,
        "idct": "orthonormal-cosine-f64-to-le-f32-v1",
        "rounding": "none",
        "clipping": "none",
    }


def crop_box(width, height):
    return [8 * ((width - SIDE) // 16), 8 * ((height - SIDE) // 16), SIDE, SIDE]


def region(data):
    validate_jpeg(data)
    with Image.open(io.BytesIO(data)) as image:
        if image.mode not in {"RGB", "L"} or min(image.size) < SIDE:
            raise ValueError("float JPEG preparation needs RGB/grayscale JPEG at least 256x256")
        return {"image_size": list(image.size), "region": crop_box(*image.size)}


def reconstruct(coefficients, quantization, width, height):
    if (
        type(width) is not int
        or type(height) is not int
        or min(width, height) < SIDE
        or width * height > 4_000_000
        or not isinstance(coefficients, np.ndarray)
        or coefficients.shape != ((height + 7) // 8, (width + 7) // 8, 8, 8)
        or coefficients.dtype.kind not in "iu"
        or np.any(coefficients < -32768)
        or np.any(coefficients > 32767)
        or not isinstance(quantization, np.ndarray)
        or quantization.shape != (8, 8)
        or quantization.dtype.kind not in "iu"
        or np.any(quantization < 1)
        or np.any(quantization > 65535)
    ):
        raise ValueError("invalid bounded float JPEG coefficients/quantization")
    left, top, _, _ = crop_box(width, height)
    blocks = coefficients[top // 8 : top // 8 + 32, left // 8 : left // 8 + 32].astype(np.float64)
    basis = np.cos(np.pi * np.arange(8)[:, None] * (2 * np.arange(8)[None, :] + 1) / 16) / 2
    basis[0] /= np.sqrt(2)
    raster = np.einsum("ux,abuv,vy->abxy", basis, blocks * quantization, basis, optimize=True) + 128
    values = raster.transpose(0, 2, 1, 3).reshape(SIDE, SIDE).astype("<f4")
    if not np.isfinite(values).all() or np.any(np.abs(values) > MAX_MAGNITUDE):
        raise ValueError("invalid float JPEG raster")
    return values


def decode(data):
    import jpeglib

    info = region(data)
    jpeglib.version.set("6b")
    with tempfile.TemporaryDirectory(prefix="jpeg-float-") as directory:
        path = Path(directory) / "input.jpg"
        with path.open("xb") as stream:
            stream.write(data)
        jpeg = jpeglib.read_dct(str(path))
        # jpeglib 1.0.2 enum equality aliases distinct color spaces; use names.
        if jpeg.jpeg_color_space.name not in {"JCS_GRAYSCALE", "JCS_YCbCr"}:
            raise ValueError("float JPEG preparation requires Y component")
        jpeg.load()
        return reconstruct(jpeg.Y, jpeg.get_component_qt(0), *info["image_size"]).tobytes()


def pixel_batch(files):
    if not 1 <= len(files) <= MAX_BATCH:
        raise ValueError("float JPEG batch outside limits")
    decoder_contract()
    for data in files:
        region(data)
    payload = len(files).to_bytes(4, "little") + b"".join(
        len(d).to_bytes(4, "little") + d for d in files
    )
    with tempfile.TemporaryFile() as output:
        try:
            result = subprocess.run(  # noqa: S603 - fixed bounded module, no shell
                [sys.executable, "-m", "core.jpeg_float256"],
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
                },
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("float JPEG worker exceeded 15 seconds") from exc
        output.seek(0)
        raw = output.read(len(files) * PIXELS * 4 + 1)
    if result.returncode != 0 or len(raw) != len(files) * PIXELS * 4:
        raise RuntimeError("float JPEG worker failed or exceeded output limits")
    values = np.frombuffer(raw, dtype="<f4").reshape(len(files), 1, SIDE, SIDE)
    if not np.isfinite(values).all() or np.any(np.abs(values) > MAX_MAGNITUDE):
        raise ValueError("invalid float JPEG worker tensor")
    return values


def main():
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
            raise ValueError("invalid float JPEG frame")
        for _ in range(count):
            header = stream.read(4)
            size = int.from_bytes(header, "little")
            if len(header) != 4 or not 0 < size <= MAX_IMAGE_BYTES:
                raise ValueError("invalid float JPEG frame size")
            data = stream.read(size)
            if len(data) != size:
                raise ValueError("truncated float JPEG frame")
            sys.stdout.buffer.write(decode(data))
        if stream.read(1):
            raise ValueError("trailing float JPEG frame")
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
