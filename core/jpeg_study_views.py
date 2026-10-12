"""Bounded, unrounded gray JPEG views; no content/label-based region choice."""

from __future__ import annotations

import io
import os
import resource
import signal
import subprocess
import sys
import tempfile
from contextlib import suppress
from pathlib import Path

import numpy as np
from PIL import Image

from core import jpeg_float256
from core.jpeg_features import validate_jpeg

CENTER = "center256"
FIVE = "center-four-corners256"
SCHEMA = "jpeg-learning-study-five-view-data-v1"
REGIONS = ((128, 128, 256, 256), (0, 0, 256, 256), (256, 0, 256, 256),
           (0, 256, 256, 256), (256, 256, 256, 256))
ROW_BYTES = 5 * 256 * 256 * 4
MAX_JPEG = 256 * 1024


def contract():
    return {"feature_version": "jpeg-gray-five-view-phase0-unrounded-f32-v1",
            "regions": [list(r) for r in REGIONS], "view_count": 5,
            "cache_layout": "sample-view-channel-y-x-le-f32",
            "training_views": "lineage-seeded-offset-cyclic-per-pair-exposure-v1",
            "inference": "arithmetic-mean-five-singleton-stego-probabilities-v1",
            "color": "native grayscale; not RGB", "view_selection": "label-independent"}


def validate(data):
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_JPEG:
        raise ValueError("five-view JPEG byte bounds")
    validate_jpeg(data)
    with Image.open(io.BytesIO(data)) as image:
        if image.mode != "L" or image.size != (512, 512):
            raise ValueError("five-view JPEG requires native gray 512x512")


def reconstruct(coefficients, qt):
    # The existing bounded coefficient validator is reused, not its region choice.
    jpeg_float256.reconstruct(coefficients, qt, 512, 512)
    basis = np.cos(np.pi * np.arange(8)[:, None] * (2 * np.arange(8)[None, :] + 1) / 16) / 2
    basis[0] /= np.sqrt(2)
    blocks = coefficients.astype(np.float64) * qt
    raster = np.einsum("ux,abuv,vy->abxy", basis, blocks, basis, optimize=True) + 128
    full = raster.transpose(0, 2, 1, 3).reshape(512, 512).astype("<f4")
    if not np.isfinite(full).all() or np.any(np.abs(full) > jpeg_float256.MAX_MAGNITUDE):
        raise ValueError("five-view nonfinite/out-of-bounds raster")
    return np.stack([full[top:top + height, left:left + width]
                     for left, top, width, height in REGIONS])[:, None]


def decode(data):
    import jpeglib

    validate(data)
    jpeglib.version.set("6b")
    with tempfile.TemporaryDirectory(prefix="jpeg-five-view-") as folder:
        path = Path(folder) / "input.jpg"
        with path.open("xb") as stream:
            stream.write(data)
        jpeg = jpeglib.read_dct(str(path))
        jpeg.load()
        if (jpeg.jpeg_color_space.name != "JCS_GRAYSCALE"
            or (jpeg.width, jpeg.height) != (512, 512)):
            raise ValueError("five-view native JPEG structure mismatch")
        return reconstruct(jpeg.Y, jpeg.get_component_qt(0)).tobytes()


def pixel_batch(files):
    if not isinstance(files, (list, tuple)) or not 1 <= len(files) <= 2:
        raise ValueError("five-view batch outside limits")
    jpeg_float256.decoder_contract()
    for data in files:
        validate(data)
    payload = len(files).to_bytes(4, "little") + b"".join(
        len(data).to_bytes(4, "little") + data for data in files
    )
    maximum = len(files) * ROW_BYTES
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(  # noqa: S603 - fixed resource-limited module, no shell
            [sys.executable, "-m", "core.jpeg_study_views"], stdin=subprocess.PIPE,
            stdout=output, stderr=subprocess.DEVNULL, start_new_session=True,
            cwd=Path(__file__).resolve().parents[1],
            env={"PATH": os.environ.get("PATH", ""), "OPENBLAS_NUM_THREADS": "1",
                 "OMP_NUM_THREADS": "1"},
        )
        try:
            process.communicate(payload, timeout=20)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("five-view worker timeout") from exc
        finally:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        output.seek(0)
        raw = output.read(maximum + 1)
    if process.returncode or len(raw) != maximum:
        raise RuntimeError("five-view worker failed/output bounds")
    values = np.frombuffer(raw, dtype="<f4").reshape(len(files), 5, 1, 256, 256)
    if not np.isfinite(values).all() or np.any(np.abs(values) > jpeg_float256.MAX_MAGNITUDE):
        raise ValueError("five-view worker tensor invalid")
    return values


def main():
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (20, 21))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2 * ROW_BYTES, 2 * ROW_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        stream = sys.stdin.buffer
        header = stream.read(4)
        count = int.from_bytes(header, "little")
        if len(header) != 4 or not 1 <= count <= 2:
            raise ValueError("five-view worker frame invalid")
        for _ in range(count):
            header = stream.read(4)
            size = int.from_bytes(header, "little")
            if len(header) != 4 or not 0 < size <= MAX_JPEG:
                raise ValueError("five-view worker byte limit")
            raw = stream.read(size)
            if len(raw) != size:
                raise ValueError("five-view worker truncated frame")
            sys.stdout.buffer.write(decode(raw))
        if stream.read(1):
            raise ValueError("five-view worker trailing input")
        return 0
    except (ValueError, OSError, ImportError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
