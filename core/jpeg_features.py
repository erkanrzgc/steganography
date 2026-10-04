"""Experimental JPEG DCT summary features, not SRNet/DCTR or a verdict."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

FEATURE_VERSION = "jpeg-dct-summary-v1"
AC = tuple((u, v) for u in range(8) for v in range(8) if (u, v) != (0, 0))
LOW = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (2, 1), (1, 2), (2, 2))
FEATURE_NAMES = (
    tuple(f"y_{u}_{v}_abs_{b}" for u, v in AC for b in range(8))
    + tuple(
        f"y_{u}_{v}_{d}_pair_{a}_{b}"
        for u, v in LOW
        for d in ("h", "v")
        for a in range(5)
        for b in range(5)
    )
    + tuple(f"q_y_{u}_{v}" for u in range(8) for v in range(8))
)
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_PIXELS = 4_000_000
MAX_OUTPUT_BYTES = 128 * 1024


def coefficient_features(coefficients: np.ndarray, quantization: np.ndarray) -> list[float]:
    """Y[block-row, block-column, u, v]; signed pairs clipped to [-2,2]."""
    if (
        coefficients.ndim != 4
        or coefficients.shape[2:] != (8, 8)
        or min(coefficients.shape[:2]) < 2
        or coefficients.size > MAX_PIXELS + 64_000
        or coefficients.dtype.kind not in "iu"
        or quantization.shape != (8, 8)
        or not np.isfinite(quantization).all()
        or np.any(quantization < 1)
        or np.any(quantization > 65535)
    ):
        raise ValueError("invalid bounded JPEG coefficient/quantization arrays")
    # int64 avoids overflow on signed coefficient magnitude conversion.
    if np.any(coefficients < -32768) or np.any(coefficients > 32767):
        raise ValueError("JPEG coefficient outside signed 16-bit range")
    values = coefficients.astype(np.int64)
    features = []
    for u, v in AC:
        histogram = np.bincount(np.minimum(np.abs(values[:, :, u, v]), 7).ravel(), minlength=8)
        features.extend((histogram / histogram.sum()).tolist())
    for u, v in LOW:
        plane = np.clip(values[:, :, u, v], -2, 2) + 2
        for first, second in ((plane[:, :-1], plane[:, 1:]), (plane[:-1, :], plane[1:, :])):
            histogram = np.bincount((first * 5 + second).ravel(), minlength=25)
            features.extend((histogram / histogram.sum()).tolist())
    features.extend((quantization.astype(np.float64).ravel() / 65535.0).tolist())
    return features


def validate_jpeg(data: bytes) -> None:
    if not 0 < len(data) <= MAX_IMAGE_BYTES:
        raise ValueError("JPEG feature input exceeds byte limits")
    with Image.open(io.BytesIO(data)) as image:
        if (
            image.format != "JPEG"
            or min(image.size) < 16
            or image.width * image.height > MAX_PIXELS
        ):
            raise ValueError("JPEG feature format or dimensions outside limits")


def jpeg_features(data: bytes) -> list[float]:
    validate_jpeg(data)
    if importlib.util.find_spec("jpeglib") is None:
        raise RuntimeError("JPEG features require the optional dct extra")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    with tempfile.TemporaryFile() as output:
        try:
            result = subprocess.run(  # noqa: S603 - fixed interpreter/module, bytes via stdin
                [sys.executable, "-m", "core.jpeg_features"],
                input=data,
                stdout=output,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
                cwd=Path(__file__).resolve().parents[1],
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("JPEG feature worker exceeded 15 seconds") from exc
        output.seek(0)
        raw = output.read(MAX_OUTPUT_BYTES + 1)
    if result.returncode != 0 or len(raw) > MAX_OUTPUT_BYTES:
        raise RuntimeError("JPEG feature worker failed or exceeded output limits")
    values = json.loads(raw)
    if (
        not isinstance(values, list)
        or len(values) != len(FEATURE_NAMES)
        or any(type(v) not in (float, int) or not np.isfinite(v) or not 0 <= v <= 1 for v in values)
    ):
        raise ValueError("invalid JPEG feature worker output")
    return values


def worker(data: bytes) -> list[float]:
    import jpeglib

    validate_jpeg(data)
    with tempfile.TemporaryDirectory(prefix="jpeg-features-") as directory:
        path = Path(directory) / "input.jpg"
        with path.open("xb") as stream:
            stream.write(data)
        jpeg = jpeglib.read_dct(str(path))
        jpeg.load()
        if jpeg.num_components not in (1, 3):
            raise ValueError("unsupported JPEG colorspace")
        return coefficient_features(jpeg.Y, jpeg.qt[int(jpeg.quant_tbl_no[0])])


def main() -> int:
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (15, 16))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_IMAGE_BYTES, MAX_IMAGE_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        values = worker(sys.stdin.buffer.read(MAX_IMAGE_BYTES + 1))
        print(json.dumps(values, allow_nan=False))
        return 0
    except Exception:
        return 1  # Native paths/messages never cross the worker boundary.


if __name__ == "__main__":
    raise SystemExit(main())
