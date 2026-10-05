"""Optional local-research JRM adapter; no upstream code or model is bundled."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from core import jpeg_features as jpeg

VERSION = "2024.12"
FEATURE_VERSION = "sealwatch-jrm-2024.12-v1"
DIMENSIONS = 11255
LAYOUT_SHA256 = "c3e65542745f259383b18d85574a24f0701e72879f5a2df3cef21d8b44681549"
OUTPUT_BYTES = DIMENSIONS * 4
MAX_BATCH = 8


def require_version() -> None:
    try:
        version = importlib.metadata.version("sealwatch")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("JRM reference requires the optional jrm-reference extra") from exc
    if version != VERSION:
        raise RuntimeError("JRM reference requires pinned sealwatch 2024.12")


def coefficient_features(coefficients: np.ndarray, quantization: np.ndarray) -> list[float]:
    require_version()
    # Reuse signed coefficient, quantization and pixel guards without weakening them.
    jpeg.coefficient_features(coefficients, quantization)
    if min(coefficients.shape[:2]) < 8:
        raise ValueError("JRM reference requires at least eight blocks per dimension")
    import sealwatch as sw

    models = sw.jrm.extract(coefficients.astype(np.int32), calibrated=False)
    layout = [(name, list(values.shape)) for name, values in models.items()]
    digest = hashlib.sha256(json.dumps(layout, separators=(",", ":")).encode()).hexdigest()
    if digest != LAYOUT_SHA256:
        raise ValueError("JRM upstream layout contract changed")
    values = np.concatenate([v.ravel(order="C") for v in models.values()]).astype("<f4")
    validate_vector(values)
    return values.tolist()


def validate_vector(values: np.ndarray) -> None:
    if (
        values.shape != (DIMENSIONS,)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 1)
    ):
        raise ValueError("invalid bounded JRM vector")


def jpeg_jrm_features(data: bytes) -> np.ndarray:
    return jpeg_jrm_batch([data])[0]


def jpeg_jrm_batch(files: list[bytes]) -> np.ndarray:
    if not 1 <= len(files) <= MAX_BATCH:
        raise ValueError("JRM batch outside limits")
    for data in files:
        jpeg.validate_jpeg(data)
    require_version()
    payload = len(files).to_bytes(4, "little") + b"".join(
        len(data).to_bytes(4, "little") + data for data in files
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    with tempfile.TemporaryFile() as output:
        try:
            result = subprocess.run(  # noqa: S603 - fixed module and bounded stdin
                [sys.executable, "-m", "core.jpeg_jrm"],
                input=payload,
                stdout=output,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
                cwd=Path(__file__).resolve().parents[1],
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("JRM worker exceeded 15 seconds") from exc
        output.seek(0)
        raw = output.read(OUTPUT_BYTES * len(files) + 1)
    if result.returncode != 0 or len(raw) != OUTPUT_BYTES * len(files):
        raise RuntimeError("JRM worker failed or exceeded output limits")
    values = np.frombuffer(raw, dtype="<f4").reshape(len(files), DIMENSIONS)
    for vector in values:
        validate_vector(vector)
    return values


def main() -> int:
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (15, 16))
    resource.setrlimit(resource.RLIMIT_FSIZE, (jpeg.MAX_IMAGE_BYTES, jpeg.MAX_IMAGE_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    try:
        stream = sys.stdin.buffer
        header = stream.read(4)
        count = int.from_bytes(header, "little")
        if len(header) != 4 or not 1 <= count <= MAX_BATCH:
            raise ValueError("invalid batch count")
        for _ in range(count):
            header = stream.read(4)
            size = int.from_bytes(header, "little")
            if len(header) != 4 or not 1 <= size <= jpeg.MAX_IMAGE_BYTES:
                raise ValueError("invalid frame size")
            data = stream.read(size)
            if len(data) != size:
                raise ValueError("truncated frame")
            values = jpeg.worker(data, coefficient_features)
            sys.stdout.buffer.write(np.asarray(values, dtype="<f4").tobytes())
        if stream.read(1):
            raise ValueError("trailing frame data")
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
