"""Internal subprocess worker for :mod:`modules.image_jpeg_dct`."""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

from core.placement import sample_positions

_MAGIC = b"STEGDCT1"
_VERSION = 1
_BOOTSTRAP = struct.Struct(">8sBQ16s")
_BOOTSTRAP_BITS = _BOOTSTRAP.size * 8
_ARITHMETIC_SOF = frozenset(range(0xC9, 0xCC)) | frozenset(range(0xCD, 0xD0))


def _read(path: Path):
    import jpeglib

    data = path.read_bytes()
    if not any(bytes((0xFF, marker)) in data for marker in (0xC0, 0xC1)):
        if any(bytes((0xFF, marker)) in data for marker in _ARITHMETIC_SOF):
            raise ValueError("arithmetic-coded JPEG is unsupported")
        raise ValueError("only baseline/extended sequential JPEG is supported")
    image = jpeglib.read_dct(str(path))
    if image.progressive_mode:
        raise ValueError("progressive JPEG is unsupported")
    return image


def _eligible(image) -> np.ndarray:
    coefficients = image.Y.reshape(-1)
    positions = np.arange(coefficients.size, dtype=np.int64)
    ac = positions % 64 != 0
    return positions[ac & (np.abs(coefficients.astype(np.int32)) >= 2)]


def _write_bits(coefficients: np.ndarray, positions: np.ndarray, bits: np.ndarray) -> None:
    values = coefficients[positions].astype(np.int32)
    magnitudes = np.abs(values)
    change = (magnitudes & 1) != bits
    magnitudes[change] += 1
    coefficients[positions] = np.where(values < 0, -magnitudes, magnitudes).astype(
        coefficients.dtype
    )


def capacity(src: Path) -> int:
    image = _read(src)
    return max(0, (len(_eligible(image)) - _BOOTSTRAP_BITS) // 8)


def embed(
    src: Path,
    out: Path,
    payload_path: Path,
    key: str,
    salt: bytes | None = None,
) -> None:
    if not key:
        raise ValueError("placement key is empty")
    image = _read(src)
    coefficients = image.Y.reshape(-1)
    eligible = _eligible(image)
    payload = payload_path.read_bytes()
    if len(payload) > max(0, (len(eligible) - _BOOTSTRAP_BITS) // 8):
        raise ValueError("payload exceeds JPEG DCT capacity")
    salt = salt or os.urandom(16)
    bootstrap = _BOOTSTRAP.pack(_MAGIC, _VERSION, len(payload), salt)
    bootstrap_bits = np.unpackbits(np.frombuffer(bootstrap, dtype=np.uint8))
    _write_bits(coefficients, eligible[:_BOOTSTRAP_BITS], bootstrap_bits)
    candidates = eligible[_BOOTSTRAP_BITS:]
    payload_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    selected = np.fromiter(
        sample_positions(
            candidates,
            payload_bits.size,
            key,
            salt,
            context=b"image-jpeg-dct",
        ),
        dtype=np.int64,
        count=payload_bits.size,
    )
    _write_bits(coefficients, selected, payload_bits)
    image.write_dct(str(out))
    if extract(out, key) != payload:
        Path(out).unlink(missing_ok=True)
        raise ValueError("JPEG DCT output failed round-trip validation")


def _bootstrap(image) -> tuple[bytes, int, int, bytes] | None:
    coefficients = image.Y.reshape(-1)
    eligible = _eligible(image)
    if len(eligible) < _BOOTSTRAP_BITS:
        return None
    bits = np.abs(coefficients[eligible[:_BOOTSTRAP_BITS]].astype(np.int32)) & 1
    raw = np.packbits(bits.astype(np.uint8)).tobytes()
    return _BOOTSTRAP.unpack(raw)


def extract(src: Path, key: str) -> bytes:
    if not key:
        raise ValueError("placement key is empty")
    image = _read(src)
    header = _bootstrap(image)
    if header is None:
        raise ValueError("JPEG has insufficient coefficients for a DCT bootstrap")
    magic, version, length, salt = header
    if magic != _MAGIC or version != _VERSION:
        raise ValueError("no supported STEGDCT1 bootstrap")
    coefficients = image.Y.reshape(-1)
    candidates = _eligible(image)[_BOOTSTRAP_BITS:]
    bit_count = length * 8
    if bit_count > len(candidates):
        raise ValueError("declared DCT payload exceeds coefficient capacity")
    selected = np.fromiter(
        sample_positions(
            candidates,
            bit_count,
            key,
            salt,
            context=b"image-jpeg-dct",
        ),
        dtype=np.int64,
        count=bit_count,
    )
    bits = np.abs(coefficients[selected].astype(np.int32)) & 1
    return np.packbits(bits.astype(np.uint8)).tobytes()


def analyze(src: Path) -> dict:
    try:
        image = _read(src)
    except ValueError as exc:
        return {
            "status": "unsupported",
            "suspicion": 0,
            "signals": [],
            "explanation": None,
            "error": str(exc),
        }
    header = _bootstrap(image)
    signals = []
    if header is not None:
        magic, version, length, _ = header
        if magic == _MAGIC:
            valid = version == _VERSION
            signals.append(
                {
                    "name": "stegdct_bootstrap",
                    "score": 98 if valid else 80,
                    "detail": f"version={version}, payload_length={length}",
                    "category": "known_marker",
                    "evidence": "verified" if valid else "strong",
                }
            )
    coefficients = image.Y.reshape(-1)
    eligible = _eligible(image)
    if len(eligible):
        parity_mean = float(
            np.mean(np.abs(coefficients[eligible].astype(np.int32)) & 1)
        )
        parity_score = min(60, round(abs(parity_mean - 0.5) * 120))
        signals.append(
            {
                "name": "dct_parity_bias",
                "score": parity_score,
                "detail": f"eligible AC parity mean={parity_mean:.4f}",
                "category": "jpeg_dct_statistics",
                "evidence": "heuristic" if parity_score >= 10 else "informational",
            }
        )
    suspicion = max(
        (
            item["score"]
            for item in signals
            if item["evidence"] != "informational"
        ),
        default=0,
    )
    return {
        "status": "ok",
        "suspicion": suspicion,
        "signals": signals,
        "explanation": None,
        "error": None,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print("missing operation", file=sys.stderr)
        return 2
    operation, *values = arguments
    try:
        if operation == "capacity" and len(values) == 1:
            result = {"capacity": capacity(Path(values[0]))}
        elif operation == "embed" and len(values) in {3, 4}:
            key = sys.stdin.readline().rstrip("\r\n")
            salt = bytes.fromhex(values[3]) if len(values) == 4 else None
            if salt is not None and len(salt) != 16:
                raise ValueError("placement salt must be exactly 16 bytes")
            embed(Path(values[0]), Path(values[1]), Path(values[2]), key, salt)
            result = {"ok": True}
        elif operation == "extract" and len(values) == 2:
            key = sys.stdin.readline().rstrip("\r\n")
            Path(values[1]).write_bytes(extract(Path(values[0]), key))
            result = {"ok": True}
        elif operation == "analyze" and len(values) == 1:
            result = analyze(Path(values[0]))
        else:
            raise ValueError("invalid worker arguments")
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
