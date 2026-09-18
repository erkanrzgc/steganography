"""Internal subprocess worker for :mod:`modules.image_jpeg_dct`."""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
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


def extract_jsteg(src: Path) -> bytes:
    image = _read(src)
    coefficients = image.Y.reshape(-1)
    eligible = _eligible(image)
    if len(eligible) < 72:
        raise ValueError("insufficient eligible DCT coefficients for JSteg")
    hdr_bits = np.abs(coefficients[eligible[:72]].astype(np.int32)) & 1
    hdr_bytes = bytes(
        sum(int(hdr_bits[c + b]) << b for b in range(8))
        for c in range(0, 72, 8)
    )
    if not hdr_bytes.startswith(b"jsteg"):
        raise ValueError("no JSteg magic header found")
    (length,) = struct.unpack("<I", hdr_bytes[5:9])
    total_bits = (9 + length) * 8
    if len(eligible) < total_bits:
        raise ValueError("declared JSteg payload exceeds available coefficients")
    payload_bits = np.abs(coefficients[eligible[72:total_bits]].astype(np.int32)) & 1
    return bytes(
        sum(int(payload_bits[c + b]) << b for b in range(8))
        for c in range(0, len(payload_bits), 8)
    )


def _compute_westfeld_dct_chi2(values: np.ndarray) -> tuple[float, int]:
    chi_total = 0.0
    deg_freedom = 0
    for k in range(1, 16):
        c_even = int(np.sum(values == 2 * k))
        c_odd = int(np.sum(values == 2 * k + 1))
        total = c_even + c_odd
        if total >= 6:
            exp = total / 2.0
            chi_total += ((c_even - exp) ** 2 + (c_odd - exp) ** 2) / exp
            deg_freedom += 1
        c_even_neg = int(np.sum(values == -2 * k))
        c_odd_neg = int(np.sum(values == -2 * k - 1))
        total_neg = c_even_neg + c_odd_neg
        if total_neg >= 6:
            exp = total_neg / 2.0
            chi_total += ((c_even_neg - exp) ** 2 + (c_odd_neg - exp) ** 2) / exp
            deg_freedom += 1
    normalized = chi_total / max(deg_freedom, 1)
    return normalized, deg_freedom


def _compute_calibrated_f5_metrics(src: Path, image) -> tuple[float, float] | None:
    """Perform 4-pixel spatial calibration to detect F5 shrinkage & DCT histogram deformation."""
    import jpeglib

    try:
        if getattr(image, "progressive_mode", False):
            return None
        spatial = jpeglib.read_spatial(str(src))
        h, w = spatial.spatial.shape[:2]
        if h < 16 or w < 16:
            return None
        new_h = ((h - 4) // 8) * 8
        new_w = ((w - 4) // 8) * 8
        if new_h < 8 or new_w < 8:
            return None
        cropped = spatial.spatial[4 : 4 + new_h, 4 : 4 + new_w]
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            calib_sp = jpeglib.from_spatial(cropped)
            calib_sp.write_spatial(tmp.name, qt=image.qt)
            calib_dct = jpeglib.read_dct(tmp.name)

        ac_mask = np.ones((8, 8), dtype=bool)
        ac_mask[0, 0] = False

        orig_ac = image.Y.reshape(-1, 8, 8)[:, ac_mask].reshape(-1)
        calib_ac = calib_dct.Y.reshape(-1, 8, 8)[:, ac_mask].reshape(-1)

        orig_blocks = max(1, image.Y.shape[0] * image.Y.shape[1])
        calib_blocks = max(1, calib_dct.Y.shape[0] * calib_dct.Y.shape[1])

        nz_per_blk = float(np.sum(orig_ac != 0) / orig_blocks)
        h1_orig = float(np.sum(np.abs(orig_ac) == 1) / orig_blocks)
        h1_calib = float(np.sum(np.abs(calib_ac) == 1) / calib_blocks)
        h0_orig = float(np.sum(orig_ac == 0) / orig_blocks)
        h0_calib = float(np.sum(calib_ac == 0) / calib_blocks)

        if nz_per_blk < 3.0 or h1_orig < 0.8 or h1_calib < 0.8:
            return None

        delta_zero = h0_orig - h0_calib
        ratio_ones = h1_orig / h1_calib
        return delta_zero, ratio_ones
    except Exception:
        return None


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
    if len(eligible) >= 72:
        hdr_bits = np.abs(coefficients[eligible[:72]].astype(np.int32)) & 1
        hdr_bytes = bytes(
            sum(int(hdr_bits[c + b]) << b for b in range(8))
            for c in range(0, 72, 8)
        )
        if hdr_bytes.startswith(b"jsteg"):
            (declared_len,) = struct.unpack("<I", hdr_bytes[5:9])
            max_possible = (len(eligible) - 72) // 8
            valid_len = declared_len <= max_possible
            signals.append(
                {
                    "name": "jsteg_header",
                    "score": 98 if valid_len else 80,
                    "detail": f"JSteg marker found; payload_length={declared_len}",
                    "category": "known_marker",
                    "evidence": "verified" if valid_len else "strong",
                }
            )
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
        if len(eligible) >= 100:
            values = coefficients[eligible]
            chi_norm, df = _compute_westfeld_dct_chi2(values)
            if df >= 4:
                if chi_norm <= 3.0:
                    chi_score = min(85, round(75 + (3.0 - chi_norm) / 3.0 * 10))
                elif chi_norm <= 8.5:
                    chi_score = min(75, round(50 + (8.5 - chi_norm) / 5.5 * 25))
                elif chi_norm <= 14.0:
                    chi_score = min(50, round(20 + (14.0 - chi_norm) / 5.5 * 30))
                else:
                    chi_score = 0
                signals.append(
                    {
                        "name": "westfeld_dct_chi_square",
                        "score": chi_score,
                        "detail": f"DCT pairs-of-values chi-square={chi_norm:.4f} (df={df})",
                        "category": "jpeg_dct_chi_square",
                        "evidence": "heuristic" if chi_score >= 40 else "informational",
                    }
                )

    f5_metrics = _compute_calibrated_f5_metrics(src, image)
    if f5_metrics is not None:
        delta_zero, ratio_ones = f5_metrics
        if delta_zero >= 2.0 and ratio_ones <= 0.88:
            score = min(88, round(75 + (delta_zero - 2.0) * 5))
            evidence = "strong"
        elif delta_zero >= 1.4 and ratio_ones <= 0.91:
            score = 70
            evidence = "heuristic"
        elif delta_zero >= 1.0 and ratio_ones <= 0.94:
            score = 50
            evidence = "heuristic"
        else:
            score = 0
            evidence = "informational"
        signals.append(
            {
                "name": "f5_shrinkage_anomaly",
                "score": score,
                "detail": f"excess zeros/block={delta_zero:+.3f}, ones ratio={ratio_ones:.3f}",
                "category": "jpeg_dct_f5",
                "evidence": evidence,
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
        elif operation == "extract_jsteg" and len(values) == 2:
            Path(values[1]).write_bytes(extract_jsteg(Path(values[0])))
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
