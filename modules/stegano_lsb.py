"""Recover messages hidden by the Python ``stegano`` package's LSB method.

``stegano.lsb.hide`` writes the byte-length prefix ``"<n>:"`` followed by the
encoded message, eight bits per byte (most significant first), into the least
significant bits of R, G and B of consecutive pixels. Pixels are visited in
row-major order (default generator) or at the prime indices (``eratosthenes``).

A message counts as recovered only when the decimal prefix is well formed, the
declared length fits the image and the full message decodes strictly in one of
stegano's encodings. Only its length, encoding and SHA-256 are reported.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

MAX_PREFIX_DIGITS = 9
ENCODINGS = ("UTF-32LE", "UTF-8")  # UTF-32LE text is also valid UTF-8
ORDERS = ("row-major", "eratosthenes")


@dataclass(frozen=True)
class SteganoMessage:
    """A recovered stegano LSB message (content withheld)."""

    order: str
    encoding: str
    length: int
    sha256: str


def _primes_below(limit: int) -> np.ndarray:
    sieve = np.ones(max(limit, 2), bool)
    sieve[:2] = False
    for p in range(2, int(limit**0.5) + 1):
        if sieve[p]:
            sieve[p * p :: p] = False
    return np.flatnonzero(sieve)


def _pixel_order(order: str, count: int) -> np.ndarray:
    return np.arange(count) if order == "row-major" else _primes_below(count)


def _reader(rgb: np.ndarray, order: str):
    flat = rgb.reshape(-1, 3)
    positions = _pixel_order(order, len(flat))

    def read(nbytes: int) -> bytes | None:
        pixels = -(-nbytes * 8 // 3)
        if pixels > len(positions):
            return None
        bits = (flat[positions[:pixels]] & 1).reshape(-1)[: nbytes * 8]
        return np.packbits(bits.astype(np.uint8)).tobytes()

    return read, len(positions) * 3 // 8


def find_message(rgb: np.ndarray) -> SteganoMessage | None:
    """The first stegano message found in a uint8 HxWx3 array, or ``None``."""
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3:
        raise ValueError("uint8 HxWx3 array required")
    if rgb.shape[2] != 3:
        raise ValueError("three colour channels required")
    for order in ORDERS:
        read, capacity = _reader(rgb, order)
        head = read(min(MAX_PREFIX_DIGITS + 1, capacity))
        if not head or b":" not in head:
            continue
        digits = head[: head.index(b":")]
        if not digits.isdigit() or digits.startswith(b"0"):
            continue
        length = int(digits)
        prefix = len(digits) + 1
        if prefix + length > capacity:
            continue
        data = read(prefix + length)
        if data is None:  # pragma: no cover - guarded by the capacity check
            continue
        message = data[prefix:]
        for encoding in ENCODINGS:
            try:
                message.decode(encoding, errors="strict")
            except UnicodeDecodeError:
                continue
            return SteganoMessage(order, encoding, length, hashlib.sha256(message).hexdigest())
    return None
