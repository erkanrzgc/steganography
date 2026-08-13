"""Deterministic cryptographic position sampling for keyed carriers."""
from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Any

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

from core.crypto import derive_key


def sample_positions(
    candidates: Any,
    count: int,
    steg_key: str,
    salt: bytes,
    *,
    context: bytes = b"steganography-placement",
) -> Iterator[int]:
    """Yield a keyed sample without replacement using partial Fisher-Yates."""
    if count > len(candidates):
        raise ValueError("sample is larger than candidate set")
    random = ChaChaRandom(steg_key, salt, context=context)
    swaps: dict[int, int] = {}
    size = len(candidates)
    for index in range(count):
        chosen = index + random.randbelow(size - index)
        selected_value = swaps.get(chosen, chosen)
        swaps[chosen] = swaps.get(index, index)
        yield int(candidates[selected_value])


class ChaChaRandom:
    """Minimal rejection-sampling PRNG backed by a ChaCha20 keystream."""

    def __init__(self, password: str, salt: bytes, *, context: bytes) -> None:
        key = derive_key(password, salt)
        nonce = hashlib.sha256(context + b"\x00" + salt).digest()[:16]
        self._encryptor = Cipher(algorithms.ChaCha20(key, nonce), mode=None).encryptor()
        self._buffer = bytearray()

    def _read(self, count: int) -> bytes:
        while len(self._buffer) < count:
            self._buffer.extend(self._encryptor.update(b"\x00" * 4096))
        value = bytes(self._buffer[:count])
        del self._buffer[:count]
        return value

    def randbelow(self, upper: int) -> int:
        if upper <= 0:
            raise ValueError("upper must be positive")
        byte_count = max(1, (upper.bit_length() + 7) // 8)
        space = 1 << (8 * byte_count)
        limit = space - (space % upper)
        while True:
            value = int.from_bytes(self._read(byte_count), "big")
            if value < limit:
                return value % upper
