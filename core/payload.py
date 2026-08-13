"""Versioned payload envelopes shared by every carrier.

Version 2 adds a 64-bit length and a SHA-256 digest of the stored payload.  The
reader deliberately retains support for version 1 so files produced by the
0.1.x CLI remain extractable.
"""
import hashlib
import struct
from dataclasses import dataclass

MAGIC = b"STEG"
LEGACY_VERSION = 1
VERSION = 2
FLAG_ENCRYPTED = 0x01
_KNOWN_FLAGS = FLAG_ENCRYPTED

_V1_HEADER_STRUCT = struct.Struct(">4sBBI16s12s")
_V2_HEADER_STRUCT = struct.Struct(">4sBBQ16s12s32s")
LEGACY_HEADER_OVERHEAD = _V1_HEADER_STRUCT.size
HEADER_OVERHEAD = _V2_HEADER_STRUCT.size


class InvalidPayloadError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ParsedPayload:
    payload: bytes
    encrypted: bool
    salt: bytes
    nonce: bytes
    version: int = VERSION


def pack(*, payload: bytes, encrypted: bool, salt: bytes, nonce: bytes) -> bytes:
    if len(salt) != 16 or len(nonce) != 12:
        raise ValueError("salt must be 16 bytes and nonce 12 bytes")
    flags = FLAG_ENCRYPTED if encrypted else 0
    digest = hashlib.sha256(payload).digest()
    header = _V2_HEADER_STRUCT.pack(MAGIC, VERSION, flags, len(payload), salt, nonce, digest)
    return header + payload


def pack_v1(*, payload: bytes, encrypted: bool, salt: bytes, nonce: bytes) -> bytes:
    """Create a legacy envelope for compatibility tests and migration tools."""
    if len(salt) != 16 or len(nonce) != 12:
        raise ValueError("salt must be 16 bytes and nonce 12 bytes")
    flags = FLAG_ENCRYPTED if encrypted else 0
    return _V1_HEADER_STRUCT.pack(
        MAGIC, LEGACY_VERSION, flags, len(payload), salt, nonce
    ) + payload


def unpack(blob: bytes) -> ParsedPayload:
    if len(blob) < 6:
        raise InvalidPayloadError("blob too short to contain header")
    magic = blob[:4]
    version = blob[4]
    if magic != MAGIC:
        raise InvalidPayloadError(f"bad magic: {magic!r}")
    if version == LEGACY_VERSION:
        if len(blob) < LEGACY_HEADER_OVERHEAD:
            raise InvalidPayloadError("blob too short to contain v1 header")
        _, _, flags, length, salt, nonce = _V1_HEADER_STRUCT.unpack(
            blob[:LEGACY_HEADER_OVERHEAD]
        )
        offset = LEGACY_HEADER_OVERHEAD
        digest = None
    elif version == VERSION:
        if len(blob) < HEADER_OVERHEAD:
            raise InvalidPayloadError("blob too short to contain v2 header")
        _, _, flags, length, salt, nonce, digest = _V2_HEADER_STRUCT.unpack(
            blob[:HEADER_OVERHEAD]
        )
        offset = HEADER_OVERHEAD
    else:
        raise InvalidPayloadError(f"unsupported version: {version}")
    if flags & ~_KNOWN_FLAGS:
        raise InvalidPayloadError(f"unsupported flags: 0x{flags:02x}")
    payload = blob[offset : offset + length]
    if len(payload) != length:
        raise InvalidPayloadError("truncated payload")
    if digest is not None and not hashlib.sha256(payload).digest() == digest:
        raise InvalidPayloadError("payload integrity check failed")
    return ParsedPayload(
        payload=payload,
        encrypted=bool(flags & FLAG_ENCRYPTED),
        salt=salt,
        nonce=nonce,
        version=version,
    )
