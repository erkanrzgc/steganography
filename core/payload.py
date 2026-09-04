"""Versioned payload envelopes shared by every carrier.

Version 2 adds a 64-bit length and a SHA-256 digest of the stored payload.  The
reader deliberately retains support for version 1 so files produced by the
0.1.x CLI remain extractable.
"""

import hashlib
import json
import struct
from dataclasses import dataclass

MAGIC = b"STEG"
LEGACY_VERSION = 1
VERSION = 2
VERSION_3 = 3
FLAG_ENCRYPTED = 0x01
FLAG_COMPRESSED = 0x02
FLAG_ECC = 0x04
_KNOWN_FLAGS = FLAG_ENCRYPTED | FLAG_COMPRESSED | FLAG_ECC

_V1_HEADER_STRUCT = struct.Struct(">4sBBI16s12s")
_V2_HEADER_STRUCT = struct.Struct(">4sBBQ16s12s32s")
_V3_HEADER_STRUCT = struct.Struct(">4sBB4sQ16s12s32sIH")
_V3_MARKER = b"PV3!"
LEGACY_HEADER_OVERHEAD = _V1_HEADER_STRUCT.size
HEADER_OVERHEAD = _V2_HEADER_STRUCT.size
V3_HEADER_OVERHEAD = _V3_HEADER_STRUCT.size


class InvalidPayloadError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ParsedPayload:
    payload: bytes
    encrypted: bool
    salt: bytes
    nonce: bytes
    version: int = VERSION
    compressed: bool = False
    ecc_symbols: int = 0
    metadata: dict[str, object] | None = None


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
    return _V1_HEADER_STRUCT.pack(MAGIC, LEGACY_VERSION, flags, len(payload), salt, nonce) + payload


def pack_v3(
    *,
    payload: bytes,
    encrypted: bool,
    salt: bytes,
    nonce: bytes,
    metadata: dict[str, object] | None = None,
    compressed: bool = False,
    ecc_symbols: int = 0,
) -> bytes:
    """Create a v3 envelope around already compressed/encrypted/ECC bytes."""
    if len(salt) != 16 or len(nonce) != 12:
        raise ValueError("salt must be 16 bytes and nonce 12 bytes")
    if not 0 <= ecc_symbols <= 255:
        raise ValueError("ecc_symbols must be between 0 and 255")
    metadata_blob = json.dumps(
        metadata or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if len(metadata_blob) > 1024 * 1024:
        raise ValueError("payload metadata is too large")
    flags = (FLAG_ENCRYPTED if encrypted else 0) | (FLAG_COMPRESSED if compressed else 0)
    if ecc_symbols:
        flags |= FLAG_ECC
    digest = hashlib.sha256(metadata_blob + payload).digest()
    return (
        _V3_HEADER_STRUCT.pack(
            MAGIC,
            VERSION_3,
            flags,
            _V3_MARKER,
            len(payload),
            salt,
            nonce,
            digest,
            len(metadata_blob),
            ecc_symbols,
        )
        + metadata_blob
        + payload
    )


def unpack(blob: bytes) -> ParsedPayload:
    if len(blob) < 6:
        raise InvalidPayloadError("blob too short to contain header")
    magic = blob[:4]
    version = blob[4]
    metadata: dict[str, object] | None = None
    metadata_blob = b""
    compressed = False
    ecc_symbols = 0
    if magic != MAGIC:
        raise InvalidPayloadError(f"bad magic: {magic!r}")
    if version == LEGACY_VERSION:
        if len(blob) < LEGACY_HEADER_OVERHEAD:
            raise InvalidPayloadError("blob too short to contain v1 header")
        _, _, flags, length, salt, nonce = _V1_HEADER_STRUCT.unpack(blob[:LEGACY_HEADER_OVERHEAD])
        offset = LEGACY_HEADER_OVERHEAD
        digest = None
    elif version == VERSION:
        if len(blob) < HEADER_OVERHEAD:
            raise InvalidPayloadError("blob too short to contain v2 header")
        _, _, flags, length, salt, nonce, digest = _V2_HEADER_STRUCT.unpack(blob[:HEADER_OVERHEAD])
        offset = HEADER_OVERHEAD
    elif version == VERSION_3 and blob[6:10] == _V3_MARKER:
        if len(blob) < V3_HEADER_OVERHEAD:
            raise InvalidPayloadError("blob too short to contain v3 header")
        (
            _,
            _,
            flags,
            _,
            length,
            salt,
            nonce,
            digest,
            metadata_length,
            ecc_symbols,
        ) = _V3_HEADER_STRUCT.unpack(blob[:V3_HEADER_OVERHEAD])
        if metadata_length > 1024 * 1024:
            raise InvalidPayloadError("v3 metadata is too large")
        metadata_end = V3_HEADER_OVERHEAD + metadata_length
        metadata_blob = blob[V3_HEADER_OVERHEAD:metadata_end]
        if len(metadata_blob) != metadata_length:
            raise InvalidPayloadError("truncated v3 metadata")
        try:
            metadata = json.loads(metadata_blob.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidPayloadError("invalid v3 metadata") from exc
        if not isinstance(metadata, dict):
            raise InvalidPayloadError("v3 metadata must be an object")
        offset = metadata_end
        compressed = bool(flags & FLAG_COMPRESSED)
    else:
        raise InvalidPayloadError(f"unsupported version: {version}")
    known_flags = _KNOWN_FLAGS if version == VERSION_3 else FLAG_ENCRYPTED
    if flags & ~known_flags:
        raise InvalidPayloadError(f"unsupported flags: 0x{flags:02x}")
    payload = blob[offset : offset + length]
    if len(payload) != length:
        raise InvalidPayloadError("truncated payload")
    digest_input = metadata_blob + payload if version == VERSION_3 else payload
    if digest is not None and not hashlib.sha256(digest_input).digest() == digest:
        raise InvalidPayloadError("payload integrity check failed")
    return ParsedPayload(
        payload=payload,
        encrypted=bool(flags & FLAG_ENCRYPTED),
        salt=salt,
        nonce=nonce,
        version=version,
        compressed=compressed,
        ecc_symbols=ecc_symbols,
        metadata=metadata,
    )
