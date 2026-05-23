"""Binary payload header: magic | version | flags | length | salt | nonce | payload."""
import struct
from dataclasses import dataclass

MAGIC = b"STEG"
VERSION = 1
FLAG_ENCRYPTED = 0x01

# magic(4) + version(1) + flags(1) + length(4 BE) + salt(16) + nonce(12) = 38
_HEADER_STRUCT = struct.Struct(">4sBBI16s12s")
HEADER_OVERHEAD = _HEADER_STRUCT.size  # 38


class InvalidPayloadError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class ParsedPayload:
    payload: bytes
    encrypted: bool
    salt: bytes
    nonce: bytes


def pack(*, payload: bytes, encrypted: bool, salt: bytes, nonce: bytes) -> bytes:
    if len(salt) != 16 or len(nonce) != 12:
        raise ValueError("salt must be 16 bytes and nonce 12 bytes")
    flags = FLAG_ENCRYPTED if encrypted else 0
    header = _HEADER_STRUCT.pack(MAGIC, VERSION, flags, len(payload), salt, nonce)
    return header + payload


def unpack(blob: bytes) -> ParsedPayload:
    if len(blob) < HEADER_OVERHEAD:
        raise InvalidPayloadError("blob too short to contain header")
    magic, version, flags, length, salt, nonce = _HEADER_STRUCT.unpack(blob[:HEADER_OVERHEAD])
    if magic != MAGIC:
        raise InvalidPayloadError(f"bad magic: {magic!r}")
    if version != VERSION:
        raise InvalidPayloadError(f"unsupported version: {version}")
    payload = blob[HEADER_OVERHEAD : HEADER_OVERHEAD + length]
    if len(payload) != length:
        raise InvalidPayloadError("truncated payload")
    return ParsedPayload(
        payload=payload,
        encrypted=bool(flags & FLAG_ENCRYPTED),
        salt=salt,
        nonce=nonce,
    )
