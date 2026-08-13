import struct

import pytest

from core.payload import (
    HEADER_OVERHEAD,
    LEGACY_VERSION,
    VERSION,
    InvalidPayloadError,
    pack,
    pack_v1,
    unpack,
)


def test_v2_roundtrip_and_integrity():
    blob = pack(
        payload=b"authenticated bytes",
        encrypted=False,
        salt=b"\x00" * 16,
        nonce=b"\x00" * 12,
    )
    assert len(blob) == HEADER_OVERHEAD + len(b"authenticated bytes")
    assert blob[4] == VERSION
    parsed = unpack(blob)
    assert parsed.version == VERSION
    assert parsed.payload == b"authenticated bytes"

    tampered = bytearray(blob)
    tampered[-1] ^= 1
    with pytest.raises(InvalidPayloadError, match="integrity"):
        unpack(bytes(tampered))


def test_v1_remains_readable():
    blob = pack_v1(
        payload=b"legacy",
        encrypted=True,
        salt=b"s" * 16,
        nonce=b"n" * 12,
    )
    parsed = unpack(blob)
    assert parsed.version == LEGACY_VERSION
    assert parsed.encrypted is True
    assert parsed.payload == b"legacy"


@pytest.mark.parametrize("version", [0, 3, 255])
def test_unknown_version_rejected(version):
    blob = b"STEG" + bytes([version, 0]) + b"\x00" * 100
    with pytest.raises(InvalidPayloadError, match="unsupported version"):
        unpack(blob)


def test_unknown_flag_and_truncated_v2_rejected():
    blob = bytearray(
        pack(payload=b"x", encrypted=False, salt=b"\x00" * 16, nonce=b"\x00" * 12)
    )
    blob[5] = 0x80
    with pytest.raises(InvalidPayloadError, match="unsupported flags"):
        unpack(bytes(blob))
    with pytest.raises(InvalidPayloadError, match="v2 header"):
        unpack(b"STEG" + bytes([VERSION, 0]))


def test_v1_header_and_length_validation():
    with pytest.raises(InvalidPayloadError, match="v1 header"):
        unpack(b"STEG" + bytes([LEGACY_VERSION, 0]))
    blob = bytearray(
        pack_v1(
            payload=b"x",
            encrypted=False,
            salt=b"\x00" * 16,
            nonce=b"\x00" * 12,
        )
    )
    struct.pack_into(">I", blob, 6, 500)
    with pytest.raises(InvalidPayloadError, match="truncated"):
        unpack(bytes(blob))


def test_invalid_salt_or_nonce_rejected():
    with pytest.raises(ValueError):
        pack(payload=b"x", encrypted=False, salt=b"short", nonce=b"n" * 12)
    with pytest.raises(ValueError):
        pack_v1(payload=b"x", encrypted=False, salt=b"s" * 16, nonce=b"short")
