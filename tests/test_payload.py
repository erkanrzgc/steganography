import pytest

from core.payload import (
    HEADER_OVERHEAD,
    MAGIC,
    InvalidPayloadError,
    pack,
    unpack,
)


def test_pack_unpack_unencrypted():
    blob = pack(payload=b"hello", encrypted=False, salt=b"\x00" * 16, nonce=b"\x00" * 12)
    assert blob.startswith(MAGIC)
    assert len(blob) == HEADER_OVERHEAD + len(b"hello")
    parsed = unpack(blob)
    assert parsed.payload == b"hello"
    assert parsed.encrypted is False


def test_pack_unpack_encrypted():
    salt = b"\xaa" * 16
    nonce = b"\xbb" * 12
    blob = pack(payload=b"\x01\x02\x03", encrypted=True, salt=salt, nonce=nonce)
    parsed = unpack(blob)
    assert parsed.encrypted is True
    assert parsed.salt == salt
    assert parsed.nonce == nonce
    assert parsed.payload == b"\x01\x02\x03"


def test_bad_magic_raises():
    with pytest.raises(InvalidPayloadError):
        unpack(b"XXXX" + b"\x00" * 50)


def test_truncated_raises():
    with pytest.raises(InvalidPayloadError):
        unpack(b"STEG")
