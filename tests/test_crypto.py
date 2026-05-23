import pytest

from core.crypto import (
    AES_GCM_NONCE_LEN,
    DecryptionError,
    KDF_SALT_LEN,
    derive_key,
    encrypt,
    decrypt,
)


def test_derive_key_is_deterministic():
    salt = b"\x01" * KDF_SALT_LEN
    k1 = derive_key("hunter2", salt)
    k2 = derive_key("hunter2", salt)
    assert k1 == k2
    assert len(k1) == 32


def test_derive_key_changes_with_password():
    salt = b"\x01" * KDF_SALT_LEN
    assert derive_key("a", salt) != derive_key("b", salt)


def test_roundtrip_encrypts_and_decrypts():
    salt = b"\x02" * KDF_SALT_LEN
    nonce, ct = encrypt(b"top secret", "pw", salt)
    assert len(nonce) == AES_GCM_NONCE_LEN
    pt = decrypt(ct, "pw", salt, nonce)
    assert pt == b"top secret"


def test_wrong_password_raises():
    salt = b"\x03" * KDF_SALT_LEN
    nonce, ct = encrypt(b"top secret", "pw", salt)
    with pytest.raises(DecryptionError):
        decrypt(ct, "WRONG", salt, nonce)


def test_tampered_ciphertext_raises():
    salt = b"\x04" * KDF_SALT_LEN
    nonce, ct = encrypt(b"top secret", "pw", salt)
    tampered = bytes([ct[0] ^ 0x01]) + ct[1:]
    with pytest.raises(DecryptionError):
        decrypt(tampered, "pw", salt, nonce)
