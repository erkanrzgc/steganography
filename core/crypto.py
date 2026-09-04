"""AES-256-GCM encryption with scrypt password KDF."""

import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

AES_KEY_LEN = 32
AES_GCM_NONCE_LEN = 12
KDF_SALT_LEN = 16
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_ARGON2_MEMORY_KIB = 64 * 1024
_ARGON2_ITERATIONS = 3
_ARGON2_LANES = 1


class DecryptionError(Exception):
    """Raised when authentication fails (wrong password or tampered ciphertext)."""


def derive_key(password: str, salt: bytes) -> bytes:
    if len(salt) != KDF_SALT_LEN:
        raise ValueError(f"salt must be {KDF_SALT_LEN} bytes")
    kdf = Scrypt(salt=salt, length=AES_KEY_LEN, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    return kdf.derive(password.encode("utf-8"))


def encrypt(plaintext: bytes, password: str, salt: bytes) -> tuple[bytes, bytes]:
    """Encrypt plaintext, returning (nonce, ciphertext_with_tag)."""
    key = derive_key(password, salt)
    nonce = os.urandom(AES_GCM_NONCE_LEN)
    ct = AESGCM(key).encrypt(nonce, plaintext, associated_data=None)
    return nonce, ct


def decrypt(ciphertext: bytes, password: str, salt: bytes, nonce: bytes) -> bytes:
    key = derive_key(password, salt)
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, associated_data=None)
    except InvalidTag as e:
        raise DecryptionError("authentication failed") from e


def derive_key_argon2id(password: str, salt: bytes) -> bytes:
    """Derive a payload-v3 key with an intentionally memory-hard KDF."""
    if len(salt) != KDF_SALT_LEN:
        raise ValueError(f"salt must be {KDF_SALT_LEN} bytes")
    return Argon2id(
        salt=salt,
        length=AES_KEY_LEN,
        iterations=_ARGON2_ITERATIONS,
        lanes=_ARGON2_LANES,
        memory_cost=_ARGON2_MEMORY_KIB,
    ).derive(password.encode("utf-8"))


def encrypt_v3(
    plaintext: bytes,
    password: str,
    salt: bytes,
    *,
    associated_data: bytes = b"STEG-payload-v3",
) -> tuple[bytes, bytes]:
    key = derive_key_argon2id(password, salt)
    nonce = os.urandom(AES_GCM_NONCE_LEN)
    return nonce, AESGCM(key).encrypt(nonce, plaintext, associated_data)


def decrypt_v3(
    ciphertext: bytes,
    password: str,
    salt: bytes,
    nonce: bytes,
    *,
    associated_data: bytes = b"STEG-payload-v3",
) -> bytes:
    key = derive_key_argon2id(password, salt)
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, associated_data)
    except InvalidTag as exc:
        raise DecryptionError("authentication failed") from exc
