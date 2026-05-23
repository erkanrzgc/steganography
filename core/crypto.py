"""AES-256-GCM encryption with scrypt password KDF."""
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

AES_KEY_LEN = 32
AES_GCM_NONCE_LEN = 12
KDF_SALT_LEN = 16
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1


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
