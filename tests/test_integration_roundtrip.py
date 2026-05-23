"""End-to-end: encrypted payload through the registry-selected carrier."""
import os
from pathlib import Path

import pytest

from core.crypto import KDF_SALT_LEN, decrypt, encrypt
from core.payload import pack, unpack
from registry import Registry


@pytest.fixture
def registry() -> Registry:
    r = Registry()
    r.autodiscover()
    return r


def _wrap(payload: bytes, password: str | None) -> bytes:
    if password:
        salt = os.urandom(KDF_SALT_LEN)
        nonce, ct = encrypt(payload, password, salt)
        return pack(payload=ct, encrypted=True, salt=salt, nonce=nonce)
    return pack(payload=payload, encrypted=False, salt=b"\x00" * 16, nonce=b"\x00" * 12)


def _unwrap(blob: bytes, password: str | None) -> bytes:
    parsed = unpack(blob)
    if parsed.encrypted:
        return decrypt(parsed.payload, password or "", parsed.salt, parsed.nonce)
    return parsed.payload


def test_png_lsb_encrypted_roundtrip(registry, png_64x64: Path, tmp_outdir: Path):
    carrier = registry.select_carrier_for_embed(Path("x.png"))
    blob = _wrap(b"top secret bytes", "hunter2")
    out = tmp_outdir / "stego.png"
    carrier.embed(png_64x64, blob, out)
    recovered = _unwrap(carrier.extract(out), "hunter2")
    assert recovered == b"top secret bytes"


def test_wav_encrypted_roundtrip(registry, wav_pcm16: Path, tmp_outdir: Path):
    carrier = registry.select_carrier_for_embed(Path("x.wav"))
    blob = _wrap(b"audio secret", "p@ss")
    out = tmp_outdir / "stego.wav"
    carrier.embed(wav_pcm16, blob, out)
    recovered = _unwrap(carrier.extract(out), "p@ss")
    assert recovered == b"audio secret"
