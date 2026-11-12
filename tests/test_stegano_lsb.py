"""stegano LSB message recovery: independent encoder fixture plus optional library interop."""

import hashlib

import numpy as np
import pytest
from PIL import Image

from modules import stegano_lsb
from modules.image_bitplane import ImageBitplaneAnalyzer


def cover(seed=0, size=40):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


def hide(rgb, message, encoding="UTF-8", order="row-major"):
    """Independent re-implementation of stegano's LSB layout."""
    data = message.encode(encoding)
    payload = f"{len(data)}:".encode("ascii") + data
    bits = np.unpackbits(np.frombuffer(payload, np.uint8))
    bits = np.concatenate([bits, np.zeros((-len(bits)) % 3, np.uint8)])
    flat = rgb.reshape(-1, 3).copy()
    positions = stegano_lsb._pixel_order(order, len(flat))[: len(bits) // 3]
    flat[positions] = (flat[positions] & 0xFE) | bits.reshape(-1, 3)
    return flat.reshape(rgb.shape)


@pytest.mark.parametrize("order", stegano_lsb.ORDERS)
@pytest.mark.parametrize("encoding", stegano_lsb.ENCODINGS)
def test_recovers_hidden_message(order, encoding):
    text = "attack at dawn, şifre yok"
    found = stegano_lsb.find_message(hide(cover(1, 80), text, encoding, order))
    data = text.encode(encoding)
    assert found == stegano_lsb.SteganoMessage(order, encoding, len(data),
                                               hashlib.sha256(data).hexdigest())


def test_random_covers_and_malformed_prefixes_are_clean():
    assert all(stegano_lsb.find_message(cover(seed)) is None for seed in range(200))
    rgb = cover(2)
    for raw in (b"07:abcdefg", b"x1:a", b"999999:ab", b"123456789", b"4:\xff\xfe\xfd\xfc"):
        bits = np.unpackbits(np.frombuffer(raw, np.uint8))
        bits = np.concatenate([bits, np.zeros((-len(bits)) % 3, np.uint8)])
        flat = rgb.reshape(-1, 3).copy()
        flat[: len(bits) // 3] = (flat[: len(bits) // 3] & 0xFE) | bits.reshape(-1, 3)
        found = stegano_lsb.find_message(flat.reshape(rgb.shape))
        assert found is None or found.order == "eratosthenes"


def test_tiny_images_and_bad_input():
    assert stegano_lsb.find_message(np.zeros((1, 2, 3), np.uint8)) is None
    for bad in (np.zeros((4, 4, 3), np.int16), np.zeros((4, 4), np.uint8),
                np.zeros((4, 4, 4), np.uint8), [[0]]):
        with pytest.raises(ValueError):
            stegano_lsb.find_message(bad)


def test_bitplane_analyzer_confirms_recovered_message(tmp_path):
    clean, stego = tmp_path / "c.png", tmp_path / "s.png"
    Image.fromarray(cover(3, 64)).save(clean)
    Image.fromarray(hide(cover(3, 64), "secret text")).save(stego)
    analyzer = ImageBitplaneAnalyzer()
    names = {s.name for s in analyzer.analyze(clean).signals}
    assert "stegano_lsb_message" not in names
    signal = next(s for s in analyzer.analyze(stego).signals if s.name == "stegano_lsb_message")
    assert signal.evidence == "verified" and signal.category == "known_marker"
    assert "secret" not in signal.detail and "11 bytes" in signal.detail


def test_interoperates_with_the_stegano_package():
    lsb = pytest.importorskip("stegano.lsb")
    generators = pytest.importorskip("stegano.lsb.generators")
    image = Image.fromarray(cover(4, 64))
    plain = np.asarray(lsb.hide(image.copy(), "hello from stegano"))
    primes = np.asarray(lsb.hide(image.copy(), "prime walk", generators.eratosthenes()))
    assert stegano_lsb.find_message(plain).length == 18
    assert stegano_lsb.find_message(primes).order == "eratosthenes"
