"""JPEG COM fingerprints of encoders bundled with steganography tools."""

import io

import numpy as np
from PIL import Image

from modules.image_jpeg import ImageJpeg

F5_COMMENT = b"JPEG Encoder Copyright 1998, James R. Weeks and BioElectroMech."


def jpeg_with_comment(tmp_path, comment):
    buffer = io.BytesIO()
    pixels = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=75)
    data = buffer.getvalue()
    if comment is not None:
        segment = b"\xff\xfe" + (len(comment) + 2).to_bytes(2, "big") + comment
        data = data[:2] + segment + data[2:]  # right after SOI
    path = tmp_path / "x.jpg"
    path.write_bytes(data)
    return path


def names(path):
    return {s.name: s for s in ImageJpeg().analyze(path).signals}


def test_f5_encoder_comment_is_a_strong_tool_fingerprint(tmp_path):
    signal = names(jpeg_with_comment(tmp_path, F5_COMMENT))["stego_tool_encoder_comment"]
    assert signal.score == 70 and signal.evidence == "strong"
    assert signal.category == "stego_tool_fingerprint" and "F5" in signal.detail


def test_other_comments_and_plain_jpegs_are_not_flagged(tmp_path):
    assert "stego_tool_encoder_comment" not in names(jpeg_with_comment(tmp_path, b"hello"))
    assert "stego_tool_encoder_comment" not in names(jpeg_with_comment(tmp_path, None))
