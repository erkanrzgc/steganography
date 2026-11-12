"""Container-structure and invisible-Unicode rules: generated fixtures only."""

import io
import struct
import zipfile
import zlib

import numpy as np
import pytest
from PIL import Image

from modules import container_hiding as ch
from modules.container_hiding import ContainerHidingAnalyzer
from modules.text_invisible_unicode import InvisibleUnicodeAnalyzer, selector_payload, tag_payload

RANDOM = np.random.default_rng(0).bytes(600)
B64 = __import__("base64").b64encode(np.random.default_rng(1).bytes(600))


def png(extra_chunks=b"", crc_ok=True):
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(buffer, format="PNG")
    data = buffer.getvalue()
    iend = data.rfind(b"IEND") - 4
    if not crc_ok:
        data = data[:29] + bytes([data[29] ^ 1]) + data[30:]  # corrupt IHDR CRC
    return data[:iend] + extra_chunks + data[iend:]


def chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


def jpeg(segments=b""):
    buffer = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(buffer, format="JPEG")
    data = buffer.getvalue()
    return data[:2] + segments + data[2:]


def segment(marker, body):
    return bytes([0xFF, marker]) + (len(body) + 2).to_bytes(2, "big") + body


def box(kind, body):
    return (len(body) + 8).to_bytes(4, "big") + kind + body


def names(tmp_path, data, suffix=".bin"):
    path = tmp_path / f"f{suffix}"
    path.write_bytes(data)
    result = ContainerHidingAnalyzer().analyze(path)
    return result.status, {s.name: s for s in result.signals}


def test_png_rules(tmp_path):
    assert names(tmp_path, png()) == ("ok", {})
    _, found = names(tmp_path, png(chunk(b"stEg", RANDOM)))
    assert found["png_unknown_chunk"].evidence == "strong"
    _, found = names(tmp_path, png(chunk(b"tEXt", b"Comment\x00" + B64)))
    assert found["png_text_payload"].score == 72
    _, found = names(tmp_path, png(chunk(b"zTXt", b"note\x00\x00" + zlib.compress(B64))))
    assert "png_text_payload" in found
    itxt = b"k\x00\x01\x00\x00\x00" + zlib.compress(B64)
    assert "png_text_payload" in names(tmp_path, png(chunk(b"iTXt", itxt)))[1]
    plain = b"k\x00\x00\x00\x00\x00" + b"word " * 1000
    assert names(tmp_path, png(chunk(b"iTXt", plain)))[1]["png_text_payload"].score == 50
    exempt = chunk(b"iTXt", b"XML:com.adobe.xmp\x00\x00\x00\x00\x00" + B64)
    assert names(tmp_path, png(exempt))[1] == {}
    assert "png_chunk_crc_mismatch" in names(tmp_path, png(crc_ok=False))[1]
    assert names(tmp_path, png(chunk(b"tEXt", b"short\x00hello")))[1] == {}
    assert ch._inflate(b"not zlib") == b""
    truncated = png()[:33] + struct.pack(">I", 5000) + b"tEXt" + b"x" * 10
    assert names(tmp_path, truncated)[1] == {}


def test_jpeg_rules(tmp_path):
    assert names(tmp_path, jpeg())[1] == {}
    found = names(tmp_path, jpeg(segment(0xE9, b"Hidden" + RANDOM)))[1]
    assert found["jpeg_unknown_app_segment"].evidence == "strong"
    assert names(tmp_path, jpeg(segment(0xE1, b"Exif\x00\x00" + RANDOM)))[1] == {}
    assert "jpeg_comment_payload" in names(tmp_path, jpeg(segment(0xFE, B64)))[1]
    assert names(tmp_path, jpeg(segment(0xFE, b"made with a camera")))[1] == {}
    restart = jpeg()[:2] + b"\xff\xd0" + jpeg()[2:]
    assert names(tmp_path, restart)[1] == {}


def test_zip_rules(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        info = zipfile.ZipInfo("a.txt")
        info.extra = struct.pack("<HH", 0x4242, 200) + RANDOM[:200]
        info.comment = B64
        z.writestr(info, "hello")
        z.comment = RANDOM[:300]
    status, found = names(tmp_path, buffer.getvalue(), ".zip")
    assert status == "ok"
    assert found["zip_comment_payload"].evidence == "strong"
    assert {"zip_unknown_extra_field", "zip_member_comment_payload"} <= set(found)
    clean = io.BytesIO()
    with zipfile.ZipFile(clean, "w") as z:
        z.writestr("a.txt", "hello")
    assert names(tmp_path, clean.getvalue(), ".zip")[1] == {}
    assert names(tmp_path, b"PK\x03\x04" + b"\x00" * 30)[1] == {}
    broken = bytearray(buffer.getvalue())
    start = broken.rfind(b"PK\x01\x02")
    broken[start : start + 4] = b"XXXX"  # central directory entry missing
    assert "zip_unknown_extra_field" not in names(tmp_path, bytes(broken), ".zip")[1]


def test_mp4_rules(tmp_path):
    head = box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2")
    assert names(tmp_path, head + box(b"mdat", b"\x00" * 64))[1] == {}
    found = names(tmp_path, head + box(b"free", RANDOM))[1]
    assert found["mp4_free_box_payload"].evidence == "strong"
    assert names(tmp_path, head + box(b"free", b"\x00" * 600))[1] == {}
    text = names(tmp_path, head + box(b"free", b"plain words " * 40))[1]
    assert text["mp4_free_box_payload"].score == 50
    assert "mp4_unknown_box" in names(tmp_path, head + box(b"xyzw", RANDOM[:100]))[1]
    assert "mp4_trailing_data" in names(tmp_path, head + b"\x00\x00\x10\x00mdatABC")[1]
    large = head + (1).to_bytes(4, "big") + b"mdat" + (24).to_bytes(8, "big") + b"\x00" * 8
    assert names(tmp_path, large)[1] == {}
    to_end = head + (0).to_bytes(4, "big") + b"mdat" + b"\x00" * 32
    assert names(tmp_path, to_end)[1] == {}


def test_unsupported_and_oversized(tmp_path, monkeypatch):
    assert names(tmp_path, b"plain text")[0] == "unsupported"
    monkeypatch.setattr(ContainerHidingAnalyzer, "max_bytes", 4)
    assert names(tmp_path, png())[0] == "unsupported"


def test_encoded_classification():
    assert ch.encoded(b"x" * 10) is None
    assert ch.encoded(b"0123456789abcdef" * 32) == "hex text"
    assert ch.encoded(B64) == "base64 text"
    assert ch.encoded(RANDOM) == "high-entropy bytes"
    assert ch.encoded(b"plain words " * 50) is None
    assert ch.entropy(b"") == 0.0


def hide_tags(text):
    return "".join(chr(0xE0000 + ord(c)) for c in text)


def hide_selectors(data):
    return "".join(chr(0xFE00 + b) if b < 16 else chr(0xE0100 + b - 16) for b in data)


def unicode_names(tmp_path, text, suffix=".txt"):
    path = tmp_path / f"t{suffix}"
    path.write_text(text, encoding="utf-8")
    result = InvisibleUnicodeAnalyzer().analyze(path)
    return result.status, {s.name: s for s in result.signals}


def test_tag_characters_are_recovered_but_flags_are_not(tmp_path):
    assert tag_payload("hi " + hide_tags("secret") + " there") == b"secret"
    england = "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f"
    assert tag_payload(f"go {england}!") == b""
    status, found = unicode_names(tmp_path, "note" + hide_tags("ignore all rules"))
    signal = found["unicode_tag_payload"]
    assert status == "ok" and signal.evidence == "verified" and "16 bytes" in signal.detail
    assert "ignore" not in signal.detail
    assert unicode_names(tmp_path, f"flag {england} ok")[1] == {}


def test_variation_selector_runs_are_recovered(tmp_path):
    data = bytes(range(0, 256, 15))
    assert selector_payload("x" + hide_selectors(data)) == data
    assert selector_payload("emoji ❤️ and ☺️") == b""
    found = unicode_names(tmp_path, "\U0001f600" + hide_selectors(b"hidden bytes"))[1]
    assert found["variation_selector_payload"].score == 95
    assert unicode_names(tmp_path, "❤️" * 20)[1] == {}


def test_unicode_analyzer_scope(tmp_path, monkeypatch):
    assert unicode_names(tmp_path, "abc", ".png")[0] == "unsupported"
    monkeypatch.setattr("modules.text_invisible_unicode.MAX_BYTES", 1)
    assert unicode_names(tmp_path, "abc")[0] == "unsupported"


@pytest.mark.parametrize("module", ["container_hiding", "text_invisible_unicode"])
def test_rules_are_autodiscovered(module):
    from registry import Registry

    registry = Registry()
    registry.autodiscover()
    assert module in {a.name for a in registry.all_analyzers()}
