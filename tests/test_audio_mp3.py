"""Tests for MP3 audio steganalysis, ID3v2 extraction, and CTF solving."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from core.ctf import CTFService, _trailer
from core.filetype import detect_type
from modules.audio_mp3 import (
    AudioMp3,
    extract_mp3_payloads,
    get_mpeg_frame_length,
    mp3_structural_end,
    parse_id3v2,
)
from modules.filestruct_appended import FilestructAppended


def make_mpeg_frame(private: int = 0) -> bytes:
    """Generate a single valid MPEG-1 Layer III audio frame (128 kbps, 44.1 kHz)."""
    hdr = bytes([0xFF, 0xFB, 0x90 | ((private & 1) << 0), 0x00])
    frame_len = int(144 * 128000 / 44100)  # 417 bytes
    return hdr + b"\x00" * (frame_len - 4)


def make_id3v2(frames: list[tuple[str, bytes]], padding: bytes = b"") -> bytes:
    """Construct an ID3v2.3 tag header with the given frames and padding."""
    body = bytearray()
    for fid, content in frames:
        body.extend(fid.encode("ascii")[:4])
        body.extend(struct.pack(">I", len(content)))
        body.extend(b"\x00\x00")
        body.extend(content)
    body.extend(padding)
    tag_size = len(body)
    sz_bytes = bytes([
        (tag_size >> 21) & 0x7F,
        (tag_size >> 14) & 0x7F,
        (tag_size >> 7) & 0x7F,
        tag_size & 0x7F,
    ])
    header = b"ID3\x03\x00\x00" + sz_bytes
    return header + bytes(body)


def test_filetype_detects_mp3(tmp_path: Path) -> None:
    sample_id3 = tmp_path / "song.mp3"
    sample_id3.write_bytes(make_id3v2([("TIT2", b"\x00Sample Song")]) + make_mpeg_frame() * 2)
    dt1 = detect_type(sample_id3)
    assert dt1.name == "mp3"
    assert dt1.mime_type == "audio/mpeg"

    sample_raw = tmp_path / "raw.mp3"
    sample_raw.write_bytes(make_mpeg_frame() * 3)
    dt2 = detect_type(sample_raw)
    assert dt2.name == "mp3"


def test_clean_mp3_has_zero_suspicion(tmp_path: Path) -> None:
    clean = tmp_path / "clean.mp3"
    clean.write_bytes(make_id3v2([("TIT2", b"\x00Sample Track")]) + make_mpeg_frame() * 10)

    result = AudioMp3().analyze(clean)
    assert result.status == "ok"
    assert result.suspicion == 0


def test_id3v2_comment_flag_detection_and_ctf(tmp_path: Path) -> None:
    flag = b"flag{mp3_id3v2_comment_payload}"
    content = b"\x00eng\x00" + flag
    mp3_data = make_id3v2([("COMM", content)]) + make_mpeg_frame() * 5

    mp3_path = tmp_path / "comment_flag.mp3"
    mp3_path.write_bytes(mp3_data)

    res = AudioMp3().analyze(mp3_path)
    signals = {s.name: s for s in res.signals}
    assert "id3_frame_flag" in signals
    assert signals["id3_frame_flag"].evidence == "verified"
    assert signals["id3_frame_flag"].score == 98

    # CTF automated solve
    out_dir = tmp_path / "ctf_out"
    report = CTFService().solve(mp3_path, out_dir, mode="quick")
    assert report.verdict == "confirmed"
    assert any("id3-comm" in a.name for a in report.artifacts)


def test_id3v2_padding_steganography(tmp_path: Path) -> None:
    flag = b"flag{hidden_in_id3_padding_area}"
    padding = b"\x00\x00\x00\x00" + flag + b"\x00" * 32
    mp3_data = make_id3v2([("TIT2", b"\x00Song Title")], padding=padding) + make_mpeg_frame() * 4

    mp3_path = tmp_path / "padding.mp3"
    mp3_path.write_bytes(mp3_data)

    res = AudioMp3().analyze(mp3_path)
    signals = {s.name: s for s in res.signals}
    assert "id3_padding_flag" in signals
    assert signals["id3_padding_flag"].score == 98

    payloads = extract_mp3_payloads(mp3_data)
    assert any(p[0] == "id3-padding.bin" and flag in p[1] for p in payloads)


def test_id3v2_album_art_extraction(tmp_path: Path) -> None:
    jpeg_art = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9"
    apic_content = b"\x00image/jpeg\x00\x03Cover\x00" + jpeg_art
    mp3_data = make_id3v2([("APIC", apic_content)]) + make_mpeg_frame() * 3

    mp3_path = tmp_path / "artwork.mp3"
    mp3_path.write_bytes(mp3_data)

    payloads = extract_mp3_payloads(mp3_data)
    assert any(p[0] == "mp3-artwork.jpg" and p[1] == jpeg_art for p in payloads)


def test_mp3_private_bit_steganography(tmp_path: Path) -> None:
    flag = b"flag{mp3_private_bit_covert}"
    bits = [int(b) for b in "".join(f"{byte:08b}" for byte in flag)]
    frames = b"".join(make_mpeg_frame(private=bit) for bit in bits)
    mp3_data = make_id3v2([("TIT2", b"\x00Stream")]) + frames

    mp3_path = tmp_path / "private_bits.mp3"
    mp3_path.write_bytes(mp3_data)

    res = AudioMp3().analyze(mp3_path)
    signals = {s.name: s for s in res.signals}
    assert "mp3_private_bit_flag" in signals
    assert signals["mp3_private_bit_flag"].score == 98


def test_mp3_trailer_carving_with_id3v1(tmp_path: Path) -> None:
    frames = make_mpeg_frame() * 4
    id3v1 = b"TAG" + b"Title".ljust(30, b"\x00") + b"Artist".ljust(30, b"\x00") + b"\x00" * 65
    trailer_data = b"flag{appended_after_id3v1_trailer}"
    mp3_data = frames + id3v1 + trailer_data

    mp3_path = tmp_path / "trailer.mp3"
    mp3_path.write_bytes(mp3_data)

    res = AudioMp3().analyze(mp3_path)
    signals = {s.name: s for s in res.signals}
    assert "mp3_appended_flag" in signals

    fs_res = FilestructAppended().analyze(mp3_path)
    assert fs_res.suspicion >= 70

    carved = _trailer(mp3_data, ".mp3")
    assert carved == trailer_data


def test_unsupported_and_helper_guards(tmp_path: Path) -> None:
    carrier = AudioMp3()
    assert carrier.analyze(Path("nope.bin")).status == "unsupported"
    assert carrier.capacity(Path("any.mp3")) == 0
    with pytest.raises(NotImplementedError):
        carrier.embed(Path("a"), b"", Path("b"))
    with pytest.raises(NotImplementedError):
        carrier.extract(Path("a"))

    not_mp3 = tmp_path / "test.txt"
    not_mp3.write_bytes(b"hello world")
    assert carrier.analyze(not_mp3).status == "unsupported"

    fake_mp3 = tmp_path / "fake.mp3"
    fake_mp3.write_bytes(b"not an mp3 file really")
    assert carrier.analyze(fake_mp3).status == "unsupported"

    assert get_mpeg_frame_length(b"bad") is None
    assert get_mpeg_frame_length(b"\x00\x00\x00\x00") is None
    # Bad version bit
    assert get_mpeg_frame_length(bytes([0xFF, 0xE8, 0x90, 0x00])) is None
    # Bad bitrate index 15
    assert get_mpeg_frame_length(bytes([0xFF, 0xFB, 0xF0, 0x00])) is None
    # Bad samplerate index 3
    assert get_mpeg_frame_length(bytes([0xFF, 0xFB, 0x9C, 0x00])) is None

    # MPEG-1 Layer I
    hdr_l1 = bytes([0xFF, 0xFE, 0x90, 0x00])
    parsed_l1 = get_mpeg_frame_length(hdr_l1)
    assert parsed_l1 is not None and parsed_l1[0] > 0

    # MPEG-1 Layer II
    hdr_l2 = bytes([0xFF, 0xFC, 0x90, 0x00])
    parsed_l2 = get_mpeg_frame_length(hdr_l2)
    assert parsed_l2 is not None and parsed_l2[0] > 0

    # MPEG-2 Layer I & III
    hdr_m2_l1 = bytes([0xFF, 0xF6, 0x90, 0x00])
    assert get_mpeg_frame_length(hdr_m2_l1) is not None
    hdr_m2_l3 = bytes([0xFF, 0xF2, 0x90, 0x00])
    assert get_mpeg_frame_length(hdr_m2_l3) is not None

    assert parse_id3v2(b"not_id3") == {}
    assert mp3_structural_end(b"") is None


def test_id3v24_and_apic_png(tmp_path: Path) -> None:
    # ID3v2.4 with synchsafe frame length
    body = bytearray()
    body.extend(b"APIC")
    png_magic = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    content = b"\x00image/png\x00\x03Cover\x00" + png_magic
    flen = len(content)
    # synchsafe frame len
    body.extend(bytes([0, 0, (flen >> 7) & 0x7F, flen & 0x7F]))
    body.extend(b"\x00\x00")
    body.extend(content)
    tag_size = len(body)
    header = b"ID3\x04\x00\x00" + bytes([0, 0, (tag_size >> 7) & 0x7F, tag_size & 0x7F])
    mp3_data = header + bytes(body) + make_mpeg_frame() * 2

    payloads = extract_mp3_payloads(mp3_data)
    assert any(p[0] == "mp3-artwork.png" and png_magic in p[1] for p in payloads)


def test_heuristics_and_non_flag_payloads(tmp_path: Path) -> None:
    # Frame text payload without flag pattern
    text_content = b"\x00" + b"Secret message hidden in plain sight"
    # Padding non-zero bytes without flag pattern
    non_flag_padding = b"\x00\x00\x00\x00" + b"\x41\x42\x43\x44" * 10
    mp3_data = make_id3v2([("COMM", text_content)], padding=non_flag_padding)

    # Frame private bits encoding plain ASCII text
    msg = b"Hidden ASCII message in private bits"
    bits = [int(b) for b in "".join(f"{byte:08b}" for byte in msg)]
    mp3_data += b"".join(make_mpeg_frame(private=bit) for bit in bits)

    # Appended non-flag trailer data
    trailer_bytes = b"\xde\xad\xbe\xef" * 32
    mp3_data += trailer_bytes

    mp3_path = tmp_path / "heuristics.mp3"
    mp3_path.write_bytes(mp3_data)

    res = AudioMp3().analyze(mp3_path)
    signals = {s.name: s for s in res.signals}
    assert "id3_frame_payload" in signals
    assert "id3_padding_payload" in signals
    assert "mp3_private_bit_payload" in signals
    assert "mp3_appended_data" in signals
    assert res.suspicion >= 70

    # Test extract_mp3_payloads with trailer
    payloads = extract_mp3_payloads(mp3_data)
    assert any(p[0] == "trailer.bin" for p in payloads)

    # Test ID3 with non-alphanumeric or corrupt frame ID
    corrupt_id3 = b"ID3\x03\x00\x00\x00\x00\x00\x20\xff\xfe\xfd\xfc\x00\x00\x00\x04\x00\x00test"
    parsed_corrupt = parse_id3v2(corrupt_id3)
    assert parsed_corrupt["padding"] != b""

