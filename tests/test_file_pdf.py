"""Tests for PDF steganalysis, stream/metadata extraction, and enhanced CTF decoders."""

from __future__ import annotations

import bz2
import lzma
import zlib
from pathlib import Path

import pytest

from core.ctf import CTFReport, CTFService, _decoded_candidates, _decompress, _trailer
from core.filetype import detect_type
from modules.file_pdf import (
    FilePdf,
    decompress_pdf_stream,
    extract_pdf_payloads,
    parse_pdf_comments,
    pdf_structural_end,
)
from modules.filestruct_appended import FilestructAppended


def make_minimal_pdf(
    body_objs: list[str] | None = None,
    comments: list[str] | None = None,
    trailer_dict: str = "",
    appended: bytes = b"",
) -> bytes:
    """Construct a syntactically valid minimal PDF."""
    pdf = bytearray(b"%PDF-1.4\n")
    if comments:
        for c in comments:
            pdf.extend(f"% {c}\n".encode())

    offsets: list[int] = [0]
    default_objs = [
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj",
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj",
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj",
    ]
    objs = default_objs + (body_objs or [])
    for obj_str in objs:
        offsets.append(len(pdf))
        pdf.extend(obj_str.encode("latin1") + b"\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("ascii"))
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("ascii"))

    t_dict = trailer_dict if trailer_dict else "<< /Size 4 /Root 1 0 R >>"
    pdf.extend(f"trailer\n{t_dict}\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii"))
    pdf.extend(appended)
    return bytes(pdf)


def test_filetype_detects_pdf(tmp_path: Path) -> None:
    pdf_file = tmp_path / "sample.pdf"
    pdf_file.write_bytes(make_minimal_pdf())
    dt = detect_type(pdf_file)
    assert dt.name == "pdf"
    assert dt.mime_type == "application/pdf"


def test_clean_pdf_has_zero_suspicion(tmp_path: Path) -> None:
    clean = tmp_path / "clean.pdf"
    clean.write_bytes(make_minimal_pdf())
    res = FilePdf().analyze(clean)
    assert res.status == "ok"
    assert res.suspicion == 0


def test_pdf_comment_flag_and_ctf(tmp_path: Path) -> None:
    flag = "flag{pdf_comment_hidden_channel}"
    pdf_data = make_minimal_pdf(comments=[f"Secret note: {flag}"])
    pdf_path = tmp_path / "comment.pdf"
    pdf_path.write_bytes(pdf_data)

    res = FilePdf().analyze(pdf_path)
    signals = {s.name: s for s in res.signals}
    assert "pdf_comment_flag" in signals
    assert signals["pdf_comment_flag"].score == 98
    assert signals["pdf_comment_flag"].evidence == "verified"

    # CTF automated solve
    out_dir = tmp_path / "ctf_out"
    report = CTFService().solve(pdf_path, out_dir, mode="quick")
    assert report.verdict == "confirmed"
    assert any("pdf-comment" in a.name for a in report.artifacts)


def test_pdf_metadata_flag_and_payload(tmp_path: Path) -> None:
    flag = "flag{pdf_metadata_title_channel}"
    body = [
        f"4 0 obj\n<< /Title ({flag}) /Author <666c61677b6865785f617574686f727d> /Subject (A" * 10
        + ") >>\nendobj"
    ]
    pdf_data = make_minimal_pdf(body_objs=body)
    pdf_path = tmp_path / "meta.pdf"
    pdf_path.write_bytes(pdf_data)

    res = FilePdf().analyze(pdf_path)
    signals = {s.name: s for s in res.signals}
    assert "pdf_metadata_flag" in signals
    assert signals["pdf_metadata_flag"].score == 98

    payloads = extract_pdf_payloads(pdf_data)
    assert any("pdf-meta-title.txt" in p[0] and p[3] for p in payloads)


def test_pdf_flatedecode_stream_flag_and_ctf(tmp_path: Path) -> None:
    flag = b"flag{pdf_flatedecode_stream_flag}"
    compressed = zlib.compress(b"BT /F1 12 Tf 72 712 Td (" + flag + b") Tj ET")
    stream_obj = (
        f"4 0 obj\n<< /Length {len(compressed)} /Filter /FlateDecode >>\nstream\n".encode("latin1")
        + compressed
        + b"\nendstream\nendobj"
    )

    pdf = bytearray(b"%PDF-1.4\n")
    pdf.extend(stream_obj + b"\n")
    pdf.extend(b"xref\n0 5\n0000000000 65535 f \ntrailer\n<< >>\nstartxref\n100\n%%EOF\n")

    pdf_path = tmp_path / "stream_flag.pdf"
    pdf_path.write_bytes(pdf)

    res = FilePdf().analyze(pdf_path)
    signals = {s.name: s for s in res.signals}
    assert "pdf_stream_flag" in signals
    assert signals["pdf_stream_flag"].score == 98

    # CTF extraction
    out_dir = tmp_path / "ctf_out_stream"
    report = CTFService().solve(pdf_path, out_dir, mode="quick")
    assert report.verdict == "confirmed"


def test_pdf_embedded_file_attachment(tmp_path: Path) -> None:
    attachment = b"secret data inside attachment"
    stream_obj = (
        b"4 0 obj\n<< /Type /EmbeddedFile /F (secret.txt) /Length "
        + str(len(attachment)).encode()
        + b" >>\nstream\n"
        + attachment
        + b"\nendstream\nendobj"
    )
    pdf = bytearray(b"%PDF-1.4\n")
    pdf.extend(stream_obj + b"\n")
    pdf.extend(b"xref\n0 5\n0000000000 65535 f \ntrailer\n<< >>\nstartxref\n100\n%%EOF\n")

    pdf_path = tmp_path / "attachment.pdf"
    pdf_path.write_bytes(pdf)

    res = FilePdf().analyze(pdf_path)
    signals = {s.name: s for s in res.signals}
    assert "pdf_embedded_file" in signals
    assert signals["pdf_embedded_file"].score == 90

    payloads = extract_pdf_payloads(pdf)
    assert any(p[0] == "secret.txt" and p[1] == attachment for p in payloads)


def test_pdf_incremental_updates(tmp_path: Path) -> None:
    pdf1 = make_minimal_pdf()
    pdf_revised = pdf1 + b"\n% Second revision\n%%EOF\n"
    pdf_path = tmp_path / "revisions.pdf"
    pdf_path.write_bytes(pdf_revised)

    res = FilePdf().analyze(pdf_path)
    signals = {s.name: s for s in res.signals}
    assert "pdf_incremental_updates" in signals
    assert signals["pdf_incremental_updates"].score == 50


def test_pdf_appended_trailer(tmp_path: Path) -> None:
    trailer_bytes = b"flag{pdf_appended_after_eof_marker}"
    pdf_data = make_minimal_pdf(appended=trailer_bytes)
    pdf_path = tmp_path / "trailer.pdf"
    pdf_path.write_bytes(pdf_data)

    res = FilePdf().analyze(pdf_path)
    signals = {s.name: s for s in res.signals}
    assert "pdf_appended_flag" in signals
    assert signals["pdf_appended_flag"].score == 98

    # filestruct_appended integration
    fs_res = FilestructAppended().analyze(pdf_path)
    assert fs_res.suspicion >= 70

    carved = _trailer(pdf_data, ".pdf")
    assert carved == trailer_bytes


def test_pdf_guards_and_helpers(tmp_path: Path) -> None:
    carrier = FilePdf()
    assert carrier.analyze(Path("nope.bin")).status == "unsupported"
    assert carrier.capacity(Path("any.pdf")) == 0
    with pytest.raises(NotImplementedError):
        carrier.embed(Path("a"), b"", Path("b"))
    with pytest.raises(NotImplementedError):
        carrier.extract(Path("a"))

    not_pdf = tmp_path / "file.txt"
    not_pdf.write_bytes(b"some text")
    assert carrier.analyze(not_pdf).status == "unsupported"

    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"not a valid pdf header")
    assert carrier.analyze(fake_pdf).status == "unsupported"

    assert pdf_structural_end(b"") is None
    assert pdf_structural_end(b"not pdf") is None
    assert pdf_structural_end(b"%PDF-1.4\nno eof here") is None

    # Test decoders
    assert decompress_pdf_stream(b"", b"/FlateDecode") is None
    assert decompress_pdf_stream(b"invalid", b"/FlateDecode") is None
    # ASCIIHexDecode (even and odd length)
    hex_stream = b"48656c6c6f>"
    assert decompress_pdf_stream(hex_stream, b"/ASCIIHexDecode") == b"Hello"
    odd_hex = b"48656c6c6f6>"
    assert decompress_pdf_stream(odd_hex, b"/ASCIIHexDecode") is not None
    # ASCII85Decode (with and without brackets)
    a85_stream = b"<~87cURDZ~>"
    assert decompress_pdf_stream(a85_stream, b"/ASCII85Decode") == b"Hello"
    assert decompress_pdf_stream(b"87cURDZ", b"/ASCII85Decode") == b"Hello"
    assert decompress_pdf_stream(b"bad", b"/Unknown") is None
    # FlateDecode with exceeded max_size
    large_compressed = zlib.compress(b"A" * 100)
    assert decompress_pdf_stream(large_compressed, b"/FlateDecode", max_size=10) is None

    # Test comments parser
    assert parse_pdf_comments(b"%PDF-1.4\n%%EOF\n% normal comment\n") == [b"normal comment"]


def test_pdf_heuristics_and_edge_cases(tmp_path: Path) -> None:
    # 1. Non-flag comment >= 16 chars
    comment = "This is a suspicious comment without flag pattern"
    # 2. Non-flag metadata >= 32 chars and invalid hex in Author
    body = [
        "4 0 obj\n<< /Title (A" * 20
        + ") /Author <INVALID_HEX> /Subject (Plain Subject) >>\nendobj"
    ]
    # 3. Non-flag appended trailer
    trailer = b"\xaa\xbb\xcc\xdd" * 16

    pdf_data = make_minimal_pdf(body_objs=body, comments=[comment], appended=trailer)
    pdf_path = tmp_path / "heuristics.pdf"
    pdf_path.write_bytes(pdf_data)

    res = FilePdf().analyze(pdf_path)
    signals = {s.name: s for s in res.signals}
    assert "pdf_comment_payload" in signals
    assert "pdf_metadata_payload" in signals
    assert "pdf_appended_data" in signals
    assert res.suspicion >= 60

    payloads = extract_pdf_payloads(pdf_data)
    assert any(p[0] == "trailer.bin" for p in payloads)
    assert any("pdf-comment-0.txt" in p[0] for p in payloads)



def test_enhanced_ctf_decoders() -> None:
    # 1. BZ2 decompression in _decompress
    bz2_data = bz2.compress(b"Hello from BZ2 world!")
    decompressed_bz2 = _decompress(bz2_data)
    assert decompressed_bz2 == b"Hello from BZ2 world!"

    # 2. LZMA/XZ decompression in _decompress
    xz_data = lzma.compress(b"Hello from XZ/LZMA world!")
    decompressed_xz = _decompress(xz_data)
    assert decompressed_xz == b"Hello from XZ/LZMA world!"

    # 3. Raw Deflate decompression in _decompress (wbits=-15)
    compressor = zlib.compressobj(wbits=-15)
    raw_deflate = compressor.compress(b"Hello raw deflate payload!") + compressor.flush()
    decompressed_raw = _decompress(raw_deflate)
    assert decompressed_raw == b"Hello raw deflate payload!"

    # 4. Hex candidate in _decoded_candidates
    hex_str = b"666c61677b6865785f6465636f6465645f737563636573737d"  # flag{hex_decoded_success}
    cands = _decoded_candidates(hex_str, deep=False)
    assert any(c[1] == "decoded-hex.bin" and b"flag{hex_decoded_success}" in c[0] for c in cands)

    # 5. Byte-reversed flag in _decoded_candidates
    flag = b"flag{reversed_flag_test}"
    rev_str = flag[::-1]
    cands_rev = _decoded_candidates(rev_str, deep=False)
    assert any(c[1] == "decoded-reversed.bin" and flag in c[0] for c in cands_rev)


def test_exif_metadata_ctf_recovery(tmp_path: Path) -> None:
    import piexif
    from PIL import Image

    img_path = tmp_path / "exif_flag.jpg"
    img = Image.new("RGB", (32, 32), color=(100, 100, 100))
    img.save(img_path)

    flag = b"flag{exif_artist_payload_found}"
    exif_dict = {"0th": {piexif.ImageIFD.Artist: flag}, "Exif": {}, "GPS": {}, "1st": {}}
    piexif.insert(piexif.dump(exif_dict), str(img_path))

    report: CTFReport = CTFService().solve(img_path, tmp_path / "exif_out", mode="quick")
    assert report.verdict == "confirmed"
    assert any("exif-metadata" in a.name for a in report.artifacts)
