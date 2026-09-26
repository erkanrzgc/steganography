import base64
import zlib

import pytest

from core.ctf import CTFLimits, CTFService
from modules import file_pdf


def stream_object(index, payload):
    return (
        str(index).encode()
        + b" 0 obj\n<< /Filter /FlateDecode >>\nstream\n"
        + zlib.compress(payload)
        + b"\nendstream\nendobj\n"
    )


@pytest.mark.parametrize("size", [0, 1, 2, 3, 4, 5, 16, 31, 64])
@pytest.mark.parametrize("method", ["hex", "a85", "flate"])
def test_all_decoders_enforce_exact_output_boundary(size, method):
    payload = bytes(range(size))
    if method == "hex":
        encoded = payload.hex().encode() + b">"
        filter_name = b"/ASCIIHexDecode"
    elif method == "a85":
        encoded = base64.a85encode(payload, adobe=True)
        filter_name = b"/ASCII85Decode"
    else:
        encoded = zlib.compress(payload)
        filter_name = b"/FlateDecode"
    assert file_pdf.decompress_pdf_stream(encoded, filter_name, max_size=size) == payload
    if size:
        assert file_pdf.decompress_pdf_stream(encoded, filter_name, max_size=size - 1) is None


def test_ascii85_zero_runs_are_rejected_before_decoder_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("oversized ASCII85 output reached allocating decoder")

    monkeypatch.setattr(file_pdf.base64, "a85decode", forbidden)
    assert file_pdf.decompress_pdf_stream(b"zz", b"ASCII85Decode", max_size=7) is None
    assert file_pdf.decompress_pdf_stream(b"<~zz~>", b"ASCII85Decode", max_size=7) is None
    assert file_pdf.decompress_pdf_stream(b"z" * 300, b"ASCII85Decode", max_size=1) is None
    assert file_pdf.decompress_pdf_stream(b"!", b"ASCII85Decode", max_size=1) is None


def test_pdf_decoder_malformed_whitespace_and_odd_nibbles():
    assert file_pdf.decompress_pdf_stream(b"\x00 4\n1\t4>", b"ASCIIHexDecode", max_size=2) == b"A@"
    assert file_pdf.decompress_pdf_stream(b"414>", b"ASCIIHexDecode", max_size=1) is None
    assert file_pdf.decompress_pdf_stream(b"41\xff42>", b"ASCIIHexDecode", max_size=2) is None
    assert file_pdf.decompress_pdf_stream(b"41>>", b"ASCIIHexDecode", max_size=2) is None
    assert (
        file_pdf.decompress_pdf_stream(b" " * 300 + b"41>", b"ASCIIHexDecode", max_size=1) is None
    )
    assert (
        file_pdf.decompress_pdf_stream(b"<~\x00z \t~>", b"ASCII85Decode", max_size=4) == b"\x00" * 4
    )
    assert file_pdf.decompress_pdf_stream(b"!z!!!", b"ASCII85Decode", max_size=8) is None
    assert file_pdf.decompress_pdf_stream(b"vvvvv", b"ASCII85Decode", max_size=4) is None
    compressed = zlib.compress(b"complete output")
    assert file_pdf.decompress_pdf_stream(compressed[:-1], b"FlateDecode") is None
    assert file_pdf.decompress_pdf_stream(compressed + b"trailing junk", b"FlateDecode") is None
    with pytest.raises(ValueError, match="nonnegative"):
        file_pdf.decompress_pdf_stream(b"", b"FlateDecode", max_size=-1)


def test_multi_stream_budget_is_shared_and_raw_fallback_marked():
    data = b"%PDF-1.4\n" + b"".join(
        stream_object(i, value) for i, value in enumerate((b"abcd", b"12345", b"z"))
    )
    streams = file_pdf.parse_pdf_streams(data, max_decoded_bytes=5)
    assert [s["decode_status"] for s in streams] == ["decoded", "unavailable", "decoded"]
    assert sum(len(s["content"]) for s in streams if s["decompressed"]) == 5
    assert streams[1]["content"] == streams[1]["raw_stream"]
    assert len(file_pdf.parse_pdf_streams(data, max_streams=1)) == 1
    assert not any(s["decompressed"] for s in file_pdf.parse_pdf_streams(data, max_decoded_bytes=0))
    for kwargs in ({"max_streams": -1}, {"max_decoded_bytes": -1}):
        with pytest.raises(ValueError, match="nonnegative"):
            file_pdf.parse_pdf_streams(data, **kwargs)


def test_extraction_and_ctf_forward_remaining_budget(tmp_path, monkeypatch):
    payload = b"flag{decoded_stream}"
    data = b"%PDF-1.4\n" + stream_object(1, payload) + b"%%EOF\n"
    assert any(
        p[1] == payload for p in file_pdf.extract_pdf_payloads(data, max_decoded_bytes=len(payload))
    )
    assert not file_pdf.extract_pdf_payloads(data, max_decoded_bytes=0)
    seen = []
    original = file_pdf.extract_pdf_payloads

    def tracked(data, *, max_decoded_bytes):
        seen.append(max_decoded_bytes)
        return original(data, max_decoded_bytes=max_decoded_bytes)

    monkeypatch.setattr(file_pdf, "extract_pdf_payloads", tracked)
    source = tmp_path / "challenge.pdf"
    source.write_bytes(data)
    report = CTFService().solve(
        source, tmp_path / "job", mode="quick", limits=CTFLimits(max_bytes=4096)
    )
    assert seen and 0 <= seen[0] <= 4096
    assert report.verdict == "confirmed"
