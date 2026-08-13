from pathlib import Path

import pytest

from core.payload import pack
from modules.filestruct_trailer import FilestructTrailer
from modules.image_lsb_scatter import ImageLsbScatter


def test_pdf_and_gif_trailer_roundtrip_and_analysis(tmp_path: Path):
    carrier = FilestructTrailer()
    payload = pack(
        payload=b"trailer secret",
        encrypted=False,
        salt=b"\x00" * 16,
        nonce=b"\x00" * 12,
    )
    for name, content in (
        ("cover.pdf", b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n%%EOF\n"),
        ("cover.gif", b"GIF89a" + b"\x00" * 20 + b";"),
    ):
        cover = tmp_path / name
        cover.write_bytes(content)
        out = tmp_path / f"out-{name}"
        carrier.embed(cover, payload, out)
        assert carrier.extract(out) == payload
        result = carrier.analyze(out)
        assert result.suspicion >= 95
        assert result.signals[0].evidence == "verified"


def test_trailer_invalid_files_and_tamper(tmp_path: Path):
    carrier = FilestructTrailer()
    cover = tmp_path / "bad.pdf"
    cover.write_bytes(b"not a PDF")
    with pytest.raises(ValueError, match="end marker"):
        carrier.embed(cover, b"x", tmp_path / "out.pdf")
    clean = tmp_path / "clean.pdf"
    clean.write_bytes(b"%PDF-1.4\n%%EOF")
    assert carrier.analyze(clean).suspicion == 0
    out = tmp_path / "stego.pdf"
    carrier.embed(clean, b"payload", out)
    data = bytearray(out.read_bytes())
    data[-1] ^= 1
    out.write_bytes(data)
    assert carrier.analyze(out).suspicion == 80
    with pytest.raises(ValueError, match="integrity"):
        carrier.extract(out)


def test_scatter_roundtrip_wrong_key_and_marker(
    png_64x64: Path, tmp_path: Path
):
    carrier = ImageLsbScatter()
    payload = b"keyed payload" * 3
    out = tmp_path / "scatter.png"
    carrier.embed_with_options(
        png_64x64,
        payload,
        out,
        steg_key="placement-key",
        options={"channels": "rb"},
    )
    assert carrier.extract_with_options(out, steg_key="placement-key") == payload
    assert carrier.analyze(out).suspicion >= 95
    assert carrier.capacity(out) > len(payload)
    assert carrier.extract_with_options(out, steg_key="wrong") != payload
    with pytest.raises(ValueError, match="requires"):
        carrier.embed(png_64x64, payload, out)
    with pytest.raises(ValueError, match="requires"):
        carrier.extract(out)


def test_scatter_rejects_missing_key_channels_and_capacity(
    png_64x64: Path, tmp_path: Path
):
    carrier = ImageLsbScatter()
    with pytest.raises(ValueError, match="requires"):
        carrier.embed_with_options(png_64x64, b"x", tmp_path / "x.png")
    with pytest.raises(ValueError, match="channels"):
        carrier.embed_with_options(
            png_64x64,
            b"x",
            tmp_path / "x.png",
            steg_key="k",
            options={"channels": "x"},
        )
    with pytest.raises(Exception, match="capacity"):
        carrier.embed_with_options(
            png_64x64,
            b"x" * 10000,
            tmp_path / "x.png",
            steg_key="k",
            options={"channels": "r"},
        )
