"""Payloads hidden in container structure: PNG chunks, JPEG APPn/COM, ZIP fields, MP4 boxes.

Each format is walked with hard limits and nothing is executed or extracted to
disk. A structure is flagged when it is unknown to the format, or when a free-form
field carries an unusually large, encoded-looking (base64/hex/high-entropy) blob.
Well-known large metadata (ICC, XMP, EXIF, ImageMagick raw profiles) is exempt.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from pathlib import Path

from core.analyzer import Analyzer
from core.result import AnalysisResult, Signal

MAX_ITEMS = 10_000
MAX_INFLATE = 1 << 20
MIN_UNKNOWN = 64
MIN_FREEFORM = 256
_BASE64 = re.compile(rb"^[A-Za-z0-9+/=\s]+$")
_HEX = re.compile(rb"^[0-9A-Fa-f\s]+$")

PNG_KNOWN = {
    b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS", b"cHRM", b"gAMA", b"iCCP", b"sBIT", b"sRGB",
    b"cICP", b"mDCV", b"cLLI", b"tEXt", b"zTXt", b"iTXt", b"bKGD", b"hIST", b"pHYs", b"sPLT",
    b"eXIf", b"tIME", b"acTL", b"fcTL", b"fdAT", b"oFFs", b"pCAL", b"sCAL", b"sTER", b"gIFg",
    b"gIFx", b"dSIG", b"iDOT", b"CgBI", b"vpAg", b"caNv", b"orNT", b"exIf",
}
PNG_EXEMPT_KEYS = (b"XML:com.adobe.xmp", b"Raw profile type", b"icc", b"ICC")
JPEG_KNOWN_APP = (
    b"JFIF\x00", b"JFXX\x00", b"Exif\x00", b"http://ns.adobe.com/", b"ICC_PROFILE\x00",
    b"FPXR\x00", b"MPF\x00", b"Photoshop 3.0\x00", b"Adobe", b"Ducky", b"Meta", b"JP",
    b"PrintIM", b"HPQ", b"AROT", b"GoPro", b"XMP\x00", b"QVCI", b"FLIR", b"Ocad",
)
ZIP_KNOWN_EXTRA = {0x0001, 0x0007, 0x0008, 0x0009, 0x000A, 0x000C, 0x000D, 0x000E, 0x000F,
                   0x0014, 0x0015, 0x0016, 0x0017, 0x0018, 0x0019, 0x0020, 0x0021, 0x0022,
                   0x0023, 0x0065, 0x0066, 0x4154, 0x5455, 0x5855, 0x6375, 0x7075, 0x7855,
                   0x7875, 0x9901, 0xA220, 0xCAFE, 0xD935}
MP4_TOP = {b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide", b"uuid", b"meta", b"pdin",
           b"moof", b"mfra", b"styp", b"sidx", b"ssix", b"prft", b"emsg", b"pnot", b"junk"}


def entropy(data: bytes) -> float:
    """Shannon entropy in bits per byte."""
    if not data:
        return 0.0
    total = len(data)
    return -sum(c / total * math.log2(c / total) for c in Counter(data).values())


def encoded(data: bytes) -> str | None:
    """Describe ``data`` if it looks like an encoded or encrypted blob."""
    stripped = data.strip(b"\x00")
    if len(stripped) < MIN_FREEFORM:
        return None
    if _HEX.match(stripped) and len(stripped.split()) <= len(stripped) // 16:
        return "hex text"
    mixed = re.search(rb"[0-9]", stripped) and re.search(rb"[A-Z]", stripped)
    if _BASE64.match(stripped) and mixed:
        return "base64 text"
    if entropy(stripped) >= 7.2:
        return "high-entropy bytes"
    return None


def _signal(name: str, kind: str | None, detail: str, category: str,
            base: int = 55) -> Signal:
    score = 72 if kind else base
    note = f"; looks like {kind}" if kind else ""
    return Signal(name, score, detail + note, category=category,
                  evidence="strong" if kind else "heuristic")


def png_signals(data: bytes) -> list[Signal]:
    out: list[Signal] = []
    pos, items, bad_crc = 8, 0, 0
    while pos + 12 <= len(data) and items < MAX_ITEMS:
        length = int.from_bytes(data[pos : pos + 4], "big")
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if len(body) < length or pos + 12 + length > len(data):
            break
        crc = int.from_bytes(data[pos + 8 + length : pos + 12 + length], "big")
        bad_crc += zlib.crc32(kind + body) & 0xFFFFFFFF != crc
        items += 1
        pos += 12 + length
        if kind == b"IEND":
            break
        if kind in (b"tEXt", b"zTXt", b"iTXt"):
            out.extend(_png_text(kind, body))
        elif kind not in PNG_KNOWN and length >= MIN_UNKNOWN:
            out.append(_signal("png_unknown_chunk", encoded(body),
                               f"unregistered chunk {kind.decode('latin-1')!r} of {length} bytes",
                               "container_hiding"))
    if bad_crc:
        out.append(Signal("png_chunk_crc_mismatch", 40, f"{bad_crc} chunk(s) with a wrong CRC",
                          category="container_structure", evidence="heuristic"))
    return out


def _inflate(blob: bytes) -> bytes:
    decompressor = zlib.decompressobj()
    try:
        return decompressor.decompress(blob, MAX_INFLATE)
    except zlib.error:
        return b""


def _png_text(kind: bytes, body: bytes) -> list[Signal]:
    key, _, rest = body.partition(b"\x00")
    if any(key.startswith(k) for k in PNG_EXEMPT_KEYS):
        return []
    if kind == b"zTXt":
        text = _inflate(rest[1:])
    elif kind == b"iTXt":
        flag, rest = rest[:1], rest[2:]
        rest = rest.split(b"\x00", 2)[-1]
        text = _inflate(rest) if flag == b"\x01" else rest
    else:
        text = rest
    blob = encoded(text)
    if blob or len(text) >= 4096:
        return [_signal("png_text_payload", blob,
                        f"{kind.decode()} chunk {key[:40].decode('latin-1')!r} holds "
                        f"{len(text)} bytes", "container_hiding", base=50)]
    return []


def jpeg_signals(data: bytes) -> list[Signal]:
    out: list[Signal] = []
    pos, items, comments = 2, 0, b""
    while pos + 4 <= len(data) and items < MAX_ITEMS and data[pos] == 0xFF:
        marker = data[pos + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        if marker in (0xDA, 0xD9):
            break
        length = int.from_bytes(data[pos + 2 : pos + 4], "big")
        body = data[pos + 4 : pos + 2 + length]
        items += 1
        pos += 2 + length
        if 0xE0 <= marker <= 0xEF and len(body) >= MIN_UNKNOWN:
            if not any(body.startswith(k) for k in JPEG_KNOWN_APP):
                out.append(_signal("jpeg_unknown_app_segment", encoded(body),
                                   f"APP{marker - 0xE0} segment of {len(body)} bytes with an "
                                   "unrecognised identifier", "container_hiding"))
        elif marker == 0xFE:
            comments += body
    blob = encoded(comments)
    if blob:
        out.append(_signal("jpeg_comment_payload", blob, f"{len(comments)} bytes of COM text",
                           "container_hiding"))
    return out


def zip_signals(data: bytes) -> list[Signal]:
    out: list[Signal] = []
    end = data.rfind(b"PK\x05\x06", max(0, len(data) - 65_557))
    if end < 0 or end + 22 > len(data):
        return out
    count = int.from_bytes(data[end + 10 : end + 12], "little")
    offset = int.from_bytes(data[end + 16 : end + 20], "little")
    comment = data[end + 22 : end + 22 + int.from_bytes(data[end + 20 : end + 22], "little")]
    if len(comment) >= 32:
        out.append(_signal("zip_comment_payload", encoded(comment),
                           f"archive comment of {len(comment)} bytes", "container_hiding",
                           base=45))
    pos, odd_extra, member_comments = offset, 0, b""
    for _ in range(min(count, MAX_ITEMS)):
        if data[pos : pos + 4] != b"PK\x01\x02" or pos + 46 > len(data):
            break
        name_len, extra_len, comment_len = (int.from_bytes(data[pos + i : pos + i + 2], "little")
                                            for i in (28, 30, 32))
        extra = data[pos + 46 + name_len : pos + 46 + name_len + extra_len]
        cursor = 0
        while cursor + 4 <= len(extra):
            header = int.from_bytes(extra[cursor : cursor + 2], "little")
            size = int.from_bytes(extra[cursor + 2 : cursor + 4], "little")
            if header not in ZIP_KNOWN_EXTRA and size >= MIN_UNKNOWN:
                odd_extra += size
            cursor += 4 + size
        start = pos + 46 + name_len + extra_len
        member_comments += data[start : start + comment_len]
        pos = start + comment_len
    if odd_extra:
        out.append(Signal("zip_unknown_extra_field", 55,
                          f"{odd_extra} bytes in unregistered extra fields",
                          category="container_hiding", evidence="heuristic"))
    blob = encoded(member_comments)
    if blob:
        out.append(_signal("zip_member_comment_payload", blob,
                           f"{len(member_comments)} bytes of member comments", "container_hiding"))
    return out


def mp4_signals(data: bytes) -> list[Signal]:
    out: list[Signal] = []
    pos, items = 0, 0
    while pos + 8 <= len(data) and items < MAX_ITEMS:
        size = int.from_bytes(data[pos : pos + 4], "big")
        kind = data[pos + 4 : pos + 8]
        header = 8
        if size == 1 and pos + 16 <= len(data):
            size, header = int.from_bytes(data[pos + 8 : pos + 16], "big"), 16
        elif size == 0:
            size = len(data) - pos
        if size < header or pos + size > len(data):
            out.append(Signal("mp4_trailing_data", 70,
                              f"{len(data) - pos} bytes after the last complete box",
                              category="appended_data", evidence="strong"))
            break
        body = data[pos + header : pos + size]
        items += 1
        pos += size
        if kind in (b"free", b"skip", b"junk", b"wide") and len(body) >= MIN_FREEFORM:
            blob = encoded(body)
            if blob or body.strip(b"\x00"):
                out.append(_signal("mp4_free_box_payload", blob,
                                   f"{kind.decode()} box with {len(body)} non-zero bytes",
                                   "container_hiding", base=50))
        elif kind not in MP4_TOP and len(body) >= MIN_UNKNOWN:
            out.append(_signal("mp4_unknown_box", encoded(body),
                               f"unregistered top-level box {kind.decode('latin-1')!r} of "
                               f"{len(body)} bytes", "container_hiding"))
    return out


class ContainerHidingAnalyzer(Analyzer):
    """Flag payload-sized blobs in PNG, JPEG, ZIP and MP4/MOV structures."""

    name = "container_hiding"
    max_bytes = 512 * 1024 * 1024

    def analyze(self, src: Path) -> AnalysisResult:
        if src.stat().st_size > self.max_bytes:
            return AnalysisResult(self.name, 0, (), None, status="unsupported")
        data = src.read_bytes()
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            signals = png_signals(data)
        elif data.startswith(b"\xff\xd8"):
            signals = jpeg_signals(data)
        elif data[4:8] in (b"ftyp", b"moov", b"mdat", b"free", b"wide"):
            signals = mp4_signals(data)
        elif data.startswith(b"PK\x03\x04") or data.startswith(b"PK\x05\x06"):
            signals = zip_signals(data)
        else:
            return AnalysisResult(self.name, 0, (), None, status="unsupported")
        suspicion = max((s.score for s in signals), default=0)
        return AnalysisResult(self.name, suspicion, tuple(signals), None)
