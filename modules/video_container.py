"""Hiding places in video containers: MP4 unreferenced media data, MKV attachments/tags/voids,
AVI JUNK and unknown chunks.

- **MP4/MOV**: TCSteg-style tools hide a volume inside ``mdat`` and shift the
  chunk offsets so the video still plays. Every sample's byte range is rebuilt
  from ``stco``/``co64``, ``stsc`` and ``stsz``; a large share of ``mdat`` that
  no sample covers is flagged.
- **Matroska/WebM**: attachments other than fonts and cover images, binary tag
  values and non-zero ``Void`` padding.
- **AVI**: non-zero ``JUNK`` padding, unregistered top-level chunks and data
  after the RIFF size.

Parsing is bounded by element/box counts and never decodes media.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from core.analyzer import Analyzer
from core.result import AnalysisResult, Signal
from modules.container_hiding import encoded

MAX_ITEMS = 200_000
MIN_GAP = 64 * 1024
MIN_BLOB = 256
CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf", b"udta", b"mvex"}
BENIGN_ATTACHMENTS = ("font", "truetype", "opentype", "image/")
AVI_KNOWN = {b"LIST", b"JUNK", b"idx1", b"indx", b"ix00", b"ix01", b"RIFF"}

# Matroska element IDs
SEGMENT, ATTACHMENTS, ATTACHED_FILE, MIME, FILE_DATA = (0x18538067, 0x1941A469, 0x61A7,
                                                         0x4660, 0x465C)
TAGS, TAG, SIMPLE_TAG, TAG_BINARY, VOID = 0x1254C367, 0x7373, 0x67C8, 0x4485, 0xEC
MKV_DESCEND = {SEGMENT, ATTACHMENTS, ATTACHED_FILE, TAGS, TAG, SIMPLE_TAG}


def _signal(name: str, blob: str | None, detail: str, base: int = 55) -> Signal:
    return Signal(name, 72 if blob else base, detail + (f"; looks like {blob}" if blob else ""),
                  category="container_hiding", evidence="strong" if blob else "heuristic")


# --- MP4 -------------------------------------------------------------------------------

def boxes(data: bytes, start: int, end: int) -> Iterator[tuple[bytes, int, int]]:
    """(type, body start, body end) of the boxes in ``data[start:end]``."""
    pos = start
    while pos + 8 <= end:
        size, header = int.from_bytes(data[pos : pos + 4], "big"), 8
        kind = data[pos + 4 : pos + 8]
        if size == 1 and pos + 16 <= end:
            size, header = int.from_bytes(data[pos + 8 : pos + 16], "big"), 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            return
        yield kind, pos + header, pos + size
        pos += size


def _walk(data: bytes, start: int, end: int, budget: list[int]) -> Iterator[tuple[bytes, int, int]]:
    for kind, body, stop in boxes(data, start, end):
        budget[0] -= 1
        if budget[0] < 0:
            return
        yield kind, body, stop
        if kind in CONTAINERS:
            yield from _walk(data, body, stop, budget)


def _table(data: bytes, body: int, stop: int, width: int, fields: int) -> list[tuple[int, ...]]:
    count = int.from_bytes(data[body + 4 : body + 8], "big")
    pos, rows = body + 8, []
    for _ in range(min(count, MAX_ITEMS)):
        if pos + width * fields > stop:
            break
        rows.append(tuple(int.from_bytes(data[pos + i * width : pos + (i + 1) * width], "big")
                          for i in range(fields)))
        pos += width * fields
    return rows


def sample_ranges(data: bytes) -> list[tuple[int, int]]:
    """Byte ranges of every sample of every track, from the sample tables."""
    ranges: list[tuple[int, int]] = []
    tables: dict[bytes, tuple[int, int]] = {}
    for kind, body, stop in _walk(data, 0, len(data), [MAX_ITEMS]):
        if kind == b"trak":
            ranges += _track(data, tables)
            tables = {}
        elif kind in (b"stco", b"co64", b"stsc", b"stsz"):
            tables[kind] = (body, stop)
    return ranges + _track(data, tables)


def _track(data: bytes, tables: dict[bytes, tuple[int, int]]) -> list[tuple[int, int]]:
    if b"stsz" not in tables or b"stsc" not in tables:
        return []
    if b"stco" in tables:
        offsets = [r[0] for r in _table(data, *tables[b"stco"], 4, 1)]
    elif b"co64" in tables:
        offsets = [r[0] for r in _table(data, *tables[b"co64"], 8, 1)]
    else:
        return []
    body, stop = tables[b"stsz"]
    fixed = int.from_bytes(data[body + 4 : body + 8], "big")
    count = int.from_bytes(data[body + 8 : body + 12], "big")
    sizes = ([fixed] * min(count, MAX_ITEMS) if fixed else
             [r[0] for r in _table(data, body + 4, stop, 4, 1)])
    runs = _table(data, *tables[b"stsc"], 4, 3)
    out, sample = [], 0
    for index, offset in enumerate(offsets, start=1):
        per_chunk = next((r[1] for r in reversed(runs) if r[0] <= index), 0)
        length = sum(sizes[sample : sample + per_chunk])
        out.append((offset, offset + length))
        sample += per_chunk
    return out


def mp4_signals(data: bytes) -> list[Signal]:
    mdats = [(body, stop) for kind, body, stop in boxes(data, 0, len(data)) if kind == b"mdat"]
    if not mdats:
        return []
    ranges = sorted(sample_ranges(data))
    if not ranges:
        return []
    total = sum(stop - body for body, stop in mdats)
    covered = 0
    for body, stop in mdats:
        cursor = body
        for start, end in ranges:
            start, end = max(start, cursor), min(end, stop)
            if end > start:
                covered += end - start
                cursor = end
    gap = total - covered
    if gap >= MIN_GAP and gap >= 0.05 * total:
        return [Signal("mp4_unreferenced_media_data", 80,
                       f"{gap} of {total} mdat bytes are not referenced by any sample "
                       "(TCSteg-style hidden volume)", category="container_hiding",
                       evidence="strong")]
    return []


# --- Matroska --------------------------------------------------------------------------

def _vint(data: bytes, pos: int, keep_marker: bool) -> tuple[int, int] | None:
    if pos >= len(data) or data[pos] == 0:
        return None
    length = 8 - data[pos].bit_length() + 1
    if pos + length > len(data):
        return None
    value = int.from_bytes(data[pos : pos + length], "big")
    if not keep_marker:
        value &= (1 << (7 * length)) - 1
        if value == (1 << (7 * length)) - 1:
            value = -1  # unknown size
    return value, pos + length


def ebml(data: bytes, start: int, end: int) -> Iterator[tuple[int, int, int]]:
    pos = start
    while pos < end:
        head = _vint(data, pos, True)
        size = head and _vint(data, head[1], False)
        if not head or not size:
            return
        stop = end if size[0] < 0 else min(end, size[1] + size[0])
        yield head[0], size[1], stop
        pos = stop


def mkv_signals(data: bytes) -> list[Signal]:
    out: list[Signal] = []
    budget = MAX_ITEMS

    def walk(start: int, end: int, mime: list[str]) -> None:
        nonlocal budget
        for element, body, stop in ebml(data, start, end):
            budget -= 1
            if budget < 0:
                return
            if element == ATTACHED_FILE:
                found: list[str] = []
                walk(body, stop, found)
                kind = found[0] if found else "unknown"
                if not any(b in kind for b in BENIGN_ATTACHMENTS):
                    out.append(Signal("mkv_attachment", 70, f"attached file of type {kind!r}, "
                                      f"{stop - body} bytes", category="container_hiding",
                                      evidence="strong"))
            elif element in MKV_DESCEND:
                walk(body, stop, mime)
            elif element == MIME:
                mime.append(data[body:stop].decode("latin-1").lower())
            elif element == TAG_BINARY and stop - body >= MIN_BLOB:
                blob = data[body:stop]
                out.append(_signal("mkv_binary_tag", encoded(blob),
                                   f"binary tag value of {len(blob)} bytes"))
            elif element == VOID and data[body:stop].strip(b"\x00") and stop - body >= MIN_BLOB:
                out.append(_signal("mkv_void_payload", encoded(data[body:stop]),
                                   f"Void element with {stop - body} non-zero bytes", base=50))

    walk(0, len(data), [])
    return out


# --- AVI -------------------------------------------------------------------------------

def avi_signals(data: bytes) -> list[Signal]:
    out: list[Signal] = []
    riff_end = 8 + int.from_bytes(data[4:8], "little")
    if len(data) > riff_end:
        out.append(Signal("avi_trailing_data", 70, f"{len(data) - riff_end} bytes after RIFF",
                          category="appended_data", evidence="strong"))
    pos, items = 12, 0
    while pos + 8 <= min(riff_end, len(data)) and items < MAX_ITEMS:
        kind = data[pos : pos + 4]
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        body = data[pos + 8 : pos + 8 + size]
        items += 1
        if kind == b"JUNK" and body.strip(b"\x00") and size >= MIN_BLOB:
            out.append(_signal("avi_junk_payload", encoded(body),
                               f"JUNK chunk with {size} non-zero bytes", base=50))
        elif kind not in AVI_KNOWN and size >= MIN_BLOB:
            out.append(_signal("avi_unknown_chunk", encoded(body),
                               f"unregistered chunk {kind.decode('latin-1')!r} of {size} bytes"))
        pos += 8 + size + (size & 1)
    return out


class VideoContainerAnalyzer(Analyzer):
    """Hidden data in MP4/MOV, Matroska/WebM and AVI container structure."""

    name = "video_container"
    max_bytes = 2 * 1024 * 1024 * 1024

    def analyze(self, src: Path) -> AnalysisResult:
        if src.stat().st_size > self.max_bytes:
            return AnalysisResult(self.name, 0, (), None, status="unsupported")
        data = src.read_bytes()
        if data[4:8] in (b"ftyp", b"moov", b"mdat", b"free", b"wide"):
            found = mp4_signals(data)
        elif data.startswith(b"\x1a\x45\xdf\xa3"):
            found = mkv_signals(data)
        elif data.startswith(b"RIFF") and data[8:12] == b"AVI ":
            found = avi_signals(data)
        else:
            return AnalysisResult(self.name, 0, (), None, status="unsupported")
        return AnalysisResult(self.name, max((s.score for s in found), default=0),
                              tuple(found), None)
