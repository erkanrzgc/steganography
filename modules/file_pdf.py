"""PDF carrier and steganalysis module.

Detects hidden comments, metadata payloads, decompressable object streams,
embedded file attachments, incremental revision shadows, and structural trailer
carving without external dependencies.
"""
from __future__ import annotations

import base64
import re
import zlib
from pathlib import Path
from typing import Any

from core.carrier import Carrier
from core.context import AnalysisContext
from core.result import AnalysisResult, EmbedResult, Signal

_FLAG_PATTERN = re.compile(rb"([a-zA-Z0-9_]{3,24}\{[ -~]{4,120}\})")
_OBJ_PATTERN = re.compile(rb"(\d+)\s+(\d+)\s+obj\b(.*?)endobj", re.DOTALL)
_STREAM_PATTERN = re.compile(rb"<<(.*?)>>\s*stream[\r\n]+(.*?)[\r\n]+endstream", re.DOTALL)
_COMMENT_PATTERN = re.compile(rb"(?m)^%([^\r\n]*)")
_META_KEYS = (b"Title", b"Author", b"Subject", b"Keywords", b"Creator", b"Producer")


def pdf_structural_end(data: bytes) -> int | None:
    """Return the structural end offset of a PDF after the last %%EOF and trailing whitespace."""
    if not data or not data.startswith(b"%PDF-"):
        return None
    idx = data.rfind(b"%%EOF")
    if idx < 0:
        return None
    end = idx + len(b"%%EOF")
    while end < len(data) and data[end : end + 1] in b" \t\r\n":
        end += 1
    return end


def parse_pdf_comments(data: bytes) -> list[bytes]:
    """Extract comment lines from PDF, skipping %PDF- header and %%EOF markers."""
    comments: list[bytes] = []
    for match in _COMMENT_PATTERN.finditer(data):
        line = match.group(1).strip()
        if not line or line.startswith(b"PDF-") or line.startswith(b"%EOF"):
            continue
        comments.append(line)
    return comments


def decompress_pdf_stream(
    stream_data: bytes, filter_name: bytes, max_size: int = 16 * 1024 * 1024
) -> bytes | None:
    """Decompress a PDF stream with bounded memory usage."""
    if not stream_data:
        return None
    try:
        if b"FlateDecode" in filter_name:
            decompressor = zlib.decompressobj()
            decompressed = decompressor.decompress(stream_data, max_size + 1)
            if len(decompressed) <= max_size:
                return decompressed
            return None
        if b"ASCIIHexDecode" in filter_name:
            clean = re.sub(rb"\s+", b"", stream_data).rstrip(b">")
            if len(clean) % 2 != 0:
                clean += b"0"
            return bytes.fromhex(clean.decode("ascii", errors="ignore"))
        if b"ASCII85Decode" in filter_name:
            clean = re.sub(rb"\s+", b"", stream_data)
            if not clean.startswith(b"<~"):
                clean = b"<~" + clean
            if not clean.endswith(b"~>"):
                clean = clean + b"~>"
            return base64.a85decode(clean, adobe=True)
    except Exception:  # noqa: BLE001
        return None
    return None


def parse_pdf_streams(
    data: bytes, max_streams: int = 200
) -> list[dict[str, Any]]:
    """Parse object streams from PDF data, extracting metadata and decoded content."""
    results: list[dict[str, Any]] = []
    count = 0
    for match in _OBJ_PATTERN.finditer(data):
        if count >= max_streams:
            break
        obj_id = f"{match.group(1).decode('ascii')}_{match.group(2).decode('ascii')}"
        body = match.group(3)
        stream_match = _STREAM_PATTERN.search(body)
        if not stream_match:
            continue
        dict_part = stream_match.group(1)
        raw_stream = stream_match.group(2)

        # Detect filter
        filter_match = re.search(rb"/Filter\s*(?:/[A-Za-z0-9]+|\[\s*/[A-Za-z0-9]+\s*\])", dict_part)
        filter_name = filter_match.group(0) if filter_match else b""

        # Detect embedded file specification
        is_embedded = bool(
            re.search(rb"/(?:EmbeddedFile|EF|FileSpec)\b", dict_part)
            or re.search(rb"/Type\s*/EmbeddedFile", dict_part)
        )
        fname_match = re.search(rb"/F\s*\(([^)]+)\)", dict_part)
        filename = fname_match.group(1).decode("latin1", errors="replace") if fname_match else None

        decompressed: bytes | None = None
        if filter_name:
            decompressed = decompress_pdf_stream(raw_stream, filter_name)

        content = decompressed if decompressed is not None else raw_stream
        results.append({
            "obj_id": obj_id,
            "dict": dict_part,
            "filter": filter_name.decode("latin1", errors="replace"),
            "raw_stream": raw_stream,
            "decompressed": decompressed is not None,
            "content": content,
            "is_embedded": is_embedded,
            "filename": filename,
        })
        count += 1
    return results


def parse_pdf_metadata(data: bytes) -> dict[str, str]:
    """Extract metadata text fields from PDF /Info or object dictionaries."""
    metadata: dict[str, str] = {}
    for key in _META_KEYS:
        # Search for /Key (Value) or /Key <hex>
        m_lit = re.search(rb"/" + key + rb"\s*\(([^)]+)\)", data)
        if m_lit:
            metadata[key.decode("ascii")] = m_lit.group(1).decode("latin1", errors="replace")
            continue
        m_hex = re.search(rb"/" + key + rb"\s*<([0-9a-fA-F]+)>", data)
        if m_hex:
            try:
                decoded = bytes.fromhex(m_hex.group(1).decode("ascii")).decode(
                    "latin1", errors="replace"
                )
                metadata[key.decode("ascii")] = decoded
            except Exception:  # noqa: BLE001, S110
                pass
    return metadata


def extract_pdf_payloads(data: bytes) -> list[tuple[str, bytes, str, bool]]:
    """Extract candidate CTF payloads from PDF comments, streams, metadata, and trailer."""
    candidates: list[tuple[str, bytes, str, bool]] = []

    # 1. Comments
    comments = parse_pdf_comments(data)
    for idx, c in enumerate(comments):
        is_flag = bool(_FLAG_PATTERN.search(c))
        if is_flag or (len(c) >= 6 and any(32 <= b <= 126 for b in c)):
            candidates.append((
                f"pdf-comment-{idx}.txt",
                c,
                "PDF comment line payload",
                is_flag,
            ))

    # 2. Metadata
    meta = parse_pdf_metadata(data)
    for k, val in meta.items():
        bval = val.encode("latin1", errors="replace")
        is_flag = bool(_FLAG_PATTERN.search(bval))
        if is_flag or len(bval) >= 6:
            candidates.append((
                f"pdf-meta-{k.lower()}.txt",
                bval,
                f"PDF metadata /{k} payload",
                is_flag,
            ))

    # 3. Object Streams & Embedded Files
    streams = parse_pdf_streams(data)
    for stream in streams:
        content = stream["content"]
        obj_id = stream["obj_id"]
        is_flag = bool(_FLAG_PATTERN.search(content))
        if stream["is_embedded"]:
            fname = stream["filename"] or f"embedded-{obj_id}.bin"
            candidates.append((
                fname,
                content,
                f"PDF embedded file attachment (obj {obj_id})",
                is_flag,
            ))
        elif is_flag:
            candidates.append((
                f"pdf-stream-{obj_id}.txt",
                content,
                f"flag recovered in PDF stream obj {obj_id}",
                True,
            ))
        elif stream["decompressed"] and len(content) >= 8:
            printable = sum(32 <= b <= 126 or b in (9, 10, 13) for b in content) / len(content)
            if printable >= 0.85:
                candidates.append((
                    f"pdf-stream-{obj_id}.txt",
                    content,
                    f"decompressed text stream obj {obj_id}",
                    False,
                ))

    # 4. Appended Trailer
    end = pdf_structural_end(data)
    if end is not None and len(data) > end:
        trailer = data[end:]
        is_flag = bool(_FLAG_PATTERN.search(trailer))
        candidates.append((
            "trailer.bin",
            trailer,
            "data appended after PDF %%EOF",
            is_flag,
        ))

    return candidates


class FilePdf(Carrier):
    """Carrier and steganalysis for PDF documents."""

    name = "file_pdf"
    method_id = "file_pdf"
    extensions = (".pdf",)
    can_embed = False
    can_extract = False
    priority = 160
    requires_explicit = False

    def capacity(self, src: Path) -> int:
        del src
        return 0

    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult:
        del src, payload, out
        raise NotImplementedError("file_pdf is analysis-only")

    def extract(self, src: Path) -> bytes:
        del src
        raise NotImplementedError("file_pdf is analysis-only")

    def analyze(self, src: Path) -> AnalysisResult:
        if not src.is_file():
            return AnalysisResult(
                self.name, 0, (), None, status="unsupported", error="file does not exist"
            )
        return self.analyze_context(AnalysisContext(src))

    def analyze_context(self, context: AnalysisContext) -> AnalysisResult:
        if context.path.suffix.lower() != ".pdf":
            return AnalysisResult(self.name, 0, (), None, status="unsupported")

        data = context.data
        if not data.startswith(b"%PDF-"):
            return AnalysisResult(self.name, 0, (), None, status="unsupported")

        signals: list[Signal] = []

        # 1. Comment Analysis
        comments = parse_pdf_comments(data)
        for c in comments:
            m = _FLAG_PATTERN.search(c)
            if m:
                flag_str = m.group(1).decode("latin1", errors="replace")
                signals.append(
                    Signal(
                        "pdf_comment_flag",
                        98,
                        f"flag detected in PDF comment: {flag_str}",
                        category="pdf_comment",
                        evidence="verified",
                    )
                )
            elif len(c) >= 16 and any(32 <= b <= 126 for b in c):
                signals.append(
                    Signal(
                        "pdf_comment_payload",
                        65,
                        f"PDF comment contains text payload ({len(c)} bytes): {bytes(c[:24])!r}",
                        category="pdf_comment",
                        evidence="heuristic",
                    )
                )

        # 2. Metadata Analysis
        meta = parse_pdf_metadata(data)
        for k, v in meta.items():
            bv = v.encode("latin1", errors="replace")
            m = _FLAG_PATTERN.search(bv)
            if m:
                flag_str = m.group(1).decode("latin1", errors="replace")
                signals.append(
                    Signal(
                        "pdf_metadata_flag",
                        98,
                        f"flag detected in PDF metadata /{k}: {flag_str}",
                        category="pdf_metadata",
                        evidence="verified",
                    )
                )
            elif len(bv) >= 32 and any(32 <= b <= 126 for b in bv):
                signals.append(
                    Signal(
                        "pdf_metadata_payload",
                        60,
                        f"PDF metadata /{k} contains payload ({len(bv)} bytes)",
                        category="pdf_metadata",
                        evidence="heuristic",
                    )
                )

        # 3. Stream & Embedded File Analysis
        streams = parse_pdf_streams(data)
        for stream in streams:
            content = stream["content"]
            obj_id = stream["obj_id"]
            if stream["is_embedded"]:
                signals.append(
                    Signal(
                        "pdf_embedded_file",
                        90,
                        f"PDF embedded file attachment detected in obj {obj_id}"
                        + (f" ({stream['filename']})" if stream["filename"] else ""),
                        category="pdf_attachment",
                        evidence="verified",
                    )
                )
            m = _FLAG_PATTERN.search(content)
            if m:
                flag_str = m.group(1).decode("latin1", errors="replace")
                signals.append(
                    Signal(
                        "pdf_stream_flag",
                        98,
                        f"flag detected in PDF stream obj {obj_id}: {flag_str}",
                        category="pdf_stream",
                        evidence="verified",
                    )
                )

        # 4. Incremental Updates Check
        eof_count = data.count(b"%%EOF")
        if eof_count > 1:
            signals.append(
                Signal(
                    "pdf_incremental_updates",
                    50,
                    f"PDF contains {eof_count} %%EOF revisions (possible hidden shadowed objects)",
                    category="pdf_structure",
                    evidence="heuristic",
                )
            )

        # 5. Trailer / Appended Data
        end = pdf_structural_end(data)
        if end is not None and len(data) > end:
            trailer = data[end:]
            m_trl = _FLAG_PATTERN.search(trailer[:4096])
            if m_trl:
                flag_str = m_trl.group(1).decode("latin1", errors="replace")
                signals.append(
                    Signal(
                        "pdf_appended_flag",
                        98,
                        f"flag detected in PDF trailer ({len(trailer)} bytes): {flag_str}",
                        category="pdf_trailer",
                        evidence="verified",
                    )
                )
            else:
                signals.append(
                    Signal(
                        "pdf_appended_data",
                        70,
                        f"{len(trailer)} bytes appended after PDF structural end",
                        category="pdf_trailer",
                        evidence="strong",
                    )
                )

        suspicion = max(
            (s.score for s in signals if s.evidence != "informational"),
            default=0,
        )
        return AnalysisResult(self.name, suspicion, tuple(signals), None)
