"""Explicit PDF/GIF trailer carrier with a self-validating marker."""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path

from core.carrier import Carrier, InsufficientCapacityError
from core.result import AnalysisResult, EmbedResult, Signal

_END_MARKERS = {".pdf": b"%%EOF", ".gif": b";"}
_MAGIC = b"STEGTRLR"
_HEADER = struct.Struct(">8sBQ32s")
_VERSION = 1
_MAX_PAYLOAD = 10 * 1024 * 1024


class FilestructTrailer(Carrier):
    name = "filestruct_trailer"
    method_id = "filestruct_trailer"
    extensions = tuple(_END_MARKERS)
    priority = 200
    requires_explicit = True

    def capacity(self, src: Path) -> int:
        del src
        return _MAX_PAYLOAD

    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult:
        if len(payload) > _MAX_PAYLOAD:
            raise InsufficientCapacityError(
                f"payload {len(payload)} > trailer limit {_MAX_PAYLOAD}"
            )
        marker = _END_MARKERS.get(src.suffix.lower())
        if marker is None:
            raise ValueError(f"unsupported trailer extension: {src.suffix}")
        cover = src.read_bytes()
        structural_end = cover.rfind(marker)
        if structural_end < 0:
            raise ValueError(f"no structural end marker for {src.suffix.lower()}")
        structural_end += len(marker)
        header = _HEADER.pack(
            _MAGIC, _VERSION, len(payload), hashlib.sha256(payload).digest()
        )
        out.write_bytes(cover[:structural_end] + header + payload)
        return EmbedResult(self.identifier, out, len(payload), encrypted=False)

    def extract(self, src: Path) -> bytes:
        data = src.read_bytes()
        index = data.rfind(_MAGIC)
        if index < 0 or len(data) < index + _HEADER.size:
            raise ValueError("no STEGTRLR trailer present")
        magic, version, length, digest = _HEADER.unpack(
            data[index : index + _HEADER.size]
        )
        if magic != _MAGIC or version != _VERSION:
            raise ValueError("unsupported STEGTRLR header")
        start = index + _HEADER.size
        payload = data[start : start + length]
        if len(payload) != length:
            raise ValueError("truncated STEGTRLR payload")
        if hashlib.sha256(payload).digest() != digest:
            raise ValueError("STEGTRLR integrity check failed")
        return payload

    def analyze(self, src: Path) -> AnalysisResult:
        data = src.read_bytes()
        index = data.rfind(_MAGIC)
        if index < 0:
            return AnalysisResult(self.identifier, 0, (), None)
        try:
            self.extract(src)
        except ValueError as exc:
            signal = Signal(
                "malformed_stegtrlr_marker",
                80,
                str(exc),
                category="known_marker",
                evidence="strong",
            )
            return AnalysisResult(self.identifier, 80, (signal,), str(exc))
        signal = Signal(
            "stegtrlr_marker",
            98,
            f"validated trailer at byte offset {index}",
            category="known_marker",
            evidence="verified",
        )
        return AnalysisResult(self.identifier, 98, (signal,), None)
