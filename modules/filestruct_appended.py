"""Detect data appended after a file's structural end marker.

Carrier base used for registry uniformity but embed/extract are disabled
(analysis-only module).
"""
from pathlib import Path

from core.carrier import Carrier
from core.result import AnalysisResult, EmbedResult, Signal

_END_MARKERS = {
    ".png": b"IEND\xaeB`\x82",
    ".jpg": b"\xff\xd9",
    ".jpeg": b"\xff\xd9",
    ".gif": b";",
    ".pdf": b"%%EOF",
}

_APPENDED_SIGNATURES = {
    b"PK\x03\x04": "ZIP archive",
    b"Rar!\x1a\x07": "RAR archive",
    b"7z\xbc\xaf\x27\x1c": "7z archive",
    b"MZ": "PE executable",
    b"\x7fELF": "ELF executable",
}


class FilestructAppended(Carrier):
    name = "filestruct_appended"
    extensions = tuple(_END_MARKERS.keys())
    can_embed = False
    can_extract = False

    def capacity(self, src: Path) -> int:
        return 0

    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult:
        raise NotImplementedError("filestruct_appended is analysis-only")

    def extract(self, src: Path) -> bytes:
        raise NotImplementedError("filestruct_appended is analysis-only")

    def analyze(self, src: Path) -> AnalysisResult:
        ext = src.suffix.lower()
        marker = _END_MARKERS.get(ext)
        if marker is None:
            return AnalysisResult(self.name, 0, (), None)
        data = src.read_bytes()
        idx = data.rfind(marker)
        signals: list[Signal] = []
        if idx == -1:
            return AnalysisResult(self.name, 0, (), None)
        trailer = data[idx + len(marker) :]
        if not trailer:
            return AnalysisResult(self.name, 0, (), None)
        signals.append(
            Signal(name="appended_data", score=70, detail=f"{len(trailer)} bytes after EOF marker")
        )
        for sig, label in _APPENDED_SIGNATURES.items():
            if trailer.startswith(sig):
                signals.append(
                    Signal(
                        name=f"embedded_{label.replace(' ', '_')}",
                        score=95,
                        detail=label,
                    )
                )
                break
        suspicion = max(s.score for s in signals)
        return AnalysisResult(self.name, suspicion, tuple(signals), None)
