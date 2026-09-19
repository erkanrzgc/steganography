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
    extensions = tuple(_END_MARKERS.keys()) + (".wav", ".wave", ".mp3")
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
        if marker is None and ext not in {".wav", ".wave", ".mp3"}:
            return AnalysisResult(self.name, 0, (), None)
        data = src.read_bytes()
        if ext == ".gif":
            from modules.image_gif import gif_structural_end

            idx = gif_structural_end(data)
            if idx is None or idx >= len(data):
                return AnalysisResult(self.name, 0, (), None)
            trailer = data[idx:]
        elif ext in {".wav", ".wave"}:
            if data.startswith(b"RIFF") and len(data) >= 8:
                riff_size = int.from_bytes(data[4:8], "little")
                structural_len = 8 + riff_size
                if len(data) <= structural_len:
                    return AnalysisResult(self.name, 0, (), None)
                trailer = data[structural_len:]
            else:
                return AnalysisResult(self.name, 0, (), None)
        elif ext == ".mp3":
            from modules.audio_mp3 import mp3_structural_end

            idx = mp3_structural_end(data)
            if idx is None or idx >= len(data):
                return AnalysisResult(self.name, 0, (), None)
            trailer = data[idx:]
        elif ext == ".pdf":
            from modules.file_pdf import pdf_structural_end

            idx = pdf_structural_end(data)
            if idx is None or idx >= len(data):
                return AnalysisResult(self.name, 0, (), None)
            trailer = data[idx:]
        else:
            if marker is None:
                return AnalysisResult(self.name, 0, (), None)
            idx = data.rfind(marker)
            if idx == -1:
                return AnalysisResult(self.name, 0, (), None)
            trailer = data[idx + len(marker) :]
        signals: list[Signal] = []
        if not trailer:
            return AnalysisResult(self.name, 0, (), None)
        signals.append(
            Signal(
                name="appended_data",
                score=70,
                detail=f"{len(trailer)} bytes after EOF marker",
                category="appended_data",
                evidence="strong",
            )
        )
        for sig, label in _APPENDED_SIGNATURES.items():
            if trailer.startswith(sig):
                signals.append(
                    Signal(
                        name=f"embedded_{label.replace(' ', '_')}",
                        score=95,
                        detail=label,
                        category="polyglot",
                        evidence="verified",
                    )
                )
                break
        suspicion = max(s.score for s in signals)
        return AnalysisResult(self.name, suspicion, tuple(signals), None)
