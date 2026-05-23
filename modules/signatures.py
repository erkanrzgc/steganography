"""Known-tool signature detection. Declarative — add a row, get detection."""
from dataclasses import dataclass
from pathlib import Path

from core.analyzer import Analyzer
from core.result import AnalysisResult, Signal


@dataclass(frozen=True, slots=True)
class _Sig:
    name: str
    needle: bytes
    score: int
    detail: str


_SIGNATURES: tuple[_Sig, ...] = (
    _Sig("openstego", b"OPENSTEGO", 90, "OpenStego marker"),
    _Sig("steghide", b"shsh", 70, "possible steghide artifact"),
    _Sig("outguess", b"OutGuess", 85, "OutGuess marker"),
    _Sig("stegapp", b"STEGAPP", 95, "STEGAPP (this tool's JPEG trailer)"),
    _Sig("stegexif", b"STEGEXIF", 95, "STEGEXIF (this tool's EXIF marker)"),
)


class Signatures(Analyzer):
    name = "signatures"

    def analyze(self, src: Path) -> AnalysisResult:
        data = src.read_bytes()
        hits = [
            Signal(name=s.name, score=s.score, detail=s.detail)
            for s in _SIGNATURES
            if s.needle in data
        ]
        suspicion = max((h.score for h in hits), default=0)
        return AnalysisResult(self.name, suspicion, tuple(hits), None)
