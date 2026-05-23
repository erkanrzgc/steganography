"""Zero-width unicode steganography.

Encodes payload bits as ZWSP (U+200B = 0) and ZWNJ (U+200C = 1).
Appended at the end of the cover text, terminated by ZWJ (U+200D).
"""
from pathlib import Path

from core.carrier import Carrier
from core.result import AnalysisResult, EmbedResult, Signal

_ZERO = "​"        # ZWSP
_ONE = "‌"         # ZWNJ
_TERMINATOR = "‍"  # ZWJ — marks end of encoded run


class TextZeroWidth(Carrier):
    name = "text_zerowidth"
    extensions = (".txt", ".md")

    def capacity(self, src: Path) -> int:
        return 1_000_000  # effectively unbounded; appended

    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult:
        cover = src.read_text(encoding="utf-8")
        bits = "".join(f"{b:08b}" for b in payload)
        encoded = "".join(_ONE if c == "1" else _ZERO for c in bits) + _TERMINATOR
        out.write_text(cover + encoded, encoding="utf-8")
        return EmbedResult(self.name, out, len(payload), encrypted=False)

    def extract(self, src: Path) -> bytes:
        text = src.read_text(encoding="utf-8")
        run = []
        for ch in text:
            if ch == _TERMINATOR:
                break
            if ch in (_ZERO, _ONE):
                run.append("1" if ch == _ONE else "0")
        bits = "".join(run)
        out = bytearray()
        for i in range(0, len(bits) - 7, 8):
            out.append(int(bits[i : i + 8], 2))
        return bytes(out)

    def analyze(self, src: Path) -> AnalysisResult:
        text = src.read_text(encoding="utf-8", errors="ignore")
        count = sum(text.count(ch) for ch in (_ZERO, _ONE, _TERMINATOR))
        if count == 0:
            return AnalysisResult(self.name, 0, (), None)
        # Zero-width chars almost never appear in legitimate text; any presence
        # is highly suspicious, with score increasing as count grows.
        score = min(100, 70 + count // 2)
        sig = Signal(name="zero_width_chars", score=score, detail=f"{count} zero-width chars")
        return AnalysisResult(self.name, score, (sig,), None)
