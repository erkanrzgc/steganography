"""Trailing-whitespace steganography.

Each line carries one bit: trailing TAB = 1, trailing SPACE = 0.
Requires cover to have at least 8 * len(payload) lines.
"""
from pathlib import Path

from core.carrier import Carrier, InsufficientCapacityError
from core.result import AnalysisResult, EmbedResult, Signal

_ONE = "\t"
_ZERO = " "


class TextWhitespace(Carrier):
    name = "text_whitespace"
    extensions = (".txt", ".md")

    def capacity(self, src: Path) -> int:
        lines = src.read_text(encoding="utf-8").splitlines()
        return max(0, len(lines) // 8)

    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult:
        cap = self.capacity(src)
        if len(payload) > cap:
            raise InsufficientCapacityError(f"payload {len(payload)} > capacity {cap}")
        lines = src.read_text(encoding="utf-8").splitlines()
        bits = [(b >> (7 - i)) & 1 for b in payload for i in range(8)]
        new_lines = []
        for idx, line in enumerate(lines):
            if idx < len(bits):
                marker = _ONE if bits[idx] else _ZERO
                new_lines.append(line.rstrip(" \t") + marker)
            else:
                new_lines.append(line)
        out.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return EmbedResult(self.name, out, len(payload), encrypted=False)

    def extract(self, src: Path) -> bytes:
        lines = src.read_text(encoding="utf-8").splitlines()
        bits = []
        for line in lines:
            if line.endswith(_ONE):
                bits.append(1)
            elif line.endswith(_ZERO):
                bits.append(0)
            else:
                break
        out = bytearray()
        for i in range(0, len(bits) - 7, 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            out.append(byte)
        return bytes(out)

    def analyze(self, src: Path) -> AnalysisResult:
        lines = src.read_text(encoding="utf-8").splitlines()
        if not lines:
            return AnalysisResult(self.name, 0, (), None)
        flagged = sum(1 for ln in lines if ln.endswith((_ONE, _ZERO)) and ln.strip())
        ratio = flagged / len(lines)
        score = int(min(100, ratio * 100))
        sig = Signal(
            name="trailing_whitespace_ratio",
            score=score,
            detail=f"{flagged}/{len(lines)}",
        )
        return AnalysisResult(self.name, score, (sig,), None)
