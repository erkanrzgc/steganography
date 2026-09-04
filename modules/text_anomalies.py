"""Unicode directionality, homoglyph, whitespace and encoding indicators."""

from __future__ import annotations

import unicodedata
from pathlib import Path

from core.analyzer import Analyzer
from core.result import AnalysisResult, Signal

_BIDI_CONTROLS = frozenset(chr(value) for value in range(0x202A, 0x202F)) | frozenset(
    chr(value) for value in range(0x2066, 0x206A)
)


class TextAnomalyAnalyzer(Analyzer):
    name = "text_anomalies"

    def analyze(self, src: Path) -> AnalysisResult:
        if src.suffix.lower() not in {".txt", ".md", ".csv", ".log", ".eml"}:
            return AnalysisResult(self.name, 0, (), None, status="unsupported")
        raw = src.read_bytes()
        text = raw.decode("utf-8", errors="replace")
        signals: list[Signal] = []
        replacements = text.count("\ufffd")
        bidi = sum(character in _BIDI_CONTROLS for character in text)
        trailing = sum(line.endswith((" ", "\t")) for line in text.splitlines())
        non_ascii_letters = sum(
            ord(character) > 127 and unicodedata.category(character).startswith("L")
            for character in text
        )
        if replacements:
            signals.append(
                Signal(
                    "invalid_utf8",
                    min(70, 25 + replacements),
                    f"{replacements} invalid UTF-8 replacement characters",
                    category="text_encoding",
                    evidence="heuristic",
                )
            )
        if bidi:
            signals.append(
                Signal(
                    "bidi_controls",
                    min(90, 55 + bidi * 5),
                    f"{bidi} bidirectional control characters",
                    category="text_unicode",
                    evidence="strong",
                )
            )
        if trailing >= 8:
            signals.append(
                Signal(
                    "trailing_whitespace_pattern",
                    min(65, 20 + trailing),
                    f"{trailing} lines carry trailing spaces or tabs",
                    category="text_whitespace",
                    evidence="heuristic",
                )
            )
        if non_ascii_letters >= 4:
            scripts = sorted(
                {
                    unicodedata.name(character, "UNKNOWN").split()[0]
                    for character in text
                    if ord(character) > 127 and unicodedata.category(character).startswith("L")
                }
            )
            if len(scripts) > 1:
                signals.append(
                    Signal(
                        "mixed_script_homoglyphs",
                        45,
                        "multiple non-ASCII scripts: " + ", ".join(scripts[:8]),
                        category="text_homoglyph",
                        evidence="heuristic",
                    )
                )
        suspicion = max((signal.score for signal in signals), default=0)
        return AnalysisResult(self.name, suspicion, tuple(signals), None)
