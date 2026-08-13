"""Content/extension validation, entropy and embedded-file heuristics."""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from core.analyzer import Analyzer
from core.filetype import detect_type, extension_mismatch
from core.result import AnalysisResult, Signal

_READ_LIMIT = 8 * 1024 * 1024
_EMBEDDED_MAGICS: tuple[tuple[bytes, str], ...] = (
    (b"PK\x03\x04", "zip"),
    (b"Rar!\x1a\x07", "rar"),
    (b"7z\xbc\xaf\x27\x1c", "7z"),
    (b"\x7fELF", "elf"),
    (b"%PDF-", "pdf"),
    (b"\x89PNG\r\n\x1a\n", "png"),
)


class FileStructure(Analyzer):
    name = "file_structure"

    def analyze(self, src: Path) -> AnalysisResult:
        detected = detect_type(src)
        signals: list[Signal] = []
        if extension_mismatch(src, detected):
            signals.append(
                Signal(
                    "extension_mismatch",
                    65,
                    f"extension {src.suffix.lower() or '(none)'} contains {detected.name}",
                    category="file_identity",
                    evidence="strong",
                )
            )

        with src.open("rb") as stream:
            data = stream.read(_READ_LIMIT)
        for magic, label in _EMBEDDED_MAGICS:
            first = data.find(magic)
            if first > 0:
                # ZIP signatures inside OOXML/PDF and ordinary binary data are
                # common. Only elevate when the secondary header is substantial.
                score = 55 if label == "zip" else 70
                signals.append(
                    Signal(
                        f"embedded_{label}_header",
                        score,
                        f"{label.upper()} header at byte offset {first}",
                        category="polyglot",
                        evidence="strong" if score >= 70 else "heuristic",
                    )
                )
                break

        entropy = _entropy(data)
        if detected.name in {"unknown", "text"} and entropy >= 7.6:
            score = min(65, round(35 + (entropy - 7.6) * 75))
            signals.append(
                Signal(
                    "high_entropy_content",
                    score,
                    f"Shannon entropy={entropy:.3f} bits/byte",
                    category="entropy",
                )
            )
        elif data:
            signals.append(
                Signal(
                    "content_entropy",
                    min(20, round(entropy * 2)),
                    f"Shannon entropy={entropy:.3f} bits/byte",
                    category="entropy",
                    evidence="informational",
                )
            )
        scored = (signal.score for signal in signals if signal.evidence != "informational")
        suspicion = max(scored, default=0)
        return AnalysisResult(self.name, suspicion, tuple(signals), None)


def _entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())
