"""Invisible Unicode payloads: tag characters and variation-selector byte runs.

- **Tag characters** (U+E0020-E007E) mirror printable ASCII and render as nothing
  ("ASCII smuggling"). The only legitimate use is a subdivision-flag sequence:
  U+1F3F4 followed by lowercase tag letters/digits and U+E007F. Such sequences
  are skipped.
- **Variation selectors** (U+FE00-FE0F, U+E0100-E01EF) can each carry one byte:
  0-15 for the first block and 16-255 for the supplement. Natural text uses at
  most one selector after a base character, so runs of two or more are payloads.

A recovered payload counts as successful extraction. Only its length and SHA-256
are reported, never the content.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from core.analyzer import Analyzer
from core.result import AnalysisResult, Signal

TEXT_SUFFIXES = {".txt", ".md", ".csv", ".log", ".json", ".html", ".htm", ".xml", ".yaml",
                 ".yml", ".srt", ".rtf", ".tex", ".ini", ".cfg"}
MAX_BYTES = 64 * 1024 * 1024
MIN_PAYLOAD = 4
_FLAG = re.compile("\U0001f3f4[\U000e0030-\U000e0039\U000e0061-\U000e007a]{2,7}\U000e007f")
_TAGS = re.compile("[\U000e0000-\U000e007f]+")
_SELECTORS = re.compile("[︀-️\U000e0100-\U000e01ef]{2,}")


def tag_payload(text: str) -> bytes:
    """ASCII hidden in tag characters outside legitimate flag sequences."""
    cleaned = _FLAG.sub("", text)
    return "".join(chr(ord(c) - 0xE0000) for run in _TAGS.findall(cleaned) for c in run
                   if 0xE0020 <= ord(c) <= 0xE007E).encode("ascii")


def selector_payload(text: str) -> bytes:
    """Bytes carried by runs of two or more variation selectors."""
    out = bytearray()
    for run in _SELECTORS.findall(text):
        for c in run:
            code = ord(c)
            out.append(code - 0xFE00 if code <= 0xFE0F else code - 0xE0100 + 16)
    return bytes(out)


def _signal(name: str, payload: bytes, how: str) -> Signal:
    return Signal(name, 95, f"{len(payload)} bytes recovered from {how}; "
                  f"sha256={hashlib.sha256(payload).hexdigest()}",
                  category="known_marker", evidence="verified")


class InvisibleUnicodeAnalyzer(Analyzer):
    """Recover payloads hidden with Unicode tag characters or variation selectors."""

    name = "text_invisible_unicode"

    def analyze(self, src: Path) -> AnalysisResult:
        if src.suffix.lower() not in TEXT_SUFFIXES or src.stat().st_size > MAX_BYTES:
            return AnalysisResult(self.name, 0, (), None, status="unsupported")
        text = src.read_bytes().decode("utf-8", errors="replace")
        signals = []
        tags = tag_payload(text)
        if len(tags) >= MIN_PAYLOAD:
            signals.append(_signal("unicode_tag_payload", tags, "Unicode tag characters"))
        selectors = selector_payload(text)
        if len(selectors) >= MIN_PAYLOAD:
            signals.append(_signal("variation_selector_payload", selectors,
                                   "variation-selector runs"))
        return AnalysisResult(self.name, max((s.score for s in signals), default=0),
                              tuple(signals), None)
