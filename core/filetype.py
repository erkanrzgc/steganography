"""Small, dependency-free content type detection used for dispatch and evidence."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DetectedType:
    name: str
    mime_type: str
    extension: str | None
    compatible_extensions: frozenset[str]


_UNKNOWN = DetectedType("unknown", "application/octet-stream", None, frozenset())


def detect_type(path: Path) -> DetectedType:
    """Identify supported formats by magic bytes, falling back to UTF-8 text."""
    try:
        with path.open("rb") as stream:
            head = stream.read(8192)
    except OSError:
        return _UNKNOWN

    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return DetectedType("png", "image/png", ".png", frozenset({".png"}))
    if head.startswith(b"\xff\xd8\xff"):
        return DetectedType(
            "jpeg", "image/jpeg", ".jpg", frozenset({".jpg", ".jpeg"})
        )
    if head.startswith((b"GIF87a", b"GIF89a")):
        return DetectedType("gif", "image/gif", ".gif", frozenset({".gif"}))
    if head.startswith(b"%PDF-"):
        return DetectedType(
            "pdf", "application/pdf", ".pdf", frozenset({".pdf"})
        )
    if head.startswith(b"BM"):
        return DetectedType("bmp", "image/bmp", ".bmp", frozenset({".bmp"}))
    if head.startswith((b"II*\x00", b"MM\x00*")):
        return DetectedType(
            "tiff", "image/tiff", ".tiff", frozenset({".tif", ".tiff"})
        )
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return DetectedType("wav", "audio/wav", ".wav", frozenset({".wav"}))
    if head.startswith(b"ID3") or (
        len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0
    ):
        return DetectedType("mp3", "audio/mpeg", ".mp3", frozenset({".mp3"}))
    if _looks_like_text(head):
        suffix = path.suffix.lower()
        canonical = suffix if suffix in {".txt", ".md"} else ".txt"
        return DetectedType(
            "text", "text/plain", canonical, frozenset({".txt", ".md"})
        )
    return _UNKNOWN


def extension_mismatch(path: Path, detected: DetectedType) -> bool:
    suffix = path.suffix.lower()
    return bool(detected.compatible_extensions and suffix not in detected.compatible_extensions)


def _looks_like_text(data: bytes) -> bool:
    if not data or b"\x00" in data:
        return False
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    printable = sum(char.isprintable() or char in "\r\n\t" for char in text)
    return printable / max(len(text), 1) >= 0.95
