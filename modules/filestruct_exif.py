"""EXIF UserComment-based steganography (JPEG/TIFF)."""
from pathlib import Path

import piexif

from core.carrier import Carrier
from core.result import AnalysisResult, EmbedResult, Signal

_MAGIC = b"STEGEXIF"


class FilestructExif(Carrier):
    name = "filestruct_exif"
    extensions = (".jpg", ".jpeg", ".tiff")

    def capacity(self, src: Path) -> int:
        return 60_000  # EXIF UserComment practical limit

    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult:
        try:
            exif_dict = piexif.load(str(src))
        except piexif.InvalidImageDataError:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        exif_dict.setdefault("Exif", {})[piexif.ExifIFD.UserComment] = _MAGIC + payload
        exif_bytes = piexif.dump(exif_dict)
        piexif.insert(exif_bytes, str(src), str(out))
        return EmbedResult(self.name, out, len(payload), encrypted=False)

    def extract(self, src: Path) -> bytes:
        exif_dict = piexif.load(str(src))
        comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment, b"")
        if not comment.startswith(_MAGIC):
            raise ValueError("no STEGEXIF marker in EXIF UserComment")
        return comment[len(_MAGIC) :]

    def analyze(self, src: Path) -> AnalysisResult:
        try:
            exif_dict = piexif.load(str(src))
        except piexif.InvalidImageDataError:
            return AnalysisResult(self.name, 0, (), None)
        comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment, b"")
        signals = []
        if comment.startswith(_MAGIC):
            signals.append(Signal("stegexif_marker", 95, "STEGEXIF marker in UserComment"))
        elif len(comment) > 64:
            signals.append(Signal("large_usercomment", 60, f"UserComment len={len(comment)}"))
        suspicion = max((s.score for s in signals), default=0)
        return AnalysisResult(self.name, suspicion, tuple(signals), None)
