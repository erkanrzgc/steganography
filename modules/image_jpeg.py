"""JPEG carrier using the appended-data technique (after FFD9 EOI marker).

Full DCT-coefficient LSB is deferred (see design spec §15).
"""
import struct
from pathlib import Path

from core.carrier import Carrier
from core.result import AnalysisResult, EmbedResult, Signal

_JPEG_EOI = b"\xff\xd9"
_APP_MAGIC = b"STEGAPP"
_LEN_PREFIX = 4


class ImageJpeg(Carrier):
    name = "image_jpeg"
    extensions = (".jpg", ".jpeg")

    def capacity(self, src: Path) -> int:
        # Appended-data path: practically unbounded; expose a generous cap.
        return 10 * 1024 * 1024  # 10 MiB

    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult:
        cover = src.read_bytes()
        eoi = cover.rfind(_JPEG_EOI)
        if eoi == -1:
            raise ValueError("not a JPEG (no FFD9)")
        head = cover[: eoi + len(_JPEG_EOI)]
        trailer = _APP_MAGIC + struct.pack(">I", len(payload)) + payload
        out.write_bytes(head + trailer)
        return EmbedResult(self.name, out, len(payload), encrypted=False)

    def extract(self, src: Path) -> bytes:
        data = src.read_bytes()
        idx = data.find(_APP_MAGIC)
        if idx == -1:
            raise ValueError("no STEGAPP trailer present")
        cursor = idx + len(_APP_MAGIC)
        (length,) = struct.unpack(">I", data[cursor : cursor + _LEN_PREFIX])
        cursor += _LEN_PREFIX
        return data[cursor : cursor + length]

    def analyze(self, src: Path) -> AnalysisResult:
        data = src.read_bytes()
        eoi = data.rfind(_JPEG_EOI)
        signals: list[Signal] = []
        if eoi == -1:
            return AnalysisResult(self.name, 0, (), None)
        trailer_len = len(data) - (eoi + len(_JPEG_EOI))
        if trailer_len > 0:
            signals.append(
                Signal(
                    name="appended_data",
                    score=min(100, 50 + trailer_len // 32),
                    detail=f"{trailer_len} bytes after JPEG EOI",
                )
            )
        if _APP_MAGIC in data:
            signals.append(
                Signal(name="stegapp_marker", score=95, detail="STEGAPP marker found"),
            )
        suspicion = max((s.score for s in signals), default=0)
        return AnalysisResult(self.name, suspicion, tuple(signals), None)
