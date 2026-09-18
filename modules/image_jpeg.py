"""JPEG carrier using the appended-data technique (after FFD9 EOI marker).

Full DCT-coefficient LSB is deferred (see design spec §15).
"""

import struct
from pathlib import Path

from core.carrier import Carrier
from core.context import AnalysisContext
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
        return self.analyze_context(AnalysisContext(src))

    def analyze_context(self, context: AnalysisContext) -> AnalysisResult:
        data = context.data
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
                    category="appended_data",
                    evidence="strong",
                )
            )
        if _APP_MAGIC in data:
            signals.append(
                Signal(
                    name="stegapp_marker",
                    score=95,
                    detail="STEGAPP marker found",
                    category="known_marker",
                    evidence="verified",
                ),
            )
        segments = context.jpeg_segments
        dqt_segments = [segment for segment in segments if segment["marker"] == 0xDB]
        if dqt_segments:
            dqt_details = []
            all_ones = False
            for seg in dqt_segments:
                start = seg["offset"] + 4
                end = start + seg["length"] - 2
                table_bytes = data[start:end]
                if len(table_bytes) >= 65:
                    table_id = table_bytes[0] & 0x0F
                    q_vals = table_bytes[1:65]
                    if all(v == 1 for v in q_vals):
                        all_ones = True
                    dqt_details.append(f"T{table_id}:min={min(q_vals)},max={max(q_vals)}")
            score = 65 if all_ones else 0
            signals.append(
                Signal(
                    name="jpeg_quantization_tables",
                    score=score,
                    detail=f"{len(dqt_segments)} DQT segment(s); "
                    + ("; ".join(dqt_details) if dqt_details else "unparsed")
                    + ("; flat unit quantization detected" if all_ones else ""),
                    category="jpeg_structure",
                    evidence="heuristic" if all_ones else "informational",
                )
            )
        truncated = [segment for segment in segments if segment.get("truncated")]
        if truncated:
            signals.append(
                Signal(
                    name="truncated_jpeg_segment",
                    score=65,
                    detail=f"truncated marker at byte offset {truncated[0]['offset']}",
                    category="jpeg_structure",
                    evidence="strong",
                )
            )
        suspicion = max((s.score for s in signals), default=0)
        return AnalysisResult(self.name, suspicion, tuple(signals), None)
