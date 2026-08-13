"""LSB steganography for lossless images (PNG, BMP)."""
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from core.carrier import Carrier, InsufficientCapacityError
from core.payload import MAGIC
from core.result import AnalysisResult, EmbedResult, Signal

_LEN_PREFIX = 4  # bytes used to encode payload length in-image


class ImageLsb(Carrier):
    name = "image_lsb"
    extensions = (".png", ".bmp")

    def _load_rgb(self, src: Path) -> np.ndarray:
        img = Image.open(src).convert("RGB")
        return np.array(img, dtype=np.uint8)

    def _bits_from_bytes(self, data: bytes) -> np.ndarray:
        return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    def _bytes_from_bits(self, bits: np.ndarray) -> bytes:
        return np.packbits(bits).tobytes()

    def capacity(self, src: Path) -> int:
        arr = self._load_rgb(src)
        total_bytes = arr.size // 8
        return max(0, total_bytes - _LEN_PREFIX)

    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult:
        if len(payload) > self.capacity(src):
            raise InsufficientCapacityError(
                f"payload {len(payload)} > capacity {self.capacity(src)}"
            )
        arr = self._load_rgb(src)
        flat = arr.reshape(-1)
        blob = struct.pack(">I", len(payload)) + payload
        bits = self._bits_from_bytes(blob)
        flat = flat.copy()
        flat[: bits.size] = (flat[: bits.size] & 0xFE) | bits
        fmt = src.suffix.lstrip(".").upper()
        Image.fromarray(flat.reshape(arr.shape), "RGB").save(out, format=fmt)
        return EmbedResult(self.name, out, len(payload), encrypted=False)

    def extract(self, src: Path) -> bytes:
        arr = self._load_rgb(src).reshape(-1)
        length_bits = arr[: _LEN_PREFIX * 8] & 1
        (length,) = struct.unpack(">I", self._bytes_from_bits(length_bits))
        start = _LEN_PREFIX * 8
        end = start + length * 8
        if end > arr.size:
            raise ValueError("declared length exceeds image LSB capacity")
        return self._bytes_from_bits(arr[start:end] & 1)

    def analyze(self, src: Path) -> AnalysisResult:
        array = self._load_rgb(src)
        flat = array.reshape(-1)
        signals: list[Signal] = []
        channel_means = [float(np.mean(array[:, :, index] & 1)) for index in range(3)]
        maximum_deviation = max(abs(value - 0.5) for value in channel_means)
        bias_score = min(60, max(0, round((maximum_deviation - 0.02) * 1000)))
        signals.append(
            Signal(
                name="lsb_channel_bias",
                score=bias_score,
                detail="RGB LSB means=" + ",".join(f"{value:.4f}" for value in channel_means),
                category="image_lsb_statistics",
                evidence="heuristic" if bias_score >= 10 else "informational",
            )
        )
        if flat.size >= (_LEN_PREFIX + len(MAGIC)) * 8:
            length_bits = flat[: _LEN_PREFIX * 8] & 1
            (length,) = struct.unpack(">I", self._bytes_from_bits(length_bits))
            start = _LEN_PREFIX * 8
            marker_end = start + len(MAGIC) * 8
            marker = self._bytes_from_bits(flat[start:marker_end] & 1)
            if marker == MAGIC and length <= self.capacity(src):
                signals.append(
                    Signal(
                        "steg_payload_header",
                        98,
                        f"validated STEG envelope prefix; embedded length={length}",
                        category="known_marker",
                        evidence="verified",
                    )
                )
        suspicion = max(
            (signal.score for signal in signals if signal.evidence != "informational"),
            default=0,
        )
        return AnalysisResult(
            analyzer=self.name,
            suspicion=suspicion,
            signals=tuple(signals),
            explanation=None,
        )
