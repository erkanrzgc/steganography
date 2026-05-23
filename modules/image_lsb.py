"""LSB steganography for lossless images (PNG, BMP)."""
import struct
from pathlib import Path

import numpy as np
from PIL import Image

from core.carrier import Carrier, InsufficientCapacityError
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
        arr = self._load_rgb(src).reshape(-1)
        even = arr[::2].astype(np.int64)
        odd = arr[1::2].astype(np.int64)
        pairs = np.minimum(even, odd)
        denom = np.maximum(even + odd, 1)
        ratio = float(np.mean(pairs * 2 / denom))
        lsb_mean = float(np.mean(arr & 1))
        lsb_dev = abs(lsb_mean - 0.5) * 200  # 0..100
        chi_signal = Signal(
            name="lsb_pair_ratio",
            score=int(min(100, ratio * 100)),
            detail=f"adjacent-pair LSB ratio = {ratio:.3f} (~1.0 suggests embedding)",
        )
        bias_signal = Signal(
            name="lsb_bit_bias",
            score=int(min(100, lsb_dev)),
            detail=f"LSB mean = {lsb_mean:.3f}",
        )
        suspicion = max(chi_signal.score, bias_signal.score)
        return AnalysisResult(
            analyzer=self.name,
            suspicion=suspicion,
            signals=(chi_signal, bias_signal),
            explanation=None,
        )
