"""Lazy, per-input caches shared by native analysis and CTF stages."""

from __future__ import annotations

import math
import wave
from collections import Counter
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.filetype import DetectedType, detect_type


class AnalysisContext:
    """One immutable input with lazily decoded, reusable representations."""

    def __init__(
        self,
        path: Path,
        *,
        detected: DetectedType | None = None,
        max_bytes: int = 512 * 1024 * 1024,
        max_pixels: int = 50_000_000,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file() or self.path.is_symlink():
            raise ValueError("analysis input must be a regular, non-symlink file")
        self.detected = detected or detect_type(self.path)
        self.max_bytes = max_bytes
        self.max_pixels = max_pixels
        if self.path.stat().st_size > max_bytes:
            raise ValueError(f"file exceeds the {max_bytes}-byte context limit")

    @cached_property
    def data(self) -> bytes:
        return self.path.read_bytes()

    @cached_property
    def entropy(self) -> float:
        if not self.data:
            return 0.0
        counts = Counter(self.data)
        total = len(self.data)
        return -sum((count / total) * math.log2(count / total) for count in counts.values())

    def entropy_map(self, *, block_size: int = 4096) -> tuple[float, ...]:
        if block_size < 64:
            raise ValueError("entropy-map block size must be at least 64 bytes")
        values: list[float] = []
        for offset in range(0, len(self.data), block_size):
            block = self.data[offset : offset + block_size]
            if not block:
                continue
            counts = Counter(block)
            total = len(block)
            values.append(
                -sum((count / total) * math.log2(count / total) for count in counts.values())
            )
        return tuple(values)

    @cached_property
    def image_rgba(self) -> np.ndarray:
        with Image.open(self.path) as image:
            if image.width * image.height > self.max_pixels:
                raise ValueError(f"image exceeds the {self.max_pixels}-pixel context limit")
            image.load()
            return np.asarray(image.convert("RGBA"), dtype=np.uint8)

    @cached_property
    def jpeg_segments(self) -> tuple[dict[str, Any], ...]:
        data = self.data
        if not data.startswith(b"\xff\xd8"):
            return ()
        segments: list[dict[str, Any]] = []
        offset = 2
        while offset + 1 < len(data):
            if data[offset] != 0xFF:
                offset += 1
                continue
            while offset < len(data) and data[offset] == 0xFF:
                offset += 1
            if offset >= len(data):
                break
            marker = data[offset]
            marker_at = offset - 1
            offset += 1
            if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                segments.append({"marker": marker, "offset": marker_at, "length": 0})
                if marker == 0xD9:
                    break
                continue
            if offset + 2 > len(data):
                break
            length = int.from_bytes(data[offset : offset + 2], "big")
            if length < 2 or offset + length > len(data):
                segments.append(
                    {"marker": marker, "offset": marker_at, "length": length, "truncated": True}
                )
                break
            segments.append({"marker": marker, "offset": marker_at, "length": length})
            offset += length
            if marker == 0xDA:  # entropy-coded scan; locate the next marker conservatively
                end = data.find(b"\xff\xd9", offset)
                offset = len(data) if end < 0 else end
        return tuple(segments)

    @cached_property
    def wav_samples(self) -> tuple[np.ndarray, int, int]:
        with wave.open(str(self.path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            rate = source.getframerate()
            frames = source.getnframes()
            if frames * channels * sample_width > self.max_bytes:
                raise ValueError("decoded WAV samples exceed the context limit")
            raw = source.readframes(frames)
        if sample_width == 1:
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
        elif sample_width == 2:
            samples = np.frombuffer(raw, dtype="<i2")
        elif sample_width == 4:
            samples = np.frombuffer(raw, dtype="<i4")
        else:
            raise ValueError(f"unsupported WAV sample width: {sample_width}")
        if channels > 1:
            samples = samples.reshape(-1, channels)
        return samples, rate, sample_width
