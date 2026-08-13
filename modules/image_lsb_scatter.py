"""Keyed LSB scattering for lossless RGB images.

A small bootstrap is stored in fixed LSB positions. The payload positions are
then selected without replacement using a ChaCha20 keystream derived from the
user's placement key and a random salt.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from core.carrier import Carrier, InsufficientCapacityError
from core.placement import sample_positions
from core.result import AnalysisResult, EmbedResult, Signal

_MAGIC = b"STEGSCAT"
_VERSION = 1
_BOOTSTRAP = struct.Struct(">8sBQB16s")
_BOOTSTRAP_BITS = _BOOTSTRAP.size * 8
_CHANNEL_BITS = {"r": 0b001, "g": 0b010, "b": 0b100}


class ImageLsbScatter(Carrier):
    name = "image_lsb_scatter"
    method_id = "image_lsb_scatter"
    extensions = (".png", ".bmp")
    priority = 200
    requires_explicit = True

    def _load_rgb(self, src: Path) -> np.ndarray:
        with Image.open(src) as image:
            return np.array(image.convert("RGB"), dtype=np.uint8)

    def capacity(self, src: Path) -> int:
        array = self._load_rgb(src)
        return max(0, (array.size - _BOOTSTRAP_BITS) // 8)

    def embed(self, src: Path, payload: bytes, out: Path) -> EmbedResult:
        raise ValueError("image_lsb_scatter requires embed_with_options and a steg key")

    def embed_with_options(
        self,
        src: Path,
        payload: bytes,
        out: Path,
        *,
        steg_key: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> EmbedResult:
        if not steg_key:
            raise ValueError("image_lsb_scatter requires --steg-key or --password")
        channels, mask = _channel_selection(options)
        array = self._load_rgb(src)
        flat = array.reshape(-1).copy()
        candidates = _candidate_positions(flat.size, mask)
        if len(candidates) < len(payload) * 8:
            raise InsufficientCapacityError(
                f"payload {len(payload)} bytes exceeds keyed LSB capacity "
                f"{len(candidates) // 8} bytes for channels {channels}"
            )
        salt = os.urandom(16)
        bootstrap = _BOOTSTRAP.pack(_MAGIC, _VERSION, len(payload), mask, salt)
        bootstrap_bits = np.unpackbits(np.frombuffer(bootstrap, dtype=np.uint8))
        flat[:_BOOTSTRAP_BITS] = (
            flat[:_BOOTSTRAP_BITS] & 0xFE
        ) | bootstrap_bits

        payload_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
        positions = np.fromiter(
            sample_positions(
                candidates,
                payload_bits.size,
                steg_key,
                salt,
                context=b"image-lsb-scatter",
            ),
            dtype=np.int64,
            count=payload_bits.size,
        )
        flat[positions] = (flat[positions] & 0xFE) | payload_bits
        fmt = src.suffix.lstrip(".").upper()
        Image.fromarray(flat.reshape(array.shape), "RGB").save(out, format=fmt)
        return EmbedResult(self.identifier, out, len(payload), encrypted=False)

    def extract(self, src: Path) -> bytes:
        raise ValueError("image_lsb_scatter requires extract_with_options and a steg key")

    def extract_with_options(
        self,
        src: Path,
        *,
        steg_key: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> bytes:
        del options
        if not steg_key:
            raise ValueError("image_lsb_scatter requires --steg-key or --password")
        flat = self._load_rgb(src).reshape(-1)
        magic, version, length, mask, salt = _read_bootstrap(flat)
        if magic != _MAGIC or version != _VERSION:
            raise ValueError("no supported STEGSCAT bootstrap")
        if mask == 0 or mask & ~0b111:
            raise ValueError("invalid STEGSCAT channel mask")
        candidates = _candidate_positions(flat.size, mask)
        bit_count = length * 8
        if bit_count > len(candidates):
            raise ValueError("declared STEGSCAT length exceeds image capacity")
        positions = np.fromiter(
            sample_positions(
                candidates,
                bit_count,
                steg_key,
                salt,
                context=b"image-lsb-scatter",
            ),
            dtype=np.int64,
            count=bit_count,
        )
        return np.packbits(flat[positions] & 1).tobytes()

    def analyze(self, src: Path) -> AnalysisResult:
        flat = self._load_rgb(src).reshape(-1)
        if flat.size < _BOOTSTRAP_BITS:
            return AnalysisResult(self.identifier, 0, (), None)
        magic, version, length, mask, _ = _read_bootstrap(flat)
        if magic != _MAGIC:
            return AnalysisResult(self.identifier, 0, (), None)
        valid = version == _VERSION and mask != 0 and not mask & ~0b111
        score = 98 if valid else 80
        signal = Signal(
            "stegscat_bootstrap",
            score,
            f"version={version}, payload_length={length}, channel_mask={mask:#05b}",
            category="known_marker",
            evidence="verified" if valid else "strong",
        )
        return AnalysisResult(self.identifier, score, (signal,), None)


def _read_bootstrap(flat: np.ndarray) -> tuple[bytes, int, int, int, bytes]:
    if flat.size < _BOOTSTRAP_BITS:
        raise ValueError("image is too small for a STEGSCAT bootstrap")
    raw = np.packbits(flat[:_BOOTSTRAP_BITS] & 1).tobytes()
    return _BOOTSTRAP.unpack(raw)


def _channel_selection(options: dict[str, Any] | None) -> tuple[str, int]:
    value = str((options or {}).get("channels", "rgb")).lower()
    normalized = "".join(channel for channel in "rgb" if channel in value)
    if not normalized or any(channel not in "rgb" for channel in value):
        raise ValueError("channels must be a combination of r, g and b")
    mask = 0
    for channel in normalized:
        mask |= _CHANNEL_BITS[channel]
    return normalized, mask


def _candidate_positions(size: int, mask: int) -> np.ndarray:
    positions = np.arange(_BOOTSTRAP_BITS, size, dtype=np.int64)
    channels = positions % 3
    allowed = (
        ((mask & 0b001) != 0) & (channels == 0)
        | ((mask & 0b010) != 0) & (channels == 1)
        | ((mask & 0b100) != 0) & (channels == 2)
    )
    return positions[allowed]
